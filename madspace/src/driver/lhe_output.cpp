#include "madspace/driver/lhe_output.hpp"

#include "madspace/util.hpp"

#include <algorithm>
#include <cstdio>
#include <span>

using namespace madspace;
using json = nlohmann::json;

namespace {

std::size_t cantor_pairing(std::size_t i, std::size_t j) {
    return (i + j) * (i + j + 1) / 2 + i;
}

std::size_t cantor_pairing(std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return cantor_pairing(cantor_pairing(i, j), cantor_pairing(k, l));
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
            "{:4} {:4} {:4} {:4} {:4} {:4} {:+.10e} {:+.10e} {:+.10e} {:.10e} {:.10e} "
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
    buffer += "</event>\n";
}

LHECompleter::LHECompleter(
    const std::vector<SubprocArgs>& subproc_args, double bw_cutoff
) :
    _bw_cutoff(bw_cutoff), _max_particle_count(0) {
    std::size_t color_offset = 0, pdg_id_offset = 0, helicity_offset = 0,
                mass_offset = 0;
    _max_particle_count = 0;
    for (std::size_t subproc_index = 0; auto& args : subproc_args) {
        std::size_t particle_count = args.helicities.at(0).size();
        if (_max_particle_count < particle_count) {
            _max_particle_count = particle_count;
        }

        for (auto& helicities : args.helicities) {
            if (helicities.size() != particle_count) {
                throw std::invalid_argument("Invalid number of helicities");
            }
            _helicities.insert(_helicities.end(), helicities.begin(), helicities.end());
        }

        std::size_t matrix_flavor_count = args.color_flows.size();
        std::size_t color_count = args.color_flows.at(0).size();
        for (auto& flavor_color_flows : args.color_flows) {
            if (flavor_color_flows.size() != color_count) {
                throw std::invalid_argument("Invalid number of colors per flavor");
            }
            for (auto& color_flows : flavor_color_flows) {
                if (color_flows.size() != particle_count) {
                    throw std::invalid_argument(
                        "Invalid number of particles per color"
                    );
                }
                _colors.insert(_colors.end(), color_flows.begin(), color_flows.end());
            }
        }

        if (args.pdg_ids.size() != args.matrix_flavor_indices.size()) {
            throw std::invalid_argument(
                "pdg_ids and matrix_flavor_indices must have same size"
            );
        }

        for (auto [pdg_id_options, matrix_flavor_index] :
             zip(args.pdg_ids, args.matrix_flavor_indices)) {
            if (pdg_id_options.size() == 0) {
                throw std::invalid_argument(
                    "Must provide at least one option per flavor index"
                );
            }
            if (matrix_flavor_index >= matrix_flavor_count) {
                throw std::invalid_argument("Invalid matrix element flavor index");
            }
            _pdg_id_index_and_count.push_back(
                {_pdg_ids.size(), matrix_flavor_index, pdg_id_options.size()}
            );
            for (auto& pdg_ids : pdg_id_options) {
                if (pdg_ids.size() != particle_count) {
                    throw std::invalid_argument("Invalid number of particles ids");
                }
                _pdg_ids.insert(_pdg_ids.end(), pdg_ids.begin(), pdg_ids.end());
            }
        }

        auto& first_topo = args.topologies.at(0);
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

        std::vector<double> e_min;
        std::vector<int> momentum_masks;
        std::vector<std::tuple<int, int>> prop_colors;
        std::vector<int> decay_colors, decay_anti_colors;
        std::vector<int> resonant_prop_indices;

        std::size_t diagram_count = 0;
        for (auto [topo, permutations, diag_indices, diag_colors] :
             zip(args.topologies,
                 args.permutations,
                 args.diagram_indices,
                 args.diagram_color_indices)) {
            std::size_t prop_offset = _propagators.size();
            for (auto [permutation, diag_index, colors] :
                 zip(permutations, diag_indices, diag_colors)) {
                diagram_count += diag_indices.size();

                for (std::size_t matrix_flavor_index = 0;
                     matrix_flavor_index < matrix_flavor_count;
                     ++matrix_flavor_index) {
                    e_min.clear();
                    e_min.resize(topo.decays().size());
                    momentum_masks.clear();
                    momentum_masks.resize(topo.decays().size());
                    prop_colors.clear();
                    prop_colors.resize(topo.decays().size() * colors.size());
                    resonant_prop_indices.clear();
                    resonant_prop_indices.resize(topo.decays().size(), -1);
                    for (auto [index, mass, perm_index] :
                         zip(topo.outgoing_indices(),
                             topo.outgoing_masses(),
                             std::span(permutation.begin() + 2, permutation.end()))) {
                        e_min.at(index) = mass;
                        momentum_masks.at(index) = 1 << perm_index;
                        for (std::size_t i = 0; std::size_t color_index : colors) {
                            prop_colors.at(colors.size() * index + i) =
                                args.color_flows.at(matrix_flavor_index)
                                    .at(color_index)
                                    .at(perm_index);
                            ++i;
                        }
                    }
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
                            int child_prop_index =
                                resonant_prop_indices.at(child_index);
                            if (child_prop_index != -1) {
                                child_prop_mask |= 1 << child_prop_index;
                            }
                        }
                        if (e_min_item >= decay.mass) {
                            continue;
                        }

                        resonant_prop_indices.at(decay.index) =
                            _propagators.size() - prop_offset;
                        _propagators.push_back({
                            .pdg_id = decay.pdg_id,
                            .momentum_mask = momentum_mask,
                            .child_prop_mask = child_prop_mask,
                            .mass = decay.mass,
                            .width = decay.width,
                        });
                        int color_type = args.pdg_color_types.at(decay.pdg_id);
                        for (std::size_t i = 0; std::size_t color_index : colors) {
                            decay_colors.clear();
                            decay_anti_colors.clear();
                            for (std::size_t child_index : decay.child_indices) {
                                auto [color, anti_color] =
                                    prop_colors.at(colors.size() * child_index + i);
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
                            auto& prop_color =
                                prop_colors.at(colors.size() * decay.index + i);
                            if (color_type == 1) {
                                if (decay_colors.size() > 0 ||
                                    decay_anti_colors.size() > 0) {
                                    throw std::runtime_error(
                                        "Incompatible with color singlet"
                                    );
                                }
                                prop_color = {0, 0};
                            } else if (color_type == 3) {
                                if (decay_colors.size() != 1 ||
                                    decay_anti_colors.size() > 0) {
                                    throw std::runtime_error(
                                        "Incompatible with color triplet"
                                    );
                                }
                                prop_color = {decay_colors.at(0), 0};
                            } else if (color_type == -3) {
                                if (decay_colors.size() > 0 ||
                                    decay_anti_colors.size() != 1) {
                                    throw std::runtime_error(
                                        "Incompatible with anti-color triplet"
                                    );
                                }
                                prop_color = {0, decay_anti_colors.at(0)};
                            } else if (color_type == 8) {
                                if (decay_colors.size() != 1 ||
                                    decay_anti_colors.size() != 1) {
                                    throw std::runtime_error(
                                        "Incompatible with color octet"
                                    );
                                }
                                prop_color = {
                                    decay_colors.at(0), decay_anti_colors.at(0)
                                };
                            } else {
                                throw std::runtime_error("Invalid color type");
                            }
                            ++i;
                        }
                    }
                    std::size_t prop_count = _propagators.size() - prop_offset;
                    if (prop_count > 0) {
                        for (std::size_t i = 0; std::size_t color : colors) {
                            std::size_t prop_color_offset = _propagator_colors.size();
                            for (std::size_t j = resonant_prop_indices.size();
                                 int prop_index :
                                 std::views::reverse(resonant_prop_indices)) {
                                --j;
                                if (prop_index != -1) {
                                    _propagator_colors.push_back(
                                        prop_colors.at(colors.size() * j + i)
                                    );
                                }
                            }
                            _propagator_index_and_count[cantor_pairing(
                                subproc_index, diag_index, color, matrix_flavor_index
                            )] = {prop_offset, prop_color_offset, prop_count};
                            ++i;
                        }
                    }
                }
            }
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
            .matrix_flavor_count = matrix_flavor_count,
            .diagram_count = diagram_count,
            .helicity_count = args.helicities.size(),
        });

        helicity_offset += particle_count * args.helicities.size();
        color_offset += particle_count * color_count * matrix_flavor_count;
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
    std::mt19937& rand_gen
) {
    auto& subproc_data = _subproc_data.at(subprocess_index);
    if (event.particles.size() != subproc_data.particle_count) {
        throw std::runtime_error("Invalid particle number for subprocess");
    }
    if (diagram_index < 0 || diagram_index >= subproc_data.diagram_count) {
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

    std::size_t color_offset =
        subproc_data.color_offset + subproc_data.particle_count * color_index;
    std::size_t helicity_offset =
        subproc_data.helicity_offset + subproc_data.particle_count * helicity_index;
    std::size_t mass_offset = subproc_data.mass_offset;

    auto [pdg_index, matrix_flavor_index, pdg_count] =
        _pdg_id_index_and_count.at(subproc_data.pdg_id_offset + flavor_index);
    std::uniform_int_distribution<std::size_t> dist(0, pdg_count - 1);
    std::size_t pdg_random = dist(rand_gen);
    std::size_t pdg_offset = pdg_index + subproc_data.particle_count * pdg_random;

    for (std::size_t particle_index = 0; auto& particle : event.particles) {
        std::tie(particle.color, particle.anti_color) =
            _colors.at(color_offset + particle_index);
        particle.pdg_id = _pdg_ids.at(pdg_offset + particle_index);
        if (particle_index < 2) {
            particle.status_code = -1;
            particle.mother1 = 0;
            particle.mother2 = 0;
        } else {
            particle.status_code = 1;
            particle.mother1 = 1;
            particle.mother2 = 2;
        }
        particle.mass = _masses.at(mass_offset + particle_index);
        particle.lifetime = 0;
        particle.spin = _helicities.at(helicity_offset + particle_index);
        ++particle_index;
    }

    auto find_propagators = _propagator_index_and_count.find(cantor_pairing(
        subprocess_index, diagram_index, color_index, matrix_flavor_index
    ));
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
        double m_min = propagator.mass - _bw_cutoff * propagator.width;
        double m_max = propagator.mass + _bw_cutoff * propagator.width;
        if (m2 > m_min * m_min && m2 < m_max * m_max) {
            auto [color, anti_color] = prop_color;
            resonant_prop_mask |= 1 << prop_index;
            new_particles.push_back({
                .pdg_id = propagator.pdg_id,
                .status_code = 2,
                .mother1 = 1,
                .mother2 = 2,
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
        event.particles.begin() + 2, new_particles.rbegin(), new_particles.rend()
    );
    for (std::size_t prop_index = prop_count, res_index = 0;
         auto& propagator : std::views::reverse(
             std::span(
                 _propagators.begin() + prop_offset,
                 _propagators.begin() + prop_offset + prop_count
             )
         )) {
        --prop_index;
        if (resonant_prop_mask & (1 << prop_index)) {
            int child_prop_mask = propagator.child_prop_mask;
            for (int child_prop_index = prop_index - 1, child_res_index = res_index + 1;
                 child_prop_index >= 0;
                 --child_prop_index) {
                if (resonant_prop_mask & (1 << child_prop_index)) {
                    auto& child_particle = event.particles.at(child_res_index + 2);
                    child_particle.mother1 = res_index + 3;
                    child_particle.mother2 = res_index + 3;
                    ++child_res_index;
                }
            }

            int momentum_mask = propagator.momentum_mask >> 2;
            for (auto& particle : std::span(
                     event.particles.begin() + 2 + new_particles.size(),
                     event.particles.end()
                 )) {
                if (momentum_mask & 1) {
                    particle.mother1 = res_index + 3;
                    particle.mother2 = res_index + 3;
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
        {"pdg_id_index_and_count", lhe_completer._pdg_id_index_and_count},
        {"pdg_ids", lhe_completer._pdg_ids},
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
    lhe_completer._pdg_id_index_and_count =
        j.at("pdg_id_index_and_count").get<std::vector<std::array<std::size_t, 3>>>();
    lhe_completer._pdg_ids = j.at("pdg_ids").get<std::vector<int>>();
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
        subproc_data.matrix_flavor_count,
        subproc_data.diagram_count,
        subproc_data.helicity_count,
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
        .matrix_flavor_count = j.at(8).get<std::size_t>(),
        .diagram_count = j.at(9).get<std::size_t>(),
        .helicity_count = j.at(10).get<std::size_t>(),
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
