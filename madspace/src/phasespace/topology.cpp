#include "madspace/phasespace/topology.hpp"

#include <algorithm>
#include <numeric>

#include "madspace/util.hpp"

using namespace madspace;

namespace {

bool find_t_vertices(
    const Diagram& diagram,
    std::vector<bool>& visited,
    std::vector<std::size_t>& t_vertices,
    std::vector<Diagram::LineRef>& lines_after_t,
    std::vector<int>& integration_order,
    std::vector<double>& _t_propagator_masses,
    std::vector<double>& _t_propagator_widths,
    std::size_t current_index,
    int source_propagator
) {
    if (visited.at(current_index)) {
        throw std::invalid_argument("Diagram must not have loops");
    }
    visited.at(current_index) = true;

    bool is_t_vertex = false;
    int t_integ_order = 0;
    std::vector<Diagram::LineRef> out_lines;
    for (auto& line_ref : diagram.vertices().at(current_index)) {
        switch (line_ref.type()) {
        case Diagram::incoming:
            if (line_ref.index() == 0) {
                is_t_vertex = true;
            }
            break;
        case Diagram::outgoing:
            out_lines.push_back(line_ref);
            break;
        case Diagram::propagator:
            if (line_ref.index() != source_propagator) {
                auto& vertices = diagram.propagator_vertices().at(line_ref.index());
                if (find_t_vertices(
                        diagram,
                        visited,
                        t_vertices,
                        lines_after_t,
                        integration_order,
                        _t_propagator_masses,
                        _t_propagator_widths,
                        vertices.at(0) == current_index
                            ? vertices.at(1)
                            : vertices.at(0),
                        line_ref.index()
                    )) {
                    is_t_vertex = true;
                    t_integ_order =
                        diagram.propagators().at(line_ref.index()).integration_order;
                    auto& propagator = diagram.propagators().at(line_ref.index());
                    _t_propagator_masses.push_back(propagator.mass);
                    _t_propagator_widths.push_back(propagator.width);
                } else {
                    out_lines.push_back(line_ref);
                }
            }
            break;
        }
    }
    if (is_t_vertex) {
        lines_after_t.insert(lines_after_t.end(), out_lines.begin(), out_lines.end());
        for (std::size_t i = 0; i < out_lines.size(); ++i) {
            t_vertices.push_back(current_index);
            integration_order.push_back(t_integ_order);
        }
    }
    return is_t_vertex;
}

void build_decays(
    const Diagram& diagram,
    std::vector<Topology::Decay>& decays,
    std::vector<std::size_t>& decay_indices,
    std::vector<int>& integration_order,
    std::vector<std::size_t>& outgoing_indices,
    std::size_t vertex_index,
    Diagram::LineRef line_ref,
    std::size_t parent_decay_index
) {
    std::size_t decay_index = decays.size();
    switch (line_ref.type()) {
    case Diagram::outgoing:
        decays.push_back(
            {decay_index,
             parent_decay_index,
             {},
             diagram.outgoing_masses().at(line_ref.index()),
             0.,
             0.,
             0.}
        );
        outgoing_indices.at(line_ref.index()) = decay_index;
        break;
    case Diagram::propagator: {
        auto& propagator = diagram.propagators().at(line_ref.index());
        decays.push_back({
            decay_index,
            parent_decay_index,
            {},
            propagator.mass,
            propagator.width,
            propagator.e_min,
            propagator.e_max,
            propagator.pdg_id,
        });
        decay_indices.push_back(decay_index);
        integration_order.push_back(propagator.integration_order);

        auto& vertices = diagram.propagator_vertices().at(line_ref.index());
        std::size_t next_vertex_index =
            vertices.at(0) == vertex_index ? vertices.at(1) : vertices.at(0);
        for (auto& child_line : diagram.vertices().at(next_vertex_index)) {
            if (child_line.type() == line_ref.type() &&
                child_line.index() == line_ref.index()) {
                continue;
            }
            decays.at(decay_index).child_indices.push_back(decays.size());
            build_decays(
                diagram,
                decays,
                decay_indices,
                integration_order,
                outgoing_indices,
                next_vertex_index,
                child_line,
                decay_index
            );
        }
        break;
    }
    default:
        throw std::logic_error("unreachable");
    }
}

} // namespace

Diagram::LineRef::LineRef(std::string str) {
    if (str.size() < 2) {
        throw std::invalid_argument("Invalid line index");
    }
    switch (str.front()) {
    case 'i':
        _type = Diagram::incoming;
        break;
    case 'o':
        _type = Diagram::outgoing;
        break;
    case 'p':
        _type = Diagram::propagator;
        break;
    default:
        throw std::invalid_argument("Invalid line type");
    }
    _index = std::stoul(str.substr(1));
}

Diagram::Diagram(
    const std::vector<double>& incoming_masses,
    const std::vector<double>& outgoing_masses,
    const std::vector<Propagator>& propagators,
    const std::vector<Vertex>& vertices
) :
    _incoming_masses(incoming_masses),
    _outgoing_masses(outgoing_masses),
    _propagators(propagators),
    _vertices(vertices),
    _incoming_vertices{-1, -1},
    _outgoing_vertices(outgoing_masses.size(), -1),
    _propagator_vertices(propagators.size()) {
    if (incoming_masses.size() != 2) {
        throw std::invalid_argument("Diagram must have two incoming particles");
    }
    if (outgoing_masses.size() < 2) {
        throw std::invalid_argument(
            "Diagram must have at least two outgoing particles"
        );
    }

    std::size_t index = 0;
    for (auto& vertex : vertices) {
        for (auto& line_ref : vertex) {
            switch (line_ref.type()) {
            case Diagram::incoming:
                _incoming_vertices.at(line_ref.index()) = index;
                break;
            case Diagram::outgoing:
                _outgoing_vertices.at(line_ref.index()) = index;
                break;
            case Diagram::propagator:
                _propagator_vertices.at(line_ref.index()).push_back(index);
                break;
            }
        }
        ++index;
    }

    // TODO: run more checks
}

std::ostream& madspace::operator<<(std::ostream& out, const Diagram::LineRef& value) {
    switch (value.type()) {
    case Diagram::incoming:
        out << "i";
        break;
    case Diagram::outgoing:
        out << "o";
        break;
    case Diagram::propagator:
        out << "p";
        break;
    }
    out << value.index();
    return out;
}

std::vector<Topology> Topology::topologies(const Diagram& diagram) {
    // TODO: restructure this to account for subchannels etc
    Topology topo;

    topo._outgoing_indices.resize(diagram.outgoing_masses().size());
    topo._incoming_masses = diagram.incoming_masses();
    topo._outgoing_masses = diagram.outgoing_masses();

    std::vector<bool> visited(diagram.vertices().size());
    std::vector<std::size_t> t_vertices;
    std::vector<Diagram::LineRef> lines_after_t;
    std::vector<int> integration_order;
    find_t_vertices(
        diagram,
        visited,
        t_vertices,
        lines_after_t,
        integration_order,
        topo._t_propagator_masses,
        topo._t_propagator_widths,
        diagram.incoming_vertices().at(1),
        -1
    );

    // sort by integration order and propagator mass, while preventing
    // impossible integration orders
    bool choose_low = false;
    std::size_t index_low = 0, index_high = integration_order.size() - 1;
    while (index_low != index_high) {
        int order_low = integration_order.at(index_low);
        int order_high = integration_order.at(index_high - 1);
        double mass_low = topo._t_propagator_masses.at(index_low);
        double mass_high = topo._t_propagator_masses.at(index_high - 1);
        if (order_low != order_high) {
            choose_low = order_low < order_high;
        } else if (mass_low != mass_high) { // TODO: maybe smarter heuristic here?
            choose_low = mass_low < mass_high;
        }
        if (choose_low) {
            topo._t_integration_order.push_back(index_low);
            ++index_low;
        } else {
            topo._t_integration_order.push_back(index_high - 1);
            --index_high;
        }
    }

    integration_order.clear();
    std::vector<std::size_t> decay_indices;
    // check if diagram is pure s-channel
    if (lines_after_t.size() == 1) {
        build_decays(
            diagram,
            topo._decays,
            decay_indices,
            integration_order,
            topo._outgoing_indices,
            t_vertices.at(0),
            lines_after_t.at(0),
            0
        );
    } else {
        topo._decays.push_back({0, 0, {}, 0., 0.});
        decay_indices.push_back(0);
        integration_order.push_back(0);
        for (auto [t_vertex, line] : zip(t_vertices, lines_after_t)) {
            topo._decays.at(0).child_indices.push_back(topo._decays.size());
            build_decays(
                diagram,
                topo._decays,
                decay_indices,
                integration_order,
                topo._outgoing_indices,
                t_vertex,
                line,
                0
            );
        }
    }

    std::size_t massive_decays = 0;
    for (std::size_t index : decay_indices) {
        if (topo._decays.at(index).mass != 0) {
            ++massive_decays;
        }
    }
    int on_shell_configs = 1 << massive_decays;
    std::vector<Topology> topos;
    for (int i = on_shell_configs - 1; i >= 0; --i) {
        Topology new_topo = topo;
        auto& decays = new_topo._decays;
        std::vector<double> e_min(new_topo.decays().size());
        for (auto [index, m] :
             zip(new_topo.outgoing_indices(), new_topo.outgoing_masses())) {
            e_min.at(index) = m;
        }
        bool possible = true;
        for (std::size_t j = 0;
             std::size_t index : std::views::reverse(decay_indices)) {
            auto& decay = decays.at(index);
            double& e_min_item = e_min.at(index);
            for (std::size_t child_index : decay.child_indices) {
                e_min_item += e_min.at(child_index);
            }
            if (decay.mass == 0) {
                continue;
            }
            if ((i >> j) & 1) {
                if (e_min_item >= decay.mass) {
                    possible = false;
                    break;
                } else {
                    decay.on_shell = true;
                    e_min_item = decay.mass;
                }
            }
            ++j;
        }
        if (!possible) {
            continue;
        }

        bool redundant = false;
        for (auto& other_topo : topos) {
            bool subset = true;
            for (auto [this_decay, other_decay] : zip(decays, other_topo.decays())) {
                if (this_decay.on_shell > other_decay.on_shell) {
                    subset = false;
                    break;
                }
            }
            if (subset) {
                redundant = true;
                break;
            }
        }
        if (redundant) {
            continue;
        }

        std::vector<std::size_t> decay_perm(integration_order.size());
        std::iota(decay_perm.rbegin(), decay_perm.rend(), 0);
        std::stable_sort(
            decay_perm.begin(), decay_perm.end(), [&](std::size_t i, std::size_t j) {
                std::size_t index_i = decay_indices.at(i);
                std::size_t index_j = decay_indices.at(j);
                auto& decay_i = decays.at(index_i);
                auto& decay_j = decays.at(index_j);
                return 4 * integration_order.at(i) - 2 * decay_i.on_shell -
                    (index_i == 0) <
                    4 * integration_order.at(j) - 2 * decay_j.on_shell - (index_j == 0);
            }
        );
        for (std::size_t index : decay_perm) {
            new_topo._decay_integration_order.push_back(decay_indices.at(index));
        }
        topos.push_back(new_topo);
    }

    return topos;
}

Topology::Topology(const Diagram& diagram) { *this = topologies(diagram).at(0); }

std::vector<std::tuple<std::vector<int>, double, double>>
Topology::propagator_momentum_terms(bool only_decays) const {
    std::vector<std::tuple<std::vector<int>, double, double>> ret;
    std::vector<std::vector<std::size_t>> decay_indices(_decays.size());
    std::size_t n_ext = _outgoing_masses.size() + 2;
    std::size_t ext_index = 2;
    for (std::size_t index : _outgoing_indices) {
        decay_indices.at(index).push_back(ext_index);
        ++ext_index;
    }
    for (auto& decay : std::views::reverse(_decays)) {
        if (decay.index == 0) {
            if (_t_integration_order.size() == 0) {
                std::vector<int> factors(n_ext);
                factors.at(0) = 1;
                factors.at(1) = 1;
                ret.push_back({factors, decay.mass, decay.width});
            }
        } else if (decay.child_indices.size() != 0) {
            auto& indices = decay_indices.at(decay.index);
            for (std::size_t index : decay.child_indices) {
                auto& child_indices = decay_indices.at(index);
                indices.insert(
                    indices.end(), child_indices.begin(), child_indices.end()
                );
            }
            std::vector<int> factors(n_ext);
            for (std::size_t index : indices) {
                factors.at(index) = 1;
            }
            ret.push_back({factors, decay.mass, decay.width});
        }
    }
    if (_t_integration_order.size() == 0 || only_decays) {
        return ret;
    }

    std::size_t left_count = 0, right_count = 0;
    auto& child_indices = _decays.at(0).child_indices;
    for (std::size_t index : child_indices) {
        std::size_t current_count = decay_indices.at(index).size();
        if (left_count == 0) {
            left_count = current_count;
        } else {
            right_count += current_count;
        }
    }
    std::size_t child_count = 1;
    for (auto [index, mass, width] :
         zip(child_indices | std::views::drop(1),
             _t_propagator_masses,
             _t_propagator_widths)) {
        std::size_t current_count = decay_indices.at(index).size();
        std::vector<int> factors(n_ext);
        if (left_count <= right_count) {
            factors.at(0) = 1;
            for (std::size_t child_index :
                 child_indices | std::views::take(child_count)) {
                for (std::size_t ext_index : decay_indices.at(child_index)) {
                    factors.at(ext_index) = -1;
                }
            }
        } else {
            factors.at(1) = 1;
            for (std::size_t child_index :
                 child_indices | std::views::drop(child_count)) {
                for (std::size_t ext_index : decay_indices.at(child_index)) {
                    factors.at(ext_index) = -1;
                }
            }
        }
        ret.push_back({factors, mass, width});
        left_count += current_count;
        right_count -= current_count;
        ++child_count;
    }
    return ret;
}
