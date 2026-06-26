#pragma once

#include <array>
#include <ostream>
#include <string>
#include <vector>

namespace madspace {

struct Propagator {
    double mass;
    double width;
    int integration_order;
    double e_min;
    double e_max;
    int pdg_id;
};

class Diagram {
public:
    enum LineType { incoming, outgoing, propagator };
    class LineRef {
    public:
        LineRef(LineType type, std::size_t index) : _type(type), _index(index) {}
        LineRef(std::string str);
        LineType type() const { return _type; }
        std::size_t index() const { return _index; }

    private:
        LineType _type;
        std::size_t _index;
    };
    using Vertex = std::vector<LineRef>;

    Diagram(
        const std::vector<double>& incoming_masses,
        const std::vector<double>& outgoing_masses,
        const std::vector<Propagator>& propagators,
        const std::vector<Vertex>& vertices
    );

    const std::vector<double>& incoming_masses() const { return _incoming_masses; }
    const std::vector<double>& outgoing_masses() const { return _outgoing_masses; }
    const std::vector<Propagator>& propagators() const { return _propagators; }
    const std::vector<Vertex>& vertices() const { return _vertices; }
    const std::array<int, 2>& incoming_vertices() const { return _incoming_vertices; };
    const std::vector<int>& outgoing_vertices() const { return _outgoing_vertices; };
    const std::vector<std::vector<std::size_t>>& propagator_vertices() const {
        return _propagator_vertices;
    }

private:
    std::vector<double> _incoming_masses;
    std::vector<double> _outgoing_masses;
    std::vector<Propagator> _propagators;
    std::vector<Vertex> _vertices;
    std::array<int, 2> _incoming_vertices;
    std::vector<int> _outgoing_vertices;
    std::vector<std::vector<std::size_t>> _propagator_vertices;
};

std::ostream& operator<<(std::ostream& out, const Diagram::LineRef& value);

class Topology {
public:
    struct Decay {
        std::size_t index;
        std::size_t parent_index;
        std::vector<std::size_t> child_indices;
        double mass;
        double width;
        double e_min;
        double e_max;
        int pdg_id;
        bool on_shell;
    };

    static std::vector<Topology> topologies(const Diagram& diagram);
    Topology(const Diagram& diagram);

    std::size_t t_propagator_count() const { return _t_integration_order.size(); }
    const std::vector<std::size_t>& t_integration_order() const {
        return _t_integration_order;
    }
    const std::vector<double>& t_propagator_masses() const {
        return _t_propagator_masses;
    }
    const std::vector<double>& t_propagator_widths() const {
        return _t_propagator_widths;
    }
    const std::vector<Decay>& decays() const { return _decays; }
    const std::vector<std::size_t>& decay_integration_order() const {
        return _decay_integration_order;
    }
    const std::vector<std::size_t>& outgoing_indices() const {
        return _outgoing_indices;
    }
    const std::vector<double>& incoming_masses() const { return _incoming_masses; }
    const std::vector<double>& outgoing_masses() const { return _outgoing_masses; }
    std::vector<std::tuple<std::vector<int>, double, double>>
    propagator_momentum_terms(bool only_decays = false) const;

private:
    Topology() = default;

    std::vector<std::size_t> _t_integration_order;
    std::vector<double> _t_propagator_masses;
    std::vector<double> _t_propagator_widths;
    std::vector<Decay> _decays;
    std::vector<std::size_t> _decay_integration_order;
    std::vector<std::size_t> _outgoing_indices;
    std::vector<double> _incoming_masses;
    std::vector<double> _outgoing_masses;
};

} // namespace madspace
