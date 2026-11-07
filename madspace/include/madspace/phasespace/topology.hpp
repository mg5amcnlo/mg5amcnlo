#pragma once

#include <array>
#include <ostream>
#include <string>
#include <vector>

namespace madspace {

/**
 * An internal line of a @ref Diagram.
 *
 * Holds the propagator mass, width and PDG id, and the energy window
 * `[e_min, e_max]` its invariant is restricted to. The integer
 * `integration_order` hint orders the `s`-channel invariants, a lower value
 * being sampled first. See @ref Topology.
 */
struct Propagator {
    /// Propagator mass.
    double mass;
    /// Propagator width (0 for a stable / off-shell propagator).
    double width;
    /// Sampling-order hint; lower values are generated first.
    int integration_order;
    /// Lower bound on the propagator energy.
    double e_min;
    /// Upper bound on the propagator energy.
    double e_max;
    /// PDG id of the propagating particle.
    int pdg_id;
    /// Half-width, in units of `width`, of the window where the Breit-Wigner
    /// sampling is flattened (`$`-excluded propagator); `0` for none.
    double flat_window = 0.;
};

/**
 * A tree-level Feynman diagram: external masses, internal @ref Propagator lines
 * and the vertices connecting them.
 *
 * Incoming and outgoing lines and propagators are each 0-indexed. A collision
 * has two incoming lines, the beams 0 and 1; a decay has the single decaying
 * particle as incoming line 0. A `Diagram` is turned into one or
 * more @ref Topology integration channels. @ref PhaseSpaceMapping walks those
 * to build the recursive phase-space decomposition (Sec. 2.2 of [1]).
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 */
class Diagram {
public:
    /// Which family of lines a @ref LineRef points into.
    enum LineType { incoming, outgoing, propagator };
    /// A reference to one line of a @ref Diagram by @ref LineType and 0-based
    /// index. The string form is a type letter (`i`, `o`, `p`) followed by the
    /// index, e.g. `"i0"`, `"o2"`, `"p1"`.
    class LineRef {
    public:
        /// @param type   Which line family.
        /// @param index  0-based index within that family.
        LineRef(LineType type, std::size_t index) : _type(type), _index(index) {}
        /// Parse the `"i0"` / `"o2"` / `"p1"` string form.
        LineRef(std::string str);
        /// The line family.
        LineType type() const { return _type; }
        /// The 0-based index within the family.
        std::size_t index() const { return _index; }

    private:
        LineType _type;
        std::size_t _index;
    };
    /// A vertex, given as the lines meeting at it.
    using Vertex = std::vector<LineRef>;

    /**
     * @param incoming_masses  Masses of the incoming particles: the two beams
     *                         of a collision, or the single decaying particle.
     * @param outgoing_masses  Masses of the outgoing particles.
     * @param propagators      The internal lines.
     * @param vertices         The vertices, each listing its @ref LineRef lines.
     */
    Diagram(
        const std::vector<double>& incoming_masses,
        const std::vector<double>& outgoing_masses,
        const std::vector<Propagator>& propagators,
        const std::vector<Vertex>& vertices
    );

    /// Masses of the incoming particles.
    const std::vector<double>& incoming_masses() const { return _incoming_masses; }
    /// Masses of the outgoing particles.
    const std::vector<double>& outgoing_masses() const { return _outgoing_masses; }
    /// The internal propagator lines.
    const std::vector<Propagator>& propagators() const { return _propagators; }
    /// The vertices.
    const std::vector<Vertex>& vertices() const { return _vertices; }
    /// Index of the vertex each incoming line attaches to. One entry per
    /// incoming particle: two for a collision, one for a decay.
    const std::vector<int>& incoming_vertices() const { return _incoming_vertices; };
    /// Index of the vertex each outgoing line attaches to.
    const std::vector<int>& outgoing_vertices() const { return _outgoing_vertices; };
    /// Whether this diagram is a decay, with a single incoming particle.
    bool is_decay() const { return _incoming_masses.size() == 1; }
    /// For each propagator, the indices of its two end vertices.
    const std::vector<std::vector<std::size_t>>& propagator_vertices() const {
        return _propagator_vertices;
    }

private:
    std::vector<double> _incoming_masses;
    std::vector<double> _outgoing_masses;
    std::vector<Propagator> _propagators;
    std::vector<Vertex> _vertices;
    std::vector<int> _incoming_vertices;
    std::vector<int> _outgoing_vertices;
    std::vector<std::vector<std::size_t>> _propagator_vertices;
};

std::ostream& operator<<(std::ostream& out, const Diagram::LineRef& value);

/**
 * One integration channel derived from a @ref Diagram.
 *
 * Splits the diagram into its `t`-channel momentum transfers (an ordered list
 * of propagators) and its `s`-channel @ref Topology::Decay tree, and records
 * the order in which their invariants are sampled. `Topology::topologies()`
 * enumerates the sub-channels of a diagram, one per set of propagators that can
 * be simultaneously on shell (Sec. 2.2.7 of [1]). The constructor builds the
 * single canonical channel. All particle indices are 0-based, with the two
 * beams at 0 and 1, or the single decaying particle at 0.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2)
 */
class Topology {
public:
    /// `Decay::flat_propagator_index` value for a node with no entry in
    /// `Diagram::propagators()` (an outgoing leaf or the virtual `t`-channel
    /// root).
    static constexpr std::size_t no_propagator = static_cast<std::size_t>(-1);

    /// One node of the `s`-channel decay tree built by @ref Topology.
    struct Decay {
        /// Index of this node in the decay list.
        std::size_t index;
        /// Index of the parent node.
        std::size_t parent_index;
        /// Indices of the child nodes.
        std::vector<std::size_t> child_indices;
        /// Propagator mass of this node.
        double mass;
        /// Propagator width of this node.
        double width;
        /// Lower bound on the node energy.
        double e_min;
        /// Upper bound on the node energy.
        double e_max;
        /// PDG id of the propagator.
        int pdg_id;
        /// Whether this node's invariant is fixed to `mass` (a resonance).
        bool on_shell;
        /// Whether the on-shell condition sits on the integration boundary.
        bool on_shell_boundary;
        /// Position in the flat `Diagram::propagators()` list, or
        /// @ref no_propagator; lets callers apply a per-diagram PDG override.
        std::size_t flat_propagator_index;
        /// Flattened Breit-Wigner window of the propagator, see
        /// @ref Propagator::flat_window.
        double flat_window = 0.;
    };

    /// Enumerate the sub-channels of a diagram, one per on-shell configuration.
    /// @param diagram  The diagram to decompose.
    static std::vector<Topology> topologies(const Diagram& diagram);
    /// Build the single canonical channel of a diagram.
    /// @param diagram  The diagram to decompose.
    Topology(const Diagram& diagram);

    /// Number of space-like momentum transfers.
    std::size_t t_propagator_count() const { return _t_integration_order.size(); }
    /// Sampling order of the `t`-channel propagators.
    const std::vector<std::size_t>& t_integration_order() const {
        return _t_integration_order;
    }
    /// Masses of the `t`-channel propagators.
    const std::vector<double>& t_propagator_masses() const {
        return _t_propagator_masses;
    }
    /// Widths of the `t`-channel propagators.
    const std::vector<double>& t_propagator_widths() const {
        return _t_propagator_widths;
    }
    /// The pdg ids of the `t`-channel propagators, in the same order as
    /// t_propagator_masses(). Needed to tell a QCD splitting off the initial
    /// state from a QED/EW one.
    const std::vector<int>& t_propagator_pdg_ids() const {
        return _t_propagator_pdg_ids;
    }
    /// The `s`-channel decay-tree nodes.
    const std::vector<Decay>& decays() const { return _decays; }
    /**
     * Raise the lower energy bound of one `s`-channel decay node.
     *
     * Hands a cut that bounds the pair this propagator decays into straight to
     * the sampler, so the region the cut forbids is never generated. Lowering
     * the bound is a no-op.
     *
     * @param index  Index of the node in @ref decays().
     * @param e_min  The new lower bound.
     */
    void raise_decay_e_min(std::size_t index, double e_min) {
        auto& decay = _decays.at(index);
        if (e_min > decay.e_min) {
            decay.e_min = e_min;
        }
    }
    /// Sampling order of the `s`-channel invariants.
    const std::vector<std::size_t>& decay_integration_order() const {
        return _decay_integration_order;
    }
    /// Decay-tree node index of each outgoing particle, in diagram order.
    const std::vector<std::size_t>& outgoing_indices() const {
        return _outgoing_indices;
    }
    /// Masses of the incoming particles.
    const std::vector<double>& incoming_masses() const { return _incoming_masses; }
    /// Masses of the outgoing particles.
    const std::vector<double>& outgoing_masses() const { return _outgoing_masses; }
    /// Whether this channel is a decay, with a single incoming particle.
    /// A decay has no `t`-channel chain and a root virtuality fixed by the
    /// mass of the decaying particle.
    bool is_decay() const { return _incoming_masses.size() == 1; }
    /// For each propagator, a tuple `(factors, mass, width)`. `factors` has one
    /// entry per external particle (incoming first): +1 or -1 for the momenta
    /// summed to form the propagator momentum, 0 otherwise.
    /// @param only_decays  Restrict the result to the `s`-channel decay nodes.
    std::vector<std::tuple<std::vector<int>, double, double>>
    propagator_momentum_terms(bool only_decays = false) const;
    /// Human-readable description of the channel.
    std::string to_string() const;

private:
    Topology() = default;

    std::vector<std::size_t> _t_integration_order;
    std::vector<double> _t_propagator_masses;
    std::vector<double> _t_propagator_widths;
    std::vector<int> _t_propagator_pdg_ids;
    std::vector<Decay> _decays;
    std::vector<std::size_t> _decay_integration_order;
    std::vector<std::size_t> _outgoing_indices;
    std::vector<double> _incoming_masses;
    std::vector<double> _outgoing_masses;
};

} // namespace madspace
