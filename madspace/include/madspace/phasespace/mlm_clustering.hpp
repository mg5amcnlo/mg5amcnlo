#pragma once

#include <unordered_map>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/topology.hpp"

namespace madspace {

// Number of words each clustering transition occupies in the compiled state
// machine: (data, next_offset, trace_data).
inline constexpr int state_machine_item_size = 3;

// How a clustering scale is assigned to an outgoing jet.
enum class JetScaleScheme {
    // The scale of the vertex at which the leg itself was emitted, i.e. the
    // softest clustering it takes part in.
    emission = 0,
    // The hardest vertex on the parton line the leg belongs to, which is the
    // scale at which that line was produced. This is what madevent writes, and
    // what Pythia uses as the shower starting scale for the leg when MLM
    // merging turns on Beams:setProductionScalesFromLHEF.
    production = 1,
};

// How the renormalisation and factorisation scales are read off the clustering
// history once it has been chosen.
enum class ScaleScheme {
    // Geometric mean of every clustering scale, with the non-QCD ones replaced
    // by the largest, and a single factorisation scale taken as the smallest
    // QCD clustering scale capped at mu_R.
    clustering_mean = 0,
    // What madevent does: follow each beam's parton line through the
    // clustering and take
    //     mu_R   = (s[jlast1] s[jcentral1] s[jlast2] s[jcentral2])^(1/4)
    //     mu_F,b = sqrt(s[jlast_b] s[jcentral_b])
    // where jlast is the last initial-state clustering while the beam line is
    // still a jet, and jcentral the last one while it is still coloured. The
    // two beams get different factorisation scales.
    madevent = 1,
};

// How the beam parton line is decided to carry on past a vertex. Only read by
// ScaleScheme::madevent.
enum class PartonLineScheme {
    // From the flavour of the object emitted at that vertex alone.
    flavor = 0,
    // From goodjet of reweight.f: a line counts as a parton line only while
    // every clustering it has been through was a jet vertex, so one non-jet
    // vertex anywhere in its history stops it for good. This is what madevent
    // does, and it stops beam lines earlier than flavor does.
    goodjet = 1,
};

// How alpha_s is evaluated for a merged event.
enum class AlphasScheme {
    // One coupling at the event's renormalisation scale, for the whole event.
    none = 0,
    // alphas(pt_i) at each clustering vertex where a parton is produced, which
    // is what madevent does (the rewgt loop in reweight.f). A merged sample
    // without it is short by one factor per emission.
    per_vertex = 1,
    // One coupling at the geometric mean of those vertex scales, raised to the
    // number of them: the same idea with a single scale for the whole ladder.
    geometric_mean = 2,
};

// Which clustering measure scores a candidate final-state pair. Only the pairs
// the two definitions disagree on are affected: a non-resonant final-state
// clustering whose mother is massless with one massive and one massless
// daughter (q* > q W, g* > g h), or whose mother is massive with two massless
// daughters. Initial-state and resonant clusterings are the same in both.
enum class ClusteringMeasure {
    // cluster_scale of Template/NLO/SubProcesses/cluster.f, the FxFx
    // definition: sqrt(|p_j . (p_i + p_j)|) / 2 for the massless-mother case and
    // the invariant mass for the massive-mother one.
    fxfx = 0,
    // DJ of Template/LO/Source/kin_functions.f, which madevent's LO cluster.f
    // uses for every final-state pair: a massless-massive pair scores the
    // massless one's transverse mass (times 1 + 1e-6), and anything else the
    // kt measure with the larger mass squared added.
    madevent = 1,
};

// Which diagrams a clustering history may follow.
enum class ClusteringHistory {
    // The smallest-measure history over the union of every diagram, with the
    // properties of each line taken from the first diagram that has it.
    all_diagrams = 0,
    // Pick one diagram per event, with probability proportional to its
    // single-diagram weight |A_i|^2, and cluster along that diagram alone: its
    // own line flavours, vertices and merging jets. What an integration
    // channel is to madevent's reclustering, without its first-event
    // bookkeeping.
    diagram = 1,
    // setclscales of Template/LO/SubProcesses/reweight.f: keep the history
    // over every diagram unless it calls a different number of legs merging
    // jets than the picked diagram does, and fall back to that diagram then.
    madevent = 2,
};

class MLMClustering : public FunctionGenerator {
public:
    MLMClustering(
        std::vector<Topology> topologies,
        nested_vector3<std::size_t> permutations,
        nested_vector2<std::size_t> diagram_indices,
        // Collider energy. An outgoing leg that no QCD clustering assigned a
        // scale to is reported at this value rather than at zero, so that an
        // MLM veto can never trip on it.
        double cm_energy,
        JetScaleScheme jet_scale_scheme = JetScaleScheme::production,
        // Unlike jet_scale_scheme, this choice changes the cross section.
        ScaleScheme scale_scheme = ScaleScheme::clustering_mean,
        // Signed color representation per pdg id, as exported in the
        // subprocess metadata. Used to follow a parton line through the
        // clustering; falls back to the Standard Model assignment for a pdg id
        // that is not listed.
        std::unordered_map<int, int> pdg_color_types = {},
        // Generation-level merging cut, the counterpart of madevent's xqcut.
        // A clustering that emitted a jet below this scale sets xqcut_weight
        // to 0. Zero disables it.
        double xqcut = 0,
        double bw_cutoff = 15,
        // D of the final-state kt measure; 1 as madevent's run_card d and FxFx.
        double jet_radius = 1.0,
        bool hadronic = true,
        // pdg ids of the external particles, in leg order. When empty, every
        // clustering is assumed to be a QCD splitting between jets.
        std::vector<int> external_pdg_ids = {},
        int max_jet_flavor = 4,
        PartonLineScheme parton_line_scheme = PartonLineScheme::goodjet,
        AlphasScheme alphas_scheme = AlphasScheme::per_vertex,
        // Re-evaluate the beam densities along the clustering ladder instead of
        // once at the factorisation scale, which is what madevent does for a
        // merged sample (pdfwgt in the run card, hidden and on by default).
        bool pdf_reweighting = true,
        ClusteringMeasure clustering_measure = ClusteringMeasure::fxfx,
        // Anything but all_diagrams also compiles one state machine per
        // diagram, which build_along_diagram walks.
        ClusteringHistory clustering_history = ClusteringHistory::all_diagrams,
        // With pdf_reweighting, also walk the densities along the clustering
        // ladder under scale_scheme = clustering_mean: the densities (their
        // lowered leading scale, the ladder ratios and the per-beam scale the
        // ladder ends at) are then madevent's, while mu_R, the alpha_s vertex
        // scales and the jet bookkeeping stay clustering_mean's. On by default:
        // without it clustering_mean takes both densities at the softest
        // clustering scale, which is what moves its merged rates near qcut
        // away from madevent's.
        bool clustering_mean_pdf_ladder = true
    );

    // The same outputs as the function itself, for a history restricted to
    // the given diagram index (as the matrix element numbers diagrams). A
    // diagram no topology was supplied for falls back to the history over
    // every diagram.
    NamedVector<Value>
    build_along_diagram(FunctionBuilder& fb, Value momenta, Value diagram) const;
    // The same from an offset into cluster_state_machine, as
    // diagram_start_states lists them, for a caller that keeps its own table.
    // flavor_index and leg_flavors restrict the walk to the clusterings that
    // exist for the flavours of each event: leg_flavors holds, row-major, the
    // abs pdg id of every external leg for each flavour index. Left out, every
    // clustering of the union stays allowed.
    NamedVector<Value> build_from_start_state(
        FunctionBuilder& fb,
        Value momenta,
        Value start_state,
        Value flavor_index = Value(),
        const std::vector<me_int_t>& leg_flavors = {}
    ) const;
    // The history over every diagram, restricted to the flavours of each event
    // as build_from_start_state does.
    NamedVector<Value> build_with_flavors(
        FunctionBuilder& fb,
        Value momenta,
        Value flavor_index,
        const std::vector<me_int_t>& leg_flavors
    ) const;
    // Whether any clustering of the compiled state machines depends on the
    // flavours of the event. Without one, the flavours change nothing.
    bool has_flavor_dependent_clusterings() const { return _has_same_flavor; }

    // The compiled clustering state machine, in the flat encoding the kernel
    // walks. Exposed so that its structure can be checked directly.
    const std::vector<me_int_t>& cluster_state_machine() const {
        return _cluster_state_machine;
    }
    const std::vector<double>& external_masses() const { return _external_masses; }
    const std::vector<double>& bw_masses() const { return _bw_masses; }
    const std::vector<double>& bw_widths() const { return _bw_widths; }
    AlphasScheme alphas_scheme() const { return _alphas_scheme; }
    bool pdf_reweighting() const { return _pdf_reweighting; }
    bool clustering_mean_pdf_ladder() const { return _clustering_mean_pdf_ladder; }
    ScaleScheme scale_scheme() const { return _scale_scheme; }
    ClusteringMeasure clustering_measure() const { return _clustering_measure; }
    ClusteringHistory clustering_history() const { return _clustering_history; }
    // Offset of each diagram's own state machine in cluster_state_machine,
    // indexed by diagram index, or 0 for a diagram that has none. Empty under
    // ClusteringHistory::all_diagrams.
    const std::vector<me_int_t>& diagram_start_states() const {
        return _diagram_start_states;
    }
    // The flavours the pdf reweighting asks for that are neither the gluon nor
    // a beam's own, in the order the kernel's flavour classes index them. The
    // consumer turns these, the gluon and the per-channel beam flavours into
    // the density it evaluates; see the flavour-class comment in
    // mlm_clustering.cpp.
    const std::vector<int>& pdf_absolute_pdgs() const { return _pdf_absolute_pdgs; }

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;
    NamedVector<Value> build_kernel(
        FunctionBuilder& fb,
        Value momenta,
        Value start_state,
        me_int_t history_mode,
        Value flavor_index,
        const std::vector<me_int_t>& leg_flavors
    ) const;

    std::vector<me_int_t> _cluster_state_machine;
    std::vector<me_int_t> _diagram_start_states;
    std::vector<double> _external_masses;
    std::vector<double> _bw_masses;
    std::vector<double> _bw_widths;
    double _cm_energy;
    JetScaleScheme _jet_scale_scheme;
    ScaleScheme _scale_scheme;
    PartonLineScheme _parton_line_scheme;
    AlphasScheme _alphas_scheme;
    bool _pdf_reweighting;
    bool _clustering_mean_pdf_ladder;
    ClusteringMeasure _clustering_measure;
    ClusteringHistory _clustering_history;
    std::vector<int> _pdf_absolute_pdgs;
    bool _has_same_flavor = false;
    int _beam_flags;
    int _jet_leg_mask;
    double _xqcut;
    double _bw_cutoff;
    double _jet_radius;
    bool _hadronic;
};

// The clustering along one diagram as a function of its own, taking the
// diagram index as a second argument. The integrand calls
// build_from_start_state directly; this is how that path is reached from Python.
class MLMClusteringAlongDiagram : public FunctionGenerator {
public:
    MLMClusteringAlongDiagram(const MLMClustering& clustering);

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    MLMClustering _clustering;
};

// The clustering over every diagram restricted to the flavours of each event,
// taking the flavour index as a second argument and the table of leg flavours
// (see MLMClustering::build_with_flavors) at construction. The integrand calls
// build_with_flavors directly; this is how that path is reached from Python.
class MLMClusteringWithFlavors : public FunctionGenerator {
public:
    MLMClusteringWithFlavors(
        const MLMClustering& clustering, std::vector<me_int_t> leg_flavors
    );

private:
    NamedVector<Value> build_function_impl(
        FunctionBuilder& fb, const NamedVector<Value>& args
    ) const override;

    MLMClustering _clustering;
    std::vector<me_int_t> _leg_flavors;
};

} // namespace madspace
