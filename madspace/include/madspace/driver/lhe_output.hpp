#pragma once

#include <array>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "madspace/driver/random.hpp"
#include "madspace/driver/thread_pool.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/util.hpp"

namespace madspace {

/**
 * One `<header>` block of an LHE file's `<init>` section.
 *
 * A free-form, generator-specific block of extra metadata, written verbatim
 * under `name`. `escape_content` XML-escapes `content` before writing it.
 */
struct LHEHeader {
    /// The `<header>` block's tag name.
    std::string name;
    /// The block's text content.
    std::string content;
    /// Whether to XML-escape `content` before writing it.
    bool escape_content;
};

/**
 * One process line of an LHE file's `<init>` section: cross section and
 * maximum weight for the events tagged with `process_id`.
 */
struct LHEProcess {
    /// Cross section of this subprocess, in pb.
    double cross_section;
    /// Statistical uncertainty of `cross_section`.
    double cross_section_error;
    /// Maximum event weight, used to unweight LHEF v1 weighted events.
    double max_weight;
    /// Process id tagging this subprocess's events.
    int process_id;
};

/**
 * The run-level `<init>` block of an LHE file: beam setup, one @ref
 * LHEProcess per subprocess, and free-form @ref LHEHeader blocks.
 */
struct LHEMeta {
    /// PDG ids of the two beam particles.
    int beam1_pdg_id, beam2_pdg_id;
    /// Energies of the two beams, in GeV.
    double beam1_energy, beam2_energy;
    /// PDF group ids (LHAPDF's `PDFAuthors`) of the two beams.
    int beam1_pdf_authors, beam2_pdf_authors;
    /// PDF set ids (LHAPDF's `LHAPDF ID`) of the two beams.
    int beam1_pdf_id, beam2_pdf_id;
    /// LHEF `IDWTUP` event-weight convention.
    int weight_mode;
    /// One entry per subprocess.
    std::vector<LHEProcess> processes;
    /// Free-form metadata blocks.
    std::vector<LHEHeader> headers;
};

/**
 * One particle line of an LHE `<event>` block, as defined by the Les Houches
 * Accord [1].
 */
struct LHEParticle {
    /// `status_code` of an incoming particle.
    inline static const int status_incoming = -1;
    /// `status_code` of a final-state (outgoing) particle.
    inline static const int status_outgoing = 1;
    /// `status_code` of an intermediate resonance, kept for information only.
    inline static const int status_intermediate_resonance = 2;

    /// Particle Data Group id.
    int pdg_id;
    /// One of `status_incoming`, `status_outgoing` or
    /// `status_intermediate_resonance`.
    int status_code;
    /// 1-based indices (into the same event) of the two mother particles;
    /// `0` if not applicable.
    int mother1, mother2;
    /// Color and anti-color line indices; `0` for a color singlet.
    int color, anti_color;
    /// Momentum components and energy, in GeV.
    double px, py, pz, energy, mass;
    /// Proper lifetime (`c` times `tau`), in mm; `0` if not tracked.
    double lifetime;
    /// Cosine of the spin angle, or `9` if the spin is not specified.
    double spin;
    double cluster_scale;
};

/// Leading-order reweighting inputs of one event, written as the `<mgrwt>`
/// block: the format MadEvent uses, read by `lhe_parser.Event.parse_lo_weight`.
struct LOReweightInfo {
    /// Power of `alpha_s` in the Born matrix element.
    int qcd_power;
    /// Renormalization scale, in GeV.
    double ren_scale;
    /// Whether each beam took part in the hard process.
    bool has_beam1, has_beam2;
    /// PDG ids of the two initial partons.
    int pdg1, pdg2;
    /// Parton momentum fractions of the two beams.
    double x1, x2;
    /// Factorization scales of the two beams, in GeV.
    double fact_scale1, fact_scale2;
    /// MLM pdf reweighting ladder of each beam, as madevent's `<pdfrwt>`
    /// books it: the beam parton at its leading scale, then one entry per
    /// ladder step (line flavour, momentum fraction before the step, scale).
    /// Empty: a single entry from pdg/x/fact_scale.
    std::array<std::vector<int>, 2> ladder_pdgs{};
    /// See @ref ladder_pdgs.
    std::array<std::vector<double>, 2> ladder_x{};
    /// See @ref ladder_pdgs.
    std::array<std::vector<double>, 2> ladder_q{};
    /// Scales of the MLM alpha_s vertices (`<asrwt>`); `qcd_power` counts
    /// them too, the `<rscale>` power is what is left at the
    /// renormalisation scale.
    std::vector<double> asrwt{};
};

/**
 * One `<event>` block of an LHE file, as defined by the Les Houches Accord
 * [1].
 *
 * **References**
 * - [1] E. Boos et al., "Generic user process interface for event
 *   generators", https://arxiv.org/abs/hep-ph/0109068
 */
struct LHEEvent {
    /// Process id, matching one of `LHEMeta::processes`.
    int process_id;
    /// Event weight.
    double weight;
    /// Event scale, in GeV.
    double scale;
    /// QED coupling at the event scale.
    double alpha_qed;
    /// QCD coupling at the event scale.
    double alpha_qcd;
    /// One entry per particle, incoming and outgoing.
    std::vector<LHEParticle> particles;
    /// Whether the MLM clustering scales of the outgoing particles are written
    /// (`pt_clust_<i>` attributes of `<scales>`).
    bool has_cluster_scales;
    /// The renormalisation and factorisation scales, written as the standard
    /// LHEF3 `<scales>` attributes. SCALUP carries only one number, and the two
    /// codes do not agree on which: unwgt.f writes sqrt(max(q2fact)) there,
    /// i.e. mu_F, while this writer puts mu_R. Emitting both explicitly means a
    /// comparison does not have to guess.
    double mu_r;
    double mu_f1;
    double mu_f2;
    bool has_scales;
    /// Optional LHEF v3 weights (`<rwgt>` block); `rwgt_ids` and `rwgt` have
    /// the same length.
    std::vector<int> rwgt_ids;
    std::vector<double> rwgt;
    /// Optional `<mgrwt>` block.
    std::optional<LOReweightInfo> lo_info;

    /// Append this event's LHE text to `buffer`.
    void format_to(std::string& buffer) const;
};

/**
 * Fills in the non-kinematic data of an @ref LHEEvent from sampled indices.
 *
 * The matrix element only returns momenta and the sampled diagram, color,
 * flavor and helicity indices; a `LHECompleter` turns those into the PDG
 * ids, color flow, mother/daughter structure, mass and spin of every @ref
 * LHEParticle, per subprocess. Built once from the topology and color/flavor
 * bookkeeping of every subprocess (@ref SubprocArgs), then reused for every
 * event via @ref complete_event_data.
 */
class LHECompleter {
public:
    /// Per-subprocess topology, permutation, color and flavor bookkeeping
    /// needed to complete its events.
    struct SubprocArgs {
        /// Process id shared by every event of this subprocess.
        int process_id;
        /// Decay topology of each diagram channel. A subprocess without diagrams
        /// has one topology and no permutations and diagram indices.
        std::vector<Topology> topologies;
        /// Final-state particle permutation of each channel.
        nested_vector3<std::size_t> permutations;
        /// Sampled-diagram index available to each channel.
        nested_vector2<std::size_t> diagram_indices;
        /// Color-configuration indices available to each diagram.
        nested_vector3<std::size_t> diagram_color_indices;
        /// Color-line assignment of each sampled color configuration.
        nested_vector2<std::tuple<int, int>> color_flows;
        /// Color representation (singlet, triplet, octet, ...) of each PDG id.
        std::unordered_map<int, int> pdg_color_types;
        /// Per-particle helicity of each sampled helicity configuration.
        nested_vector2<double> helicities;
        /// Per-particle PDG id of each sampled flavor configuration.
        nested_vector3<int> pdg_ids;
        /// Per-diagram PDG override, indexed like `diagram_color_indices`
        /// then by `Topology::Decay::flat_propagator_index`. Falls back to
        /// `Topology::Decay::pdg_id` if empty.
        nested_vector3<int> diagram_propagator_pdgs;
        /// Relative weight of each PDG-id option of each flavor: an option is
        /// written with probability proportional to its weight. Uniform if empty,
        /// or for a flavor whose weights are empty.
        nested_vector2<double> pdg_id_weights;
    };

    /// @param subproc_args  Bookkeeping for every subprocess, indexed like
    ///                      `subprocess_index` in @ref complete_event_data.
    /// @param bw_cutoff     Number of widths within which a propagator is
    ///                      sampled on its Breit-Wigner resonance.
    LHECompleter(const std::vector<SubprocArgs>& subproc_args, double bw_cutoff);
    /**
     * Fill in `event`'s PDG ids, color flow, mother/daughter links, masses
     * and spins from the sampled indices. `event.particles` must already
     * hold the momenta, one entry per particle.
     *
     * @param event              The event to complete, modified in place.
     * @param subprocess_index   Index into the `subproc_args` passed to the
     *                           constructor.
     * @param diagram_index      Sampled diagram, -1 for a subprocess without
     *                           diagrams.
     * @param color_index        Sampled color configuration.
     * @param flavor_index       Sampled flavor assignment.
     * @param helicity_index     Sampled helicity configuration.
     * @param rand_gen           RNG stream, used to sample resonance masses.
     */
    void complete_event_data(
        LHEEvent& event,
        int subprocess_index,
        int diagram_index,
        int color_index,
        int flavor_index,
        int helicity_index,
        MixMaxRandom& rand_gen
    );
    /// Largest particle count over every subprocess.
    std::size_t max_particle_count() const { return _max_particle_count; }
    /// Serialize this completer's bookkeeping to `file`.
    void save(const std::string& file) const;
    /// Load a completer previously written with @ref save.
    static LHECompleter load(const std::string& file);

private:
    struct SubprocData {
        int process_id;
        std::size_t color_offset, pdg_id_offset, helicity_offset, mass_offset;
        std::size_t particle_count, color_count, flavor_count;
        std::size_t diagram_count, helicity_count;
        // 2 for a collision, 1 for a decay. Decides which leading particles are
        // written as initial state and what the outgoing ones point at.
        std::size_t incoming_count;
    };
    struct PropagatorData {
        int pdg_id;
        int momentum_mask;
        int child_prop_mask;
        double mass, width;
    };
    std::vector<SubprocData> _subproc_data;
    std::vector<int> _process_indices;
    std::vector<double> _masses;
    std::vector<std::tuple<int, int>> _colors;
    std::vector<double> _helicities;
    std::vector<std::array<std::size_t, 2>> _pdg_id_and_count;
    std::vector<int> _pdg_ids;
    // flavor slot -> offset of its cumulative option probabilities, if not uniform
    std::unordered_map<std::size_t, std::size_t> _pdg_id_weight_offsets;
    std::vector<double> _pdg_id_cumulative;
    std::unordered_map<std::size_t, std::array<std::size_t, 3>>
        _propagator_index_and_count;
    std::vector<PropagatorData> _propagators;
    std::vector<std::tuple<int, int>> _propagator_colors;
    double _bw_cutoff;
    std::size_t _max_particle_count;

    std::size_t append_helicities(const SubprocArgs& args);
    std::size_t append_colors(const SubprocArgs& args, std::size_t particle_count);
    void append_pdg_ids(const SubprocArgs& args, std::size_t particle_count);
    void append_masses(const Topology& first_topo);
    std::pair<std::size_t, std::size_t>
    build_propagators(std::size_t subproc_index, const SubprocArgs& args);
    void init_propagator_data(
        const Topology& topo,
        const SubprocArgs& args,
        const std::vector<std::size_t>& colors,
        const std::vector<std::size_t>& permutation,
        std::vector<double>& e_min,
        std::vector<int>& momentum_masks,
        std::vector<std::tuple<int, int>>& prop_colors,
        std::vector<int>& resonant_prop_indices
    ) const;
    void find_resonant_propagators(
        const Topology& topo,
        const SubprocArgs& args,
        const std::vector<std::size_t>& colors,
        const std::vector<int>& propagator_pdgs,
        std::size_t prop_offset,
        std::vector<double>& e_min,
        std::vector<int>& momentum_masks,
        std::vector<std::tuple<int, int>>& prop_colors,
        std::vector<int>& resonant_prop_indices
    );
    void record_propagator_colors(
        std::size_t subproc_index,
        std::size_t diag_index,
        const std::vector<std::size_t>& colors,
        std::size_t prop_offset,
        const std::vector<std::tuple<int, int>>& prop_colors,
        const std::vector<int>& resonant_prop_indices
    );

    LHECompleter() = default;
    friend void to_json(nlohmann::json& j, const LHECompleter& lhe_completer);
    friend void from_json(const nlohmann::json& j, LHECompleter& lhe_completer);
    friend void
    to_json(nlohmann::json& j, const LHECompleter::SubprocData& subproc_data);
    friend void
    from_json(const nlohmann::json& j, LHECompleter::SubprocData& subproc_data);
    friend void
    to_json(nlohmann::json& j, const LHECompleter::PropagatorData& prop_data);
    friend void
    from_json(const nlohmann::json& j, LHECompleter::PropagatorData& prop_data);
};

void to_json(nlohmann::json& j, const LHECompleter& lhe_completer);
void from_json(const nlohmann::json& j, LHECompleter& lhe_completer);
void to_json(nlohmann::json& j, const LHECompleter::SubprocData& subproc_data);
void from_json(const nlohmann::json& j, LHECompleter::SubprocData& subproc_data);
void to_json(nlohmann::json& j, const LHECompleter::PropagatorData& prop_data);
void from_json(const nlohmann::json& j, LHECompleter::PropagatorData& prop_data);

/**
 * Streams events to an LHE file.
 *
 * Opens `file_name`, writes the `<init>` header built from `meta`, and lets
 * @ref write append one `<event>` block at a time; the destructor closes the
 * `</LesHouchesEvents>` tag.
 */
class LHEFileWriter {
public:
    /// Opens `file_name` and writes its `<init>` header from `meta`.
    LHEFileWriter(const std::string& file_name, const LHEMeta& meta);
    /// Append `event` as one `<event>` block.
    void write(const LHEEvent& event);
    /// Append `str` to the file verbatim, unescaped.
    void write_string(const std::string& str);
    /// Closes the file, appending the closing `</LesHouchesEvents>` tag.
    ~LHEFileWriter();

private:
    std::ofstream _file_stream;
    std::string _buffer;
};

} // namespace madspace
