#include "madspace/phasespace/mlm_clustering.hpp"

#include <bit>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <unordered_map>

using namespace madspace;

namespace {

// Largest number of external particles the kernel can handle (mlm.hpp
// N_EXT_MAX), and the number of bits available for a clustering mask.
constexpr std::size_t n_ext_max = 12;

// Properties of the line a clustering mask stands for: the external particle
// itself for a single-bit mask, the propagator recombining them otherwise.
struct LineMeta {
    double mass = 0.;
    double width = 0.;
    int pdg_id = 0;
    bool known = false;
};

// Mirrors isjet() in Template/LO/SubProcesses/reweight.f: a final state
// particle a parton shower would turn into a jet, and hence one that gets its
// own clustering scale in the LHE output. 81 is the merged-flavor placeholder
// used by flavor grouping.
bool is_jet_pdg(int pdg_id, int max_jet_flavor) {
    int a = std::abs(pdg_id);
    return (a >= 1 && a <= max_jet_flavor) || a == 21 || a == 81;
}

// Mirrors is_octet() in Template/LO/SubProcesses/reweight.f: a colour octet,
// which in the QCD sector means the gluon. The iqjets bookkeeping treats an
// emitted gluon as a jet unconditionally - it has a soft singularity the shower
// must be allowed to fill - and uses octet-ness again to decide which leg of a
// non-jet vertex to demote.
bool is_octet_pdg(int pdg_id, const std::unordered_map<int, int>& pdg_color_types);

// Mirrors isparton() in Template/LO/SubProcesses/reweight.f: a line that can
// be a beam parton, i.e. one the pdf reweighting can follow back into a beam.
// Wider than is_jet_pdg: madevent tests against max(asrwgtflavor, maxjetflavor)
// and asrwgtflavor defaults to 5, so the b is a parton even when it is not a
// jet.
bool is_parton_pdg(int pdg_id) {
    int a = std::abs(pdg_id);
    return (a >= 1 && a <= 5) || a == 21 || a == 81;
}

// Which density the pdf reweighting should ask for on a given line.
//
// The flavour cannot simply be baked in. One MLMClustering serves every flavour
// channel of a subprocess, and external_pdg_ids is a single representative
// assignment with the signs stripped (clean_pids in launch.py), so "the beam is
// a u" is only true of one channel out of many. What is stable across the
// channels is the *structure*: whether a line is a gluon, and whether it still
// carries the flavour the beam came in with. Those two are baked here as
// classes; the integrand resolves them per event once the flavour has been
// sampled.
//
// Anything else - a quark line whose flavour changed along the way, which for
// W+jets means past the W vertex - falls back to an absolute pdg taken from the
// representative assignment. That is a real approximation, but it is confined
// to lines whose flavour the representative was already guessing at, and a pdf
// ratio between two nearby scales is far more sensitive to quark-versus-gluon
// than to which quark.
constexpr int flavor_class_none = -1;
constexpr int flavor_class_gluon = 0;
constexpr int flavor_class_beam1 = 1;
constexpr int flavor_class_beam2 = 2;
constexpr int flavor_class_first_absolute = 3;
// _MERGED_PID_REPRESENTATIVE in launch.py, which is what external_pdg_ids has
// already been mapped through by the time it gets here.
constexpr int merged_quark_representative = 1;

// Color representation of a line, as the absolute value of the signed color of
// the model (1 for a singlet, 3 for a (anti)triplet, 8 for an octet, 6 for a
// sextet). The
// merged-flavor placeholders that survive in propagator pdg ids stand for the
// particles clean_pids() maps them to in launch.py: 81 is a quark, 82 an
// electron and 83 a neutrino.
int color_rep(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    if (auto search = pdg_color_types.find(pdg_id); search != pdg_color_types.end()) {
        return std::abs(search->second);
    }
    int a = std::abs(pdg_id);
    if ((a >= 1 && a <= 6) || a == 81) {
        return 3;
    }
    if (a == 21) {
        return 8;
    }
    return 1;
}

// Whether a line carries color. Used to decide whether a *vertex* is a QCD
// splitting, which is what selects the clustering scales entering mu_R/mu_F.
bool is_colored_pdg(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    return color_rep(pdg_id, pdg_color_types) != 1;
}

bool is_octet_pdg(int pdg_id, const std::unordered_map<int, int>& pdg_color_types) {
    return color_rep(pdg_id, pdg_color_types) == 8;
}

// How the parton line of the mother of a clustering continues into its two
// daughters, so that a clustering scale can be booked onto the external legs
// the line ends at. Mirrors ipartupdate() in
// Template/LO/SubProcesses/reweight.f, which decides the same thing from the
// color representations (its leading pdg-equality tests give the same answer
// as the color ones for every case they cover).
enum TraceMode {
    trace_first = 0,  // the line continues into daughter 1
    trace_second = 1, // ... into daughter 2
    trace_harder = 2, // ... into whichever daughter is harder (decided per event)
    trace_both = 3,   // the mother stands for both daughters' lines
};

TraceMode trace_mode_for(int color_in, int color_1, int color_2) {
    if (color_in == 8) {
        // g -> g g, or an octet splitting to two octets: follow the harder one
        if (color_1 == 8 && color_2 == 8) {
            return trace_harder;
        }
        // g -> q qbar: both daughters carry the line on
        if (color_1 == 3 && color_2 == 3) {
            return trace_both;
        }
        // g -> g H and friends: follow the colored daughter
        if (color_1 == 8) {
            return trace_first;
        }
        if (color_2 == 8) {
            return trace_second;
        }
    } else if (color_in == 3) {
        // an epsilon^ijk vertex, 3 -> 3 3, is treated like a photon emission
        // and follows the first daughter
        if (color_1 == 3 && color_2 == 3) {
            return trace_first;
        }
        // q -> q g, q -> q Z/H/W, and the exotic q -> q' S variants
        if (color_1 == 3) {
            return trace_first;
        }
        if (color_2 == 3) {
            return trace_second;
        }
    } else if (color_in == 6) {
        if (color_1 == 3 && color_2 == 3) {
            return trace_both;
        }
    } else if (color_in == 1) {
        return trace_both;
    }
    // Nothing matched. The Fortran stops the run here; keeping the line on the
    // first daughter is wrong but local, and only costs the clustering scale of
    // one leg rather than the whole event.
    return trace_first;
}

struct CompileContext {
    const nested_vector2<int>& valid_diags;
    const std::vector<LineMeta>& mask_meta;
    std::vector<double>& bw_masses;
    std::vector<double>& bw_widths;
    std::map<std::pair<double, double>, int>& bw_indices;
    const std::unordered_map<int, int>& pdg_color_types;
    bool have_pdg_ids;
    int max_jet_flavor;
    // |pdg| of each beam, to recognise a line that still carries the flavour
    // the beam came in with.
    int beam_pdg[2];
    // The absolute flavours needed beyond gluon and the two beams, in the
    // order the classes above index them.
    std::vector<int>& absolute_pdgs;
    // The propagator pdg ids each diagram puts on a line, keyed by (diagram,
    // mask): what the union of mask_meta keeps only one of.
    const std::map<std::pair<int, int>, std::vector<int>>& line_pdgs;
    // Set once any clustering is marked same_flavor.
    bool& has_same_flavor;
};

// Whether a line is a fermion whose flavour a g, a, Z or h vertex preserves.
bool is_flavored_fermion_pdg(int pdg_id) {
    int a = std::abs(pdg_id);
    return (a >= 1 && a <= 6) || (a >= 11 && a <= 16);
}

// Whether every diagram among diags that has the line mask puts a g, a, Z or h
// on it, so that the two bare external legs it joins must share their flavour.
// madevent's cluster.f checks the same per graph, see flavor_filter there.
bool joins_same_flavor(
    const CompileContext& ctx,
    int mask,
    int mask_1,
    int mask_2,
    const std::vector<int>& diags
) {
    if (!ctx.have_pdg_ids || std::popcount(static_cast<unsigned int>(mask_1)) != 1 ||
        std::popcount(static_cast<unsigned int>(mask_2)) != 1 ||
        !is_flavored_fermion_pdg(ctx.mask_meta.at(mask_1).pdg_id) ||
        !is_flavored_fermion_pdg(ctx.mask_meta.at(mask_2).pdg_id)) {
        return false;
    }
    bool any = false;
    for (int diag : diags) {
        auto search = ctx.line_pdgs.find({diag, mask});
        if (search == ctx.line_pdgs.end()) {
            return false;
        }
        for (int pdg_id : search->second) {
            int a = std::abs(pdg_id);
            if (a != 21 && a != 22 && a != 23 && a != 25) {
                return false;
            }
            any = true;
        }
    }
    return any;
}

// The flavour class of one line, allocating a new absolute class if it needs
// one. flavor_class_none for anything the reweighting cannot follow.
//
// on_beam is the beam a line sits on, or -1 for one that sits on neither. It
// only matters for the flavour-merged placeholder 81, which says "a light
// quark" without saying which: on a quark beam that is the beam's own flavour
// and resolves exactly, and anywhere else it falls back to the representative
// quark that clean_pids in launch.py would have picked.
int flavor_class(CompileContext& ctx, int pdg_id, int on_beam = -1) {
    if (!ctx.have_pdg_ids || !is_parton_pdg(pdg_id)) {
        return flavor_class_none;
    }
    int a = std::abs(pdg_id);
    if (a == 21) {
        return flavor_class_gluon;
    }
    if (a == 81) {
        if (on_beam >= 0 && std::abs(ctx.beam_pdg[on_beam]) != 21 &&
            is_parton_pdg(ctx.beam_pdg[on_beam])) {
            return flavor_class_beam1 + on_beam;
        }
        a = merged_quark_representative;
    }
    for (int beam = 0; beam < 2; ++beam) {
        if (a == std::abs(ctx.beam_pdg[beam])) {
            return flavor_class_beam1 + beam;
        }
    }
    auto found = std::find(ctx.absolute_pdgs.begin(), ctx.absolute_pdgs.end(), a);
    if (found == ctx.absolute_pdgs.end()) {
        ctx.absolute_pdgs.push_back(a);
        return flavor_class_first_absolute +
            static_cast<int>(ctx.absolute_pdgs.size()) - 1;
    }
    return flavor_class_first_absolute +
        static_cast<int>(found - ctx.absolute_pdgs.begin());
}

using StateKey = std::pair<std::vector<int>, std::vector<int>>;
struct StateItem {
    int next_state;
    int particle1;
    int particle2;
    int mass_index;
    bool massive_in;
    bool massive_out1;
    bool massive_out2;
    bool is_qcd;
    bool is_jet1;
    bool is_jet2;
    TraceMode trace_mode;
    // Flavour of the *mother*. madevent's scale definition needs to know where
    // a beam's parton line stops being a jet and where it stops being coloured.
    bool is_jet_in;
    bool is_colored_in;
    // isjetvx() needs to know whether the mother repeats one daughter's
    // flavour - a gluon emission off a quark line, say - and the iqjets
    // demotion needs to know whether the mother is an octet.
    bool is_octet_in;
    bool mother_is_daughter1;
    bool mother_is_daughter2;
    // None of the three lines carries colour - a W W Z or h h h vertex. The
    // demotion leaves those alone rather than treating them as a QCD vertex
    // that failed the jet test.
    bool all_colorless;
    // Flavour class of the mother, i.e. of the beam line as it stands after
    // this clustering. flavor_class_none ends the pdf reweighting chain,
    // which is also what stops madevent following ibeam(j) any further.
    int flavor_class_in;
    // The two daughters are bare external fermions joined by a g, a, Z or h in
    // every diagram left, so the clustering only exists for an event in which
    // they carry the same flavour (TRACE_SAME_FLAVOR in kernels/mlm.hpp).
    bool same_flavor;
};

// 1-based index into the Breit-Wigner tables handed to the kernel, or 0 for
// "never treat this clustering as resonant".
int breit_wigner_index(CompileContext& ctx, const LineMeta& meta) {
    if (!meta.known || meta.width <= 0. || meta.mass <= 0.) {
        return 0;
    }
    auto key = std::pair{meta.mass, meta.width};
    if (auto search = ctx.bw_indices.find(key); search != ctx.bw_indices.end()) {
        return search->second;
    }
    int index = static_cast<int>(ctx.bw_masses.size()) + 1;
    ctx.bw_masses.push_back(meta.mass);
    ctx.bw_widths.push_back(meta.width);
    ctx.bw_indices[key] = index;
    return index;
}

StateItem make_state_item(
    CompileContext& ctx,
    int next_state,
    int mask_in,
    int mask_1,
    int mask_2,
    const std::vector<int>& diags
) {
    int particle1 = std::countr_zero(static_cast<unsigned int>(mask_1));
    int particle2 = std::countr_zero(static_cast<unsigned int>(mask_2));
    auto& meta_in = ctx.mask_meta.at(mask_in);
    auto& meta_1 = ctx.mask_meta.at(mask_1);
    auto& meta_2 = ctx.mask_meta.at(mask_2);
    // The state machine always names the merged pseudo-particle by the lowest
    // bit of the combined mask, so particle1 < particle2 and an initial-state
    // (t-channel) clustering is exactly the case particle1 < 2.
    bool is_initial = particle1 < 2;

    // A t-channel propagator is spacelike and can never go on shell, so the
    // resonance test is only meaningful for a final-state clustering.
    int mass_index = is_initial ? 0 : breit_wigner_index(ctx, meta_in);

    int color_1 = color_rep(meta_1.pdg_id, ctx.pdg_color_types);
    int color_2 = color_rep(meta_2.pdg_id, ctx.pdg_color_types);
    int color_in = color_rep(meta_in.pdg_id, ctx.pdg_color_types);
    if (meta_in.pdg_id == 0) {
        // The propagator's flavor was not supplied (the topology fixtures used
        // by the tests carry masses only), so color_rep would call it a
        // singlet. Infer it from the daughters instead, which is unambiguous
        // for a QCD vertex.
        if (color_1 == 3 && color_2 == 3) {
            color_in = 8;
        } else if (color_1 == 8 && color_2 == 8) {
            color_in = 8;
        } else if (color_1 == 3 || color_2 == 3) {
            color_in = 3;
        }
    }
    // An initial-state clustering carries the beam line on, which is daughter 1
    // by construction, so the color-based table does not apply to it.
    TraceMode trace_mode =
        is_initial ? trace_first : trace_mode_for(color_in, color_1, color_2);

    bool is_qcd, is_jet1, is_jet2;
    if (!ctx.have_pdg_ids) {
        // No flavor information supplied: fall back to assuming every
        // clustering is a massless QCD splitting between jets.
        is_qcd = is_jet1 = is_jet2 = true;
    } else {
        // A vertex is a QCD splitting when all three of its lines carry color.
        // Checking the parent matters: q qbar -> Z has two colored children and
        // a colorless parent, and is not a QCD vertex.
        //
        // A pdg id of 0 means the propagator's flavor was not supplied (the
        // topology fixtures used by the tests carry masses only). Fall back to
        // the two children then: a splitting of two colored lines is QCD unless
        // the parent says otherwise, which is also the only test available for
        // t-channel lines whose flavor changes along the chain.
        is_qcd = color_1 != 1 && color_2 != 1 && (meta_in.pdg_id == 0 || color_in != 1);
        // Only final state particles get a clustering scale in the LHE output,
        // so a beam leg is never a jet here.
        is_jet1 = !is_initial && is_jet_pdg(meta_1.pdg_id, ctx.max_jet_flavor);
        is_jet2 = is_jet_pdg(meta_2.pdg_id, ctx.max_jet_flavor);
    }

    bool same_flavor = joins_same_flavor(ctx, mask_in, mask_1, mask_2, diags);
    if (same_flavor) {
        ctx.has_same_flavor = true;
    }

    // "The mother repeats a daughter's flavour" compares |pdg|: the external
    // legs come without their sign (clean_pids in launch.py) while a propagator
    // keeps the one MG5 gave it, so the t~* of t~* > t~ g is -6 against the t~
    // leg's 6. For a light quark is_jet_in makes the vertex a jet vertex anyway;
    // for a top this is the only way it is one, and without it the gluon was
    // written at sqrt(s) and never counted as a merging jet.
    bool mother_is_daughter1 =
        meta_in.pdg_id != 0 && std::abs(meta_in.pdg_id) == std::abs(meta_1.pdg_id);
    bool mother_is_daughter2 =
        meta_in.pdg_id != 0 && std::abs(meta_in.pdg_id) == std::abs(meta_2.pdg_id);

    return {
        .next_state = next_state,
        .particle1 = particle1,
        .particle2 = particle2,
        .mass_index = mass_index,
        .massive_in = meta_in.mass > 0.,
        .massive_out1 = meta_1.mass > 0.,
        .massive_out2 = meta_2.mass > 0.,
        .is_qcd = is_qcd,
        .is_jet1 = is_jet1,
        .is_jet2 = is_jet2,
        .trace_mode = trace_mode,
        .is_jet_in =
            !ctx.have_pdg_ids || is_jet_pdg(meta_in.pdg_id, ctx.max_jet_flavor),
        .is_colored_in = !ctx.have_pdg_ids || color_in != 1,
        .is_octet_in =
            !ctx.have_pdg_ids || is_octet_pdg(meta_in.pdg_id, ctx.pdg_color_types),
        .mother_is_daughter1 = mother_is_daughter1,
        .mother_is_daughter2 = mother_is_daughter2,
        .all_colorless =
            ctx.have_pdg_ids && color_in == 1 && color_1 == 1 && color_2 == 1,
        .flavor_class_in =
            flavor_class(ctx, meta_in.pdg_id, is_initial ? particle1 : -1),
        .same_flavor = same_flavor,
    };
}

// A terminal state: nothing left to cluster, only the diagrams the history
// can have come from, one of which the walk picks at random.
void append_terminal_items(
    std::vector<StateItem>& state, const std::vector<int>& diags
) {
    for (int diag_index : diags) {
        state.push_back({
            .next_state = diag_index,
            .particle1 = 0,
            .particle2 = 0,
            .mass_index = 0,
            .massive_in = false,
            .massive_out1 = false,
            .massive_out2 = false,
            .is_qcd = false,
            .is_jet1 = false,
            .is_jet2 = false,
            .trace_mode = trace_first,
            .is_jet_in = false,
            .is_colored_in = false,
            .is_octet_in = false,
            .mother_is_daughter1 = false,
            .mother_is_daughter2 = false,
            .all_colorless = false,
            .flavor_class_in = flavor_class_none,
            .same_flavor = false,
        });
    }
}

void find_clusterings(
    CompileContext& ctx,
    const std::vector<int>& particle_masks,
    const std::vector<int>& diagrams,
    nested_vector2<StateItem>& states,
    std::map<StateKey, int>& state_map,
    std::set<int>& dead_states,
    int prev_index
) {
    int n_masks = particle_masks.size();
    for (int i = 0; i < n_masks - 1; ++i) {
        for (int j = i + 1; j < n_masks; ++j) {
            int mask_i = particle_masks.at(i), mask_j = particle_masks.at(j);
            int mask = mask_i | mask_j;
            auto& valid = ctx.valid_diags.at(mask);
            if (valid.size() == 0) {
                continue;
            }
            std::vector<int> new_masks, new_diags;
            new_masks.insert(
                new_masks.end(), particle_masks.begin(), particle_masks.begin() + i
            );
            new_masks.push_back(mask);
            new_masks.insert(
                new_masks.end(),
                particle_masks.begin() + i + 1,
                particle_masks.begin() + j
            );
            new_masks.insert(
                new_masks.end(), particle_masks.begin() + j + 1, particle_masks.end()
            );
            for (int index : diagrams) {
                if (std::find(valid.begin(), valid.end(), index) != valid.end()) {
                    new_diags.push_back(index);
                }
            }
            // valid_diags says some diagram allows this clustering, not that
            // any of the diagrams still in play does. With none left there is
            // no clustering history to continue, and following it anyway
            // builds a state the walk can never leave.
            if (new_diags.size() == 0) {
                continue;
            }

            bool is_terminal = new_masks.size() == 3;
            StateKey key;
            if (is_terminal) {
                key = {{}, new_diags};
            } else {
                key = {new_masks, new_diags};
            }
            int index;
            bool is_new_state;
            if (auto search = state_map.find(key); search != state_map.end()) {
                index = search->second;
                is_new_state = false;
                // A state already known to be a dead end stays one.
                if (dead_states.contains(index)) {
                    continue;
                }
            } else {
                index = states.size();
                state_map[key] = index;
                states.push_back({});
                is_new_state = true;
            }

            // Only expand a state the first time it is reached. Expanding it
            // again would append a second copy of the same transitions, once
            // per path leading to it.
            if (is_new_state) {
                if (is_terminal) {
                    append_terminal_items(states.at(index), new_diags);
                } else {
                    find_clusterings(
                        ctx, new_masks, new_diags, states, state_map, dead_states, index
                    );
                }
                // The expansion may have found nothing: every clustering left
                // is unsupported by the remaining diagrams. Drop the
                // transition rather than pointing it at a state the walk
                // cannot leave.
                if (states.at(index).size() == 0) {
                    dead_states.insert(index);
                    continue;
                }
            }

            states.at(prev_index)
                .push_back(
                    make_state_item(ctx, index, mask, mask_i, mask_j, new_diags)
                );
        }
    }
}

// Record the properties of the line a mask stands for. Different diagrams can
// put different propagators on the same partition of external legs (a photon
// and a Z, say); keep the one with a width, since that is the one the
// resonance test is about.
void set_mask_meta(std::vector<LineMeta>& mask_meta, int mask, LineMeta meta) {
    auto& current = mask_meta.at(mask);
    if (!current.known || (current.width <= 0. && meta.width > 0.)) {
        meta.known = true;
        current = meta;
    }
}

// Append compiled states to the flat state machine. A clustering state is a
// run of (data, next_offset, trace_data) triples ending at the triple whose data
// has the is_last bit set; a terminal state is a count followed by that many
// diagram indices. Returns the offset each state landed at.
std::vector<int> append_state_layout(
    std::vector<me_int_t>& machine, const nested_vector2<StateItem>& states
) {
    std::vector<int> first_indices;
    first_indices.reserve(states.size());
    for (int offset = static_cast<int>(machine.size()); auto& state : states) {
        first_indices.push_back(offset);
        if (state.size() == 0) {
            // a dropped dead end: nothing points at it, so it takes no space
        } else if (state.at(0).particle1 == 0 && state.at(0).particle2 == 0) {
            offset += 1 + state.size();
        } else {
            offset += state_machine_item_size * state.size();
        }
    }
    for (auto& state : states) {
        if (state.size() == 0) {
            continue;
        }
        if (state.at(0).particle1 == 0 && state.at(0).particle2 == 0) {
            machine.push_back(state.size());
            for (auto& item : state) {
                machine.push_back(item.next_state);
            }
        } else {
            for (auto& item : state) {
                machine.push_back(
                    (item.particle1 << 0) + (item.particle2 << 8) +
                    (item.mass_index << 16) + (item.massive_in << 24) +
                    (item.massive_out1 << 25) + (item.massive_out2 << 26) +
                    (item.is_qcd << 27) + (item.is_jet1 << 28) + (item.is_jet2 << 29) +
                    ((&item == &state.back()) << 30)
                );
                machine.push_back(first_indices.at(item.next_state));
                machine.push_back(
                    static_cast<int>(item.trace_mode) + (item.is_jet_in << 2) +
                    (item.is_colored_in << 3) + ((item.flavor_class_in + 1) << 4) +
                    (item.is_octet_in << 12) + (item.mother_is_daughter1 << 13) +
                    (item.mother_is_daughter2 << 14) + (item.all_colorless << 15) +
                    (item.same_flavor << 16)
                );
            }
        }
    }
    return first_indices;
}

} // namespace

MLMClustering::MLMClustering(
    std::vector<Topology> topologies,
    nested_vector3<std::size_t> permutations,
    nested_vector2<std::size_t> diagram_indices,
    double cm_energy,
    JetScaleScheme jet_scale_scheme,
    ScaleScheme scale_scheme,
    std::unordered_map<int, int> pdg_color_types,
    double xqcut,
    double bw_cutoff,
    double jet_radius,
    bool hadronic,
    std::vector<int> external_pdg_ids,
    int max_jet_flavor,
    PartonLineScheme parton_line_scheme,
    AlphasScheme alphas_scheme,
    bool pdf_reweighting,
    ClusteringMeasure clustering_measure,
    ClusteringHistory clustering_history,
    bool clustering_mean_pdf_ladder
) :
    FunctionGenerator(
        "MLMClustering",
        {{"momenta",
          batch_four_vec_array(topologies.at(0).outgoing_masses().size() + 2)}},
        {{"ren_scale", batch_float},
         {"fact_scale1", batch_float},
         {"fact_scale2", batch_float},
         {"outgoing_scales",
          batch_float_array(topologies.at(0).outgoing_masses().size())},
         {"diagram_index", batch_int},
         {"xqcut_weight", batch_float},
         {"alphas_scales",
          batch_float_array(topologies.at(0).outgoing_masses().size() - 1)},
         {"alphas_weight", batch_float},
         // The scale the beam density itself is evaluated at. Not the same as
         // fact_scale under pdf reweighting: madevent drops the density low on
         // the clustering ladder and walks it back up, so the reported
         // factorisation scale and the one the pdf is asked for differ.
         {"pdf_scale1", batch_float},
         {"pdf_scale2", batch_float},
         // One slot per clustering that can put a pdf ratio on a beam line,
         // plus one per beam for the 2 -> 1 root, which lies on both.
         {"pdf_rw_flavor",
          batch_int_array(topologies.at(0).outgoing_masses().size() + 1)},
         {"pdf_rw_x", batch_float_array(topologies.at(0).outgoing_masses().size() + 1)},
         {"pdf_rw_q_num",
          batch_float_array(topologies.at(0).outgoing_masses().size() + 1)},
         {"pdf_rw_q_den",
          batch_float_array(topologies.at(0).outgoing_masses().size() + 1)},
         {"pdf_rw_active",
          batch_float_array(topologies.at(0).outgoing_masses().size() + 1)},
         {"pdf_rw_beam",
          batch_float_array(topologies.at(0).outgoing_masses().size() + 1)},
         // the beam line's momentum fraction before each step's z, which
         // madevent books for its systematics
         {"pdf_rw_x_prev",
          batch_float_array(topologies.at(0).outgoing_masses().size() + 1)}}
    ),
    _cm_energy(cm_energy),
    _jet_scale_scheme(jet_scale_scheme),
    _scale_scheme(scale_scheme),
    _parton_line_scheme(parton_line_scheme),
    _alphas_scheme(alphas_scheme),
    _pdf_reweighting(pdf_reweighting),
    _clustering_mean_pdf_ladder(clustering_mean_pdf_ladder),
    _clustering_measure(clustering_measure),
    _clustering_history(clustering_history),
    _beam_flags(0),
    _jet_leg_mask(0),
    _xqcut(xqcut),
    _bw_cutoff(bw_cutoff),
    _jet_radius(jet_radius),
    _hadronic(hadronic) {
    std::size_t n_ext = topologies.at(0).outgoing_masses().size() + 2;
    if (n_ext > n_ext_max) {
        throw std::invalid_argument(
            std::format(
                "MLM clustering supports at most {} external particles, got {}",
                n_ext_max,
                n_ext
            )
        );
    }
    bool have_pdg_ids = external_pdg_ids.size() > 0;
    if (have_pdg_ids && external_pdg_ids.size() != n_ext) {
        throw std::invalid_argument(
            std::format(
                "expected {} external pdg ids, got {}", n_ext, external_pdg_ids.size()
            )
        );
    }

    // madevent seeds its walk with the flavour of each beam: qcdline from
    // isqcd(beam) and partonline from the same. Bake those two bits per beam
    // in, since they are a property of the subprocess.
    for (int beam = 0; beam < 2; ++beam) {
        int pdg = have_pdg_ids ? external_pdg_ids.at(beam) : 21;
        if (color_rep(pdg, pdg_color_types) != 1) {
            _beam_flags |= 1 << (2 * beam);
        }
        if (is_jet_pdg(pdg, max_jet_flavor)) {
            _beam_flags |= 1 << (2 * beam + 1);
        }
        // The flavour class the pdf reweighting starts each beam line at,
        // stored as class + 1 so that 0 means "not a parton" and the chain
        // never starts. A beam that is a parton is by construction either the
        // gluon or its own class, so this needs no lookup.
        int beam_class = !have_pdg_ids || !is_parton_pdg(pdg) ? flavor_class_none
            : std::abs(pdg) == 21
            ? flavor_class_gluon
            : flavor_class_beam1 + beam;
        _beam_flags |= (beam_class + 1) << (8 * (beam + 1));
    }

    for (std::size_t leg = 0; leg < n_ext; ++leg) {
        int pdg = have_pdg_ids ? external_pdg_ids.at(leg) : 21;
        if (is_jet_pdg(pdg, max_jet_flavor)) {
            _jet_leg_mask |= 1 << leg;
        }
        // The high half carries octet-ness for the same leg. The kernel needs
        // both and n_ext_max is 12, so one word holds them without another
        // kernel argument.
        if (is_octet_pdg(pdg, pdg_color_types)) {
            _jet_leg_mask |= 1 << (leg + 16);
        }
    }

    nested_vector2<int> valid_diags(1 << n_ext);
    std::vector<LineMeta> mask_meta(1 << n_ext);
    std::vector<int> particle_masks;
    std::vector<int> all_diags;
    // The lines of each diagram on its own, as (mask, properties): what the
    // union above merges away. Only needed for the per-diagram histories.
    std::vector<std::pair<std::size_t, std::vector<std::pair<int, LineMeta>>>>
        diagram_lines;
    auto add_line = [&](std::size_t diag_index, int mask, const LineMeta& meta) {
        valid_diags.at(mask).push_back(diag_index);
        diagram_lines.back().second.push_back({mask, meta});
    };

    // The kernel indexes momenta and external_masses by external leg, while a
    // topology's masses are indexed by topology slot, so the permutation has to
    // be applied. Start from a sentinel so the first diagram fills the table and
    // every later one is checked against it.
    _external_masses.assign(n_ext, -1.);

    // create a list of all diagram indices that are possible for a given clustering,
    // where a binary encoding of the clustering is used
    for (auto [topo, permutations, diag_indices] :
         zip(topologies, permutations, diagram_indices)) {
        auto& incoming_masses = topo.incoming_masses();
        auto& outgoing_masses = topo.outgoing_masses();
        for (auto [permutation, diag_index] : zip(permutations, diag_indices)) {
            all_diags.push_back(diag_index);
            diagram_lines.push_back({diag_index, {}});
            particle_masks.assign(topo.decays().size(), 0);
            for (std::size_t i = 2; i < permutation.size(); ++i) {
                particle_masks.at(topo.outgoing_indices().at(permutation.at(i) - 2)) = 1
                    << i;
            }

            // permutation maps external leg -> topology slot, so the mass of leg
            // i is the mass of the slot it is mapped to.
            for (std::size_t leg = 0; leg < n_ext; ++leg) {
                double mass = leg < 2
                    ? incoming_masses.at(leg)
                    : outgoing_masses.at(permutation.at(leg) - 2);
                double& stored = _external_masses.at(leg);
                if (stored < 0.) {
                    stored = mass;
                } else if (stored != mass) {
                    // The kernel keeps a single external mass per leg, so a
                    // permutation that moves a mass between legs would silently
                    // give the wrong clustering measure.
                    throw std::invalid_argument(
                        std::format(
                            "MLM clustering needs the mass of external leg {} to be "
                            "the "
                            "same for every diagram and permutation, got {} and {}",
                            leg,
                            stored,
                            mass
                        )
                    );
                }
                set_mask_meta(
                    mask_meta,
                    1 << leg,
                    {.mass = mass,
                     .width = 0.,
                     .pdg_id = have_pdg_ids ? external_pdg_ids.at(leg) : 0}
                );
            }

            bool has_t_channel = topo.t_integration_order().size() > 0;
            for (auto& decay : std::views::reverse(topo.decays())) {
                if (decay.child_indices.size() == 0) {
                    continue;
                }
                if (decay.index == 0 && has_t_channel) {
                    continue;
                }
                if (decay.child_indices.size() > 2) {
                    throw std::logic_error("does not support 1->n decays with n > 2");
                }
                int mask =
                    (particle_masks.at(decay.child_indices.at(0)) |
                     particle_masks.at(decay.child_indices.at(1)));
                particle_masks.at(decay.index) = mask;
                LineMeta meta{
                    .mass = decay.mass, .width = decay.width, .pdg_id = decay.pdg_id
                };
                add_line(diag_index, mask, {meta.mass, meta.width, meta.pdg_id, true});
                set_mask_meta(mask_meta, mask, meta);
            }

            if (!has_t_channel) {
                continue;
            }

            // t-channel lines: find_t_vertices collects the propagators from beam 1
            // on, so the k-th mask accumulated from beam 1 (mask 1) is propagator
            // k, and the k-th from beam 2 (mask 2) is propagator t_count - 1 - k.
            auto& t_masses = topo.t_propagator_masses();
            auto& t_pdg_ids = topo.t_propagator_pdg_ids();
            std::size_t t_count = t_masses.size();
            auto& t_children = topo.decays().at(0).child_indices;
            std::size_t k = 0;
            for (int mask = 1; std::size_t index : t_children) {
                mask |= particle_masks.at(index);
                if (k < t_count) {
                    // width stays 0: a spacelike propagator is never resonant
                    LineMeta meta{
                        .mass = t_masses.at(k), .width = 0., .pdg_id = t_pdg_ids.at(k)
                    };
                    add_line(diag_index, mask, {meta.mass, 0., meta.pdg_id, true});
                    set_mask_meta(mask_meta, mask, meta);
                } else {
                    add_line(diag_index, mask, {});
                }
                ++k;
            }
            k = 0;
            for (int mask = 2; std::size_t index : std::views::reverse(t_children)) {
                mask |= particle_masks.at(index);
                if (k < t_count) {
                    LineMeta meta{
                        .mass = t_masses.at(t_count - 1 - k),
                        .width = 0.,
                        .pdg_id = t_pdg_ids.at(t_count - 1 - k)
                    };
                    add_line(diag_index, mask, {meta.mass, 0., meta.pdg_id, true});
                    set_mask_meta(mask_meta, mask, meta);
                } else {
                    add_line(diag_index, mask, {});
                }
                ++k;
            }
        }
    }

    std::map<std::pair<int, int>, std::vector<int>> line_pdgs;
    for (auto& [diag_index, lines] : diagram_lines) {
        for (auto& [mask, meta] : lines) {
            line_pdgs[{static_cast<int>(diag_index), mask}].push_back(meta.pdg_id);
        }
    }

    std::vector<int> masks;
    masks.reserve(n_ext);
    for (int i = 0; i < n_ext; ++i) {
        masks.push_back(1 << i);
    }
    nested_vector2<StateItem> states{{}};
    std::map<StateKey, int> state_map;
    std::map<std::pair<double, double>, int> bw_indices;
    state_map[{{masks}, {all_diags}}] = 0;
    CompileContext ctx{
        .valid_diags = valid_diags,
        .mask_meta = mask_meta,
        .bw_masses = _bw_masses,
        .bw_widths = _bw_widths,
        .bw_indices = bw_indices,
        .pdg_color_types = pdg_color_types,
        .have_pdg_ids = have_pdg_ids,
        .max_jet_flavor = max_jet_flavor,
        .beam_pdg =
            {have_pdg_ids ? external_pdg_ids.at(0) : 21,
             have_pdg_ids ? external_pdg_ids.at(1) : 21},
        .absolute_pdgs = _pdf_absolute_pdgs,
        .line_pdgs = line_pdgs,
        .has_same_flavor = _has_same_flavor,
    };
    std::set<int> dead_states;
    if (n_ext == 3) {
        // A 2 -> 1 process (the 0-jet sample of p p > z + jets) has nothing to
        // cluster: the walk starts at a terminal state, and the kernel takes
        // the scales off the single outgoing line, as cluster.f's "special
        // 2 -> 1 case" does.
        append_terminal_items(states.at(0), all_diags);
    } else {
        find_clusterings(ctx, masks, all_diags, states, state_map, dead_states, 0);
    }

    if (states.at(0).size() == 0) {
        throw std::logic_error(
            "MLM clustering found no valid clustering for this process"
        );
    }
    for (std::size_t i = 0; auto& state : states) {
        // Dead ends are unreachable: no transition points at them any more, but
        // they still occupy a slot in `states` so that the offsets stay stable.
        if (state.size() == 0 && !dead_states.contains(static_cast<int>(i))) {
            throw std::logic_error(
                "MLM clustering reached a state with no valid clustering left"
            );
        }
        ++i;
    }

    append_state_layout(_cluster_state_machine, states);

    if (clustering_history == ClusteringHistory::all_diagrams) {
        return;
    }

    // One state machine per diagram, appended after the one over every
    // diagram, so that offset 0 can keep meaning "no diagram of its own". Each
    // is compiled exactly as the union is, only from that diagram's lines:
    // where the union takes a line's flavour from whichever diagram came first
    // - the gluon of gluon fusion on a t-channel line that a VBF diagram puts
    // a W on - these see the diagram's own, and with it its own QCD vertices,
    // parton lines and merging jets.
    std::size_t diagram_table_size = 0;
    for (auto& [diag_index, lines] : diagram_lines) {
        diagram_table_size = std::max(diagram_table_size, diag_index + 1);
    }
    _diagram_start_states.assign(diagram_table_size, 0);
    nested_vector2<int> diagram_valid(1 << n_ext);
    std::vector<LineMeta> diagram_meta(1 << n_ext);
    for (std::size_t leg = 0; leg < n_ext; ++leg) {
        diagram_meta.at(1 << leg) = mask_meta.at(1 << leg);
    }
    CompileContext diagram_ctx{
        .valid_diags = diagram_valid,
        .mask_meta = diagram_meta,
        .bw_masses = _bw_masses,
        .bw_widths = _bw_widths,
        .bw_indices = bw_indices,
        .pdg_color_types = pdg_color_types,
        .have_pdg_ids = have_pdg_ids,
        .max_jet_flavor = max_jet_flavor,
        .beam_pdg =
            {have_pdg_ids ? external_pdg_ids.at(0) : 21,
             have_pdg_ids ? external_pdg_ids.at(1) : 21},
        .absolute_pdgs = _pdf_absolute_pdgs,
        .line_pdgs = line_pdgs,
        .has_same_flavor = _has_same_flavor,
    };
    nested_vector2<StateItem> diagram_states;
    std::map<StateKey, int> diagram_state_map;
    std::set<int> diagram_dead_states;
    std::vector<std::pair<std::size_t, int>> diagram_roots;
    std::set<std::size_t> compiled;
    for (auto& [diag_index, lines] : diagram_lines) {
        if (!compiled.insert(diag_index).second) {
            continue;
        }
        int diag = static_cast<int>(diag_index);
        for (auto& [mask, meta] : lines) {
            diagram_valid.at(mask) = {diag};
            diagram_meta.at(mask) = meta;
        }
        // The state keys carry the diagram, so no state is shared between two
        // diagrams: their lines differ even where their masks agree.
        int root = static_cast<int>(diagram_states.size());
        diagram_states.push_back({});
        diagram_state_map[{masks, {diag}}] = root;
        if (n_ext == 3) {
            append_terminal_items(diagram_states.at(root), {diag});
        } else {
            find_clusterings(
                diagram_ctx,
                masks,
                {diag},
                diagram_states,
                diagram_state_map,
                diagram_dead_states,
                root
            );
        }
        if (diagram_states.at(root).size() > 0) {
            diagram_roots.push_back({diag_index, root});
        }
        for (auto& [mask, meta] : lines) {
            diagram_valid.at(mask).clear();
            diagram_meta.at(mask) = {};
        }
    }
    auto diagram_offsets = append_state_layout(_cluster_state_machine, diagram_states);
    for (auto [diag_index, root] : diagram_roots) {
        _diagram_start_states.at(diag_index) = diagram_offsets.at(root);
    }
}

NamedVector<Value> MLMClustering::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    auto start_state = fb.full({static_cast<me_int_t>(0), fb.batch_size({args.at(0)})});
    return build_kernel(fb, args.at(0), start_state, 0, Value(), {});
}

NamedVector<Value> MLMClustering::build_with_flavors(
    FunctionBuilder& fb,
    Value momenta,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    auto start_state = fb.full({static_cast<me_int_t>(0), fb.batch_size({momenta})});
    return build_kernel(fb, momenta, start_state, 0, flavor_index, leg_flavors);
}

NamedVector<Value> MLMClustering::build_along_diagram(
    FunctionBuilder& fb, Value momenta, Value diagram
) const {
    if (_clustering_history == ClusteringHistory::all_diagrams) {
        throw std::logic_error(
            "a clustering along one diagram needs a clustering_history other "
            "than all_diagrams"
        );
    }
    return build_from_start_state(
        fb, momenta, fb.gather_int(diagram, _diagram_start_states)
    );
}

NamedVector<Value> MLMClustering::build_from_start_state(
    FunctionBuilder& fb,
    Value momenta,
    Value start_state,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    if (_clustering_history == ClusteringHistory::all_diagrams) {
        throw std::logic_error(
            "a clustering along one diagram needs a clustering_history other "
            "than all_diagrams"
        );
    }
    return build_kernel(
        fb,
        momenta,
        start_state,
        static_cast<me_int_t>(_clustering_history),
        flavor_index,
        leg_flavors
    );
}

NamedVector<Value> MLMClustering::build_kernel(
    FunctionBuilder& fb,
    Value momenta,
    Value start_state,
    me_int_t history_mode,
    Value flavor_index,
    const std::vector<me_int_t>& leg_flavors
) const {
    std::array<Value, 17> mlm_out;
    Value random = fb.squeeze(fb.random(fb.batch_size({momenta}), 1));
    // Without the flavours of the event a single row of zeros, which leaves
    // every clustering allowed: the history over the union of all diagrams.
    if (!flavor_index || leg_flavors.empty()) {
        flavor_index = fb.full({static_cast<me_int_t>(0), fb.batch_size({momenta})});
    }
    std::vector<me_int_t> flavor_table = leg_flavors.empty()
        ? std::vector<me_int_t>(_external_masses.size(), 0)
        : leg_flavors;
    if (_hadronic) {
        mlm_out = fb.mlm_clustering_hadronic(
            momenta,
            random,
            start_state,
            _cluster_state_machine,
            _external_masses,
            _bw_masses,
            _bw_widths,
            _bw_cutoff,
            _jet_radius,
            _cm_energy,
            static_cast<me_int_t>(_jet_scale_scheme),
            _xqcut,
            static_cast<me_int_t>(_scale_scheme),
            static_cast<me_int_t>(_beam_flags),
            static_cast<me_int_t>(_jet_leg_mask),
            static_cast<me_int_t>(_parton_line_scheme),
            static_cast<me_int_t>(_alphas_scheme),
            // 2: the ladder under clustering_mean too (see the kernel)
            static_cast<me_int_t>(
                _pdf_reweighting ? (_clustering_mean_pdf_ladder ? 2 : 1) : 0
            ),
            static_cast<me_int_t>(_clustering_measure),
            history_mode,
            flavor_index,
            flavor_table
        );
    } else {
        mlm_out = fb.mlm_clustering_leptonic(
            momenta,
            random,
            start_state,
            _cluster_state_machine,
            _external_masses,
            _bw_masses,
            _bw_widths,
            _bw_cutoff,
            _jet_radius,
            _cm_energy,
            static_cast<me_int_t>(_jet_scale_scheme),
            _xqcut,
            static_cast<me_int_t>(_scale_scheme),
            static_cast<me_int_t>(_beam_flags),
            static_cast<me_int_t>(_jet_leg_mask),
            static_cast<me_int_t>(_parton_line_scheme),
            static_cast<me_int_t>(_alphas_scheme),
            // 2: the ladder under clustering_mean too (see the kernel)
            static_cast<me_int_t>(
                _pdf_reweighting ? (_clustering_mean_pdf_ladder ? 2 : 1) : 0
            ),
            static_cast<me_int_t>(_clustering_measure),
            history_mode,
            flavor_index,
            flavor_table
        );
    }
    return {return_types().keys(), {mlm_out.begin(), mlm_out.end()}};
}

MLMClusteringAlongDiagram::MLMClusteringAlongDiagram(const MLMClustering& clustering) :
    FunctionGenerator(
        "MLMClusteringAlongDiagram",
        [&] {
            auto types = clustering.arg_types();
            types.push_back("diagram", batch_int);
            return types;
        }(),
        clustering.return_types()
    ),
    _clustering(clustering) {}

NamedVector<Value> MLMClusteringAlongDiagram::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _clustering.build_along_diagram(fb, args.at(0), args.at(1));
}

MLMClusteringWithFlavors::MLMClusteringWithFlavors(
    const MLMClustering& clustering, std::vector<me_int_t> leg_flavors
) :
    FunctionGenerator(
        "MLMClusteringWithFlavors",
        [&] {
            auto types = clustering.arg_types();
            types.push_back("flavor_index", batch_int);
            return types;
        }(),
        clustering.return_types()
    ),
    _clustering(clustering),
    _leg_flavors(std::move(leg_flavors)) {}

NamedVector<Value> MLMClusteringWithFlavors::build_function_impl(
    FunctionBuilder& fb, const NamedVector<Value>& args
) const {
    return _clustering.build_with_flavors(fb, args.at(0), args.at(1), _leg_flavors);
}
