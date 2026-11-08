#pragma once

#include "definitions.hpp"
#include "kinematics.hpp"

namespace madspace {
namespace kernels {

constexpr int N_EXT_MAX = 12;
constexpr double ONE_PLUS_TINY = 1.000001;
// Stand-in for a clustering measure that came out inf or NaN; still far
// above any physical scale, and small enough that SCALE_MAX * 10 is finite.
constexpr double SCALE_MAX = 1e307;
// Words per clustering transition in the compiled state machine:
// (data, next_offset, trace_data). Must match state_machine_item_size in
// madspace/phasespace/mlm_clustering.hpp.
constexpr int STATE_ITEM_SIZE = 3;
// trace_data values, mirroring TraceMode in mlm_clustering.cpp: which daughter
// of a clustering the mother's parton line continues into.
constexpr int TRACE_FIRST = 0;
constexpr int TRACE_SECOND = 1;
constexpr int TRACE_HARDER = 2;
constexpr int TRACE_BOTH = 3;
// JetScaleScheme in mlm_clustering.hpp.
constexpr int SCHEME_EMISSION = 0;
constexpr int SCHEME_PRODUCTION = 1;
// ScaleScheme in mlm_clustering.hpp.
constexpr int SCALES_CLUSTERING_MEAN = 0;
constexpr int SCALES_MADEVENT = 1;
// How the beam parton line is decided to carry on past a vertex: from each
// daughter's own flavour, or from the propagated goodjet flag of reweight.f.
constexpr int LINE_FLAVOR = 0;
constexpr int LINE_GOODJET = 1;
// ClusteringMeasure in mlm_clustering.hpp: which definition scores a
// non-resonant final-state clustering.
constexpr int MEASURE_FXFX = 0;
constexpr int MEASURE_MADEVENT = 1;
// How alpha_s is evaluated for a merged event.
constexpr int ALPHAS_NONE = 0;       // one coupling at mu_R
constexpr int ALPHAS_PER_VERTEX = 1; // alphas(pt_i) at each vertex, as madevent
constexpr int ALPHAS_GEOMETRIC = 2;  // alphas at the geometric mean of the pt_i
// trace_data bits above the 2-bit trace mode: the flavour of the mother.
constexpr int TRACE_IS_JET_IN = 1 << 2;
constexpr int TRACE_IS_COLORED_IN = 1 << 3;
// Bits 4..11 hold the mother's pdf flavour class plus one; zero means the line
// is not a parton and the pdf reweighting chain stops. The classes are assigned
// in mlm_clustering.cpp and only carried here.
constexpr int TRACE_FLAVOR_SHIFT = 4;
constexpr int TRACE_FLAVOR_MASK = 0xFF;
// More of the mother's flavour, for isjetvx and the iqjets demotion.
constexpr int TRACE_IS_OCTET_IN = 1 << 12;
constexpr int TRACE_MOTHER_IS_DAU1 = 1 << 13;
constexpr int TRACE_MOTHER_IS_DAU2 = 1 << 14;
constexpr int TRACE_ALL_COLORLESS = 1 << 15;
// The clustering joins two bare external fermions through a flavour-diagonal
// boson (g, a, Z, h), so it only exists when both carry the same flavour. The
// flavour is a per-event input, see flavor_index and leg_flavors.
constexpr int TRACE_SAME_FLAVOR = 1 << 16;
// jet_leg_mask carries is_jet per leg in the low half and is_octet in the high
// half; n_ext_max is 12, so one word holds both.
constexpr int LEG_OCTET_SHIFT = 16;

// mT^2 = E^2 - pz^2 (hadronic) or E^2 (lepton collider).
// based on djb_clus from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> djb_clus(const FourMom<T>& p, bool hadronic) {
    auto r = hadronic ? (p[0] - p[3]) * (p[0] + p[3]) : p[0] * p[0];
    return max(r, 0.0);
}

// kt/Durham clustering measure for two partons.
// mass1/mass2 are the tracked clustering masses kept in a separate array and updated by
// update_momenta - NOT the Lorentz-invariant masses computed from the 4-momentum.
// jet_radius is the jet-radius parameter (Fortran common /to_dj/D).
// based on dj_clus from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> dj_clus(
    const FourMom<T>& p1,
    const FourMom<T>& p2,
    FVal<T> mass1,
    FVal<T> mass2,
    bool hadronic,
    FVal<T> jet_radius
) {
    if (!hadronic) {
        // Durham measure 2*min(E1^2, E2^2)*(1 - cos_theta), evaluated as
        // 4*min(E1^2, E2^2)*sin^2(theta/2) to avoid cancellation for collinear pairs.
        auto p1a = sqrt(esquare<T>(p1));
        auto p2a = sqrt(esquare<T>(p2));
        if (p1a * p2a == 0.0) {
            return 0.0;
        }
        auto cross_x = p1[2] * p2[3] - p1[3] * p2[2];
        auto cross_y = p1[3] * p2[1] - p1[1] * p2[3];
        auto cross_z = p1[1] * p2[2] - p1[2] * p2[1];
        auto theta = atan2(
            sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z),
            edot<T>(p1, p2)
        );
        auto sin_half = sin(0.5 * theta);
        return 4.0 * min(p1[0] * p1[0], p2[0] * p2[0]) * sin_half * sin_half;
    }

    // hadronic: massless+massive pair scores the massless parton's mT^2
    bool massive1 = (mass1 > 0.0);
    bool massive2 = (mass2 > 0.0);
    if (!massive1 && massive2) {
        return djb_clus<T>(p1, true) * ONE_PLUS_TINY;
    }
    if (massive1 && !massive2) {
        return djb_clus<T>(p2, true) * ONE_PLUS_TINY;
    }

    // both massless or both massive: generalised kt measure
    // min(pt1^2, pt2^2) * 2 * (cosh(deta) - cos(dphi)) / D^2 as in cluster.f,
    // evaluated as 4 * (sinh^2(deta/2) + sin^2(dphi/2)) to avoid cancellation
    // for collinear pairs; eta = asinh(pz/pt), dphi from atan2(cross, dot).
    auto pt1_sq = p1[1] * p1[1] + p1[2] * p1[2];
    auto pt2_sq = p2[1] * p2[1] + p2[2] * p2[2];
    if (pt1_sq == 0.0 || pt2_sq == 0.0) {
        return 0.0;
    }
    auto deta = asinh(p1[3] / sqrt(pt1_sq)) - asinh(p2[3] / sqrt(pt2_sq));
    auto dphi = atan2(p1[1] * p2[2] - p1[2] * p2[1], p1[1] * p2[1] + p1[2] * p2[2]);
    auto sinh_half = sinh(0.5 * deta);
    auto sin_half = sin(0.5 * dphi);
    auto m_max_sq = max(mass1 * mass1, mass2 * mass2);
    auto result = m_max_sq +
        min(pt1_sq, pt2_sq) * 4.0 * (sinh_half * sinh_half + sin_half * sin_half) /
            (jet_radius * jet_radius);
    // The Fortran zeroes negative and NaN results; max() alone would pass a NaN.
    if (!(result > 0.0)) {
        return 0.0;
    }
    return result;
}

// Clustering scale for the pair (momentum1=pi, momentum2=pj).
// mass1/mass2 are the tracked clustering masses for momentum1/momentum2.
//
// Parameters replacing Fortran globals / derived quantities:
//   is_initial   : momentum1 is a beam particle (state machine particle1 < 2)
//   hadronic     : hadronic collider (Fortran: lpp[] from run.inc)
//   jet_radius   : jet-radius parameter (Fortran: common /to_dj/D)
//   massive_in/out1/out2 replace the Fortran get_clustering_type cl[0:2] bit-array
//   resonant     : replaces the iBWlist lookup in Fortran cluster_scale
//   measure      : MEASURE_FXFX or MEASURE_MADEVENT
//
// based on cluster_scale from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC FVal<T> compute_scale(
    const FourMom<T>& momentum1,
    const FourMom<T>& momentum2,
    const FourMom<T>& momentum_sum,
    FVal<T> mass1,
    FVal<T> mass2,
    bool resonant,
    bool is_initial,
    bool massive_in,
    bool massive_out1,
    bool massive_out2,
    bool hadronic,
    FVal<T> jet_radius,
    int measure
) {
    if (is_initial) {
        // scale = mT of the final-state parton
        // small penalty when it goes against the beam
        if (measure == MEASURE_MADEVENT) {
            // LO cluster.f scales the squared measure by 1 + 1e-6; the same
            // product is square-rooted here so ties with DJ match bit-for-bit.
            auto djb = djb_clus<T>(momentum2, hadronic);
            if ((momentum1[3] < 0.0) != (momentum2[3] < 0.0)) {
                djb = djb * ONE_PLUS_TINY;
            }
            return sqrt(djb);
        }
        auto scale = sqrt(djb_clus<T>(momentum2, hadronic));
        if ((momentum1[3] < 0.0) != (momentum2[3] < 0.0)) {
            scale = scale * ONE_PLUS_TINY;
        }
        return scale;
    }
    if (resonant) {
        return sqrt(max(lsquare<T>(momentum_sum), 0.0));
    }
    if (measure == MEASURE_MADEVENT) {
        // DJ of Template/LO/Source/kin_functions.f, used by madevent's LO
        // cluster.f for every final-state pair. The branches below are the NLO
        // cluster_scale's, which score q* > q W and g* > g h differently.
        return sqrt(
            dj_clus<T>(momentum1, momentum2, mass1, mass2, hadronic, jet_radius)
        );
    }
    if (!massive_in && massive_out1 && !massive_out2) {
        return sqrt(fabs(ldot<T>(momentum2, momentum_sum))) / 2.0;
    }
    if (!massive_in && !massive_out1 && massive_out2) {
        return sqrt(fabs(ldot<T>(momentum1, momentum_sum))) / 2.0;
    }
    if (massive_in && !massive_out1 && !massive_out2) {
        return sqrt(max(lsquare<T>(momentum_sum), 0.0));
    }
    return sqrt(dj_clus<T>(momentum1, momentum2, mass1, mass2, hadronic, jet_radius));
}

// Candidate measures agreeing to this relative precision are treated as ties.
constexpr double TIE_TOLERANCE = 1e-12;

// Whether a candidate clustering takes over from the best one found so far.
//
// A resonant candidate beats a non-resonant one, then the smaller measure wins.
// Ties (equal in exact arithmetic, e.g. two balanced massless partons clustered
// into either beam) are settled as in cluster.f: the first candidate in loop
// order, i.e. by (particle2, particle1), is kept.
template <typename T>
KERNELSPEC bool candidate_wins(
    FVal<T> scale,
    int data,
    bool resonant,
    int best_next_state,
    FVal<T> best_scale,
    int best_data,
    bool best_resonant
) {
    if (best_next_state == -1) {
        return true;
    }
    if (resonant != best_resonant) {
        return resonant;
    }
    if (scale < best_scale * (1.0 - TIE_TOLERANCE)) {
        return true;
    }
    if (scale > best_scale * (1.0 + TIE_TOLERANCE)) {
        return false;
    }
    int particle1 = data & 0xFF, particle2 = (data >> 8) & 0xFF;
    int best_particle1 = best_data & 0xFF, best_particle2 = (best_data >> 8) & 0xFF;
    if (particle2 != best_particle2) {
        return particle2 < best_particle2;
    }
    // The same pair reached through another graph scores the same; the first
    // one listed stands.
    return particle1 < best_particle1;
}

// Update momenta and the separately-tracked masses after one clustering step.
// Based on update_momenta from Template/NLO/SubProcesses/cluster.f
template <typename T>
KERNELSPEC void update_momenta(
    int n_part,
    FourMom<T>* momenta,
    FVal<T>* masses,
    int& alive,
    int i_remove,
    int i_keep,
    bool resonant,
    bool allow_boost
) {
    alive &= ~(1 << i_remove);

    if (i_keep < 2) { // initial-state clustering
        int j_other = 1 - i_keep;
        for (int k = 0; k < 4; ++k) {
            momenta[i_keep][k] -= momenta[i_remove][k];
        }

        masses[i_keep] = (masses[i_keep] > 0.0) != (masses[i_remove] > 0.0)
            ? max(masses[i_keep], masses[i_remove])
            : 0.0;

        FourMom<T> com_boost_vector = {
            momenta[i_keep][0] + momenta[j_other][0],
            -(momenta[i_keep][1] + momenta[j_other][1]),
            -(momenta[i_keep][2] + momenta[j_other][2]),
            -(momenta[i_keep][3] + momenta[j_other][3]),
        };
        // cluster.f skips the boost at the last clustering ("Don't boost if boost
        // vector too lightlike or last vertex"), see allow_boost.
        if (allow_boost && lsquare<T>(com_boost_vector) > 100.0) {
            // boost i_keep to COM frame to define the rotation axis, then apply to all
            // alive particles
            auto jkeep_cm = boost<T>(momenta[i_keep], com_boost_vector, 1.0);
            for (int j = 0; j < n_part; ++j) {
                if (alive & (1 << j)) {
                    momenta[j] = rotate_inverse<T>(
                        boost<T>(momenta[j], com_boost_vector, 1.0), jkeep_cm
                    );
                }
            }
        }
    } else { // final-state clustering
        for (int k = 0; k < 4; ++k) {
            momenta[i_keep][k] += momenta[i_remove][k];
        }

        if (resonant) {
            masses[i_keep] = sqrt(max(lsquare<T>(momenta[i_keep]), 0.0));
        } else {
            masses[i_keep] = max(masses[i_keep], masses[i_remove]);
        }
    }
}

// One clustering history, walked from start_state: 0 is the root of the
// state machine over every diagram, anything else the root of the one compiled
// for a single diagram. Writes every output and returns the number of outgoing
// legs the history calls merging jets.
template <typename T>
KERNELSPEC int mlm_clustering_walk(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    IIn<T, 0> clustering_measure,
    IIn<T, 0> flavor_index,
    IIn<T, 1> leg_flavors,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam,
    FOut<T, 1> pdf_rw_x_prev,
    int start_state,
    bool hadronic
) {
    // we do not support SIMD for now, so we can assume simple types
    static_assert(std::is_same_v<IVal<T>, int>);
    static_assert(std::is_same_v<FVal<T>, double>);

    int state = start_state, cluster_count = 0;
    int cluster_max = momenta.size() - 3;
    int n_part = momenta.size();
    FourMom<T> momenta_tmp[N_EXT_MAX];
    // The same clustering in the lab frame, without update_momenta's boosts and
    // rotations. Used for the 2 -> 1 root after a final-state last clustering
    // (cluster.f boosts back first); see last_is_initial.
    FourMom<T> momenta_lab[N_EXT_MAX];
    // Whether the last recorded clustering was an initial-state one.
    bool last_is_initial = false;
    FVal<T> masses_tmp[N_EXT_MAX];
    int alive = (1 << n_part) - 1;
    int cluster_history[N_EXT_MAX - 3];
    int cluster_trace[N_EXT_MAX - 3];
    FVal<T> cluster_scales[N_EXT_MAX - 3];
    // mT of the final-state daughter of each initial-state clustering,
    // zero for a final-state one. This is mt2ij in cluster.f, kept
    // unsquared like every other scale here.
    FVal<T> cluster_mt[N_EXT_MAX - 3];
    // The Pythia ISR momentum fraction of each initial-state clustering,
    // zcl in cluster.f, zero for a final-state one. The pdf reweighting walks
    // the beam's momentum fraction down the ladder with it.
    FVal<T> cluster_z[N_EXT_MAX - 3];
    // sqrt(mT(d1) mT(d2)) of the last clustering if it is a final-state one,
    // zero otherwise. mt2last in cluster.f, kept unsquared.
    FVal<T> mt_last = 0.0;

    for (int i = 0; i < n_part; ++i) {
        for (int j = 0; j < 4; ++j) {
            momenta_tmp[i][j] = momenta[i][j];
            momenta_lab[i][j] = momenta[i][j];
        }
        masses_tmp[i] = external_masses[i];
    }

    // abs pdg id of each external leg for this event, zero where it is not
    // known, which leaves every clustering touching that leg allowed.
    int leg_flavor[N_EXT_MAX];
    for (int i = 0; i < n_part; ++i) {
        int entry = flavor_index * n_part + i;
        leg_flavor[i] =
            flavor_index >= 0 && entry < leg_flavors.size() ? leg_flavors[entry] : 0;
    }

    int win_next_state = -1, win_data = 0, win_trace = TRACE_FIRST;
    bool win_resonant = false;
    FVal<T> win_scale = SCALE_MAX * 10.;
    // Best candidate forbidden by the event flavours; used only if nothing else
    // is left, so that the walk can always continue.
    int alt_next_state = -1, alt_data = 0, alt_trace = TRACE_FIRST;
    bool alt_resonant = false;
    FVal<T> alt_scale = SCALE_MAX * 10.;
    while (cluster_count < cluster_max) {
        int data = state_machine[state];
        int next_state = state_machine[state + 1];
        int trace_data = state_machine[state + 2];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        int mass_index = (data >> 16) & 0xFF;
        bool massive_in = (data >> 24) & 1;
        bool massive_out1 = (data >> 25) & 1;
        bool massive_out2 = (data >> 26) & 1;
        bool is_last = (data >> 30) & 1;
        bool is_initial = (particle1 < 2);

        FourMom<T> momentum_sum{
            momenta_tmp[particle1][0] + momenta_tmp[particle2][0],
            momenta_tmp[particle1][1] + momenta_tmp[particle2][1],
            momenta_tmp[particle1][2] + momenta_tmp[particle2][2],
            momenta_tmp[particle1][3] + momenta_tmp[particle2][3],
        };

        bool resonant = false;
        if (mass_index != 0) {
            FVal<T> prop_m2 = lsquare<T>(momentum_sum);
            FVal<T> mass = bw_masses[mass_index - 1];
            FVal<T> width = bw_widths[mass_index - 1];
            FVal<T> half_window = FVal<T>(bw_cutoff) * width;
            FVal<T> m_min = max(mass - half_window, 0.0);
            FVal<T> m_max = mass + half_window;
            resonant = (prop_m2 >= m_min * m_min) && (prop_m2 <= m_max * m_max);
        }

        FVal<T> scale = compute_scale<T>(
            momenta_tmp[particle1],
            momenta_tmp[particle2],
            momentum_sum,
            masses_tmp[particle1],
            masses_tmp[particle2],
            resonant,
            is_initial,
            massive_in,
            massive_out1,
            massive_out2,
            hadronic,
            jet_radius,
            clustering_measure
        );

        // A collinear pair can make the measure inf or NaN; map it to a large finite
        // value so it loses but a clustering is still always chosen.
        if (!(scale < SCALE_MAX)) {
            scale = SCALE_MAX;
        }

        // MG5 took the resonance structure from the integration channel, which is
        // not always possible in MG7, so resonant candidates are preferred.
        bool forbidden = (trace_data & TRACE_SAME_FLAVOR) &&
            leg_flavor[particle1] != 0 && leg_flavor[particle2] != 0 &&
            leg_flavor[particle1] != leg_flavor[particle2];
        if (forbidden) {
            if (candidate_wins<T>(
                    scale,
                    data,
                    resonant,
                    alt_next_state,
                    alt_scale,
                    alt_data,
                    alt_resonant
                )) {
                alt_next_state = next_state;
                alt_scale = scale;
                alt_data = data;
                alt_trace = trace_data;
                alt_resonant = resonant;
            }
        } else {
            if (candidate_wins<T>(
                    scale,
                    data,
                    resonant,
                    win_next_state,
                    win_scale,
                    win_data,
                    win_resonant
                )) {
                win_next_state = next_state;
                win_scale = scale;
                win_data = data;
                win_trace = trace_data;
                win_resonant = resonant;
            }
        }
        if (is_last && win_next_state == -1) {
            win_next_state = alt_next_state;
            win_scale = alt_scale;
            win_data = alt_data;
            win_trace = alt_trace;
            win_resonant = alt_resonant;
        }
        if (is_last) {
            int p1_win = win_data & 0xFF;
            int p2_win = (win_data >> 8) & 0xFF;
            // Read before update_momenta boosts it; only initial-state clusterings
            // have one (iwin < 3 guard in cluster.f).
            cluster_mt[cluster_count] = p1_win < 2
                ? sqrt(djb_clus<T>(momenta_tmp[p2_win], hadronic))
                : FVal<T>(0.0);
            // mt2last of cluster.f: set when this final-state clustering leaves
            // the two beams and one object, from the momenta before merging.
            if (cluster_count == cluster_max - 1 && p1_win >= 2) {
                mt_last = sqrt(sqrt(
                    djb_clus<T>(momenta_tmp[p1_win], hadronic) *
                    djb_clus<T>(momenta_tmp[p2_win], hadronic)
                ));
            }
            // zclus() of Template/LO/Source/kin_functions.f: ratio of the partonic
            // invariants after and before the emission is removed from the beam.
            cluster_z[cluster_count] = 0.0;
            if (p1_win < 2) {
                int other_beam = 1 - p1_win;
                FourMom<T> sum_prev, sum_red;
                for (int k = 0; k < 4; ++k) {
                    sum_prev[k] = momenta_tmp[p1_win][k] + momenta_tmp[other_beam][k];
                    sum_red[k] = momenta_tmp[p1_win][k] - momenta_tmp[p2_win][k] +
                        momenta_tmp[other_beam][k];
                }
                FVal<T> s_prev = lsquare<T>(sum_prev);
                FVal<T> s_red = lsquare<T>(sum_red);
                // Like the Fortran, give up below 1 GeV^2; z = 0 is ignored downstream.
                if (s_red >= 1.0 && s_prev > 0.0) {
                    cluster_z[cluster_count] = s_red / s_prev;
                }
            }
            // The mother keeps slot p1_win; initial-state clusterings subtract
            // the emission, as pcl(imo) = pcl(ida1) - pcl(ida2) in cluster.f.
            for (int k = 0; k < 4; ++k) {
                momenta_lab[p1_win][k] = p1_win < 2
                    ? momenta_lab[p1_win][k] - momenta_lab[p2_win][k]
                    : momenta_lab[p1_win][k] + momenta_lab[p2_win][k];
            }
            if (cluster_count == cluster_max - 1) {
                last_is_initial = p1_win < 2;
            }
            update_momenta<T>(
                n_part,
                momenta_tmp,
                masses_tmp,
                alive,
                p2_win,
                p1_win,
                win_resonant,
                cluster_count < cluster_max - 1
            );
            state = win_next_state;
            cluster_history[cluster_count] = win_data;
            cluster_trace[cluster_count] = win_trace;
            cluster_scales[cluster_count] = win_scale;
            ++cluster_count;
            // Reset the whole selection for the next step, not just the scale.
            win_next_state = -1;
            win_data = 0;
            win_trace = TRACE_FIRST;
            win_resonant = false;
            win_scale = SCALE_MAX * 10.;
            alt_next_state = -1;
            alt_data = 0;
            alt_trace = TRACE_FIRST;
            alt_resonant = false;
            alt_scale = SCALE_MAX * 10.;
        } else {
            state += STATE_ITEM_SIZE;
        }
    }

    // Renormalization scale: geometric mean of QCD clustering scales
    // (non-QCD entries replaced by the maximum scale).
    // Factorization scale: smallest QCD clustering scale.
    FVal<T> fac_scale = 1e308, max_scale = 0.0;
    for (int i = 0; i < cluster_max; ++i) {
        FVal<T> scale = cluster_scales[i];
        bool is_qcd = (cluster_history[i] >> 27) & 1;
        if (is_qcd && scale < fac_scale) {
            fac_scale = scale;
        }
        if (scale > max_scale) {
            max_scale = scale;
        }
    }

    for (int i = 0; i < n_part - 2; ++i) {
        outgoing_scales[i] = 0.0;
    }

    // Which external legs the parton line of each surviving slot ends at.
    // madevent calls this ipart; a slot stands for two legs after a
    // g -> q qbar splitting, and for none once its line has been absorbed.
    int rep1[N_EXT_MAX], rep2[N_EXT_MAX];
    bool slot_is_jet[N_EXT_MAX];
    // Colour of the line currently in each slot, tracked the same way. Only
    // the two beam slots are ever read back, at the 2 -> 1 root.
    bool slot_is_colored[N_EXT_MAX];
    // and whether it is an octet, which decides whether an emission counts as
    // a jet unconditionally: a gluon has a soft singularity the shower has to
    // be left free to fill.
    bool slot_is_octet[N_EXT_MAX];
    for (int i = 0; i < n_part; ++i) {
        rep1[i] = i;
        rep2[i] = -1;
        slot_is_jet[i] = (jet_leg_mask >> i) & 1;
        slot_is_colored[i] = i < 2 ? (((beam_flags >> (2 * i)) & 1) != 0) : true;
        slot_is_octet[i] = (jet_leg_mask >> (i + LEG_OCTET_SHIFT)) & 1;
    }
    // What the iqjets bookkeeping below needs from each step but cannot
    // reconstruct afterwards: which slots were still bare external legs, and
    // where the mother's line ended up. Recorded here rather than walked a
    // second time.
    int step_bare[N_EXT_MAX - 3];
    int step_mother_leg1[N_EXT_MAX - 3], step_mother_leg2[N_EXT_MAX - 3];
    // The external legs each daughter's line ends at, as the step found them:
    // (daughter 1 first, daughter 2 second, -1 for none). madevent's ptclus
    // loop reads ipart of the daughters, which the slots no longer hold once
    // the mother has taken them over.
    int step_daughter_legs[N_EXT_MAX - 3][4];

    bool by_production = jet_scale_scheme == SCHEME_PRODUCTION;
    FVal<T> ren_scale_val = 1.0;
    // Bitmask of slots that are still bare external legs (bits 2..n_part-1).
    // Used for the emission scheme and the xqcut cut below.
    int is_last_cluster = 0b11111111'11111111'11111100;
    for (int i = 0; i < cluster_max; ++i) {
        FVal<T> scale = cluster_scales[i];
        int data = cluster_history[i];
        int particle1 = data & 0xFF;
        int particle2 = (data >> 8) & 0xFF;
        bool is_qcd = (data >> 27) & 1;
        bool is_jet1 = (data >> 28) & 1;
        bool is_jet2 = (data >> 29) & 1;
        step_bare[i] = is_last_cluster;
        if (is_qcd) {
            if (by_production) {
                // Book this vertex onto every external leg the two daughters'
                // lines end at, keeping the hardest vertex each leg takes part
                // in. That is the scale at which the leg's line was produced.
                for (int k = 0; k < 2; ++k) {
                    if (!(k == 0 ? is_jet1 : is_jet2)) {
                        continue;
                    }
                    int daughter = k == 0 ? particle1 : particle2;
                    for (int m = 0; m < 2; ++m) {
                        int leg = m == 0 ? rep1[daughter] : rep2[daughter];
                        if (leg >= 2 && scale > outgoing_scales[leg - 2]) {
                            outgoing_scales[leg - 2] = scale;
                        }
                    }
                }
            } else {
                // Book the vertex at which the leg itself was emitted, i.e.
                // the first (softest) clustering it takes part in.
                if (is_jet1 && (is_last_cluster & (1 << particle1))) {
                    outgoing_scales[particle1 - 2] = scale;
                }
                if (is_jet2 && (is_last_cluster & (1 << particle2))) {
                    outgoing_scales[particle2 - 2] = scale;
                }
            }
            ren_scale_val *= scale;
        } else {
            ren_scale_val *= max_scale;
        }
        is_last_cluster &= ~((1 << particle1) | (1 << particle2));

        step_daughter_legs[i][0] = rep1[particle1];
        step_daughter_legs[i][1] = rep2[particle1];
        step_daughter_legs[i][2] = rep1[particle2];
        step_daughter_legs[i][3] = rep2[particle2];

        // Carry the parton line into the mother, which occupies slot particle1.
        int trace = cluster_trace[i] & 0x3;
        int a1 = rep1[particle1], a2 = rep2[particle1];
        int b1 = rep1[particle2], b2 = rep2[particle2];
        if (trace == TRACE_HARDER || trace == TRACE_BOTH) {
            // madevent compares the transverse momenta of the representative
            // legs in the original event, not in the clustered one
            bool second_harder;
            if (a1 < 0) {
                second_harder = true;
            } else if (b1 < 0) {
                second_harder = false;
            } else {
                FVal<T> pt_a =
                    momenta[a1][1] * momenta[a1][1] + momenta[a1][2] * momenta[a1][2];
                FVal<T> pt_b =
                    momenta[b1][1] * momenta[b1][1] + momenta[b1][2] * momenta[b1][2];
                second_harder = pt_b > pt_a;
            }
            if (trace == TRACE_HARDER) {
                if (second_harder) {
                    rep1[particle1] = b1;
                    rep2[particle1] = b2;
                }
            } else {
                // both daughters carry the line on, hardest first
                rep1[particle1] = second_harder ? b1 : a1;
                rep2[particle1] = second_harder ? a1 : b1;
            }
        } else if (trace == TRACE_SECOND) {
            rep1[particle1] = b1;
            rep2[particle1] = b2;
        }
        // TRACE_FIRST: slot particle1 already holds daughter 1's line

        // the merged line takes the mother's flavour
        slot_is_jet[particle1] = (cluster_trace[i] & TRACE_IS_JET_IN) != 0;
        slot_is_colored[particle1] = (cluster_trace[i] & TRACE_IS_COLORED_IN) != 0;
        slot_is_octet[particle1] = (cluster_trace[i] & TRACE_IS_OCTET_IN) != 0;
        step_mother_leg1[i] = rep1[particle1];
        step_mother_leg2[i] = rep2[particle1];
    }

    // Legs without a booked QCD scale get sqrt(s), as the "ptclus = etot"
    // fallback in reweight.f; zero would wrongly trip an MLM veto.
    for (int i = 0; i < n_part - 2; ++i) {
        if (outgoing_scales[i] == 0.0) {
            outgoing_scales[i] = cm_energy;
        }
    }

    if (cluster_max == 0) {
        // 2 -> 1: no clustering; use the outgoing mass as in cluster.f's special case.
        FourMom<T> p_out{momenta[2][0], momenta[2][1], momenta[2][2], momenta[2][3]};
        ren_scale_val = sqrt(max(lsquare<T>(p_out), 0.0));
    } else {
        ren_scale_val = pow(ren_scale_val, 1.0 / cluster_max);
    }
    if (fac_scale > ren_scale_val) {
        fac_scale = ren_scale_val;
    }
    FVal<T> fac_scale1 = fac_scale, fac_scale2 = fac_scale;

    // The scale each clustering is reweighted at: rewgt reads pt2ijcl after
    // setclscales has overwritten it, so this is not the raw clustering measure.
    // Defaults to that measure when the madevent scale table is not built.
    FVal<T> alphas_step[N_EXT_MAX - 3];
    for (int i = 0; i < cluster_max; ++i) {
        alphas_step[i] = cluster_scales[i];
    }

    // Which outgoing legs the merging cut applies to; filled from iqjets by the
    // madevent walk, otherwise the leg and vertex flavours are used.
    bool merging_jet[N_EXT_MAX];
    bool have_merging_jets = false;
    for (int i = 0; i < n_part; ++i) {
        merging_jet[i] = false;
    }

    // One pdf reweighting slot per clustering that can sit on a beam line, plus
    // one per beam for the 2 -> 1 root. An unclaimed slot is inert (equal scales).
    int pdf_slot_count = n_part - 1;
    int rw_flavor[N_EXT_MAX - 1];
    FVal<T> rw_x[N_EXT_MAX - 1];
    FVal<T> rw_q_num[N_EXT_MAX - 1];
    FVal<T> rw_q_den[N_EXT_MAX - 1];
    FVal<T> rw_active[N_EXT_MAX - 1];
    // Which beam a slot sits on; known statically only for the two root slots.
    FVal<T> rw_beam[N_EXT_MAX - 1];
    // Beam momentum fraction before the step's own z (s_xpdf = xnow / zcl in
    // madevent), used by systematics to rebuild the ladder.
    FVal<T> rw_x_prev[N_EXT_MAX - 1];
    for (int i = 0; i < pdf_slot_count; ++i) {
        rw_flavor[i] = 0;
        rw_x[i] = 1.0;
        rw_q_num[i] = 1.0;
        rw_q_den[i] = 1.0;
        rw_active[i] = 0.0;
        rw_beam[i] = 0.0;
        rw_x_prev[i] = 1.0;
    }
    FVal<T> pdf_scale_val[2] = {-1.0, -1.0};

    // pdf_reweighting == 2 under clustering_mean: do the madevent beam-line walk
    // below but keep only its pdf part (ladder and per-beam final scale, which
    // is also the reported factorisation scale).
    bool mean_with_ladder = scale_scheme != SCALES_MADEVENT && pdf_reweighting == 2;
    // The scales and densities the walk below writes are used only then.
    bool walk_sets_scales = scale_scheme == SCALES_MADEVENT || mean_with_ladder;

    // The core-process scale: madevent's factorisation scale (the central scale
    // of each beam line, mT of what the core produces), whatever scale scheme
    // and pdf reweighting the event uses. The walk below is taken for it in
    // every mode; it only starts the shower of the outgoing particles that are
    // not light partons (see the end of the walk).
    FVal<T> core_scale = 0.0;

    {
        // Writes other than the densities and mu_F (mu_R, alpha_s vertex scales,
        // jet scales, merging-jet gating) happen only when this is set.
        bool madevent_scales = scale_scheme == SCALES_MADEVENT;
        // madevent clusters one step further (to 2 -> 1), absorbing what is left
        // of the final state into a beam at its transverse mass (for Drell-Yan
        // the boson mT). That step is taken here: exactly one final-state slot
        // is still alive after the walk.
        int leftover = -1;
        for (int i = 2; i < n_part; ++i) {
            if (alive & (1 << i)) {
                leftover = i;
                break;
            }
        }
        // cluster.f's pt2ijcl(n+1) = djb(pcl(imap(3,2))): lab frame after a
        // final-state last clustering, current frame after an initial-state one.
        FVal<T> extra_scale = leftover < 0
            ? FVal<T>(0.0)
            : sqrt(
                  djb_clus<T>(
                      last_is_initial ? momenta_tmp[leftover] : momenta_lab[leftover],
                      hadronic
                  )
              );
        // Follow each beam's parton line through the clustering, exactly as
        // setclscales does in Template/LO/SubProcesses/reweight.f.
        //
        //   jfirst   the first initial-state clustering on this beam
        //   jlast    the last one while the line is still a jet
        //   jcentral the last one while the line is still coloured, which
        //            carries on past jlast through e.g. a W emission
        //
        // Beam j's line always occupies slot j here, because a clustering
        // keeps the lower of the two indices, so the slot index is madevent's
        // ibeam(j) without any extra bookkeeping.
        int jfirst[2] = {-1, -1}, jlast[2] = {-1, -1}, jcentral[2] = {-1, -1};
        bool partonline[2], qcdline[2];
        for (int j = 0; j < 2; ++j) {
            qcdline[j] = (beam_flags >> (2 * j)) & 1;
            partonline[j] = qcdline[j];
        }
        // goodjet(i) of reweight.f: every clustering the line went through was a
        // jet one. Starts from the external flavour (QCD for a beam, jet for an
        // outgoing leg) and is only ever cleared. LINE_FLAVOR does not use it.
        bool goodjet[N_EXT_MAX];
        for (int i = 0; i < n_part; ++i) {
            goodjet[i] = i < 2
                ? ((beam_flags >> (2 * i)) & 1) != 0
                : ((jet_leg_mask >> i) & 1) != 0;
        }
        bool use_goodjet = parton_line_scheme == LINE_GOODJET;

        // iqjets of reweight.f: which outgoing legs count as merging jets (the
        // gate on the xqcut check).
        //
        //   1      certainly a jet: emitted while the beam line was still a
        //          parton line, or a gluon, which has a soft singularity the
        //          shower must be left free to fill
        //   jcode  possibly a jet, resolved at the end
        //   0      not a merging jet
        //
        // jcode counts crossings between QCD and non-QCD vertices; increasecode
        // makes the first jet vertex after non-jet ones count as another.
        int iqjets[N_EXT_MAX];
        for (int i = 0; i < n_part; ++i) {
            iqjets[i] = 0;
        }
        int jcode = 1;
        bool increasecode = false;
        // isjet of the beam line itself as it stands, which is ipdgcl(ida(i))
        // in the jcode test - the daughter on the beam side, before the vertex
        // turns it into the mother.
        bool line_is_jet[2];
        for (int j = 0; j < 2; ++j) {
            line_is_jet[j] = ((beam_flags >> (2 * j + 1)) & 1) != 0;
        }

        // One step of the beam-line bookkeeping, for beam j at step i.
        // emitted_leg is ipart(1, ida(3-i)), the first external leg of the
        // emitted line, or -1 where there is none; is_jet_in and is_colored_in
        // are the mother's flavour.
        auto walk_line =
            [&](int j,
                int i,
                bool is_emitted_jet,
                bool emitted_is_octet,
                int emitted_leg,
                bool is_jet_in,
                bool is_colored_in) {
                if (partonline[j]) {
                    if (jfirst[j] < 0) {
                        jfirst[j] = i;
                    }
                    jlast[j] = i;
                    partonline[j] = is_emitted_jet && is_jet_in;
                } else {
                    if (jfirst[j] < 0) {
                        jfirst[j] = i;
                    }
                    // the beam line has stopped being a parton line, so whatever
                    // it becomes is not a good jet either
                    goodjet[j] = false;
                }
                if (!is_emitted_jet || !line_is_jet[j] || !is_jet_in) {
                    ++jcode;
                    increasecode = true;
                } else if (increasecode) {
                    ++jcode;
                    increasecode = false;
                }
                if (is_emitted_jet && emitted_leg >= 2) {
                    iqjets[emitted_leg] =
                        (partonline[j] || emitted_is_octet) ? 1 : jcode;
                }
                line_is_jet[j] = is_jet_in;
                if (qcdline[j]) {
                    jcentral[j] = i;
                    qcdline[j] = is_colored_in;
                }
            };

        int steps = cluster_max + (leftover >= 0 ? 1 : 0);
        // goodjet of the two daughters as each step meets them, before the
        // step turns the first slot into the mother. The ptclus loop needs it.
        bool step_daughter_goodjet[N_EXT_MAX - 3][2];
        for (int i = 0; i < cluster_max; ++i) {
            int data = cluster_history[i];
            int particle1 = data & 0xFF;
            int particle2 = (data >> 8) & 0xFF;
            int trace = cluster_trace[i];
            step_daughter_goodjet[i][0] = goodjet[particle1];
            step_daughter_goodjet[i][1] = goodjet[particle2];
            bool is_jet_in = (trace & TRACE_IS_JET_IN) != 0;
            if (particle1 >= 2) {
                // Final-state clustering. isjetvx() of reweight.f: a QCD vertex
                // where a jet daughter sits opposite a jet mother or a mother
                // repeating the other daughter's flavour (an emission off a line).
                bool is_qcd_vertex = (data >> 27) & 1;
                bool is_jet1 = (data >> 28) & 1;
                bool is_jet2 = (data >> 29) & 1;
                bool mother_is_dau1 = (trace & TRACE_MOTHER_IS_DAU1) != 0;
                bool mother_is_dau2 = (trace & TRACE_MOTHER_IS_DAU2) != 0;
                bool is_octet_in = (trace & TRACE_IS_OCTET_IN) != 0;
                bool is_jet_vertex = is_qcd_vertex &&
                    ((is_jet1 && (is_jet_in || mother_is_dau2)) ||
                     (is_jet2 && (is_jet_in || mother_is_dau1)));
                int mother_leg1 = step_mother_leg1[i];
                int mother_leg2 = step_mother_leg2[i];
                if (!is_jet_vertex) {
                    // "Remove non-gluon jets that lead up to non-jet vertices";
                    // vertices with all or no lines coloured are left alone.
                    if (!is_qcd_vertex && mother_leg1 >= 2) {
                        bool leg1_octet =
                            ((jet_leg_mask >> (mother_leg1 + LEG_OCTET_SHIFT)) & 1) !=
                            0;
                        if (!leg1_octet) {
                            if ((trace & TRACE_ALL_COLORLESS) != 0) {
                                // a W W Z or h h h vertex: nothing to demote
                            } else if (mother_leg2 < 0) {
                                iqjets[mother_leg1] = 0;
                            } else if (iqjets[mother_leg1] > 0 && mother_leg2 >= 2 &&
                                       iqjets[mother_leg2] > 0) {
                                // both halves of an octet's line are tagged,
                                // so one of them can go
                                iqjets[mother_leg1] = 0;
                            }
                        } else if (is_octet_in) {
                            iqjets[mother_leg1] = 0;
                        }
                    }
                    if (mother_leg2 >= 2 && !is_octet_in) {
                        bool leg2_octet =
                            ((jet_leg_mask >> (mother_leg2 + LEG_OCTET_SHIFT)) & 1) !=
                            0;
                        if (!leg2_octet) {
                            iqjets[mother_leg2] = 0;
                        }
                    }
                    goodjet[particle1] = false;
                    continue;
                }
                // a jet vertex: every daughter that is still a bare external
                // jet was emitted here
                if (is_jet1 && particle1 >= 2 && (step_bare[i] & (1 << particle1))) {
                    iqjets[particle1] = 1;
                }
                if (is_jet2 && particle2 >= 2 && (step_bare[i] & (1 << particle2))) {
                    iqjets[particle2] = 1;
                }
                goodjet[particle1] =
                    is_jet_in && goodjet[particle1] && goodjet[particle2];
                continue;
            }
            bool emitted = use_goodjet ? goodjet[particle2] : ((data >> 29) & 1) != 0;
            walk_line(
                particle1,
                i,
                emitted,
                slot_is_octet[particle2],
                rep1[particle2],
                is_jet_in,
                (trace & TRACE_IS_COLORED_IN) != 0
            );
        }
        if (leftover >= 0) {
            // The 2 -> 1 root lies on both beam lines and is walked once per beam
            // (as in reweight.f); the emitted object is the leftover and the
            // mother is the other beam's line.
            bool leftover_jet = use_goodjet ? goodjet[leftover] : slot_is_jet[leftover];
            for (int j = 0; j < 2; ++j) {
                // reweight.f reads ipdgcl(imo) with imo = idacl(n,3-i).
                walk_line(
                    j,
                    cluster_max,
                    leftover_jet,
                    slot_is_octet[leftover],
                    rep1[leftover],
                    slot_is_jet[1 - j],
                    slot_is_colored[1 - j]
                );
            }
        }
        // "Emissions with code 1 are always jets; now take care of possible
        // jets": drop provisional tags up to the last crossing.
        if (!partonline[0] || !partonline[1]) {
            if (partonline[0] || partonline[1]) {
                --jcode;
            }
            for (int leg = 2; leg < n_part; ++leg) {
                if (iqjets[leg] > 1 && iqjets[leg] <= jcode) {
                    iqjets[leg] = 0;
                }
            }
        }
        if (madevent_scales) {
            for (int leg = 0; leg < n_part; ++leg) {
                merging_jet[leg] = leg >= 2 && iqjets[leg] > 0;
            }
            have_merging_jets = true;
        }
        for (int j = 0; j < 2; ++j) {
            if (jfirst[j] < 0) {
                jfirst[j] = jlast[j];
            }
        }

        // Scale of each step incl. the extra 2 -> 1 one; overwritten in place
        // below like pt2ijcl in reweight.f.
        FVal<T> pt_step[N_EXT_MAX - 2];
        for (int i = 0; i < steps; ++i) {
            pt_step[i] = i == cluster_max ? extra_scale : cluster_scales[i];
        }

        // "If last clustering is s-channel QCD (e.g. ttbar) use mt2last instead"
        // (setclscales in reweight.f): applies at the root when both beam lines
        // are still parton lines. Only the root reads it, as jcentral is never a
        // final-state clustering.
        FVal<T> root_mt = 0.0;
        if (mt_last > 2.0 && cluster_max >= 1 && jlast[0] == cluster_max &&
            jlast[1] == cluster_max &&
            ((cluster_history[cluster_max - 1] >> 27) & 1) != 0) {
            root_mt = mt_last;
        }

        // "Set central scale to mT2": at an initial-state clustering use the
        // transverse mass of what was emitted (for Drell-Yan, mT of the leptons).
        for (int j = 0; j < 2; ++j) {
            if (jcentral[j] < 0) {
                continue;
            }
            // At the root mt2ij is the second beam's mT (zero), so only mt2last
            // overrides it.
            FVal<T> mt = jcentral[j] == cluster_max ? root_mt : cluster_mt[jcentral[j]];
            if (mt > 0.0) {
                pt_step[jcentral[j]] = mt;
            }
        }

        // "Ensure that last scales are at least as big as first scales". Raised
        // in the table itself, so a shared jlast/jcentral vertex is raised too.
        for (int j = 0; j < 2; ++j) {
            if (jlast[j] >= 0 && jfirst[j] >= 0 &&
                pt_step[jfirst[j]] > pt_step[jlast[j]]) {
                pt_step[jlast[j]] = pt_step[jfirst[j]];
            }
        }
        if (madevent_scales) {
            for (int i = 0; i < cluster_max; ++i) {
                alphas_step[i] = pt_step[i];
            }
        }

        // The "Store jet info for matching" loop of setclscales. A leg takes a
        // scale only from jet vertices while its line is still a good jet (not
        // yet a leg iqjets rejects); everything else is sqrt(s), which tells the
        // shower matching to leave that parton alone. Production scheme only.
        if (by_production && madevent_scales) {
            FVal<T> ptclus[N_EXT_MAX];
            for (int leg = 0; leg < n_part; ++leg) {
                ptclus[leg] = 0.0;
            }
            auto book = [&](int leg, bool jet_vertex, bool& good, FVal<T> scale) {
                if (leg < 2) {
                    return;
                }
                if (good && iqjets[leg] == 0) {
                    good = false;
                }
                if (jet_vertex && good) {
                    ptclus[leg] = max(ptclus[leg], scale);
                } else if (ptclus[leg] == 0.0) {
                    ptclus[leg] = FVal<T>(cm_energy);
                }
            };
            for (int i = 0; i < cluster_max; ++i) {
                int data = cluster_history[i];
                int trace = cluster_trace[i];
                bool is_qcd_vertex = (data >> 27) & 1;
                bool is_jet1 = (data >> 28) & 1;
                bool is_jet2 = (data >> 29) & 1;
                bool is_jet_in = (trace & TRACE_IS_JET_IN) != 0;
                // isjetvx: an initial-state vertex is a jet vertex when the
                // emitted daughter is a jet, a final-state one as in the
                // iqjets walk above.
                bool jet_vertex = is_qcd_vertex &&
                    ((data & 0xFF) < 2
                         ? is_jet2
                         : ((is_jet1 &&
                             (is_jet_in || (trace & TRACE_MOTHER_IS_DAU2) != 0)) ||
                            (is_jet2 &&
                             (is_jet_in || (trace & TRACE_MOTHER_IS_DAU1) != 0))));
                for (int k = 0; k < 2; ++k) {
                    bool good = step_daughter_goodjet[i][k];
                    for (int m = 0; m < 2; ++m) {
                        book(
                            step_daughter_legs[i][2 * k + m],
                            jet_vertex,
                            good,
                            pt_step[i]
                        );
                    }
                }
            }
            if (leftover >= 0) {
                // the root is never a jet vertex (islast in isjetvx)
                bool good = goodjet[leftover];
                book(rep1[leftover], false, good, 0.0);
                book(rep2[leftover], false, good, 0.0);
            }
            for (int leg = 2; leg < n_part; ++leg) {
                outgoing_scales[leg - 2] =
                    ptclus[leg] > 0.0 ? ptclus[leg] : FVal<T>(cm_energy);
            }
        }

        FVal<T> s_last[2], s_central[2];
        for (int j = 0; j < 2; ++j) {
            s_last[j] = jlast[j] < 0 ? 0.0 : pt_step[jlast[j]];
            s_central[j] = jcentral[j] < 0 ? 0.0 : pt_step[jcentral[j]];
        }

        // mu_R: geometric mean of the four scales (madevent: (pt2 pt2 pt2 pt2)^(1/8)).
        FVal<T> last_scale = pt_step[steps - 1];
        // Used as mu_F fallback; becomes the event's mu_R only under madevent_scales.
        FVal<T> walk_ren_scale;
        // Take roots before multiplying: four scales at SCALE_MAX would overflow.
        if (jlast[0] >= 0 && jlast[1] >= 0) {
            walk_ren_scale = pow(s_last[0], 0.25) * pow(s_central[0], 0.25) *
                pow(s_last[1], 0.25) * pow(s_central[1], 0.25);
        } else if (jlast[0] >= 0) {
            walk_ren_scale = sqrt(s_last[0]) * sqrt(s_central[0]);
        } else if (jlast[1] >= 0) {
            walk_ren_scale = sqrt(s_last[1]) * sqrt(s_central[1]);
        } else if (jcentral[0] >= 0 && jcentral[1] >= 0) {
            walk_ren_scale = sqrt(s_central[0]) * sqrt(s_central[1]);
        } else if (jcentral[0] >= 0) {
            walk_ren_scale = s_central[0];
        } else if (jcentral[1] >= 0) {
            walk_ren_scale = s_central[1];
        } else {
            walk_ren_scale = last_scale;
        }

        // mu_F, one per beam. madevent stores q2fact = mu_F^2 as
        // sqrt(pt2[jlast] pt2[jcentral]), i.e. mu_F is the geometric mean of
        // the two scales.
        FVal<T> walk_fac1 = jlast[0] >= 0 ? sqrt(s_last[0]) * sqrt(s_central[0]) : 0.0;
        FVal<T> walk_fac2 = jlast[1] >= 0 ? sqrt(s_last[1]) * sqrt(s_central[1]) : 0.0;
        // "We have a qcd line going through the whole event, use single scale"
        if (jcentral[0] >= 0 && jcentral[0] == jcentral[1]) {
            walk_fac1 = max(walk_fac1, walk_fac2);
            walk_fac2 = walk_fac1;
        }

        // "Take care of case when jcentral are zero". The pdfwgt branch of
        // reweight.f (capping each beam at min(pt2ijcl(jfirst), q2fact)) is not
        // ported here; the ladder is handled by the pdf reweighting block below.
        if (jcentral[0] < 0 && jcentral[1] < 0) {
            if (!(walk_fac1 > 0.0) && !(walk_fac2 > 0.0)) {
                walk_fac1 = pt_step[steps - 1];
                walk_fac2 = walk_fac1;
            }
        } else if (jcentral[0] < 0) {
            if (jfirst[0] >= 0) {
                walk_fac1 = pt_step[jfirst[0]];
            }
        } else if (jcentral[1] < 0) {
            if (jfirst[1] >= 0) {
                walk_fac2 = pt_step[jfirst[1]];
            }
        }
        if (!(walk_fac1 > 0.0)) {
            walk_fac1 = walk_ren_scale;
        }
        if (!(walk_fac2 > 0.0)) {
            walk_fac2 = walk_ren_scale;
        }
        // and keep everything inside the range the rest of the kernel uses
        if (!(walk_ren_scale > 0.0) || !(walk_ren_scale < SCALE_MAX)) {
            walk_ren_scale = last_scale;
        }
        if (!(walk_fac1 > 0.0) || !(walk_fac1 < SCALE_MAX)) {
            walk_fac1 = walk_ren_scale;
        }
        if (!(walk_fac2 > 0.0) || !(walk_fac2 < SCALE_MAX)) {
            walk_fac2 = walk_ren_scale;
        }
        if (madevent_scales) {
            ren_scale_val = walk_ren_scale;
        }
        core_scale = max(walk_fac1, walk_fac2);
        if (walk_sets_scales) {
            fac_scale1 = walk_fac1;
            fac_scale2 = walk_fac2;
        }

        // The pdf half of rewgt in Template/LO/SubProcesses/reweight.f. The
        // density is first evaluated low on the ladder, at min(pt(jfirst), mu_F),
        // then each beam line is walked up, multiplying by f(x z, Q_i) /
        // f(x z, Q_i-1) per further clustering, with x rescaled by that
        // clustering's z. A line with a single clustering keeps the lowered scale.
        //
        // Only scales, momentum fractions and flavour classes are set here; the
        // flavour is sampled later from the densities being corrected.
        bool reweight_pdf = walk_sets_scales && pdf_reweighting != 0 &&
            jcentral[0] >= 0 && jcentral[1] >= 0;
        if (walk_sets_scales) {
            pdf_scale_val[0] = fac_scale1;
            pdf_scale_val[1] = fac_scale2;
        }
        if (reweight_pdf) {
            FVal<T> q_central[2] = {fac_scale1, fac_scale2};
            for (int j = 0; j < 2; ++j) {
                if (jlast[j] >= 0 && jfirst[j] >= 0 && jfirst[j] <= jlast[j]) {
                    pdf_scale_val[j] = min(pt_step[jfirst[j]], q_central[j]);
                }
            }
            // Per beam: flavour class, product of z, and scale of the last density
            // evaluation (ibeam(j), xnow(j), pt2pdf(ibeam(j)) in reweight.f).
            int line_class[2];
            FVal<T> x_frac[2] = {1.0, 1.0};
            FVal<T> pt_pdf[2] = {0.0, 0.0};
            for (int j = 0; j < 2; ++j) {
                line_class[j] = ((beam_flags >> (8 * (j + 1))) & TRACE_FLAVOR_MASK) - 1;
            }
            for (int i = 0; i < steps; ++i) {
                for (int j = 0; j < 2; ++j) {
                    // A clustering sits on the beam of its first daughter (none if
                    // that is final-state); the root sits on both.
                    bool at_root = i >= cluster_max;
                    if (!at_root && (cluster_history[i] & 0xFF) != j) {
                        continue;
                    }
                    // Once a line stops being a parton the chain is over.
                    if (line_class[j] < 0) {
                        continue;
                    }
                    int slot = at_root ? cluster_max + j : i;
                    // zcl is 1 at the root.
                    FVal<T> z = at_root ? FVal<T>(1.0) : cluster_z[i];
                    FVal<T> x_prev = x_frac[j];
                    if (z > 0.0 && z < 1.0) {
                        x_frac[j] *= z;
                    }
                    FVal<T> q_now =
                        i == jlast[j] ? q_central[j] : min(pt_step[i], q_central[j]);
                    if (!(pt_pdf[j] > 0.0)) {
                        // the first clustering on the line only records where
                        // the density it already has was evaluated
                        pt_pdf[j] = q_now;
                    } else if (pt_pdf[j] < q_now && i <= jlast[j]) {
                        rw_flavor[slot] = line_class[j];
                        rw_x[slot] = x_frac[j];
                        rw_q_num[slot] = q_now;
                        rw_q_den[slot] = pt_pdf[j];
                        rw_active[slot] = 1.0;
                        rw_beam[slot] = j;
                        rw_x_prev[slot] = x_prev;
                        pt_pdf[j] = q_now;
                    }
                    // and the line becomes the mother; at the root that is the
                    // s-channel object, which ends the chain.
                    line_class[j] = at_root
                        ? -1
                        : ((cluster_trace[i] >> TRACE_FLAVOR_SHIFT) &
                           TRACE_FLAVOR_MASK) -
                            1;
                }
            }
        }
    }

    // Outgoing particles that are not light partons (leptons, photons, heavy
    // quarks) have no QCD scale booked and sit at sqrt(s) so far. Give them the
    // core-process scale instead: with Beams:setProductionScalesFromLHEF their
    // pt_clust is where Pythia8 starts their final-state shower, and the
    // largest pt_clust is where it starts the initial-state one. The LHE writer
    // raises it to the hardest scale of the event or, for
    // mlm_shower_start = "collider_energy", puts sqrt(s) back. Light partons
    // keep sqrt(s): there it is Pythia8's do-not-match flag.
    if (core_scale > 0.0 && core_scale < FVal<T>(cm_energy)) {
        for (int leg = 2; leg < n_part; ++leg) {
            if (!((jet_leg_mask >> leg) & 1) &&
                !(outgoing_scales[leg - 2] < FVal<T>(cm_energy))) {
                outgoing_scales[leg - 2] = core_scale;
            }
        }
    }

    // The merging cut (reweight.f "Check xqcut for vertices with jet daughters
    // only"): the event fails if a bare outgoing merging jet sits at a
    // clustering below xqcut. Legs with iqjets == 0 are exempt.
    bool passes_xqcut = true;
    if (FVal<T>(xqcut) > 0.0) {
        for (int i = 0; i < cluster_max; ++i) {
            if (!(cluster_scales[i] < FVal<T>(xqcut))) {
                continue;
            }
            int data = cluster_history[i];
            int particle1 = data & 0xFF;
            int particle2 = (data >> 8) & 0xFF;
            bool is_qcd = (data >> 27) & 1;
            for (int k = 0; k < 2; ++k) {
                int slot = k == 0 ? particle1 : particle2;
                if (slot < 2 || !(step_bare[i] & (1 << slot))) {
                    continue;
                }
                bool counts = have_merging_jets
                    ? merging_jet[slot]
                    : (is_qcd && ((data >> (28 + k)) & 1) != 0);
                if (counts) {
                    passes_xqcut = false;
                }
            }
        }
    }

    ren_scale = ren_scale_val;
    fact_scale1 = fac_scale1;
    fact_scale2 = fac_scale2;
    // Scale of each clustering vertex for the alpha_s reweighting, following
    // the loop in Template/LO/SubProcesses/reweight.f: every clustering except
    // the last one (as in cluster_max) is reweighted by alphas(pt_clust) /
    // alphas(mu_R), where a parton is produced (taken here as the QCD flag).
    // reweight.f drops an event if a reweighted vertex is at or below 2 GeV;
    // alphas_weight is that veto. ALPHAS_GEOMETRIC gives every reweighted
    // vertex the geometric mean of their scales.
    bool alphas_ok = true;
    FVal<T> log_sum = 0.0;
    int reweighted = 0;
    for (int i = 0; i < cluster_max; ++i) {
        if ((cluster_history[i] >> 27) & 1) {
            FVal<T> scale = alphas_step[i];
            if (!(scale * scale > 4.0)) {
                alphas_ok = false;
            } else {
                log_sum += log(scale);
                ++reweighted;
            }
        }
    }
    FVal<T> mean_scale = reweighted > 0 ? exp(log_sum / reweighted) : ren_scale_val;
    // A vertex that is not reweighted gets 0 rather than mu_R, so that
    // systematics can tell it from a reweighted vertex sitting at mu_R.
    for (int i = 0; i < cluster_max; ++i) {
        bool is_qcd_vertex = (cluster_history[i] >> 27) & 1;
        if (alphas_scheme == ALPHAS_NONE || !is_qcd_vertex) {
            alphas_scales[i] = 0.0;
        } else if (alphas_scheme == ALPHAS_GEOMETRIC) {
            alphas_scales[i] = mean_scale;
        } else {
            alphas_scales[i] = alphas_step[i];
        }
    }

    // Weights (0 or 1) that multiply the event weight; alphas_weight is kept
    // apart from the merging cut xqcut_weight.
    alphas_weight = (alphas_scheme == ALPHAS_NONE || alphas_ok) ? 1.0 : 0.0;
    xqcut_weight = passes_xqcut ? 1.0 : 0.0;

    // Without a ladder walk the density stays at the factorisation scale.
    if (!(pdf_scale_val[0] > 0.0)) {
        pdf_scale_val[0] = fac_scale1;
    }
    if (!(pdf_scale_val[1] > 0.0)) {
        pdf_scale_val[1] = fac_scale2;
    }
    pdf_scale1 = pdf_scale_val[0];
    pdf_scale2 = pdf_scale_val[1];
    for (int i = 0; i < pdf_slot_count; ++i) {
        if (rw_active[i] == 0.0) {
            // Keep an inert slot at a valid scale so the consumer's low-density
            // veto cannot trip.
            rw_q_num[i] = pdf_scale_val[0];
            rw_q_den[i] = pdf_scale_val[0];
        }
        pdf_rw_flavor[i] = rw_flavor[i];
        pdf_rw_x[i] = rw_x[i];
        pdf_rw_q_num[i] = rw_q_num[i];
        pdf_rw_q_den[i] = rw_q_den[i];
        pdf_rw_active[i] = rw_active[i];
        pdf_rw_beam[i] = rw_beam[i];
        pdf_rw_x_prev[i] = rw_x_prev[i];
    }

    int diag_count = state_machine[state];
    int rand_index = static_cast<int>(FVal<T>(random) * diag_count);
    if (rand_index >= diag_count) {
        rand_index = diag_count - 1;
    }
    diagram_index = state_machine[state + rand_index + 1];

    // Number of merging jets: iqjets where the madevent walk filled it, else
    // jets emitted at QCD vertices. This is njets of setclscales, compared
    // between histories under HISTORY_MADEVENT.
    int merging_jet_count = 0;
    if (have_merging_jets) {
        for (int leg = 2; leg < n_part; ++leg) {
            if (merging_jet[leg]) {
                ++merging_jet_count;
            }
        }
    } else {
        int emitted = 0;
        for (int i = 0; i < cluster_max; ++i) {
            int data = cluster_history[i];
            if (!((data >> 27) & 1)) {
                continue;
            }
            for (int k = 0; k < 2; ++k) {
                int slot = k == 0 ? (data & 0xFF) : ((data >> 8) & 0xFF);
                if (slot >= 2 && (step_bare[i] & (1 << slot)) &&
                    ((data >> (28 + k)) & 1) != 0) {
                    emitted |= 1 << slot;
                }
            }
        }
        for (int leg = 2; leg < n_part; ++leg) {
            if (emitted & (1 << leg)) {
                ++merging_jet_count;
            }
        }
    }
    return merging_jet_count;
}

// ClusteringHistory in mlm_clustering.hpp.
constexpr int HISTORY_ALL_DIAGRAMS = 0;
constexpr int HISTORY_DIAGRAM = 1;
constexpr int HISTORY_MADEVENT = 2;

template <typename T>
KERNELSPEC void mlm_clustering(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 0> start_state,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    IIn<T, 0> clustering_measure,
    IIn<T, 0> history_mode,
    IIn<T, 0> flavor_index,
    IIn<T, 1> leg_flavors,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam,
    FOut<T, 1> pdf_rw_x_prev,
    bool hadronic
) {
    auto walk = [&](int start) {
        return mlm_clustering_walk<T>(
            momenta,
            random,
            state_machine,
            external_masses,
            bw_masses,
            bw_widths,
            bw_cutoff,
            jet_radius,
            cm_energy,
            jet_scale_scheme,
            xqcut,
            scale_scheme,
            beam_flags,
            jet_leg_mask,
            parton_line_scheme,
            alphas_scheme,
            pdf_reweighting,
            clustering_measure,
            flavor_index,
            leg_flavors,
            ren_scale,
            fact_scale1,
            fact_scale2,
            outgoing_scales,
            diagram_index,
            xqcut_weight,
            alphas_scales,
            alphas_weight,
            pdf_scale1,
            pdf_scale2,
            pdf_rw_flavor,
            pdf_rw_x,
            pdf_rw_q_num,
            pdf_rw_q_den,
            pdf_rw_active,
            pdf_rw_beam,
            pdf_rw_x_prev,
            start,
            hadronic
        );
    };
    int start = start_state;
    if (history_mode == HISTORY_MADEVENT && start != 0) {
        // setclscales clusters over every graph first and keeps that history
        // unless its jet count differs from the channel's graph. Both walks write
        // all outputs, so the channel walk's outputs are saved and restored if it wins.
        int diagram_jets = walk(start);
        int n_part = momenta.size();
        FVal<T> kept_ren = ren_scale, kept_fact1 = fact_scale1,
                kept_fact2 = fact_scale2, kept_xqcut = xqcut_weight,
                kept_alphas = alphas_weight, kept_pdf1 = pdf_scale1,
                kept_pdf2 = pdf_scale2;
        IVal<T> kept_diagram = diagram_index;
        FVal<T> kept_outgoing[N_EXT_MAX], kept_alphas_scales[N_EXT_MAX];
        IVal<T> kept_rw_flavor[N_EXT_MAX];
        FVal<T> kept_rw_x[N_EXT_MAX], kept_rw_q_num[N_EXT_MAX],
            kept_rw_q_den[N_EXT_MAX], kept_rw_active[N_EXT_MAX],
            kept_rw_beam[N_EXT_MAX], kept_rw_x_prev[N_EXT_MAX];
        for (int i = 0; i < n_part - 2; ++i) {
            kept_outgoing[i] = outgoing_scales[i];
        }
        for (int i = 0; i < n_part - 3; ++i) {
            kept_alphas_scales[i] = alphas_scales[i];
        }
        for (int i = 0; i < n_part - 1; ++i) {
            kept_rw_flavor[i] = pdf_rw_flavor[i];
            kept_rw_x[i] = pdf_rw_x[i];
            kept_rw_q_num[i] = pdf_rw_q_num[i];
            kept_rw_q_den[i] = pdf_rw_q_den[i];
            kept_rw_active[i] = pdf_rw_active[i];
            kept_rw_beam[i] = pdf_rw_beam[i];
        }
        for (int i = 0; i < n_part - 1; ++i) {
            kept_rw_x_prev[i] = pdf_rw_x_prev[i];
        }
        if (walk(0) != diagram_jets) {
            ren_scale = kept_ren;
            fact_scale1 = kept_fact1;
            fact_scale2 = kept_fact2;
            xqcut_weight = kept_xqcut;
            alphas_weight = kept_alphas;
            pdf_scale1 = kept_pdf1;
            pdf_scale2 = kept_pdf2;
            diagram_index = kept_diagram;
            for (int i = 0; i < n_part - 2; ++i) {
                outgoing_scales[i] = kept_outgoing[i];
            }
            for (int i = 0; i < n_part - 3; ++i) {
                alphas_scales[i] = kept_alphas_scales[i];
            }
            for (int i = 0; i < n_part - 1; ++i) {
                pdf_rw_flavor[i] = kept_rw_flavor[i];
                pdf_rw_x[i] = kept_rw_x[i];
                pdf_rw_q_num[i] = kept_rw_q_num[i];
                pdf_rw_q_den[i] = kept_rw_q_den[i];
                pdf_rw_active[i] = kept_rw_active[i];
                pdf_rw_beam[i] = kept_rw_beam[i];
            }
            for (int i = 0; i < n_part - 1; ++i) {
                pdf_rw_x_prev[i] = kept_rw_x_prev[i];
            }
        }
    } else {
        walk(start);
    }
}

template <typename T>
KERNELSPEC void kernel_mlm_clustering_hadronic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 0> start_state,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    IIn<T, 0> clustering_measure,
    IIn<T, 0> history_mode,
    IIn<T, 0> flavor_index,
    IIn<T, 1> leg_flavors,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam,
    FOut<T, 1> pdf_rw_x_prev
) {
    mlm_clustering<T>(
        momenta,
        random,
        start_state,
        state_machine,
        external_masses,
        bw_masses,
        bw_widths,
        bw_cutoff,
        jet_radius,
        cm_energy,
        jet_scale_scheme,
        xqcut,
        scale_scheme,
        beam_flags,
        jet_leg_mask,
        parton_line_scheme,
        alphas_scheme,
        pdf_reweighting,
        clustering_measure,
        history_mode,
        flavor_index,
        leg_flavors,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        alphas_scales,
        alphas_weight,
        pdf_scale1,
        pdf_scale2,
        pdf_rw_flavor,
        pdf_rw_x,
        pdf_rw_q_num,
        pdf_rw_q_den,
        pdf_rw_active,
        pdf_rw_beam,
        pdf_rw_x_prev,
        true
    );
}

template <typename T>
KERNELSPEC void kernel_mlm_clustering_leptonic(
    FIn<T, 2> momenta,
    FIn<T, 0> random,
    IIn<T, 0> start_state,
    IIn<T, 1> state_machine,
    FIn<T, 1> external_masses,
    FIn<T, 1> bw_masses,
    FIn<T, 1> bw_widths,
    FIn<T, 0> bw_cutoff,
    FIn<T, 0> jet_radius,
    FIn<T, 0> cm_energy,
    IIn<T, 0> jet_scale_scheme,
    FIn<T, 0> xqcut,
    IIn<T, 0> scale_scheme,
    IIn<T, 0> beam_flags,
    IIn<T, 0> jet_leg_mask,
    IIn<T, 0> parton_line_scheme,
    IIn<T, 0> alphas_scheme,
    IIn<T, 0> pdf_reweighting,
    IIn<T, 0> clustering_measure,
    IIn<T, 0> history_mode,
    IIn<T, 0> flavor_index,
    IIn<T, 1> leg_flavors,
    FOut<T, 0> ren_scale,
    FOut<T, 0> fact_scale1,
    FOut<T, 0> fact_scale2,
    FOut<T, 1> outgoing_scales,
    IOut<T, 0> diagram_index,
    FOut<T, 0> xqcut_weight,
    FOut<T, 1> alphas_scales,
    FOut<T, 0> alphas_weight,
    FOut<T, 0> pdf_scale1,
    FOut<T, 0> pdf_scale2,
    IOut<T, 1> pdf_rw_flavor,
    FOut<T, 1> pdf_rw_x,
    FOut<T, 1> pdf_rw_q_num,
    FOut<T, 1> pdf_rw_q_den,
    FOut<T, 1> pdf_rw_active,
    FOut<T, 1> pdf_rw_beam,
    FOut<T, 1> pdf_rw_x_prev
) {
    mlm_clustering<T>(
        momenta,
        random,
        start_state,
        state_machine,
        external_masses,
        bw_masses,
        bw_widths,
        bw_cutoff,
        jet_radius,
        cm_energy,
        jet_scale_scheme,
        xqcut,
        scale_scheme,
        beam_flags,
        jet_leg_mask,
        parton_line_scheme,
        alphas_scheme,
        pdf_reweighting,
        clustering_measure,
        history_mode,
        flavor_index,
        leg_flavors,
        ren_scale,
        fact_scale1,
        fact_scale2,
        outgoing_scales,
        diagram_index,
        xqcut_weight,
        alphas_scales,
        alphas_weight,
        pdf_scale1,
        pdf_scale2,
        pdf_rw_flavor,
        pdf_rw_x,
        pdf_rw_q_num,
        pdf_rw_q_den,
        pdf_rw_active,
        pdf_rw_beam,
        pdf_rw_x_prev,
        false
    );
}

} // namespace kernels
} // namespace madspace
