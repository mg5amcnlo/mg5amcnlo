#pragma once

#include <vector>

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"
#include "madspace/phasespace/three_particle.hpp"
#include "madspace/phasespace/topology.hpp"
#include "madspace/phasespace/two_particle.hpp"

namespace madspace {

class ColorOrderedMapping : public Mapping {
public:
    // color_order: 0-indexed permutation of {0, ..., n-1} (n = n_out + 2).
    // Particles 0 and 1 are the two incoming beams.
    //
    // Optional cuts (all indexed by 0-based outgoing-particle index, i.e. the
    // same indexing as the masses m_out and the entries of the color order
    // minus 2):
    //   * pt_min[i]        : minimum transverse momentum of outgoing particle i.
    //   * sqrt_s_min[i][j] : minimum invariant mass of the pair (i, j).
    //   * dr_min[i][j]     : minimum delta-R separation of the pair (i, j).
    // Passing any non-empty cut container enables cut-aware sampling. The cuts
    // are translated into invariant-space bounds exactly as in the Fortran
    // reference (phase_space_gen23): per-subset invariant-mass floors
    // (invm_min) on the sampled s-channel masses, the adjacent-pair floor on
    // the 2->3 s23 invariant, and a |t| floor (pt^2) on each peeled particle.
    // Empty containers (the default) reproduce the previous cut-free behaviour
    // exactly.
    ColorOrderedMapping(
        const std::vector<std::size_t>& color_order,
        double t_invariant_power = 0.8,
        double s_invariant_power = 0.8,
        const std::vector<double>& pt_min = {},
        const std::vector<std::vector<double>>& sqrt_s_min = {},
        const std::vector<std::vector<double>>& dr_min = {}
    );

    std::size_t random_dim() const { return _random_dim; }

private:
    Result build_forward_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;
    Result build_inverse_impl(
        FunctionBuilder& fb,
        const NamedVector<Value>& inputs,
        const NamedVector<Value>& conditions
    ) const override;

    // pt^2 of outgoing particle i (0 if no pt cut on it).
    double pt2(std::size_t i) const;
    // Cut-derived invariant-mass^2 floor (gen23 invm_min, without the mass^2
    // term, which is applied separately) for a subset of outgoing particles.
    // Returns 0 when cuts are disabled or the subset has fewer than 2 members.
    double cut_floor(const std::vector<std::size_t>& subset) const;

    // 0-indexed outgoing-particle indices (values in {0,...,n_out-1}).
    // _set1 contains the outgoing particles attached to beam 0's side,
    // _set2 those attached to beam 1's side, in peel order.
    std::vector<std::size_t> _set1;
    std::vector<std::size_t> _set2;
    std::size_t _n_out;
    std::size_t _random_dim;
    // True iff exactly one of (set1, set2) has size 1 (and the other >= 2).
    // In that case the central block is DoubleT instead of 2->2.
    bool _use_double_t;
    // True iff one of (set1, set2) is empty, i.e. particles 0 and 1 are
    // adjacent in the color order and all outgoing particles sit on one side.
    // In that case there is no central block at all: the full final state is
    // produced as a single t-channel chain seeded directly off the beams.
    bool _use_single_chain;

    // Cut configuration (empty => all bounds resolve to 0 = no cut).
    std::vector<double> _pt_min;
    std::vector<std::vector<double>> _sqrt_s_min;
    std::vector<std::vector<double>> _dr_min;

    Invariant _uniform_invariant;
    TwoToTwoParticleScattering _com_scattering;
    TwoToTwoParticleScattering _lab_scattering;
    TwoToThreeParticleScattering _two_to_three;
    DoubleT _double_t;
};

} // namespace madspace
