#pragma once

#include "madspace/phasespace/base.hpp"
#include "madspace/phasespace/invariants.hpp"

namespace madspace {

/**
 * Genuine @f$1 \to 3@f$ decay block.
 *
 * Splits a parent momentum into three on-shell momenta,
 * @f$p_0 = p_1 + p_2 + p_3@f$ [1]. In the parent rest frame the five degrees of
 * freedom are the daughter energies @f$E_1, E_2@f$, the polar and azimuthal
 * angles of @f$\vec p_1@f$, and an azimuth @f$\beta@f$ of @f$\vec p_2@f$ about
 * @f$\vec p_1@f$; the opening angle between @f$\vec p_1@f$ and @f$\vec p_2@f$ is
 * fixed by energy-momentum conservation. All five variables are sampled flat
 * within their kinematic ranges, giving a constant @f$1/8@f$ measure
 * (Sec. 2.2.6 of [1], following [2]).
 *
 * `batch` is the leading batch dimension. Masses are inputs here: supplied on
 * the forward call and recovered by the inverse.
 *
 * **Inputs**
 * - `random_energy1` – `float`, shape `(batch,)` – random number for @f$E_1@f$.
 * - `random_energy2` – `float`, shape `(batch,)` – random number for @f$E_2@f$.
 * - `random_phi` – `float`, shape `(batch,)` – azimuth of @f$\vec p_1@f$.
 * - `random_cos_theta` – `float`, shape `(batch,)` – polar angle of
 *   @f$\vec p_1@f$.
 * - `random_beta` – `float`, shape `(batch,)` – azimuth of @f$\vec p_2@f$ about
 *   @f$\vec p_1@f$.
 * - `mass0` – `float`, shape `(batch,)` – parent mass.
 * - `mass1`, `mass2`, `mass3` – `float`, shape `(batch,)` – daughter masses.
 * - `com_momentum` – `float`, shape `(batch, 4)` – parent four-momentum.
 *   Present only when @p com is false.
 *
 * **Conditions**
 * - None.
 *
 * **Outputs**
 * - `momentum1`, `momentum2`, `momentum3` – `float`, shape `(batch, 4)` – the
 *   daughter momenta.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.6)
 * - [2] G. Knippen, PhD thesis, University of Freiburg (2019),
 *   https://doi.org/10.6094/UNIFR/154629
 */
class ThreeBodyDecay : public Mapping {
public:
    /// @param com  If true the decay is generated in the parent rest frame,
    ///             otherwise the parent momentum is taken from the
    ///             `com_momentum` input.
    ThreeBodyDecay(bool com);
    /// Number of uniform random inputs consumed by the forward mapping (5).
    std::size_t random_dim() const { return 5; }

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

    bool _com;
};

/**
 * Two-particle block in the double-invariant @f$(\tilde s, t)@f$ parametrization.
 *
 * Replaces both angles of the two-particle block by Lorentz invariants. The
 * polar angle becomes the momentum transfer @f$t_{i-1}@f$ and the azimuth an
 * extra time-like invariant @f$\tilde s_i@f$ [1]. The pair @f$(\tilde s, t)@f$
 * alone does not fix the momenta. A recoil momentum (`momentum3`) supplies the
 * scattering plane. This block therefore peels one on-shell momentum off a
 * composite cluster and appears inside a recursive chain. The measure carries
 * an inverse square-root Gram determinant,
 *
 * @f[
 *   \int \mathrm{d}\Phi_2^{(\tilde s, t)} =
 *   \int \mathrm{d}\tilde s_i \int \mathrm{d}|t_{i-1}|\;
 *   \frac{1}{8\sqrt{-\Delta_4}}.
 * @f]
 *
 * @f$|t_{i-1}|@f$ is drawn with an @ref Invariant (from @p t_invariant_power /
 * @p t_mass / @p t_width) and @f$\tilde s_i@f$ with another (from
 * @p s_invariant_power / @p s_mass / @p s_width). The `discrete_choice` input
 * selects one of the two @f$\cos\phi@f$ branches. See Sec. 2.2.5 of [1],
 * following [2, 3].
 *
 * On its kinematic range @f$\tilde s_i@f$ is linear in @f$\cos\phi@f$, and
 * @f$8\sqrt{-\Delta_4} = \sqrt{\lambda}\,(\tilde s^{\max} - \tilde s^{\min})
 * \,|\sin\phi|@f$, so the measure is flat in @f$\phi@f$. Any density in
 * @f$\tilde s_i@f$ that is finite at the kinematic limits leaves an
 * integrable @f$1/|\sin\phi|@f$ in the weight, whose variance diverges
 * logarithmically. With @p arcsine_s23 (the default) the random number of
 * @f$\tilde s_i@f$ is first mapped by an arcsine map that is flat in
 * @f$\phi/2@f$ between the limits of the sampled range, and only then handed
 * to the power-law sampling of @ref Invariant (a Breit-Wigner is not
 * supported there). The weight stays bounded at the kinematic limits, and
 * the importance sampling of @f$\tilde s_i@f$ is kept.
 *
 * `batch` is the leading batch dimension.
 *
 * **Inputs**
 * - `discrete_choice` – `int`, shape `(batch,)` – which of the two two-body
 *   solutions to take.
 * - `random_s23` – `float`, shape `(batch,)` – random number for
 *   @f$\tilde s_i@f$.
 * - `random_t1` – `float`, shape `(batch,)` – random number for @f$|t_{i-1}|@f$.
 * - `mass1`, `mass2` – `float`, shape `(batch,)` – the two resolved masses.
 *
 * **Conditions**
 * - `momentum_in1` – `float`, shape `(batch, 4)` – first incoming momentum.
 * - `momentum_in2` – `float`, shape `(batch, 4)` – second incoming momentum;
 *   `momentum12`, the system p1 + p2, when @p p12_condition is true.
 * - `momentum3` – `float`, shape `(batch, 4)` – recoil momentum defining the
 *   scattering plane.
 * - `etmin_1`, `etmin_2` – `float`, shape `(batch,)` – transverse-energy cuts.
 *   Present only when @p has_cut is true.
 * - `drcut` – `float`, shape `(batch,)` – minimum angular separation.
 *   Present only when @p has_cut is true.
 * - `s23_min_cut` – `float`, shape `(batch,)` – minimum @f$\tilde s@f$.
 *   Present only when @p has_cut is true.
 *
 * **Outputs**
 * - `momentum1`, `momentum2` – `float`, shape `(batch, 4)` – the two resolved
 *   momenta.
 *
 * In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
 * Jacobian of the transformation.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895 (Sec. 2.2.5)
 * - [2] E. Byckling, K. Kajantie, Phys. Rev. 187 (1969) 2008,
 *   https://doi.org/10.1103/PhysRev.187.2008
 * - [3] R. Frederix, T. Vitos, "Leading-colour-based unweighted event
 *   generation for multi-parton tree-level processes",
 *   https://arxiv.org/abs/2409.12128
 */
class TwoToThreeParticleScattering : public Mapping {
public:
    /**
     * @param t_invariant_power Exponent of the @f$1/|t|^{\,p}@f$ sampling; see
     *                          @ref Invariant.
     * @param t_mass            t-channel propagator mass; see @ref Invariant.
     * @param t_width           t-channel propagator width; see @ref Invariant.
     * @param s_invariant_power Exponent of the @f$1/\tilde s^{\,p}@f$ sampling;
     *                          see @ref Invariant.
     * @param s_mass            s-channel resonance mass; see @ref Invariant.
     * @param s_width           s-channel resonance width; see @ref Invariant.
     * @param has_cut           If true, the `etmin_*`, `drcut` and `s23_min_cut`
     *                          conditions restrict the invariants to the region
     *                          passing the cuts; see @ref Cuts.
     * @param arcsine_s23       If true, remap the random number of
     *                          @f$\tilde s@f$ with the arcsine map in
     *                          @f$\phi@f$ before the invariant sampling, which
     *                          removes the @f$1/|\sin\phi|@f$ edge peak of the
     *                          weight. Power-law sampling only: requires
     *                          @p s_width = 0 (no resonance is expected in
     *                          this block).
     * @param p12_condition     If true, the second condition is the outgoing
     *                          system `momentum12` = p1 + p2 itself instead
     *                          of the second incoming momentum. A caller that
     *                          holds it more precisely than pa + pb - p3 (a
     *                          soft system next to the beams) keeps that
     *                          precision.
     * @param incoming_recoil   If true, `momentum3` is an incoming momentum
     *                          k (the other beam), and the second invariant is
     *                          the momentum transfer @f$(p_2 - k)^2 \le 0@f$
     *                          instead of @f$(p_2 + p_3)^2@f$; its absolute
     *                          value is sampled. With @p has_cut it is bounded
     *                          by @f$|(p_2 - k)^2| \ge \texttt{etmin\_2}^2 - m_2^2@f$,
     *                          the `drcut` and `s23_min_cut` conditions are
     *                          not used. Requires @p p12_condition. AmpliCol
     *                          builds a side of two particles this way, with
     *                          the opposite beam as the previous emission.
     * @param cut_aware_branch  With @p has_cut, form both azimuthal solutions
     *                          and keep the one passing the transverse-energy
     *                          floors `etmin_1`, `etmin_2` (AmpliCol's
     *                          gen23_one_step): `discrete_choice` only picks
     *                          when both pass. The weight assumes
     *                          `discrete_choice` is sampled uniformly; with a
     *                          learned distribution of it, leave this off.
     */
    TwoToThreeParticleScattering(
        double t_invariant_power = 0,
        double t_mass = 0,
        double t_width = 0,
        double s_invariant_power = 0,
        double s_mass = 0,
        double s_width = 0,
        bool has_cut = false,
        bool arcsine_s23 = true,
        bool p12_condition = false,
        bool incoming_recoil = false,
        bool cut_aware_branch = false
    );

    /// Number of discrete inputs (1): which of the two two-body solutions.
    std::size_t discrete_dim() const override { return 1; }

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

    std::array<Value, 3>
    split_conditions(FunctionBuilder& fb, const NamedVector<Value>& conditions) const;
    // Sampled and kinematic range of the second invariant: s23 itself, or
    // |(p2 - k)^2| for an incoming recoil. Arguments: the kinematic range of
    // s23, the squared mass of p2 and the conditions.
    std::array<Value, 4> sampling_ranges(
        FunctionBuilder& fb,
        Value s23_min,
        Value s23_max,
        Value m2_squared,
        const NamedVector<Value>& conditions
    ) const;
    Invariant _t_invariant;
    Invariant _s_invariant;
    double _s_power, _s_mass;
    bool _has_cut;
    bool _arcsine_s23;
    bool _p12_condition;
    bool _incoming_recoil;
    bool _cut_aware_branch;
};

} // namespace madspace
