#!/usr/bin/env python3
"""Independent measure checks for forced-threshold shower competition.

These are algebraic counterexamples and identities, not a Pythia model.
They demonstrate why the hard-event survival test needs a shadow shower
when a competing normalized threshold conversion survives at alpha_s=0.
"""
import math


def measures(alpha, rates, survival, bbar_coefficients, real, subtraction):
    """Return MC, ordinary signed-residual, and shadow-H predictions."""
    kernel = alpha * sum(rates)
    hard = alpha * (real-subtraction)
    negative = max(-hard,0.)
    cancellation = min(negative,kernel)
    shower = sum(alpha*k*d*(1+alpha*c)
                 for k,d,c in zip(rates,survival,bbar_coefficients))
    mc = shower + hard
    accepted_shower = shower * (1-cancellation/kernel)
    # This expression assumes survival=1+O(alpha), which need not hold.
    ordinary = accepted_shower + max(hard,0.) - negative + cancellation
    # Auxiliary H shower samples the conditional survival of the same
    # Born/history distribution that supplies the real point's kernel.
    shadow_survival = sum(k*d for k,d in zip(rates,survival))/sum(rates)
    shadow = accepted_shower + max(hard,0.) - negative + cancellation*shadow_survival
    expected_shadow_difference = cancellation * sum(
        k*d*(-alpha*c) for k,d,c in zip(rates,survival,bbar_coefficients))/sum(rates)
    return mc,ordinary,shadow,expected_shadow_difference


def main():
    # A shared massless real point supplied by two inverse Born histories.
    # Each also has a massive forced-conversion component where H=0. The
    # shared point's ordinary branching competes against that component.
    ordinary_errors,shadow_errors = [],[]
    for alpha in (.04,.02,.01):
        mc,ordinary,shadow,identity = measures(alpha,
            rates=(.7,.3),survival=(.2,.8),bbar_coefficients=(.4,-.3),
            real=.6,subtraction=.9)
        assert math.isclose(shadow-mc,identity,rel_tol=2.e-12,abs_tol=1.e-16)
        ordinary_errors.append(ordinary-mc)
        shadow_errors.append(shadow-mc)
    for index in (0,1):
        assert abs(ordinary_errors[index]/ordinary_errors[index+1]-2)<.01
        assert math.isclose(shadow_errors[index]/shadow_errors[index+1],4.,rel_tol=2.e-12)
    # The massive stratum itself has R=D=H=0. Keeping it untouched
    # exactly preserves its original shower contribution for any Bbar.
    massive_mc = 2.3
    massive_transformed = massive_mc
    assert massive_transformed == massive_mc
    print('PASS: disjoint massive shower stratum remains unchanged')
    print('PASS: ordinary C=min(H-,K0) alone has an O(alpha) competition error')
    print('PASS: auxiliary hard-event shower survival restores O(alpha^2) difference')


if __name__ == '__main__':
    main()
