#!/usr/bin/env python3
"""Test one claim about the ``blocks_cnn`` pane, from the committed data alone.

The draft section says of that pane

    "the four diagonal blocks and the four mixed blocks with a single I index
     sit on zero, and the entire C_nn signal is carried by the (I,I) block"

and, of the observable,

    "C_nn ... is built entirely from off-diagonal entries of the production
     density matrix."

The second sentence is a statement about the *coefficient*; the first is a
statement about a *distribution in picobarns*, and the two are not the same
object.  This script separates them.

    Part A  the 2x2 x 2x2 algebra, done numerically: which entries of the
            production density matrix C_nn and C_kk actually touch.
    Part B  the decisive table, from ``data/histograms.npz``: for each of the
            nine blocks, the INTEGRAL of the plotted distribution and the
            block's contribution to the mean, with MC errors.
    Part C  one figure, ``plots/claim_cnn_vs_ckk.pdf``, making the point.

Usage:  check_cnn_claim.py [data_dir] [out_dir]     (defaults: data/  plots/)

Nothing here re-runs MadSpin and nothing is fitted.
"""

import json
import math
import os
import sys

import numpy as np


# ==========================================================================
# Part A -- the algebra
# ==========================================================================
# rho = 1/4 [ 1 (x) 1 + B+.sigma (x) 1 + 1 (x) B-.sigma + C_ij sigma_i (x)
# sigma_j ], Tr rho = 1, helicity basis {++, +-, -+, --} with the quantisation
# axis z = k, and (x, y, z) = (r, n, k) as MADSPIN_SEQUENTIAL_PLAN.md 13.15
# and Bernreuther-Heisler-Si (arXiv:1508.05271) both take it.  Then
# C_ij = Tr[rho sigma_i (x) sigma_j].

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.eye(2, dtype=complex)
S = {'r': SX, 'n': SY, 'k': SZ}
KET = ['++', '+-', '-+', '--']


def rho_of(bplus, bminus, C):
    """Build the 4x4 density matrix from (B+, B-, C_ij)."""
    r = np.kron(ID, ID).astype(complex)
    for a, i in enumerate('rnk'):
        r = r + bplus[a] * np.kron(S[i], ID) + bminus[a] * np.kron(ID, S[i])
        for b, j in enumerate('rnk'):
            r = r + C[a, b] * np.kron(S[i], S[j])
    return r / 4.0


def coeff(rho, i, j):
    return np.trace(rho @ np.kron(S[i], S[j])).real


def part_a(log):
    p = log.append
    p('=' * 74)
    p('Part A -- which entries of rho does C_nn touch?  (exact, numeric algebra)')
    p('=' * 74)
    p('')
    p('rho = 1/4 [1x1 + B+.sigma x 1 + 1 x B-.sigma + C_ij sigma_i x sigma_j],')
    p('helicity basis {++, +-, -+, --} quantised along k, (x,y,z) = (r,n,k),')
    p('so C_ij = Tr[rho sigma_i x sigma_j].')
    p('')
    for i, j, name in (('n', 'n', 'C_nn'), ('k', 'k', 'C_kk'), ('r', 'r', 'C_rr')):
        M = np.kron(S[i], S[j])
        p('  sigma_%s (x) sigma_%s, as a 4x4 matrix on {++, +-, -+, --}:' % (i, j))
        for a in range(4):
            p('    %-3s  %s' % (KET[a], '  '.join(
                '%6.6s' % (('%g' % M[a, b].real) if M[a, b].imag == 0
                           else ('%gi' % M[a, b].imag)) for b in range(4))))
        # C = Tr[rho M] = sum_{ab} rho_ab M_ba, so entry (a,b) of rho is
        # multiplied by M[b, a].
        terms = []
        for a in range(4):
            for b in range(4):
                if M[b, a] != 0:
                    terms.append('%+g rho(%s,%s)' % (M[b, a].real, KET[a], KET[b]))
        p('    => %s = %s' % (name, ' '.join(terms)))
        flips = set()
        for a in range(4):
            for b in range(4):
                if M[b, a] != 0:
                    flips.add(sum(1 for c in range(2) if KET[a][c] != KET[b][c]))
        p('    entries touched flip %s helicity index/indices.'
          % ' or '.join(str(f) for f in sorted(flips)))
        p('')

    # the identity recorded in MADSPIN_SEQUENTIAL_PLAN.md section 13.15
    rng = np.random.default_rng(20260819)
    worst = 0.0
    for _ in range(2000):
        bp, bm = rng.uniform(-.3, .3, 3), rng.uniform(-.3, .3, 3)
        C = rng.uniform(-.3, .3, (3, 3))
        r = rho_of(bp, bm, C)
        cnn, crr = coeff(r, 'n', 'n'), coeff(r, 'r', 'r')
        ipp, imm = KET.index('++'), KET.index('--')
        ipm, imp = KET.index('+-'), KET.index('-+')
        worst = max(worst,
                    abs(4 * r[ipp, imm].real - (crr - cnn)),
                    abs(4 * r[ipm, imp].real - (crr + cnn)),
                    abs(cnn - (2 * r[ipm, imp].real - 2 * r[ipp, imm].real)),
                    abs(cnn - C[1, 1]), abs(np.trace(r).real - 1))
    p('  identities of MADSPIN_SEQUENTIAL_PLAN.md 13.15, on 2000 random rho:')
    p('    4 Re rho(++,--) = C_rr - C_nn')
    p('    4 Re rho(+-,-+) = C_rr + C_nn')
    p('    => C_nn = 2 Re rho(+-,-+) - 2 Re rho(++,--)      VERIFIED')
    p('    worst residual over the 2000 draws (incl. Tr rho = 1): %.2e' % worst)
    p('')

    # a diagonal block is a product of helicity eigenstates
    p('  a DIAGONAL block, e.g. (D+,D+) = |+ +><+ +|, is a product of helicity')
    p('  eigenstates, so on each leg <sigma_n> = <sigma_r> = 0:')
    for ket in KET:
        v = np.zeros(4, dtype=complex)
        v[KET.index(ket)] = 1.0
        r = np.outer(v, v.conj())
        p('    (D%s,D%s): C_kk = %+.1f   C_nn = %+.1f   C_rr = %+.1f   '
          'B+_n = %+.1f   B-_n = %+.1f'
          % (ket[0], ket[1], coeff(r, 'k', 'k'), coeff(r, 'n', 'n'),
             coeff(r, 'r', 'r'),
             np.trace(r @ np.kron(SY, ID)).real,
             np.trace(r @ np.kron(ID, SY)).real))
    p('  C_nn vanishes on every diagonal block; C_kk does not.  That is the')
    p('  whole content of "C_nn is a double-helicity-flip quantity".')
    p('')
    p('  BUT: the diagonal blocks still carry a cross section, and')
    p('  d(sigma)/d(cos th_n+ cos th_n-) for them is NOT zero -- only its')
    p('  first moment is.  Part B measures both.')
    p('')


# ==========================================================================
# Part B -- the measurement
# ==========================================================================
TAGS = [('pp', '(D+,D+)'), ('pm', '(D+,D-)'), ('mp', '(D-,D+)'),
        ('mm', '(D-,D-)'), ('i_dp', '(I,D+)'), ('i_dm', '(I,D-)'),
        ('dp_i', '(D+,I)'), ('dm_i', '(D-,I)'), ('ii', '(I,I)')]
DIAG = ['pp', 'pm', 'mp', 'mm']
SINGLE_I = ['i_dp', 'i_dm', 'dp_i', 'dm_i']


def part_b(z, log):
    """Per block: integral of the plotted distribution, and its first moment.

    ``mom__<tag>__<key>`` holds [sum w, sum w O, sum w^2, sum w^2 O,
    sum w^2 O^2, N] with w already divided by N_file, so

        sum w        = the integral of the plotted curve, in pb
        sum w O      = the integral of X * dsigma/dX, in pb -- the first moment
        sum w O / sigma_9blocks = that block's ADDITIVE contribution to the
                                  mean of the observable over the full sample.
    """
    p = log.append

    def mom(tags, key):
        return sum(z['mom__%s__%s' % (t, key)] for t in tags)

    out = {}
    for key, pretty in (('cnn', 'C_nn: X = cos(th_n,l+) cos(th_n,l-)'),
                        ('ckk', 'C_kk: X = cos(th_k,l+) cos(th_k,l-)')):
        sig9 = sum(mom([t], key)[0] for t, _ in TAGS)
        p('=' * 74)
        p('Part B -- %s' % pretty)
        p('=' * 74)
        p('')
        p('  %-9s %14s %10s %16s %10s %7s' %
          ('block', 'integral [pb]', '+-', 'contrib to <X>', '+-', 'sigmas'))
        rows = []
        for t, lab in TAGS:
            sw, swo, sw2, _sw2o, sw2o2, _n = mom([t], key)
            c, dc = swo / sig9, math.sqrt(max(sw2o2, 0.0)) / sig9
            rows.append((lab, sw, math.sqrt(sw2), c, dc))
            p('  %-9s %14.4f %10.4f %16.5f %10.5f %+7.1f'
              % (lab, sw, math.sqrt(sw2), c, dc, c / dc if dc > 0 else 0.0))
        p('')
        for grp, name in ((DIAG, '4 diagonal'), (SINGLE_I, '4 single-I'),
                          (['ii'], '(I,I)'),
                          (DIAG + SINGLE_I + ['ii'], 'all 9 blocks')):
            sw, swo, sw2, _a, sw2o2, _n = mom(grp, key)
            c, dc = swo / sig9, math.sqrt(max(sw2o2, 0.0)) / sig9
            p('  %-12s integral = %9.4f +- %-7.4f   contrib to <X> = '
              '%+.5f +- %.5f  (%+.1f sd)'
              % (name, sw, math.sqrt(sw2), c, dc, c / dc if dc > 0 else 0.0))
        swu, swou, _b, _c2, su2o2, _d = mom(['unpol'], key)
        p('  %-12s integral = %9.4f              <X> = %+.5f +- %.5f'
          % ('unpolarised', swu, swou / swu, math.sqrt(su2o2) / swu))
        p('')
        out[key] = rows
    return out


def part_b_symmetry(z, log):
    """A non-zero histogram with a zero mean is a SYMMETRIC one.

    Mirror each diagonal block's histogram about X = 0 and compare bin by bin.
    For C_nn the four blocks are symmetric (so: large everywhere, zero first
    moment); for C_kk the very same blocks are grossly asymmetric.
    """
    p = log.append
    p('=' * 74)
    p('Part B2 -- why the diagonal blocks have a zero first moment but a')
    p('           very non-zero histogram: the histogram is SYMMETRIC in X')
    p('=' * 74)
    p('')
    p('  %-9s %22s %22s %14s' % ('block', 'chi2 h(-X) vs h(+X), C_nn',
                                 'the same for C_kk', 'peak bin [pb]'))
    for tag, lab in TAGS[:4]:
        c = []
        for key in ('cnn', 'ckk'):
            h = z['sumw__%s__%s' % (tag, key)]
            e = np.sqrt(z['sumw2__%s__%s' % (tag, key)])
            n = len(h)
            c.append(sum((h[n - 1 - i] - h[i]) ** 2
                         / (e[i] ** 2 + e[n - 1 - i] ** 2)
                         for i in range(n // 2)))
        h = z['sumw__%s__cnn' % tag]
        e = np.sqrt(z['sumw2__%s__cnn' % tag])
        p('  %-9s %14.1f / %-5d %14.1f / %-5d %8.3f +- %.3f'
          % (lab, c[0], len(h) // 2, c[1], len(h) // 2,
             h.max(), e[h.argmax()]))
    p('')
    p('  Symmetric in C_nn, grossly asymmetric in C_kk -- the same four samples,')
    p('  the same binning.  Zero mean, very much non-zero distribution.')
    p('')


def part_b_calibration(z, log):
    """The 1/9 factor, calibrated on the blocks whose coefficients are known.

    A diagonal block is a product of helicity eigenstates, so C_kk = +-1 and
    B+_k, B-_k have unit magnitude exactly.  Measuring <cos th^k> = 1/3 and
    <cos th^k_+ cos th^k_-> = +-1/9 on them fixes both the factor and the SIGN
    convention that relates the plotted mean to the coefficient.
    """
    p = log.append
    p('=' * 74)
    p('Part B3 -- the 1/9, calibrated on the four blocks with known C_kk')
    p('=' * 74)
    p('')
    p('  %-9s %20s %20s %22s %9s' % ('block', '<cos th^k_l+>',
                                     '<cos th^k_l-> (own sign)',
                                     '<cos th^k_l+ cos th^k_l->', 'expected'))
    mags = []
    for tag, lab, exp in (('pp', '(D+,D+)', +1), ('pm', '(D+,D-)', -1),
                          ('mp', '(D-,D+)', -1), ('mm', '(D-,D-)', +1)):
        vals = []
        for key in ('cos_k_p', 'cos_k_m', 'ckk'):
            sw, swo, sw2, sw2o, sw2o2, _n = z['mom__%s__%s' % (tag, key)]
            m = swo / sw
            var = (sw2o2 - 2 * m * sw2o + m * m * sw2) / (sw * sw)
            vals.append((m, math.sqrt(max(var, 0.0))))
        # cos_k_m is stored as -(u_- . k); print it with its own sign
        mags.append((abs(vals[2][0]), vals[2][1]))
        p('  %-9s %11.5f +- %-6.5f %11.5f +- %-6.5f %13.5f +- %-6.5f %+9.5f'
          % (lab, vals[0][0], vals[0][1], -vals[1][0], vals[1][1],
             vals[2][0], vals[2][1], exp / 9.0))
    mean = sum(m for m, _ in mags) / 4.0
    err = math.sqrt(sum(e * e for _, e in mags)) / 4.0
    p('')
    p('  mean of the four magnitudes: %.5f +- %.5f   (1/9 = %.5f, %+.1f sd)'
      % (mean, err, 1 / 9.0, (mean - 1 / 9.0) / err))
    p('  => in THIS code\'s convention, C_ij = Tr[rho sigma_i x sigma_j] with')
    p('     both leptons projected on the SAME (k,n,r) triad,')
    p('        < cos th^i_l+ cos th^j_l- > = + C_ij / 9 .')
    for key, name in (('cnn', 'C_nn'), ('ckk', 'C_kk')):
        sw, swo, sw2, sw2o, sw2o2, _n = z['mom__unpol__%s' % key]
        m = swo / sw
        var = (sw2o2 - 2 * m * sw2o + m * m * sw2) / (sw * sw)
        p('     unpolarised %s = %+.4f +- %.4f' % (name, 9 * m,
                                                   9 * math.sqrt(max(var, 0))))
    p('  Several references write the double-differential distribution with')
    p('  -C_ij cos th^i_+ cos th^j_-; their C_ij is then minus this one.  No')
    p('  conclusion here depends on that sign.')
    p('')

# ==========================================================================
# Part C -- the figure
# ==========================================================================
def part_c(z, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({'font.family': 'serif', 'font.size': 11,
                         'mathtext.fontset': 'cm'})

    def mom(t, key):
        return z['mom__%s__%s' % (t, key)]

    labels = [r'$(D^+,D^+)$', r'$(D^+,D^-)$', r'$(D^-,D^+)$', r'$(D^-,D^-)$',
              r'$(I,D^+)$', r'$(I,D^-)$', r'$(D^+,I)$', r'$(D^-,I)$',
              r'$(I,I)$']
    x = np.arange(9)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(7.4, 8.6))
    fig.subplots_adjust(hspace=0.42)

    def banner(ax, txt):
        ax.text(0.5, 1.03, txt, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=10)

    ax = axes[0]
    sig = [mom(t, 'cnn')[0] for t, _ in TAGS]
    err = [math.sqrt(mom(t, 'cnn')[2]) for t, _ in TAGS]
    cols = ['0.55'] * 4 + ['tab:blue'] * 4 + ['tab:red']
    ax.bar(x, sig, yerr=err, color=cols, edgecolor='k', linewidth=0.6)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel(r'$\int\mathrm{d}\sigma$  [pb]')
    ax.set_ylim(-1.4, 9.6)
    banner(ax, r'(a) area under the curve drawn in the $C_{nn}$ pane -- '
               r'the four diagonal blocks carry all of it')
    for i, s in enumerate(sig):
        ax.annotate('%.2f' % s, (i, s), textcoords='offset points',
                    xytext=(0, 5 if s >= 0 else -13), ha='center', fontsize=8)

    sig9 = sum(sig)
    for ax, key, ttl in (
            (axes[1], 'cnn',
             r'(b) contribution to $\langle\cos\theta^n_{\ell^+}'
             r'\cos\theta^n_{\ell^-}\rangle\ (\propto C_{nn})$ -- '
             r'only $(I,I)$'),
            (axes[2], 'ckk',
             r'(c) the contrast: $\langle\cos\theta^k_{\ell^+}'
             r'\cos\theta^k_{\ell^-}\rangle\ (\propto C_{kk})$ -- '
             r'only the diagonal blocks')):
        c = np.array([mom(t, key)[1] for t, _ in TAGS]) / sig9
        dc = np.array([math.sqrt(max(mom(t, key)[4], 0.0))
                       for t, _ in TAGS]) / sig9
        ax.bar(x, c, yerr=dc, color=cols, edgecolor='k', linewidth=0.6)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_ylabel(r'contribution to $\langle X\rangle$')
        banner(ax, ttl)
        lo, hi = min(c - dc), max(c + dc)
        pad = 0.38 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        for i in range(9):
            ax.annotate('%+.1f$\\sigma$' % (c[i] / dc[i] if dc[i] > 0 else 0),
                        (i, c[i]), textcoords='offset points',
                        xytext=(0, 6 if c[i] >= 0 else -14), ha='center',
                        fontsize=7.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25)
    axes[2].set_xlabel('grey: diagonal blocks     blue: single-$I$ blocks '
                       '    red: the doubly-interfering $(I,I)$ block',
                       fontsize=9, color='0.3', labelpad=10)
    for f in ('claim_cnn_vs_ckk.pdf', 'claim_cnn_vs_ckk.png'):
        fig.savefig(os.path.join(out_dir, f), dpi=160, bbox_inches='tight')
    plt.close(fig)
    return os.path.join(out_dir, 'claim_cnn_vs_ckk.pdf')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ddir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'data')
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, 'plots')
    z = np.load(os.path.join(ddir, 'histograms.npz'))
    log = []
    part_a(log)
    part_b(z, log)
    part_b_symmetry(z, log)
    part_b_calibration(z, log)
    fig = part_c(z, out)
    log.append('wrote %s' % fig)
    text = '\n'.join(log)
    print(text)
    with open(os.path.join(out, 'claim_cnn_numbers.txt'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
