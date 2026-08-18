#!/usr/bin/env python3
"""Closure test of the MadSpin production-polarisation restriction on p p > t t~.

Compares the *sum of the four fully polarised* samples

    p p > t{+} t~{+}   t{+} t~{-}   t{-} t~{+}   t{-} t~{-}

against the unpolarised  p p > t t~,  after decaying all five through MadSpin in
a density spinmode with leptonic tops.

For a massive fermion the helicity basis is {-1,+1}, so those four combinations
exhaust the *diagonal* of the production spin-density matrix rho(i,j).  The
unpolarised result is the full double sum, so

    sum(polarised)  =  sum_i rho(i,i) rho_dec(i,i)
    unpolarised     =  sum_i sum_j rho(i,j) rho_dec(i,j)

and the difference is exactly the i != j interference.  Observables built only
from the diagonal (helicity-axis polarisations and the k-k spin correlation)
must close; observables that pick up the transverse density-matrix elements
(n-n / r-r correlations, the lepton opening angle, lab Delta phi) must not.

Usage:  analyse_closure.py <runs_dir> <out_dir>
where <runs_dir> holds the five MadEvent directories
unpol/ pp/ pm/ mp/ mm/ each with Events/<run>_decayed_1/unweighted_events.lhe.gz
"""

import gzip
import math
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------
# sample bookkeeping
# --------------------------------------------------------------------------
POL = [('pp', r'$t\{+\}\,\bar t\{+\}$'),
       ('pm', r'$t\{+\}\,\bar t\{-\}$'),
       ('mp', r'$t\{-\}\,\bar t\{+\}$'),
       ('mm', r'$t\{-\}\,\bar t\{-\}$')]
ALL = ['unpol'] + [p[0] for p in POL]

LPLUS = (-11, -13)
LMINUS = (11, 13)


# --------------------------------------------------------------------------
# minimal 4-vector helpers (E, px, py, pz)  -- arrays of shape (N, 4)
# --------------------------------------------------------------------------
def boost_to_rest(p, ref):
    """Boost the 4-vectors p into the rest frame of the 4-vectors ref."""
    m = np.sqrt(np.maximum(ref[:, 0] ** 2 - (ref[:, 1:] ** 2).sum(1), 1e-12))
    b = -ref[:, 1:] / ref[:, 0:1]                       # boost vector
    b2 = (b ** 2).sum(1)
    gamma = ref[:, 0] / m
    bp = (b * p[:, 1:]).sum(1)
    gamma2 = np.where(b2 > 0, (gamma - 1.0) / np.maximum(b2, 1e-30), 0.0)
    out = np.empty_like(p)
    out[:, 0] = gamma * (p[:, 0] + bp)
    out[:, 1:] = p[:, 1:] + (gamma2 * bp)[:, None] * b + (gamma * p[:, 0])[:, None] * b
    return out


def unit(v):
    return v / np.linalg.norm(v, axis=1)[:, None]


# --------------------------------------------------------------------------
# LHE reading
# --------------------------------------------------------------------------
def read_lhe(path):
    """Return dict of (N,4) arrays for t, tbar, l+, l- plus the <init> xsec."""
    opener = gzip.open if path.endswith('.gz') else open
    t, tb, lp, lm = [], [], [], []
    xsec = xerr = None
    in_ev = False
    in_init = False
    init_lines = []
    with opener(path, 'rt', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.startswith('<init'):
                in_init = True
                continue
            if s.startswith('</init'):
                in_init = False
                continue
            if in_init:
                init_lines.append(s)
                continue
            if s.startswith('<event'):
                in_ev = True
                head = True
                ev = {}
                continue
            if s.startswith('</event'):
                in_ev = False
                t.append(ev['t'])
                tb.append(ev['tb'])
                lp.append(ev['lp'])
                lm.append(ev['lm'])
                continue
            if not in_ev:
                continue
            if head:
                head = False
                continue
            fl = s.split()
            if len(fl) < 13:
                continue
            pdg = int(fl[0])
            status = int(fl[1])
            p4 = (float(fl[9]), float(fl[6]), float(fl[7]), float(fl[8]))
            if pdg == 6 and status == 2:
                ev['t'] = p4
            elif pdg == -6 and status == 2:
                ev['tb'] = p4
            elif pdg in LPLUS and status == 1:
                ev['lp'] = p4
            elif pdg in LMINUS and status == 1:
                ev['lm'] = p4
    # <init>: second line holds XSECUP XERRUP XMAXUP LPRUP
    for s in init_lines[1:]:
        fl = s.split()
        if len(fl) >= 4:
            try:
                xsec, xerr = float(fl[0]), float(fl[1])
            except ValueError:
                pass
            break
    return dict(t=np.array(t), tb=np.array(tb),
                lp=np.array(lp), lm=np.array(lm), xsec=xsec, xerr=xerr)


def lhe_path(medir, run):
    """MadEvent leaves the LHE uncompressed while a run is in flight."""
    base = os.path.join(medir, 'Events', run, 'unweighted_events.lhe')
    for p in (base + '.gz', base):
        if os.path.exists(p):
            return p
    raise IOError('no LHE file in %s' % os.path.dirname(base))


def read_init_xsec(path):
    """XSECUP / XERRUP from the <init> block of an LHE file (full precision)."""
    opener = gzip.open if path.endswith('.gz') else open
    seen = False
    with opener(path, 'rt', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.startswith('<init'):
                seen = True
                next(f)                       # the beam/PDF line
                fl = next(f).split()
                return float(fl[0]), float(fl[1])
            if seen or s.startswith('<event'):
                break
    return None, None


def read_production_xsec(medir, run):
    """Cross section of the *production* (undecayed) run, with its MC error."""
    return read_init_xsec(lhe_path(medir, run))


# --------------------------------------------------------------------------
# observables
# --------------------------------------------------------------------------
def observables(ev):
    t, tb, lp, lm = ev['t'], ev['tb'], ev['lp'], ev['lm']
    ptt = t + tb                                    # the me_frame: partonic CM

    t_c = boost_to_rest(t, ptt)
    lp_c = boost_to_rest(lp, ptt)
    lm_c = boost_to_rest(lm, ptt)
    tb_c = boost_to_rest(tb, ptt)

    # lepton in its parent rest frame, reached *through* the ttbar CM so that
    # the quantisation axis is the parent direction in the me_frame
    lp_t = boost_to_rest(lp_c, t_c)
    lm_t = boost_to_rest(lm_c, tb_c)

    khat = unit(t_c[:, 1:])                         # top helicity axis
    zhat = np.zeros_like(khat)
    zhat[:, 2] = 1.0                                # beam axis (LO: preserved)
    cosT = khat[:, 2]
    sinT = np.sqrt(np.maximum(1.0 - cosT ** 2, 1e-12))
    sgn = np.sign(cosT)
    sgn[sgn == 0] = 1.0
    rhat = sgn[:, None] * (zhat - cosT[:, None] * khat) / sinT[:, None]
    nhat = sgn[:, None] * np.cross(zhat, khat) / sinT[:, None]

    up = unit(lp_t[:, 1:])
    um = unit(lm_t[:, 1:])

    o = {}
    o['cos_k_p'] = (up * khat).sum(1)               # l+ vs the t helicity axis
    o['cos_k_m'] = -(um * khat).sum(1)              # l- vs the tbar helicity axis
    o['cos_r_p'] = (up * rhat).sum(1)
    o['cos_n_p'] = (up * nhat).sum(1)
    o['ckk'] = (up * khat).sum(1) * (um * khat).sum(1)
    o['crr'] = (up * rhat).sum(1) * (um * rhat).sum(1)
    o['cnn'] = (up * nhat).sum(1) * (um * nhat).sum(1)
    o['cos_phi'] = (up * um).sum(1)                 # lepton opening angle
    dphi = np.abs(np.arctan2(lp[:, 2], lp[:, 1]) - np.arctan2(lm[:, 2], lm[:, 1]))
    o['dphi_lab'] = np.where(dphi > math.pi, 2 * math.pi - dphi, dphi)
    o['pt_t'] = np.sqrt(t[:, 1] ** 2 + t[:, 2] ** 2)
    o['m_tt'] = np.sqrt(np.maximum(ptt[:, 0] ** 2 - (ptt[:, 1:] ** 2).sum(1), 0))
    return o


PLOTS = [
    ('cos_k_p', r'$\cos\theta^{k}_{\ell^+}$  ($\ell^+$ vs. $t$ helicity axis)',
     np.linspace(-1, 1, 21), 'diagonal'),
    ('cos_k_m', r'$\cos\theta^{k}_{\ell^-}$  ($\ell^-$ vs. $\bar t$ helicity axis)',
     np.linspace(-1, 1, 21), 'diagonal'),
    ('ckk', r'$\cos\theta^{k}_{\ell^+}\cdot\cos\theta^{k}_{\ell^-}$  ($C_{kk}$)',
     np.linspace(-1, 1, 21), 'diagonal'),
    ('cnn', r'$\cos\theta^{n}_{\ell^+}\cdot\cos\theta^{n}_{\ell^-}$  ($C_{nn}$)',
     np.linspace(-1, 1, 21), 'off-diagonal'),
    ('crr', r'$\cos\theta^{r}_{\ell^+}\cdot\cos\theta^{r}_{\ell^-}$  ($C_{rr}$)',
     np.linspace(-1, 1, 21), 'off-diagonal'),
    ('cos_phi', r'$\cos\varphi_{\ell\ell}=\hat\ell^+\!\cdot\hat\ell^-$ (parent rest frames)',
     np.linspace(-1, 1, 21), 'off-diagonal'),
    ('cos_n_p', r'$\cos\theta^{n}_{\ell^+}$  ($\ell^+$ vs. normal axis)',
     np.linspace(-1, 1, 21), 'off-diagonal'),
    ('dphi_lab', r'$\Delta\phi(\ell^+,\ell^-)$ in the lab',
     np.linspace(0, math.pi, 21), 'off-diagonal'),
    ('pt_t', r'$p_T(t)$ [GeV]  (control)', np.linspace(0, 400, 21), 'control'),
    ('m_tt', r'$m(t\bar t)$ [GeV]  (control)', np.linspace(340, 1200, 21), 'control'),
]


def hist(vals, w, bins):
    n, _ = np.histogram(vals, bins=bins)
    s, _ = np.histogram(vals, bins=bins, weights=np.full(len(vals), w))
    e2, _ = np.histogram(vals, bins=bins, weights=np.full(len(vals), w * w))
    return s, np.sqrt(e2), n


def main():
    runs_dir = sys.argv[1]
    out_dir = sys.argv[2]
    run = sys.argv[3] if len(sys.argv) > 3 else 'closure'
    os.makedirs(out_dir, exist_ok=True)

    data, meta = {}, {}
    for tag in ALL:
        medir = os.path.join(runs_dir, tag)
        ev = read_lhe(lhe_path(medir, '%s_decayed_1' % run))
        n = len(ev['t'])
        pxs, perr = read_production_xsec(medir, run)
        meta[tag] = dict(n=n, xsec=ev['xsec'], xerr=ev['xerr'],
                         prod_xsec=pxs, prod_err=perr)
        data[tag] = observables(ev)
        print('%-6s  N=%6d   sigma_decayed = %10.5f +- %.5f pb   '
              'sigma_prod = %s' % (tag, n, ev['xsec'], ev['xerr'], pxs))

    # ---------------- total-rate closure -----------------------------------
    lines = []
    s_pol = sum(meta[t]['xsec'] for t, _ in POL)
    e_pol = math.sqrt(sum(meta[t]['xerr'] ** 2 for t, _ in POL))
    s_unp, e_unp = meta['unpol']['xsec'], meta['unpol']['xerr']
    lines.append('decayed cross sections (pb, from the LHE <init> block)')
    for tag in ALL:
        lines.append('  %-6s %12.5f +- %.5f   (%d events)'
                     % (tag, meta[tag]['xsec'], meta[tag]['xerr'], meta[tag]['n']))
    lines.append('  sum(4 pol) %10.5f +- %.5f' % (s_pol, e_pol))
    d = s_pol - s_unp
    e = math.sqrt(e_pol ** 2 + e_unp ** 2)
    lines.append('  sum/unpol  = %.6f +- %.6f   (%.2f sigma from 1)'
                 % (s_pol / s_unp, (s_pol / s_unp) * math.sqrt(
                     (e_pol / s_pol) ** 2 + (e_unp / s_unp) ** 2), d / e))
    lines.append('')
    lines.append('production cross sections (pb, before MadSpin)')
    sp = sum(meta[t]['prod_xsec'] for t, _ in POL)
    ep = math.sqrt(sum(meta[t]['prod_err'] ** 2 for t, _ in POL))
    for tag in ALL:
        lines.append('  %-6s %12.5f +- %.5f' % (tag, meta[tag]['prod_xsec'],
                                                meta[tag]['prod_err']))
    lines.append('  sum(4 pol) %10.5f +- %.5f' % (sp, ep))
    su, eu = meta['unpol']['prod_xsec'], meta['unpol']['prod_err']
    lines.append('  sum/unpol  = %.6f +- %.6f   (%.2f sigma from 1)'
                 % (sp / su, (sp / su) * math.sqrt((ep / sp) ** 2 + (eu / su) ** 2),
                    (sp - su) / math.sqrt(ep ** 2 + eu ** 2)))
    print('\n'.join(lines))

    # ------- per-sample analytic cross-check of the polarisation ------------
    # For a top of definite helicity h the charged lepton (alpha_l = 1) follows
    # (1 + h cos theta)/2 in the top rest frame, so <cos theta> = h/3, and the
    # two helicities are uncorrelated within one fully polarised sample, so
    # <cos+ cos-> = <cos+><cos->.
    lines.append('')
    lines.append('per-sample check against the analytic pure-helicity result')
    lines.append('  %-6s %22s %22s %22s' % ('sample', '<cos_k(l+)>',
                                            '<cos_k(l-)>', '<cos_k+ * cos_k->'))
    for tag in ALL:
        row = ['  %-6s' % tag]
        for key in ('cos_k_p', 'cos_k_m', 'ckk'):
            v = data[tag][key]
            row.append('%14.4f +- %-6.4f' % (v.mean(),
                                             v.std() / math.sqrt(len(v))))
        lines.append(' '.join(row))
    lines.append('  expected: t{+} -> <cos_k(l+)> = +1/3, t{-} -> -1/3;')
    lines.append('            tbar{+} -> <cos_k(l-)> = -1/3, tbar{-} -> +1/3')
    lines.append('            (each lepton is measured against its own parent\'s')
    lines.append('             helicity axis in the me_frame = the ttbar CM;')
    lines.append('             the sign flip on the l- side is the antitop')
    lines.append('             analysing power)')
    lines.append('            <cos_k+ * cos_k-> must factorise into the product')
    lines.append('             of the two single-particle means inside a fully')
    lines.append('             polarised sample (no correlation is left)')

    # ---------------- means -------------------------------------------------
    lines.append('')
    lines.append('means (weighted by cross section), sum of the four polarised')
    lines.append('samples vs. the unpolarised one')
    lines.append('  %-12s %18s %18s %10s' % ('observable', 'sum(pol)',
                                             'unpolarised', 'pull'))
    for key, label, bins, kind in PLOTS:
        wsum = 0.0
        m = 0.0
        sig = {}
        for tag, _ in POL:
            w = meta[tag]['xsec'] / meta[tag]['n']
            v = data[tag][key]
            m += w * v.sum()
            sig[tag] = w * v.std() * math.sqrt(len(v))
            wsum += w * len(v)
        mean_p = m / wsum
        # conservative: the two members of each parity pair share the production
        # phase-space points (same MadEvent seed), so add them linearly
        err_p = math.sqrt((sig['pp'] + sig['mm']) ** 2
                          + (sig['pm'] + sig['mp']) ** 2) / wsum
        w = meta['unpol']['xsec'] / meta['unpol']['n']
        v = data['unpol'][key]
        mean_u = v.mean()
        err_u = v.std() / math.sqrt(len(v))
        pull = (mean_p - mean_u) / math.sqrt(err_p ** 2 + err_u ** 2)
        lines.append('  %-12s %10.5f+-%-7.5f %10.5f+-%-7.5f %9.2f  [%s]'
                     % (key, mean_p, err_p, mean_u, err_u, pull, kind))

    open(os.path.join(out_dir, 'closure_numbers.txt'), 'w').write(
        '\n'.join(lines) + '\n')
    print('\n'.join(lines[-len(PLOTS) - 4:]))

    # ---------------- plots -------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'pp': '#d62728', 'pm': '#ff7f0e', 'mp': '#2ca02c', 'mm': '#9467bd'}
    kindcol = {'diagonal': '#2ca02c', 'off-diagonal': '#d62728',
               'control': '#7f7f7f'}

    def build(key, bins):
        """per-sample hists, sum, sum err, unpolarised, unpolarised err.

        MadEvent ran with one seed for all five samples, and QCD is parity
        conserving, so |M(+,+)|^2 = |M(-,-)|^2 and |M(+,-)|^2 = |M(-,+)|^2 at
        every phase-space point: pp/mm (and pm/mp) come out of MadEvent with
        *identical* production momenta and only the helicity label differs
        (verified on the LHE files).  Their decay chains are independent draws,
        but the production kinematics inside each pair is 100% correlated, so
        the naive sqrt(sum w^2 n) understates the error on the sum for anything
        that depends on the production momenta.  The conservative error adds the
        two members of each pair linearly (the fully-correlated bound) and is
        what the plots show.
        """
        per, pere = {}, {}
        tot = np.zeros(len(bins) - 1)
        for tag, _ in POL:
            w = meta[tag]['xsec'] / meta[tag]['n']
            s, e, _n = hist(data[tag][key], w, bins)
            per[tag], pere[tag] = s, e
            tot += s
        naive = np.sqrt(sum(pere[t] ** 2 for t, _ in POL))
        cons = np.sqrt((pere['pp'] + pere['mm']) ** 2
                       + (pere['pm'] + pere['mp']) ** 2)
        w = meta['unpol']['xsec'] / meta['unpol']['n']
        u, ue, _n = hist(data['unpol'][key], w, bins)
        return per, tot, cons, u, ue, naive

    def ratio(tot, tot_e, u, ue):
        r = tot / np.where(u == 0, np.nan, u)
        re = r * np.sqrt((tot_e / np.where(tot == 0, np.nan, tot)) ** 2
                         + (ue / np.where(u == 0, np.nan, u)) ** 2)
        return r, re

    def draw(ax, rax, key, label, bins, kind, legend=True, small=False):
        ctr = 0.5 * (bins[1:] + bins[:-1])
        wid = bins[1] - bins[0]
        per, tot, tot_e, u, ue, naive = build(key, bins)
        for tag, lab in POL:
            ax.step(bins, np.r_[per[tag][0], per[tag]] / wid, where='pre',
                    lw=1.0, color=colors[tag], alpha=0.85, label=lab)
        ax.errorbar(ctr, u / wid, yerr=ue / wid, fmt='o', ms=3.5, color='k',
                    lw=1.1, capsize=2, label=r'unpolarised $pp\to t\bar t$')
        ax.errorbar(ctr, tot / wid, yerr=tot_e / wid, fmt='s', ms=3.5,
                    mfc='none', color='#1f77b4', lw=1.1, capsize=2,
                    label='sum of the 4 polarised')
        ax.set_ylabel(r'$d\sigma/dX$ [pb]', fontsize=8 if small else 10)
        if kind == 'control':
            ax.set_yscale('log')
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7 if small else 9)
        if legend:
            ax.legend(fontsize=7 if small else 8, ncol=2, frameon=False)
        r, re = ratio(tot, tot_e, u, ue)
        rax.axhline(1.0, color='k', lw=1.0)
        rax.errorbar(ctr, r, yerr=re, fmt='o', ms=3.5, color=kindcol[kind],
                     lw=1.1, capsize=2)
        rax.set_ylabel('sum / unpol.', fontsize=8 if small else 9)
        rax.set_xlabel(label, fontsize=8 if small else 10)
        rax.grid(alpha=0.25)
        rax.tick_params(labelsize=7 if small else 9)
        lo, hi = np.nanmin(r - re), np.nanmax(r + re)
        pad = 0.15 * max(hi - lo, 0.02)
        rax.set_ylim(min(lo - pad, 0.97), max(hi + pad, 1.03))
        good = np.isfinite(r) & np.isfinite(re) & (re > 0)
        chi2 = (((r[good] - 1) / re[good]) ** 2).sum()
        rn, ren = ratio(tot, naive, u, ue)
        chi2n = (((rn[good] - 1) / ren[good]) ** 2).sum()
        return (chi2, good.sum(), np.nanmax(np.abs(r - 1)), np.nanmean(re),
                chi2n)

    lines.append('')
    lines.append('per-bin ratio (sum of the 4 polarised) / unpolarised')
    lines.append('  (chi2 uses the conservative correlated error on the sum;'
                 ' "naive" is the uncorrelated one)')
    lines.append('  %-10s %16s %12s %10s %12s %s'
                 % ('observable', 'chi2 / nbins', 'chi2 naive', 'max|r-1|',
                    '<stat.err>', 'expectation'))
    for key, label, bins, kind in PLOTS:
        fig, (ax, rax) = plt.subplots(
            2, 1, figsize=(7.2, 6.4), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1.25], hspace=0.06))
        chi2, nb, mx, avge, chi2n = draw(ax, rax, key, label, bins, kind)
        ax.set_title('MadSpin polarisation closure  $pp\\to t\\bar t$, 13 TeV, '
                     'dileptonic\n%s   [%s]' % (label, kind), fontsize=10)
        fig.savefig(os.path.join(out_dir, 'closure_%s.png' % key), dpi=160,
                    bbox_inches='tight')
        plt.close(fig)
        msg = ('  %-10s %10.1f / %-5d %12.1f %10.4f %12.4f  %s'
               % (key, chi2, nb, chi2n, mx, avge, kind))
        print(msg)
        lines.append(msg)

    # ---- one summary figure ------------------------------------------------
    keys = ['cos_k_p', 'ckk', 'cnn', 'cos_phi', 'dphi_lab', 'm_tt']
    sel = [p for p in PLOTS if p[0] in keys]
    sel.sort(key=lambda p: keys.index(p[0]))
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 1.3, 1.0, 3, 1.3],
                          hspace=0.30, wspace=0.26)
    rows = {0: (0, 1), 1: (3, 4)}
    handles = None
    for i, (key, label, bins, kind) in enumerate(sel):
        row, col = divmod(i, 3)
        ra, rb = rows[row]
        ax = fig.add_subplot(gs[ra, col])
        rax = fig.add_subplot(gs[rb, col], sharex=ax)
        draw(ax, rax, key, label, bins, kind, legend=False, small=True)
        if handles is None:
            handles = ax.get_legend_handles_labels()
        ax.set_title('[%s]' % kind, fontsize=9, color=kindcol[kind])
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.legend(*handles, fontsize=9, ncol=6, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 0.935))
    fig.suptitle('MadSpin production-polarisation closure: sum of '
                 r'$t\{+\}\bar t\{+\},\,t\{+\}\bar t\{-\},\,t\{-\}\bar t\{+\},'
                 r'\,t\{-\}\bar t\{-\}$  vs.  unpolarised $pp\to t\bar t$'
                 '\n(13 TeV, LO, MadSpin spinmode=madspin, '
                 r'$t\to b\,\ell\nu$ both sides)', fontsize=12, y=0.99)
    fig.savefig(os.path.join(out_dir, 'closure_summary.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    open(os.path.join(out_dir, 'closure_numbers.txt'), 'w').write(
        '\n'.join(lines) + '\n')
    print('\nwrote plots + closure_numbers.txt to', out_dir)


if __name__ == '__main__':
    main()
