#!/usr/bin/env python3
"""Histogram the ten samples of the MadSpin interference-closure test (v2).

Reads the decayed LHE files produced by ``run_interference.sh`` and writes ONE
``histograms.npz`` holding, per sample and per observable, the bin edges, the
sum of weights, the sum of weights squared and the raw entry count -- so the
plots and every number quoted can be regenerated without re-running MadSpin.

Usage:  analyse_interference.py <runs_dir> <out_dir> [run_name]

Normalisation -- there isn't one any more
-----------------------------------------
This is the whole point of the redo.  The first run of this test had to put the
pure-interference samples on the same footing as the diagonal ones by hand: the
mode accepted/rejected against a ``max_weight`` that was not reported anywhere
and wrote unit-magnitude signed weights, so every interference histogram had to
be multiplied by ``max_weight / c`` with ``c`` *measured* from the unweighting
efficiency of a separate reference run.  That machinery is gone.

The mode is now fully weighted: every trial is kept and the event carries

    w = sigma_ref * BR * W / c

with ``W`` the signed production/decay convolution and ``c = <W>`` the
decay-side constant the maximum-weight scan already measures.  Under MG5's
``IDWTUP = -4`` convention (the cross-section is the MEAN of the weights, and an
ordinary unweighted MG5 sample writes ``XWGTUP = sigma`` on every event) that
makes **one rule serve every sample in this test**:

    contribution of a bin, in pb  =  sum_(events in bin) w  /  N_file

For a diagonal sample this is just ``sigma * (fraction of events in the bin)``;
for an interference sample it is the interference contribution, signed, with
``sum_all-bins w / N_file = mean(w) = 0``.  Nothing is read out of a log, no
keep-rate bookkeeping, no ``unit`` reconstruction.  ``_check_normalisation``
below asserts the convention on every sample rather than assuming it.
"""

import gzip
import json
import math
import os
import re
import sys

import numpy as np

LPLUS = (-11, -13)
LMINUS = (11, 13)

# tag -> (label, kind).  'diag' = one diagonal-diagonal block, from a production
# brace; 'inter' = a block with at least one interference index, named directly
# from the card.
SAMPLES = [
    ('unpol', r'unpolarised $pp\to t\bar t$',  'ref'),
    ('pp',    r'$(D^+\!,D^+)$',                'diag'),
    ('pm',    r'$(D^+\!,D^-)$',                'diag'),
    ('mp',    r'$(D^-\!,D^+)$',                'diag'),
    ('mm',    r'$(D^-\!,D^-)$',                'diag'),
    ('i_dp',  r'$(I,D^+)$',                    'inter'),
    ('i_dm',  r'$(I,D^-)$',                    'inter'),
    ('dp_i',  r'$(D^+\!,I)$',                  'inter'),
    ('dm_i',  r'$(D^-\!,I)$',                  'inter'),
    ('ii',    r'$(I,I)$',                      'inter'),
]

PLOTS = [
    ('cos_k_p',  np.linspace(-1, 1, 21)),
    ('cos_k_m',  np.linspace(-1, 1, 21)),
    ('ckk',      np.linspace(-1, 1, 21)),
    ('cnn',      np.linspace(-1, 1, 21)),
    ('crr',      np.linspace(-1, 1, 21)),
    ('cos_phi',  np.linspace(-1, 1, 21)),
    ('cos_n_p',  np.linspace(-1, 1, 21)),
    ('dphi_lab', np.linspace(0, math.pi, 21)),
    ('pt_t',     np.linspace(0, 400, 21)),
    ('m_tt',     np.linspace(340, 1200, 21)),
]


# --------------------------------------------------------------------------
# 4-vector helpers (E, px, py, pz), arrays of shape (N, 4)
# --------------------------------------------------------------------------
def boost_to_rest(p, ref):
    m = np.sqrt(np.maximum(ref[:, 0] ** 2 - (ref[:, 1:] ** 2).sum(1), 1e-12))
    b = -ref[:, 1:] / ref[:, 0:1]
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
    """t, tbar, l+, l- four-vectors, the per-event XWGTUP, and the header."""
    opener = gzip.open if path.endswith('.gz') else open
    t, tb, lp, lm, wgt = [], [], [], [], []
    xsec = xerr = None
    in_ev = in_init = False
    header = []
    init_lines = []
    with opener(path, 'rt', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not in_ev and len(header) < 6000:
                header.append(s)
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
                wgt.append(ev['w'])
                continue
            if not in_ev:
                continue
            if head:
                head = False
                ev['w'] = float(s.split()[2])
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
    for s in init_lines[1:]:
        fl = s.split()
        if len(fl) >= 4:
            try:
                xsec, xerr = float(fl[0]), float(fl[1])
            except ValueError:
                pass
            break
    return dict(t=np.array(t), tb=np.array(tb), lp=np.array(lp),
                lm=np.array(lm), w=np.array(wgt), xsec=xsec, xerr=xerr,
                header='\n'.join(header))


def lhe_path(medir, run):
    base = os.path.join(medir, 'Events', run, 'unweighted_events.lhe')
    for p in (base + '.gz', base):
        if os.path.exists(p):
            return p
    raise IOError('no LHE file in %s' % os.path.dirname(base))


def read_init_xsec(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('<init'):
                next(f)
                fl = next(f).split()
                return float(fl[0]), float(fl[1])
            if line.strip().startswith('<event'):
                break
    return None, None


def read_pi_block(header):
    """Everything the <MGPureInterference> banner block records."""
    def grab(pattern, cast=float):
        m = re.search(pattern, header)
        return cast(m.group(1)) if m else None
    ref = grab(r'Reference normalisation \(pb\)\s*:\s*([-+0-9.eE]+)')
    cnt = re.search(r'Events written / read\s*:\s*(\d+)\s*/\s*(\d+)', header)
    if ref is None or cnt is None:
        return None
    return dict(
        ref=ref,
        n_written=int(cnt.group(1)), n_read=int(cnt.group(2)),
        c=grab(r'Normalisation constant\s+c\s*:\s*([-+0-9.eE]+)'),
        analytic_c=grab(r'Analytic candidate for c\s*:\s*([-+0-9.eE]+)'),
        maxwgt=grab(r'Maximum weight max\|W\| probed\s*:\s*([-+0-9.eE]+)'),
        S=grab(r'Sum of written weights\s+S\s*:\s*([-+0-9.eE]+)'),
        dS=grab(r'MC error\s+sqrt\(sum w\^2\)\s*:\s*([-+0-9.eE]+)'),
        z=grab(r'z = S / error\s*:\s*([-+0-9.eE]+)'),
        mean_w=grab(r'mean\(w\), the sample XSECUP\s*:\s*([-+0-9.eE]+)'),
        dead=grab(r'Trials with a dead weight\s*:\s*(\d+)', int),
    )


def check_normalisation(tag, kind, ev, meta):
    """Assert the one convention this analysis rests on, per sample.

    * every sample: bin contribution [pb] = sum_bin(w) / N_file, i.e. MG5's
      IDWTUP = -4 'cross-section is the mean of the weights';
    * ordinary samples: every XWGTUP equals sigma from <init>, so mean(w) is
      sigma exactly;
    * interference samples: N_file == N_read (fully weighted, nothing dropped),
      the banner's S and mean(w) agree with the file, and <init> carries
      XSECUP = 0.
    """
    w = ev['w']
    n = len(w)
    out = dict(n_file=n, mean_w=float(w.mean()), sum_w=float(w.sum()))
    if kind in ('ref', 'diag'):
        spread = float(np.ptp(w)) / abs(float(w.mean()))
        assert spread < 1e-6, '%s: XWGTUP is not constant (spread %.3g)' % (tag, spread)
        rel = abs(w.mean() - ev['xsec']) / abs(ev['xsec'])
        assert rel < 1e-5, ('%s: mean(w) = %.8g but <init> XSECUP = %.8g'
                            % (tag, w.mean(), ev['xsec']))
        out['mean_over_xsec'] = float(w.mean() / ev['xsec'])
    else:
        assert abs(ev['xsec']) < 1e-12, \
            '%s: a pure-interference file must carry XSECUP = 0, got %g' % (tag, ev['xsec'])
        assert n == meta['n_written'], \
            '%s: %d events in the file but the banner says %d' % (tag, n, meta['n_written'])
        assert n == meta['n_read'], \
            ('%s: fully weighted mode must keep every trial, but wrote %d of %d'
             % (tag, n, meta['n_read']))
        rel = abs(w.sum() - meta['S']) / max(abs(meta['dS']), 1e-300)
        assert rel < 1e-3, ('%s: sum of file weights %.8g != banner S %.8g'
                            % (tag, w.sum(), meta['S']))
        out['z_from_file'] = float(w.sum() / math.sqrt((w * w).sum()))
    return out


# --------------------------------------------------------------------------
# observables (identical to the first run and to the polarisation closure)
# --------------------------------------------------------------------------
def observables(ev):
    t, tb, lp, lm = ev['t'], ev['tb'], ev['lp'], ev['lm']
    ptt = t + tb                                    # the me_frame: partonic CM

    t_c = boost_to_rest(t, ptt)
    tb_c = boost_to_rest(tb, ptt)
    lp_c = boost_to_rest(lp, ptt)
    lm_c = boost_to_rest(lm, ptt)
    lp_t = boost_to_rest(lp_c, t_c)
    lm_t = boost_to_rest(lm_c, tb_c)

    khat = unit(t_c[:, 1:])
    zhat = np.zeros_like(khat)
    zhat[:, 2] = 1.0
    cosT = khat[:, 2]
    sinT = np.sqrt(np.maximum(1.0 - cosT ** 2, 1e-12))
    sgn = np.sign(cosT)
    sgn[sgn == 0] = 1.0
    rhat = sgn[:, None] * (zhat - cosT[:, None] * khat) / sinT[:, None]
    nhat = sgn[:, None] * np.cross(zhat, khat) / sinT[:, None]

    up = unit(lp_t[:, 1:])
    um = unit(lm_t[:, 1:])

    o = {}
    o['cos_k_p'] = (up * khat).sum(1)
    o['cos_k_m'] = -(um * khat).sum(1)
    o['cos_r_p'] = (up * rhat).sum(1)
    o['cos_n_p'] = (up * nhat).sum(1)
    o['ckk'] = (up * khat).sum(1) * (um * khat).sum(1)
    o['crr'] = (up * rhat).sum(1) * (um * rhat).sum(1)
    o['cnn'] = (up * nhat).sum(1) * (um * nhat).sum(1)
    o['cos_phi'] = (up * um).sum(1)
    dphi = np.abs(np.arctan2(lp[:, 2], lp[:, 1]) - np.arctan2(lm[:, 2], lm[:, 1]))
    o['dphi_lab'] = np.where(dphi > math.pi, 2 * math.pi - dphi, dphi)
    o['pt_t'] = np.sqrt(t[:, 1] ** 2 + t[:, 2] ** 2)
    o['m_tt'] = np.sqrt(np.maximum(ptt[:, 0] ** 2 - (ptt[:, 1:] ** 2).sum(1), 0))
    return o


# --------------------------------------------------------------------------
def main():
    runs_dir = sys.argv[1]
    out_dir = sys.argv[2]
    run = sys.argv[3] if len(sys.argv) > 3 else 'closure'
    os.makedirs(out_dir, exist_ok=True)

    meta = {}
    evs = {}
    for tag, label, kind in SAMPLES:
        medir = os.path.join(runs_dir, tag)
        ev = read_lhe(lhe_path(medir, '%s_decayed_1' % run))
        pxs, perr = read_init_xsec(lhe_path(medir, run))
        m = dict(kind=kind, label=label, n_file=len(ev['t']),
                 xsec=ev['xsec'], xerr=ev['xerr'],
                 prod_xsec=pxs, prod_err=perr)
        if kind == 'inter':
            m.update(read_pi_block(ev['header']))
        m.update(check_normalisation(tag, kind, ev, m))
        meta[tag] = m
        evs[tag] = ev

    # ---- histograms -------------------------------------------------------
    # ONE rule, every sample: pb = sum_bin(w) / N_file.
    store = {}
    for key, bins in PLOTS:
        store['bins__%s' % key] = bins
    for tag, label, kind in SAMPLES:
        o = observables(evs[tag])
        w = evs[tag]['w'] / float(meta[tag]['n_file'])
        for key, bins in PLOTS:
            v = o[key]
            n, _ = np.histogram(v, bins=bins)
            s, _ = np.histogram(v, bins=bins, weights=w)
            e2, _ = np.histogram(v, bins=bins, weights=w * w)
            store['sumw__%s__%s' % (tag, key)] = s
            store['sumw2__%s__%s' % (tag, key)] = e2
            store['n__%s__%s' % (tag, key)] = n
        # unbinned moments, for the summary table:
        #   sum w, sum w O, sum w^2, sum w^2 O, sum w^2 O^2, N
        # (the last three are what the error on a ratio of the first two needs)
        for key, _bins in PLOTS:
            ww = w * w
            store['mom__%s__%s' % (tag, key)] = np.array(
                [w.sum(), (w * o[key]).sum(), ww.sum(),
                 (ww * o[key]).sum(), (ww * o[key] ** 2).sum(), float(len(w))])

    np.savez(os.path.join(out_dir, 'histograms.npz'), **store)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=str)

    print('%-6s %-6s %9s %14s %14s %10s %8s'
          % ('tag', 'kind', 'N_file', 'sigma [pb]', 'mean(w) [pb]', 'c', 'z'))
    for tag, label, kind in SAMPLES:
        m = meta[tag]
        print('%-6s %-6s %9d %14.6g %14.6g %10s %8s'
              % (tag, kind, m['n_file'], m['xsec'], m['mean_w'],
                 ('%.4e' % m['c']) if kind == 'inter' else '-',
                 ('%+.2f' % m['z']) if kind == 'inter' else '-'))
    print('wrote histograms.npz + meta.json to', out_dir)


if __name__ == '__main__':
    main()
