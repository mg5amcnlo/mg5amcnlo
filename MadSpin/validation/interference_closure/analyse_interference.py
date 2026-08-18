#!/usr/bin/env python3
"""Histogram the eleven samples of the MadSpin interference-closure test.

Reads the decayed LHE files produced by ``run_interference.sh`` and writes ONE
``histograms.npz`` holding, per sample and per observable, the bin edges, the
sum of weights, the sum of weights squared and the raw entry count -- so the
plots and every number quoted can be regenerated without re-running MadSpin.

Usage:  analyse_interference.py <runs_dir> <out_dir>

Normalisation -- the one subtle point
-------------------------------------
An ordinary MadSpin sample redraws the decay configuration until one is
accepted, so every production event yields exactly one written event and the
per-event weight is ``sigma_decayed / N``.

The pure-interference mode cannot redraw (its mean weight over the decay phase
space is zero, and what must be allowed to vary from production event to
production event is ``<|w|>``, the local size of the interference).  It draws
ONCE and keeps the event with probability ``|W| / max_weight``, writing
``+- sigma_parent * BR``.  For any observable O the expectation of the signed
sum is therefore

    sum_kept sign * w * O   ->   (sigma_ref / N_read) * sum_p Int W O dOmega
                                 / max_weight

whereas the redraw scheme of every ordinary sample delivers

    sum_events w * O        ->   (sigma / N)          * sum_p Int W O dOmega
                                 / c ,     c = <W>_decay-phase-space

with ``c = 1 / prod_i n_i`` the SAME constant for every production event (this
is exactly the property that makes redraw-until-accept unbiased; see
MADSPIN_SEQUENTIAL_PLAN.md section 13.7b).  Putting an interference sample on
the same footing as the diagonal ones therefore needs the single factor

    max_weight / c ,

nothing else.  ``max_weight`` is read from the log line this branch adds to
``get_maxwgt_for_onshell``; ``c`` is *measured*, not assumed, from the
unpolarised sample run in the ordinary joint scheme, where the unweighting
efficiency is ``c / max_weight`` -- and is then compared against the analytic
``1 / (2 * 2) = 0.25`` for two spin-1/2 particles.
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

# tag -> (label, kind).  'diag' = one diagonal entry of rho(i,j) per particle,
# 'inter' = a pure-interference sample.
SAMPLES = [
    ('unpol', r'unpolarised $pp\to t\bar t$', 'ref'),
    ('pp',    r'$t\{+\}\,\bar t\{+\}$',       'diag'),
    ('pm',    r'$t\{+\}\,\bar t\{-\}$',       'diag'),
    ('mp',    r'$t\{-\}\,\bar t\{+\}$',       'diag'),
    ('mm',    r'$t\{-\}\,\bar t\{-\}$',       'diag'),
    ('i_tbp', r'$(I_t\otimes D^+_{\bar t})$', 'inter'),
    ('i_tbm', r'$(I_t\otimes D^-_{\bar t})$', 'inter'),
    ('i_tp',  r'$(D^+_t\otimes I_{\bar t})$', 'inter'),
    ('i_tm',  r'$(D^-_t\otimes I_{\bar t})$', 'inter'),
    ('x_t',   r'$(I_t\otimes \mathbb{1}_{\bar t})$', 'inter'),
    ('x_tb',  r'$(\mathbb{1}_t\otimes I_{\bar t})$', 'inter'),
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
            if not in_ev and len(header) < 4000:
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


# --------------------------------------------------------------------------
# run bookkeeping pulled out of the log / banner
# --------------------------------------------------------------------------
def read_maxwgt(logfile):
    """The joint accept/reject bound, from the log line this branch adds."""
    val = None
    with open(logfile, errors='ignore') as f:
        for line in f:
            m = re.search(r'joint maximum weight\s*=\s*([-+0-9.eE]+)', line)
            if m:
                val = float(m.group(1))
    return val


def read_efficiency(logfile):
    """(n_written, n_trials) of the ordinary unweighting loop."""
    out = None
    with open(logfile, errors='ignore') as f:
        for line in f:
            m = re.search(r'unweight efficiency:\s*[0-9.]+\s*\((\d+) written'
                          r'\s*/\s*(\d+) trials', line)
            if m:
                out = (int(m.group(1)), int(m.group(2)))
    return out


def read_pi_block(header):
    """Reference normalisation and written/read counts of a PI sample."""
    ref = re.search(r'Reference normalisation \(pb\)\s*:\s*([-+0-9.eE]+)', header)
    cnt = re.search(r'Events written / read\s*:\s*(\d+)\s*/\s*(\d+)', header)
    ovf = re.search(r'Trials above the max weight\s*:\s*(\d+)', header)
    zz = re.search(r'z = S / error\s*:\s*([-+0-9.eE]+)', header)
    ss = re.search(r'Sum of written weights\s+S\s*:\s*([-+0-9.eE]+)', header)
    ee = re.search(r'MC error\s+sqrt\(sum w\^2\)\s*:\s*([-+0-9.eE]+)', header)
    if not (ref and cnt):
        return None
    return dict(ref=float(ref.group(1)), n_written=int(cnt.group(1)),
                n_read=int(cnt.group(2)),
                overflow=int(ovf.group(1)) if ovf else -1,
                z=float(zz.group(1)) if zz else float('nan'),
                S=float(ss.group(1)) if ss else float('nan'),
                dS=float(ee.group(1)) if ee else float('nan'))


# --------------------------------------------------------------------------
# observables (identical to the polarisation-closure test)
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
        log = os.path.join(runs_dir, 'log_%s.txt' % tag)
        m = dict(kind=kind, label=label, n_written=len(ev['t']),
                 xsec=ev['xsec'], xerr=ev['xerr'],
                 prod_xsec=pxs, prod_err=perr,
                 maxwgt=read_maxwgt(log), eff=read_efficiency(log))
        if kind == 'inter':
            m.update(read_pi_block(ev['header']))
        meta[tag] = m
        evs[tag] = ev

    # ---- the one constant: c = <W> over the decay phase space -------------
    # measured on the unpolarised sample, which is run in the ordinary joint
    # scheme precisely so that its unweighting efficiency delivers c/max_weight
    nw, nt = meta['unpol']['eff']
    c_meas = meta['unpol']['maxwgt'] * nw / float(nt)
    c_analytic = 0.25                    # 1 / (n_t * n_tbar), both spin 1/2
    meta['_c'] = dict(measured=c_meas, analytic=c_analytic,
                      eff=nw / float(nt), maxwgt=meta['unpol']['maxwgt'])

    # ---- per-written-event normalisation, in pb ---------------------------
    for tag, label, kind in SAMPLES:
        m = meta[tag]
        if kind == 'inter':
            # +- sigma_ref / N_read, rescaled by max_weight / c
            m['unit'] = (m['ref'] / m['n_read']) * (m['maxwgt'] / c_meas)
            m['sign'] = np.sign(evs[tag]['w'])
        else:
            m['unit'] = m['xsec'] / m['n_written']
            m['sign'] = np.ones(m['n_written'])

    # ---- histograms -------------------------------------------------------
    store = {}
    for key, bins in PLOTS:
        store['bins__%s' % key] = bins
    for tag, label, kind in SAMPLES:
        o = observables(evs[tag])
        w = meta[tag]['unit'] * meta[tag]['sign']
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

    clean = {}
    for tag, m in meta.items():
        if tag.startswith('_'):
            clean[tag] = m
            continue
        clean[tag] = {k: v for k, v in m.items() if k != 'sign'}
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(clean, f, indent=1, default=str)

    print('c measured = %.6f   (analytic 0.25, ratio %.4f)'
          % (c_meas, c_meas / c_analytic))
    for tag, label, kind in SAMPLES:
        m = meta[tag]
        extra = ''
        if kind == 'inter':
            extra = ('  ref=%.5f pb  kept %d/%d (%.3f)  maxwgt=%.4g  z=%+.2f'
                     '  ovf=%d' % (m['ref'], m['n_written'], m['n_read'],
                                   m['n_written'] / float(m['n_read']),
                                   m['maxwgt'], m['z'], m['overflow']))
        print('%-6s %-6s N=%7d  unit=%.6g pb%s'
              % (tag, kind, m['n_written'], m['unit'], extra))
    print('wrote histograms.npz + meta.json to', out_dir)


if __name__ == '__main__':
    main()
