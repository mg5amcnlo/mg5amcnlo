#!/usr/bin/env python3
"""Analyse the (spinmode x pure_interference_output) sweep of run_pa_madspin.sh.

    analyse_pa_madspin.py <workdir> <out_dir>

Writes ``results.json`` (every number quoted in RESULTS.md) and prints the
tables.  Nothing is read out of a log except the ``<MGPureInterference>``
banner block, which is part of the file the mode ships.

The LHE reader, the spin-basis construction and the observables are taken
verbatim from ``MadSpin/validation/interference_closure_v2/analyse_interference.py``
(committed on branch ``claude/ms-interference-closure-v2``); they are copied
rather than imported only because the two validation directories live on
sibling branches.  Keeping them identical is what makes the numbers below
directly comparable to that test's ``<C_nn>`` and ``<C_rr>``.

One normalisation rule, both output shapes
------------------------------------------
The mode is self-normalising under MG5's ``IDWTUP = -4``:

    contribution of a bin, in pb  =  sum_(events in bin) w  /  N_file

with ``N_file`` the number of events **in the file**.  For
``pure_interference_output = weighted`` that is also ``N_read`` (nothing is
dropped); for ``unweighted`` it is smaller than ``N_read``, and the rule still
holds because the written magnitude carries the run's own
``<|W|> = (N_file/N_drawn) * max|W|``, in which ``N_file`` cancels.  So the
same three lines analyse both, and no keep rate is reconstructed by hand.
"""

import glob
import gzip
import json
import math
import os
import re
import sys

import numpy as np

LPLUS = (-11, -13)
LMINUS = (11, 13)

# tag -> (spinmode, output).  Must match run_pa_madspin.sh.
COMBOS = [
    ('onshell_w', 'onshell', 'weighted'),
    ('onshell_u', 'onshell', 'unweighted'),
    ('pa_w',      'PA',      'weighted'),
    ('pa_u',      'PA',      'unweighted'),
    ('ms_w',      'madspin', 'weighted'),
    ('ms_u',      'madspin', 'unweighted'),
]

# the committed closure numbers this sweep has to reproduce
# (interference_closure_v2/RESULTS.md section 6, five interference blocks,
#  5 x 50k events)
REFERENCE = {'cnn': (+0.03657, 0.00059), 'crr': (+0.00104, 0.00066)}


# --------------------------------------------------------------------------
# 4-vector helpers (E, px, py, pz), arrays of shape (N, 4)  [closure_v2]
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
def read_pi_block(header):
    """Everything the <MGPureInterference> banner block records."""
    def grab(pattern, cast=float):
        m = re.search(pattern, header)
        return cast(m.group(1)) if m else None
    cnt = re.search(r'Events written / read\s*:\s*(\d+)\s*/\s*(\d+)', header)
    return dict(
        ref=grab(r'Reference normalisation \(pb\)\s*:\s*([-+0-9.eE]+)'),
        n_written=int(cnt.group(1)) if cnt else None,
        n_read=int(cnt.group(2)) if cnt else None,
        c=grab(r'Normalisation constant\s+c\s*:\s*([-+0-9.eE]+)'),
        c_relerr=grab(r'Normalisation constant\s+c\s*:\s*[-+0-9.eE]+\s*\+-\s*([0-9.]+)%'),
        analytic_c=grab(r'Analytic candidate for c\s*:\s*([-+0-9.eE]+)'),
        c_over_analytic=grab(r'Analytic candidate for c\s*:[^(]*\(ratio\s*([-+0-9.eE]+)\)'),
        absw_probe=grab(r'<\|W\|> from the probe\s*:\s*([-+0-9.eE]+)'),
        absw_probe_relerr=grab(r'<\|W\|> from the probe\s*:\s*[-+0-9.eE]+\s*\+-\s*([0-9.]+)%'),
        absw_run=grab(r'<\|W\|> the run realised\s*:\s*([-+0-9.eE]+)'),
        probe_scale=grab(r'<\|W\|> the run realised\s*:[^(]*\(probe x\s*([-+0-9.eE]+)\)'),
        w_magnitude=grab(r'Weight magnitude \|w\| \(pb\)\s*:\s*([-+0-9.eE]+)'),
        maxwgt=grab(r'Maximum weight max\|W\| probed\s*:\s*([-+0-9.eE]+)'),
        S=grab(r'Sum of written weights\s+S\s*:\s*([-+0-9.eE]+)'),
        dS=grab(r'MC error\s+sqrt\(sum w\^2\)\s*:\s*([-+0-9.eE]+)'),
        z=grab(r'z = S / error\s*:\s*([-+0-9.eE]+)'),
        mean_w=grab(r'mean\(w\), the sample XSECUP\s*:\s*([-+0-9.eE]+)'),
        dead=grab(r'Trials with a dead weight\s*:\s*(\d+)', int),
        overflow=grab(r'Trials above max\|W\|\s*:\s*(\d+)', int),
    )


def jac_counters(d):
    """Merge the per-process reshuffle counters drive_madspin.py wrote."""
    tot = dict(n_calls=0, n_bad=0, n_zero=0, n_negative=0, n_nonfinite=0,
               nb_reshuffle_issue=0, n_files=0, bad_values=[])
    tot['n_forced'] = 0
    for stage in ('probe', 'unweight'):
        tot[stage] = dict(n_calls=0, n_bad=0, nb_reshuffle_issue=0)
    for f in sorted(glob.glob(os.path.join(d, 'jac.*.json'))):
        j = json.load(open(f))
        tot['n_files'] += 1
        for k in ('n_calls', 'n_bad', 'n_zero', 'n_negative', 'n_nonfinite',
                  'nb_reshuffle_issue', 'n_forced'):
            tot[k] += j.get(k) or 0
        st = tot.get(j.get('stage'))
        if st is not None:
            for k in ('n_calls', 'n_bad', 'nb_reshuffle_issue'):
                st[k] += j.get(k) or 0
        tot['bad_values'] += j.get('bad_values') or []
    tot['bad_values'] = tot['bad_values'][:20]
    return tot


def weight_shape(w):
    """Sign counts, and how many distinct weights the file actually holds.

    ``n_values`` counts distinct SIGNED weights and ``n_magnitudes`` distinct
    |w|.  ``unweighted`` must give exactly 2 and 1 -- the written magnitude is
    one constant (sigma_ref*BR*<|W|>/c) times a production XWGTUP that is
    itself constant across an unweighted MG5 sample -- and ``weighted`` must
    give as many as there are events.  Distinctness is at 1e-9 relative to the
    largest |w|, which is far below any real spread: the two values of the
    unweighted output are literally the same float up to sign.
    """
    a = np.abs(w)
    scale = float(a.max()) if a.size else 1.0
    uniq = np.unique(np.round(a / (scale or 1.0), 9))
    sig = np.unique(np.round(w / (scale or 1.0), 9))
    return dict(n_pos=int((w > 0).sum()), n_neg=int((w < 0).sum()),
                n_zero=int((w == 0).sum()),
                n_values=int(len(sig)),
                n_magnitudes=int(len(uniq)),
                signed_values=[float(u * scale) for u in sig[:10]],
                magnitudes=[float(u * scale) for u in uniq[:10]])


def moment(w, o, n_file, sigma_ref):
    """Interference contribution to <O>, and its MC error.

    value = (1/N_file) sum_i w_i O_i / sigma_ref, i.e. the mean of
    (w O / sigma_ref) over the events in the file, so the error is the sample
    standard deviation of that same quantity over sqrt(N_file).  N_file is the
    number of events in the file for both output shapes (see the module
    docstring).
    """
    x = w * o / (n_file * sigma_ref)          # x_i = y_i / N, val = sum x_i
    val = float(x.sum())
    n = len(x)
    # var(mean of y) = (1/N)(<y^2> - <y>^2) = sum x^2 - val^2 / N
    err = float(math.sqrt(max(float((x * x).sum()) - val * val / n, 0.0)))
    return val, err


def main():
    work = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for tag, mode, out in COMBOS:
        d = os.path.join(work, tag)
        path = os.path.join(d, 'events_decayed.lhe.gz')
        r = dict(tag=tag, spinmode=mode, output=out,
                 decayed_file=path, decayed_file_exists=os.path.exists(path))
        for f, k in (('exit_code', 'exit_code'), ('wallclock_s', 'wallclock_s')):
            p = os.path.join(d, f)
            r[k] = int(open(p).read().strip()) if os.path.exists(p) else None
        r['jac'] = jac_counters(d)
        if not r['decayed_file_exists']:
            results[tag] = r
            continue

        ev = read_lhe(path)
        w = ev['w']
        n_file = len(w)
        banner = read_pi_block(ev['header'])
        r['banner'] = banner
        r['init_xsec'] = ev['xsec']
        r['n_file'] = n_file
        r['weights'] = weight_shape(w)

        # zero-cross-section check, recomputed from the file itself
        S = float(w.sum())
        sumw2 = float((w * w).sum())
        mean = S / n_file
        # error on the mean: the spread of the weights, not sqrt(sum w^2)
        var = max(sumw2 / n_file - mean * mean, 0.0)
        r['zero_check'] = dict(
            S=S, sqrt_sum_w2=math.sqrt(sumw2),
            z=S / math.sqrt(sumw2) if sumw2 else 0.0,
            mean_w=mean, mean_w_err=math.sqrt(var / n_file),
            mean_w_pull=mean / math.sqrt(var / n_file) if var else 0.0)

        sigma_ref = banner['ref']
        o = observables(ev)
        r['observables'] = {}
        for key in ('cnn', 'crr', 'ckk', 'cos_phi', 'dphi_lab', 'pt_t'):
            val, err = moment(w, o[key], n_file, sigma_ref)
            entry = dict(value=val, error=err)
            if key in REFERENCE:
                ref, refe = REFERENCE[key]
                entry['reference'] = ref
                entry['reference_error'] = refe
                entry['pull'] = ((val - ref) / math.sqrt(err ** 2 + refe ** 2)
                                 if (err or refe) else 0.0)
            r['observables'][key] = entry
        results[tag] = r

    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)

    # ---------------------------------------------------------------- print
    print('\n== run and file ==')
    print('%-10s %-8s %-11s %4s %8s %8s %9s %s'
          % ('tag', 'spinmode', 'output', 'rc', 'N_file', 'N_read', 'wall[s]',
             'decayed file'))
    for tag, mode, out in COMBOS:
        r = results[tag]
        b = r.get('banner') or {}
        print('%-10s %-8s %-11s %4s %8s %8s %9s %s'
              % (tag, mode, out, r['exit_code'], r.get('n_file'),
                 b.get('n_read'), r.get('wallclock_s'),
                 'OK' if r['decayed_file_exists'] else 'MISSING'))

    print('\n== zero cross-section ==')
    print('%-10s %14s %14s %8s %14s %8s %7s %7s'
          % ('tag', 'S', 'sqrt(sum w2)', 'z', 'mean(w)', 'pull', 'XSECUP', 'dead'))
    for tag, _m, _o in COMBOS:
        r = results[tag]
        if 'zero_check' not in r:
            continue
        z = r['zero_check']
        print('%-10s %14.6e %14.6e %+8.3f %14.6e %+8.2f %7.1g %7s'
              % (tag, z['S'], z['sqrt_sum_w2'], z['z'], z['mean_w'],
                 z['mean_w_pull'], r['init_xsec'], r['banner']['dead']))

    print('\n== weight shape ==')
    print('%-10s %9s %9s %9s %9s %9s %s'
          % ('tag', 'n(w>0)', 'n(w<0)', 'n(w==0)', '#w', '#|w|',
             '|w| values [pb]'))
    for tag, _m, _o in COMBOS:
        r = results[tag]
        if 'weights' not in r:
            continue
        s = r['weights']
        print('%-10s %9d %9d %9d %9d %9d %s'
              % (tag, s['n_pos'], s['n_neg'], s['n_zero'], s['n_values'],
                 s['n_magnitudes'],
                 ' '.join('%.6e' % m for m in s['magnitudes'][:3])))

    print('\n== normalisation constant c ==')
    print('%-10s %14s %8s %14s %9s %14s %9s'
          % ('tag', 'c measured', '+-%', 'c analytic', 'ratio', '<|W|> probe',
             'probe x'))
    for tag, _m, _o in COMBOS:
        r = results[tag]
        if 'banner' not in r:
            continue
        b = r['banner']
        print('%-10s %14.6e %8.3f %14.6e %9.6f %14.6e %9s'
              % (tag, b['c'], b['c_relerr'], b['analytic_c'],
                 b['c_over_analytic'], b['absw_probe'],
                 ('%.4f' % b['probe_scale']) if b['probe_scale'] else '-'))

    print('\n== physics: interference contribution ==')
    for key in ('cnn', 'crr'):
        ref, refe = REFERENCE[key]
        print('  <C_%s>   committed closure: %+.5f +- %.5f'
              % (key[1:], ref, refe))
        print('  %-10s %22s %8s' % ('tag', 'this run', 'pull'))
        for tag, _m, _o in COMBOS:
            r = results[tag]
            if 'observables' not in r:
                continue
            e = r['observables'][key]
            print('  %-10s %+12.6f +- %.6f %+8.2f'
                  % (tag, e['value'], e['error'], e['pull']))
        print()

    print('== reshuffle instrumentation (Event.reshuffle_production) ==')
    print('%-10s %10s %10s %8s %10s %10s %14s'
          % ('tag', 'calls', 'bad(<=0)', 'zero', 'negative', 'nonfinite',
             'retries'))
    for tag, _m, _o in COMBOS:
        j = results[tag]['jac']
        print('%-10s %10d %10d %8d %10d %10d %14d'
              % (tag, j['n_calls'], j['n_bad'], j['n_zero'], j['n_negative'],
                 j['n_nonfinite'], j['nb_reshuffle_issue']))
    print('\nwrote', os.path.join(out_dir, 'results.json'))


if __name__ == '__main__':
    main()
