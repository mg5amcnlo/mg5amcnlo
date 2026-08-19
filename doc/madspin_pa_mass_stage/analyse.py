#!/usr/bin/env python3
"""Turn the streams ``probe_launcher.py`` wrote into the numbers and the plots.

Reads ``<data>/<mode>.stream.txt`` for ``PA``, ``PAnojac`` and ``madspin``,
splits the probe (max-weight scan) phase from the production phase, pairs each
``U`` record with the ``W`` that follows it, and produces

  * ``summary.json``   -- every number quoted in the write-up
  * ``jacobian_distribution.png``  -- the PA reshuffling jacobian alone
  * ``mass_weight_distribution.png`` -- the three tested weights with their bounds
  * ``pa_decomposition.png`` -- jac_prod vs the no-jacobian weight vs the product
  * ``jacobian_vs_sqrts.png`` -- where the tail comes from

Run with the python that has matplotlib::

    export PATH="$HOME/.pyenv/versions/mg-3.12/bin:$PATH"
    python3 doc/madspin_pa_mass_stage/analyse.py --data <dir> --plots <dir>
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import json
import math
import os
import re

import numpy as np

pjoin = os.path.join

MODES = ['PA', 'PAnojac', 'madspin']
LABEL = {'PA': 'PA (density_keep_jacobian=True)',
         'PAnojac': 'PA (no jac.)',
         'madspin': 'madspin (offshell)'}
COLOR = {'PA': '#c0392b', 'PAnojac': '#2980b9', 'madspin': '#27ae60'}

_RE_MASS = re.compile(
    r'MadSpin sequential mass stage:\s*([0-9.eE+\-]+)\s*mass sets per accepted '
    r'event\s*\((\d+)\s*drawn,\s*(\d+)\s*rejected')
_RE_EFF = re.compile(
    r'MadSpin unweight efficiency:\s*([0-9.eE+\-]+)\s*'
    r'\((\d+)\s*written\s*/\s*(\d+)\s*trials')
_RE_OVERFLOW_MASS = re.compile(r'(\d+)\s*weights exceeded their per-particle maximum')


def parse_stream(path):
    """(production-phase trials, probe-phase trials, bounds, infeasible counts).

    A trial is one mass set that reached the accept/reject: the ``U`` record
    that settled it and the ``W`` the accept/reject tested.  A ``U`` with no
    ``W`` before the next ``U`` is a mass set the production could not be
    reshuffled onto (it costs a draw but makes no test) -- counted, not paired.
    """
    rows = {'prod': [], 'probe': []}
    infeasible = {'prod': 0, 'probe': 0}
    unpaired = {'prod': 0, 'probe': 0}
    bounds = None
    phase = 'prod'
    pending = None
    zvals = []

    def flush():
        """A ``U`` with no ``W``: the max-weight scan makes no accept/reject, so
        every probe-phase mass set lands here. Kept (with w = nan) because those
        37500-70000 draws are precisely the sample the bound is read off."""
        if pending is None:
            return
        unpaired[phase] += 1
        jac, jbw, mass, sqrts = pending
        rows[phase].append((jac, float(np.prod(jbw)) if jbw else 1.0,
                            float('nan'), float('nan'), sqrts, mass))

    with open(path) as fp:
        for line in fp:
            tag = line[0]
            if tag == 'U':
                flush()
                body = line[2:].split()
                if body[0] == '-':
                    infeasible[phase] += 1
                    pending = None
                    zvals = []
                    continue
                jac = float(body[0])
                jbw = ([float(x) for x in body[1].split(',')]
                       if body[1] != '-' else [])
                mass = ([float(x) for x in body[2].split(',')]
                        if body[2] != '-' else [])
                sqrts = float(body[3])
                pending = (jac, jbw, mass, sqrts)
                zvals = []
            elif tag == 'Z':
                if pending is not None:
                    zvals.append(float(line.rsplit(None, 1)[1]))
            elif tag == 'W':
                if pending is None:
                    continue
                jac, jbw, mass, sqrts = pending
                rows[phase].append((jac,
                                    float(np.prod(jbw)) if jbw else 1.0,
                                    float(np.prod(zvals)) if zvals else 1.0,
                                    float(line[2:]),
                                    sqrts,
                                    mass))
                pending = None
                zvals = []
            elif tag == 'P':
                flush()
                phase = 'probe' if line.split()[1] == 'start' else 'prod'
                pending = None
                zvals = []
            elif tag == 'M':
                bounds = [float(x) for x in line[2:].split(',')]
    return rows, bounds, infeasible, unpaired


def to_arrays(rows):
    if not rows:
        return {}
    jac = np.array([r[0] for r in rows])
    jbw = np.array([r[1] for r in rows])
    zz = np.array([r[2] for r in rows])
    w = np.array([r[3] for r in rows])
    sqrts = np.array([r[4] for r in rows])
    mass = np.array([r[5] for r in rows])
    return {'jac': jac, 'jbw': jbw, 'z': zz, 'w': w, 'sqrts': sqrts,
            'mass': mass}


def quantiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return {
        'n': int(len(x)),
        'mean': float(x.mean()),
        'median': float(np.median(x)),
        'sd': float(x.std()),
        'min': float(x.min()),
        'q50': float(np.percentile(x, 50)),
        'q90': float(np.percentile(x, 90)),
        'q99': float(np.percentile(x, 99)),
        'q99.9': float(np.percentile(x, 99.9)),
        'q99.99': float(np.percentile(x, 99.99)),
        'max': float(x.max()),
        'max_over_mean': float(x.max() / x.mean()) if x.mean() else None,
        'q99.9_over_mean': (float(np.percentile(x, 99.9) / x.mean())
                            if x.mean() else None),
    }


def hill_tail_index(x, fractions=(0.01, 0.003, 0.001)):
    """Hill estimator of the Pareto index ``a`` in ``P(X > t) ~ t^-a``, on the
    top ``fraction`` of the sample.  Quoted at several depths because the whole
    point of a heavy tail is that the answer must not move much with the cut.

    ``a <= 1`` means the mean does not exist, ``a <= 2`` the variance does not,
    ``a <= 3`` the skewness does not; and the maximum of ``n`` draws grows like
    ``n**(1/a)``, which is what makes the max-weight bound sample-size
    dependent.
    """
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    out = {}
    for fraction in fractions:
        k = int(len(x) * fraction)
        if k < 20:
            continue
        tail = x[-k:]
        out['top_%g%%' % (100 * fraction)] = {
            'k': k,
            'threshold': float(tail[0]),
            'index_a': float(1.0 / np.mean(np.log(tail / tail[0])))
            if np.mean(np.log(tail / tail[0])) > 0 else None,
        }
    return out


def parse_log(path):
    out = {}
    if not os.path.exists(path):
        return out
    text = re.sub(r'[ \t]+', ' ', open(path).read())
    match = None
    for match in _RE_MASS.finditer(text):
        pass
    if match:
        out['eps_m_logged'] = float(match.group(1))
        out['mass_drawn'] = int(match.group(2))
        out['mass_rejected'] = int(match.group(3))
    match = None
    for match in _RE_EFF.finditer(text):
        pass
    if match:
        out['n_written'] = int(match.group(2))
        out['n_trials'] = int(match.group(3))
    match = _RE_OVERFLOW_MASS.search(text)
    if match:
        out['overflows'] = int(match.group(1))
    return out


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--plots', required=True)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    os.makedirs(args.plots, exist_ok=True)
    tag = ('_' + args.tag) if args.tag else ''

    data = {}
    summary = {}
    for mode in MODES:
        path = pjoin(args.data, '%s.stream.txt' % mode)
        if not os.path.exists(path):
            continue
        rows, bounds, infeasible, unpaired = parse_stream(path)
        # a production-phase mass set always reaches the accept/reject, so a
        # non-finite w there would be a parsing accident, not physics
        arr = to_arrays([r for r in rows['prod'] if math.isfinite(r[3])])
        probe = to_arrays(rows['probe'])
        data[mode] = {'arr': arr, 'probe': probe, 'bounds': bounds,
                      'infeasible': infeasible}
        log = parse_log(pjoin(args.data, '%s.log' % mode))

        w, jac = arr['w'], arr['jac']
        bound = bounds[0]
        # eps_m = (tested trials + infeasible mass sets) / accepted, and the
        # accept/reject accepts a fraction mean(w)/bound of the tested ones.
        acc_pred = float(np.minimum(w, bound).mean() / bound)
        eps_pred = (1.0 / acc_pred) * (1.0 + infeasible['prod'] / len(w))
        entry = {
            'n_mass_trials': int(len(w)),
            'n_infeasible_mass_sets': infeasible['prod'],
            'bound_maxwgts0': bound,
            'bounds': bounds,
            'w': quantiles(w),
            'jac_prod': quantiles(jac),
            'jac_bw_product': quantiles(arr['jbw']),
            'zhat_product': quantiles(arr['z']),
            'w_over_jbw_z': quantiles(w / (arr['jbw'] * arr['z'])),
            'w_over_jac_jbw_z': quantiles(w / (jac * arr['jbw'] * arr['z'])),
            'jac_prod_probe_phase': quantiles(probe['jac']) if probe else {},
            'n_probe_mass_sets': int(len(probe['jac'])) if probe else 0,
            'tail_index_w': hill_tail_index(w),
            'tail_index_jac_prod': hill_tail_index(jac),
            'eps_m_predicted_bound_over_mean': eps_pred,
            'bound_over_mean_w': float(bound / w.mean()),
            'fraction_above_bound': float((w > bound).mean()),
            'log': log,
        }
        summary[mode] = entry
        print('%-9s trials=%7d  mean(w)=%.4g  bound=%.4g  bound/mean=%.3f  '
              'eps_pred=%.3f  eps_log=%s'
              % (mode, len(w), w.mean(), bound, bound / w.mean(), eps_pred,
                 log.get('eps_m_logged')), flush=True)

    with open(pjoin(args.plots, 'summary%s.json' % tag), 'w') as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    # compact raw arrays, so the plots can be redrawn without re-running MadSpin
    np.savez_compressed(
        pjoin(args.plots, 'arrays%s.npz' % tag),
        **{'%s_%s' % (mode, name): np.asarray(values, dtype=np.float32)
           for mode, blob in data.items()
           for name, values in blob['arr'].items() if name != 'mass'})

    make_plots(data, summary, args.plots, tag)

    boot = pjoin(args.data, 'bound_bootstrap.json')
    if os.path.exists(boot):
        with open(boot) as fp:
            plot_bootstrap(json.load(fp), data, args.plots, tag)


# --------------------------------------------------------------------------
def make_plots(data, summary, outdir, tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ---- 1. the PA reshuffling jacobian alone -----------------------------
    jac = data['PA']['arr']['jac']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.hist(jac, bins=np.linspace(0, 4, 200), color='#c0392b', alpha=.85)
    ax.set_xlabel(r'production reshuffling jacobian  $J$')
    ax.set_ylabel('mass sets')
    ax.set_title('bulk (linear scale, 0 < J < 4)')
    ax.axvline(jac.mean(), color='k', ls='--', lw=1,
               label='mean = %.3f' % jac.mean())
    ax.axvline(np.median(jac), color='k', ls=':', lw=1,
               label='median = %.3f' % np.median(jac))
    ax.legend()

    ax = axes[1]
    positive = jac[jac > 0]
    ax.hist(positive, bins=np.logspace(np.log10(positive.min()),
                                       np.log10(positive.max()), 200),
            color='#c0392b', alpha=.85)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$J$')
    ax.set_title('the same, log-log: the tail')
    for q, style in ((99, ':'), (99.9, '--')):
        value = np.percentile(jac, q)
        ax.axvline(value, color='k', ls=style, lw=1,
                   label='%g%% = %.2f' % (q, value))
    ax.axvline(jac.max(), color='#e67e22', lw=1.4,
               label='max = %.1f' % jac.max())
    ax.legend()
    fig.suptitle('The PA production reshuffling jacobian, %d mass sets '
                 '(p p > t t~, both tops leptonic)' % len(jac))
    fig.tight_layout()
    fig.savefig(pjoin(outdir, 'jacobian_distribution%s.png' % tag), dpi=140)
    plt.close(fig)

    # ---- 2. the three tested weights, with their bounds --------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for mode in MODES:
        if mode not in data:
            continue
        w = data[mode]['arr']['w']
        bound = data[mode]['bounds'][0]
        for ax, scale in zip(axes, ('linear', 'log')):
            positive = w[w > 0]
            if scale == 'log':
                bins = np.logspace(np.log10(max(positive.min(), 1e-12)),
                                   np.log10(max(positive.max(), bound) * 1.1),
                                   180)
            else:
                bins = np.linspace(0, max(bound, np.percentile(w, 99.9)) * 1.05,
                                   180)
            ax.hist(w, bins=bins, histtype='step', lw=1.6, color=COLOR[mode],
                    density=True,
                    label=r'%s: $\langle w\rangle$=%.3g, C=%.3g, C/$\langle w\rangle$=%.2f'
                          % (LABEL[mode], w.mean(), bound, bound / w.mean()))
            ax.axvline(bound, color=COLOR[mode], ls='--', lw=1.2)
            ax.axvline(w.mean(), color=COLOR[mode], ls=':', lw=1.2)
    for ax, scale in zip(axes, ('linear', 'log')):
        ax.set_xscale(scale)
        ax.set_yscale('log')
        ax.set_xlabel('mass-set weight $w$ tested against $C$ = maxwgts[0]')
        ax.set_ylabel('density')
    axes[0].set_title('linear in $w$ (dashed: the bound $C$; dotted: the mean)')
    axes[1].set_title('log in $w$')
    axes[1].legend(fontsize=8, loc='lower center')
    fig.suptitle(r'The mass-stage accept/reject: $\epsilon_m \simeq C/\langle w\rangle$')
    fig.tight_layout()
    fig.savefig(pjoin(outdir, 'mass_weight_distribution%s.png' % tag), dpi=140)
    plt.close(fig)

    # ---- 3. decomposition of the PA weight ---------------------------------
    pa = data['PA']['arr']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    pieces = [('$J$ (reshuffling jacobian)', pa['jac'], '#c0392b'),
              (r'$\prod_k$ jac$_{\rm BW}\cdot\hat Z$ (the no-jac. weight)',
               pa['w'] / pa['jac'], '#2980b9'),
              (r'$w = J\cdot\prod_k$ jac$_{\rm BW}\hat Z$', pa['w'], '#7f8c8d')]
    for label, values, color in pieces:
        values = values[np.isfinite(values) & (values > 0)]
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 180)
        ax.hist(values, bins=bins, histtype='step', lw=1.7, color=color,
                density=True,
                label='%s\n   mean %.3g, max/mean %.1f'
                      % (label, values.mean(), values.max() / values.mean()))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    ax.set_title('which factor carries the spread')

    ax = axes[1]
    other = pa['w'] / pa['jac']
    ax.hist2d(np.log10(pa['jac']), np.log10(other), bins=120,
              norm=matplotlib.colors.LogNorm(), cmap='magma')
    ax.set_xlabel(r'$\log_{10} J$')
    ax.set_ylabel(r'$\log_{10}(\prod$ jac$_{\rm BW}\hat Z)$')
    ax.set_title('the two factors against each other')
    fig.suptitle('PA mass-set weight, factorised')
    fig.tight_layout()
    fig.savefig(pjoin(outdir, 'pa_decomposition%s.png' % tag), dpi=140)
    plt.close(fig)

    # ---- 4. where the tail lives -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    sqrts = pa['sqrts']
    msum = np.array([sum(r) for r in data['PA']['arr'].get('mass_list', [])]) \
        if 'mass_list' in pa else None
    ax.hist2d(sqrts, np.log10(pa['jac']), bins=(140, 140),
              range=[[340, 1400], [-3, np.log10(pa['jac'].max())]],
              norm=matplotlib.colors.LogNorm(), cmap='viridis')
    ax.set_xlabel(r'production $\sqrt{\hat s}$  [GeV]')
    ax.set_ylabel(r'$\log_{10} J$')
    ax.set_title('the jacobian blows up at the $t\\bar t$ threshold')

    ax = axes[1]
    edges = np.array([340, 350, 360, 380, 400, 450, 500, 600, 800, 1200, 3000])
    centres, means, maxima = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (sqrts >= lo) & (sqrts < hi)
        if sel.sum() < 20:
            continue
        centres.append(.5 * (lo + hi))
        means.append(pa['jac'][sel].mean())
        maxima.append(pa['jac'][sel].max())
    ax.plot(centres, means, 'o-', color='#2980b9', label=r'$\langle J\rangle$')
    ax.plot(centres, maxima, 's--', color='#c0392b', label=r'$\max J$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'production $\sqrt{\hat s}$  [GeV]')
    ax.set_ylabel('$J$')
    ax.legend()
    ax.set_title('mean and max of $J$ per $\\sqrt{\\hat s}$ slice')
    fig.tight_layout()
    fig.savefig(pjoin(outdir, 'jacobian_vs_sqrts%s.png' % tag), dpi=140)
    plt.close(fig)


def plot_bootstrap(boot, data, outdir, tag):
    """The bound, replayed.  Left: where each replica's bound landed, against
    the mean of the weight it bounds. Right: the resulting eps_m."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = sorted(boot['replicas'],
                    key=lambda k: boot['replicas'][k]['Nevents_for_max_weight'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for i, label in enumerate(labels):
        entry = boot['replicas'][label]
        values = entry['bounds']
        ax.scatter([i + 1] * len(values), values, s=18, alpha=.65,
                   color='#c0392b')
        ax.plot([i + .75, i + 1.25], [entry['bound_mean']] * 2, color='k', lw=2)
    ax.axhline(boot['mean_w'], color='#2980b9', lw=1.5, ls='--',
               label=r'$\langle w\rangle$ = %.3f (the mean the bound divides)'
                     % boot['mean_w'])
    if 'PA' in data:
        ax.axhline(data['PA']['bounds'][0], color='#e67e22', lw=1.5, ls=':',
                   label='the actual run\'s C = %.2f'
                         % data['PA']['bounds'][0])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(['%s\nN=%d, $n_\\sigma$=%.2f'
                        % (l.replace('nevents=', 'nev '),
                           boot['replicas'][l]['Nevents_for_max_weight'],
                           boot['replicas'][l]['nb_sigma']) for l in labels],
                       fontsize=8)
    ax.set_ylabel('mass-stage bound $C$ = maxwgts[0]')
    ax.set_title('%d independent replays of the max-weight scan'
                 % len(boot['replicas'][labels[0]]['bounds']))
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, label in enumerate(labels):
        entry = boot['replicas'][label]
        eps = np.array(entry['bounds']) / boot['mean_w']
        ax.scatter([i + 1] * len(eps), eps, s=18, alpha=.65, color='#c0392b')
        ax.plot([i + .75, i + 1.25], [eps.mean()] * 2, color='k', lw=2)
    ax.axhline(1.10, color='#2980b9', lw=1.5, ls='--',
               label=r'PA (no jac.): $\epsilon_m$ = 1.10, i.e. the 10% margin alone')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([l.replace('nevents=', 'nev ') for l in labels],
                       fontsize=8)
    ax.set_ylabel(r'$\epsilon_m = C/\langle w\rangle$')
    ax.set_title('the mass-stage cost that bound implies')
    ax.legend(fontsize=8)
    fig.suptitle('The PA mass-stage bound is set by a rare excursion, so it '
                 'moves with the probe')
    fig.tight_layout()
    fig.savefig(pjoin(outdir, 'bound_stability%s.png' % tag), dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
