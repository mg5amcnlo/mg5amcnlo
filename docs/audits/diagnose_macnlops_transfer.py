#!/usr/bin/env python3
"""Measure the S-veto versus negative-H balance of a MAcNLOPS first step.

The rejection stage freezes one decision per proposal. In bins of the
campaign's hard-system pT at the first-step endpoint, and using the
expected removals 1-a of those decisions rather than their Bernoulli draws,

    X = sum_S       w (1-a_S)       S weight removed by the veto,
    Y = sum_{H<0} |w| (1-a_H)       weight added by removing negative H,
    Z = sum_{H<0} |w| (1-a_H)/f     that transfer without the source factor.

f is the sampled H component's source factor: the Delta no-emission
probability for MC@NLO-Delta and the bogus P for ordinary sources. The
sampled Delta rule has v=C0/K and T=Delta*C0, so X/Z=<E_g/K> and
Y/Z=<Delta>, both weighted by C0, and the balance X/Y=1 requires
E_g/K=Delta. Ordinary sources have v=T/K, so X/Y=<E_g/K> directly.
With the Born cross section of the same cards, beta=sigma_S/sigma_B and
X/(beta*Y) is the imbalance that the S normalization does not explain.
The 1/f weights make Z heavy-tailed where f can be tiny, so each bin also
reports the largest single-event share of Z; a large share means X/Z and
Y/Z rest on a few events, while X/Y is unaffected.
Global-sum sources carry no single f per H event, so Z is unavailable.
Nothing is generated, showered or rewritten.
"""
import argparse
import bisect
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import macnlops as mc

# Reuse the validation plots' hard-object definitions.
spec = importlib.util.spec_from_file_location(
    'macnlops_validation_plots', ROOT / 'docs/audits/plot_macnlops_validation.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

PROCESSES = ('dy', 'zz', 'ttbar', 'wbb', 'single_top', 'wjet')
DEFAULT_EDGES = (0., 2., 5., 10., 20., 35., 50., 100., 200.)
TTBAR_EDGES = (0., 5., 10., 20., 40., 60., 80., 100., 150., 250.)
SUMS = ('S_events', 'S_eligible_events', 'H_negative_events', 'H_negative_factored',
        'S_eligible', 'X', 'X2', 'X_negative_S', 'X_realized',
        'H_negative', 'Y', 'Y2', 'Y_realized', 'Z', 'Z2', 'YZ', 'U')
MAXIMA = ('Z_max',)
# Result key, table column and figure legend.
RATIOS = (('S_removed_per_H_added', 'X/Y', 'X/Y: S removed per H added'),
          ('Eg_over_K', 'X/Z', r'X/Z = $\langle E_g/K\rangle$'),
          ('source_factor', 'Y/Z', r'Y/Z = $\langle f\rangle$, source $\Delta$ or $P$'),
          ('imbalance_after_beta', 'X/(bY)', r'X/($\beta$Y): imbalance after $\beta$'))


class DiagnosticError(RuntimeError):
    pass


def jobs(paths):
    """Matching jobs, given directly or listed by a collected output."""
    for path in map(Path, paths):
        if (path / 'proposals.lhe').is_file():
            yield path
            continue
        summary = path / 'summary.json'
        if not summary.is_file():
            raise DiagnosticError('%s is neither a matching job nor a collected '
                                  'MAcNLOPS output' % path)
        for record in json.loads(summary.read_text()):
            if 'input' not in record:
                continue
            if record.get('status') != 'complete':
                raise DiagnosticError('Incomplete matching job for %s' % record['input'])
            job = Path(record['output']).parent
            # Recorded paths are absolute; allow a relocated output.
            yield job if job.is_dir() else path / job.name


def antikt(momenta, radius):
    """E-scheme anti-kT jets of the few partons of a first-step endpoint."""
    objects, jets = [tuple(p) for p in momenta], []
    while objects:
        inverse = [1. / max(prior.pt(p)**2, sys.float_info.min) for p in objects]
        distance, i, j = min((inverse[a], a, None) for a in range(len(objects)))
        for a in range(len(objects)):
            for b in range(a + 1, len(objects)):
                pa, pb = objects[a], objects[b]
                phi = math.remainder(math.atan2(pa[1], pa[0]) -
                                     math.atan2(pb[1], pb[0]), 2 * math.pi)
                dij = (min(inverse[a], inverse[b]) * ((prior.rapidity(pa) -
                       prior.rapidity(pb))**2 + phi**2) / radius**2)
                if dij < distance:
                    distance, i, j = dij, a, b
        if j is None:
            jets.append(objects.pop(i))
        else:
            objects[i] = tuple(x + y for x, y in zip(objects[i], objects.pop(j)))
    return sorted(jets, key=prior.pt, reverse=True)


def hard_system(event, process):
    """The campaign's hard-system (px, py, pz, E); None outside the W+jet selection."""
    stable = [(pdg, p[1], p[2], p[3], p[0]) for pdg, status, p in event.particles
              if status == 1]
    if process in prior.PROCESS_OBJECTS:
        objects = prior.pick_objects(prior.Event(event.weight, event.kind, stable), process)
    else:
        partons = [p[1:] for p in stable if 1 <= abs(p[0]) <= 5 or p[0] == 21]
        jets = antikt(partons, .4 if process == 'single_top' else .5)
        if process == 'wjet':
            jets = [jet for jet in jets
                    if prior.pt(jet) > 25. and abs(prior.rapidity(jet)) < 4.5]
        hard = [p[1:] for p in stable if p[0] == (6 if process == 'single_top' else 24)]
        if len(hard) != 1:
            raise DiagnosticError('%s endpoint needs one stable %s' %
                                  (process, 'top' if process == 'single_top' else 'W+'))
        if not jets:
            return None
        objects = [hard[0], jets[0]]
    return [sum(column) for column in zip(*objects)]


def system_pt(event, process):
    system = hard_system(event, process)
    return None if system is None else prior.pt(system)


def contributions(event, decision):
    """One event's terms; only removed negative H needs its source factor."""
    terms = dict.fromkeys(SUMS + MAXIMA, 0.)
    removed = 1. - decision['acceptance']
    if event.kind == 'S':
        x = event.weight * removed
        terms.update(S_events=1., X=x, X2=x * x,
                     X_negative_S=x if event.weight < 0 else 0.,
                     X_realized=0. if decision['accepted'] else event.weight)
        if decision['densities'] is not None:
            terms.update(S_eligible_events=1., S_eligible=event.weight)
    elif event.weight < 0:
        y = -event.weight * removed
        terms.update(H_negative_events=1., H_negative=-event.weight, Y=y, Y2=y * y,
                     Y_realized=0. if decision['accepted'] else -event.weight)
        if event.component is not None:
            factor = event.component.source_factor
            if not factor > 0:
                raise DiagnosticError('Negative H without a positive source factor')
            z = y / factor
            terms.update(H_negative_factored=1., Z=z, Z2=z * z, YZ=y * z, Z_max=z,
                         U=-event.weight / factor)
    return terms


def ratio(a, b, var_a, var_b, cov=0.):
    """Linearized ratio and its error; event sums are Poisson-like."""
    if not b:
        return None
    r = a / b
    return [r, math.sqrt(max(var_a - 2 * r * cov + r * r * var_b, 0.)) / abs(b)]


def ratios(sums, beta=None):
    result = dict(S_removed_per_H_added=ratio(sums['X'], sums['Y'], sums['X2'], sums['Y2']))
    # A partly factored sum would mix global and sampled H in Z.
    if sums['H_negative_events'] and sums['H_negative_factored'] == sums['H_negative_events']:
        result.update(Eg_over_K=ratio(sums['X'], sums['Z'], sums['X2'], sums['Z2']),
                      source_factor=ratio(sums['Y'], sums['Z'], sums['Y2'],
                                          sums['Z2'], sums['YZ']),
                      largest_Z_share=sums['Z_max'] / sums['Z'] if sums['Z'] else None)
    if beta:
        result['imbalance_after_beta'] = ratio(sums['X'], beta * sums['Y'],
                                               sums['X2'], beta**2 * sums['Y2'])
    return result


def float_list(text):
    """Comma-separated numbers, for --edges and similar options."""
    return [float(x) for x in text.split(',')]


def default_edges(process):
    return TTBAR_EDGES if process == 'ttbar' else DEFAULT_EDGES


def check_edges(edges):
    if list(edges) != sorted(set(edges)) or edges[0] < 0:
        raise DiagnosticError('Bin edges must increase from a nonnegative value')


def job_records(job, *names):
    """Each proposal of a matching job with its records from the named JSONL files.

    Yields (sequence, event, *records) and checks that every file stays in step
    with proposals.lhe and has no trailing records.
    """
    with contextlib.ExitStack() as files:
        stream = files.enter_context(mc.LHEStream(job / 'proposals.lhe'))
        rows = [files.enter_context(open(job / name)) for name in names]
        for sequence, block in enumerate(stream, 1):
            records = [json.loads(row.readline() or 'null') for row in rows]
            if any(record is None or record['sequence'] != sequence for record in records):
                raise DiagnosticError('Proposal and %s records are out of step in %s' %
                                      (', '.join(names), job))
            yield (sequence, mc.Event(block), *records)
        if any(row.readline() for row in rows):
            raise DiagnosticError('Extra %s records in %s' % (', '.join(names), job))


def diagnose(paths, process, edges, born_xsec=None):
    check_edges(edges)
    # The last bin is the overflow above the final edge.
    bins = [dict.fromkeys(SUMS + MAXIMA, 0.) for _ in edges]
    unbinned, total = dict.fromkeys(SUMS + MAXIMA, 0.), dict.fromkeys(SUMS + MAXIMA, 0.)
    sigma = dict(input=0., S=0., H=0.)
    names, prescriptions = [], set()
    for job in jobs(paths):
        names.append(str(job))
        report = job / 'report.json'
        if report.is_file():
            prescriptions.add(json.loads(report.read_text()).get('veto_prescription'))
        for sequence, event, record, decision in job_records(
                job, 'shower.jsonl', 'events.lhe.gz.decisions.jsonl'):
            if not event.kind == record['event_kind'] == decision['kind']:
                raise DiagnosticError('S/H kind disagrees at %s:%d' % (job, sequence))
            if abs(event.weight - record['source_weight']) > 1e-12 * max(1., abs(event.weight)):
                raise DiagnosticError('Proposal weight differs from its source at %s:%d' %
                                      (job, sequence))
            sigma['input'] += event.weight
            sigma[event.kind] += event.weight
            terms = contributions(event, decision)
            value = system_pt(event, process)
            index = -1 if value is None else bisect.bisect_right(edges, value) - 1
            for target in (total, unbinned if index < 0 else bins[index]):
                for key, term in terms.items():
                    target[key] = (max(target[key], term) if key in MAXIMA
                                   else target[key] + term)
    beta = sigma['S'] / born_xsec if born_xsec else None
    upper = list(edges[1:]) + [None]
    return dict(
        jobs=names, veto_prescriptions=sorted(map(str, prescriptions)), process=process,
        observable='system_pt', edges=list(edges), born_xsec=born_xsec, sigma=sigma,
        beta=beta,
        bins=[dict(lo=lo, hi=hi, **sums, ratios=ratios(sums, beta))
              for lo, hi, sums in zip(edges, upper, bins)],
        unbinned=dict(unbinned, ratios=ratios(unbinned, beta)),
        total=dict(total, ratios=ratios(total, beta)))


def formatted(value):
    return '%15s' % ('n/a' if value is None else '%.3g+-%.2g' % tuple(value))


def bin_label(sums):
    return ('>%g' % sums['lo'] if sums['hi'] is None else
            '%g-%g' % (sums['lo'], sums['hi']))


def table(result):
    sigma, beta = result['sigma'], result['beta']
    lines = ['MAcNLOPS transfer: %s, %d jobs, prescription %s' % (
                 result['process'], len(result['jobs']),
                 ', '.join(result['veto_prescriptions']) or 'unrecorded'),
             'sigma_input %.6g pb, sigma_S %.6g pb, sigma_H %.6g pb%s' % (
                 sigma['input'], sigma['S'], sigma['H'],
                 '' if beta is None else ', beta = sigma_S/sigma_B = %.4g' % beta),
             '%-12s %11s %11s %11s %11s' % ('pT [GeV]', 'S elig [pb]', 'X [pb]',
                                             'Y [pb]', 'Z [pb]') +
             ''.join('%15s' % column for _, column, _ in RATIOS) + '%8s' % 'maxz/Z']
    rows = [(bin_label(b), b) for b in result['bins']]
    if result['unbinned']['S_events'] or result['unbinned']['H_negative_events']:
        rows.append(('unbinned', result['unbinned']))
    rows.append(('total', result['total']))
    for label, sums in rows:
        share = sums['ratios'].get('largest_Z_share')
        lines.append('%-12s %11.4g %11.4g %11.4g %11.4g' % (
            label, sums['S_eligible'], sums['X'], sums['Y'], sums['Z']) +
            ''.join(formatted(sums['ratios'].get(key)) for key, _, _ in RATIOS) +
            '%8s' % ('n/a' if share is None else '%.2f' % share))
    total = result['total']
    lines.append('expected: S removed %.6g pb (negative S %.6g pb), H added %.6g pb; '
                 'realized: S removed %.6g pb, H added %.6g pb, net %+.6g pb' % (
                     total['X'], total['X_negative_S'], total['Y'], total['X_realized'],
                     total['Y_realized'], total['Y_realized'] - total['X_realized']))
    return '\n'.join(lines)


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = result['bins']
    labels = [bin_label(b) for b in bins]
    figure, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5),
                                         gridspec_kw=dict(height_ratios=(1, 1.4)))
    x = range(len(bins))
    top.plot(x, [b['X'] for b in bins], 'o-', label='X: S removed')
    top.plot(x, [b['Y'] for b in bins], 's-', label='Y: H added')
    top.set_ylabel('pb per bin')
    top.legend()
    positive = True
    for key, _, label in RATIOS:
        points = [(i, b['ratios'][key]) for i, b in enumerate(bins)
                  if b['ratios'].get(key) is not None]
        if points:
            positive &= all(value > 0 for _, (value, _) in points)
            bottom.errorbar([i for i, _ in points], [value for _, (value, _) in points],
                            [error for _, (_, error) in points], fmt='o-', capsize=2,
                            label=label)
    bottom.axhline(1., color='grey', lw=.8)
    if positive:
        bottom.set_yscale('log')
    bottom.set_xticks(list(x))
    bottom.set_xticklabels(labels, rotation=30)
    bottom.set_xlabel('hard-system $p_T$ at the first-step endpoint [GeV]')
    bottom.legend(fontsize='small')
    figure.suptitle('MAcNLOPS transfer balance: %s' % result['process'])
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', type=Path,
                        help='collected macnlops output directories or single matching jobs')
    parser.add_argument('--process', required=True, choices=PROCESSES)
    parser.add_argument('--edges', type=float_list,
                        help='comma-separated hard-system pT edges in GeV; '
                             'an overflow bin is added')
    parser.add_argument('--born-xsec', type=float,
                        help='Born cross section in pb from the same cards, for beta')
    parser.add_argument('--json', type=Path, help='write all sums and ratios')
    parser.add_argument('--plot', type=Path, help='write a summary figure')
    args = parser.parse_args(argv)
    result = diagnose(args.paths, args.process, args.edges or default_edges(args.process),
                      args.born_xsec)
    print(table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + '\n')
    if args.plot:
        plot(result, args.plot)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (DiagnosticError, mc.MatchingError, prior.ValidationError, OSError) as error:
        print('MAcNLOPS transfer diagnostic: %s' % error, file=sys.stderr)
        sys.exit(1)
