#!/usr/bin/env python3
################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Timing benchmark for MadSpin, built on :mod:`madspin_comparator`.

Runs one production sample through MadSpin once and records the wall-time split
that :class:`~MadSpin.interface_madspin.MadSpinInterface` reports (decay-event
generation, matrix-element generation, max-weight scan, accept/reject loop,
output gzip), plus the optional LHE-parser breakdown.

The production sample lives under ``--workdir`` and is *reused* across
invocations, so re-timing after a change costs one MadSpin run, not one MG5 run
plus one MadSpin run.

Usage
-----
Baseline (default benchmark is ``p p > t t~`` with both tops fully decayed)::

    ./tests/parallel_tests/madspin_benchmark.py --label baseline

Re-time after a change and compare against the baseline record::

    ./tests/parallel_tests/madspin_benchmark.py --label mg7-decay-gen \\
        --compare-to <path to the baseline json>

Every run appends a JSON record to ``--out`` (default
``madspin_benchmark_records.json`` inside the workdir).
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import datetime
import json
import logging
import os
import platform
import resource
import sys

pjoin = os.path.join

_here = os.path.dirname(os.path.realpath(__file__))
_root = os.path.split(os.path.split(_here)[0])[0]
if _root not in sys.path:
    sys.path.insert(0, _root)

from tests.parallel_tests.madspin_comparator import (
    MadSpinFactory,
    SpinModeConfig,
)

_logger = logging.getLogger('madspin_benchmark')


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------
# 'ttbar_full' is the reference benchmark for the MadSpin performance work:
# p p > t t~ with both tops decayed all the way down (t > b w+, w+ > all all),
# which is the case that motivated moving decay-event generation off Fortran
# madevent.
BENCHMARKS = {
    'ttbar_full': dict(
        production_process='p p > t t~',
        decays=['t > b w+, w+ > all all',
                't~ > b~ w-, w- > all all'],
        multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                        'j': 'g u d s c u~ d~ s~ c~'},
        extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
    ),
    # Smaller sibling, handy for smoke-testing harness changes without paying
    # for the full W decay multiplicity.
    'ttbar_semilep': dict(
        production_process='p p > t t~',
        decays=['t > b w+, w+ > l+ vl',
                't~ > b~ w-, w- > j j'],
        multiparticles={'p': 'g u d s c u~ d~ s~ c~',
                        'j': 'g u d s c u~ d~ s~ c~',
                        'l+': 'e+ mu+', 'vl': 've vm'},
        extra_run_card={'ebeam1': 6500, 'ebeam2': 6500},
    ),
}


def _peak_rss_mb():
    """Peak RSS of this process *and its children*, in MiB.

    ru_maxrss is bytes on macOS and kibibytes on Linux.
    """
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1024. * 1024. if sys.platform == 'darwin' else 1024.
    return usage.ru_maxrss / scale


def run_benchmark(benchmark, label, workdir, nevents, seed, spinmode,
                  lhe_timers, extra_madspin_settings=None):
    """Run one MadSpin timing measurement and return the record dict."""
    spec = BENCHMARKS[benchmark]
    # The production sample only depends on (process, nevents, seed), so key
    # the working tree on that and let unrelated labels share it.
    proc_key = '%s_n%d_s%d' % (benchmark, nevents, seed)
    base_dir = pjoin(workdir, proc_key)
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    factory = MadSpinFactory(
        name=proc_key,
        nevents=nevents,
        seed=seed,
        base_dir=base_dir,
        extra_madspin_settings=extra_madspin_settings or {},
        **spec
    )

    if lhe_timers:
        os.environ['MG_LHE_TIMERS'] = '1'
    else:
        os.environ.pop('MG_LHE_TIMERS', None)

    config = SpinModeConfig('bench_%s' % label, spinmode)
    _logger.info('running benchmark %s [label=%s, spinmode=%s, nevents=%d]',
                 benchmark, label, spinmode, nevents)
    result = factory.run_mode(config)

    record = {
        'label': label,
        'benchmark': benchmark,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'spinmode': spinmode,
        'nevents': nevents,
        'seed': seed,
        'host': platform.node(),
        'platform': platform.platform(),
        'cpu_count': os.cpu_count(),
        'wall_seconds': round(result.wall_seconds, 3),
        'phase_seconds': result.phase_seconds,
        'phase_counts': result.phase_counts,
        'lhe_timers': result.lhe_timers,
        'peak_rss_mb': round(_peak_rss_mb(), 1),
        'efficiency': result.efficiency,
        'BR': result.BR,
        'cross_in': result.cross_in,
        'cross_out': result.cross_out,
        'log_path': result.log_path,
        'lhe_path': result.lhe_path,
    }
    return record


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
# Phases printed in run order rather than alphabetically. Anything MadSpin
# reports that is not listed here is appended afterwards, so a new phase shows
# up without touching this table.
_PHASE_ORDER = [
    'decay_event_generation',
    'decay_me_generate',
    'decay_me_output',
    'decay_mg7_launch',
    'decay_mg7_integrate',
    'me_generation',
    'max_weight_scan',
    'decay_loop',
    'decay_event_refill',
    'output_gzip',
    'total',
]

# Phases that run *inside* another phase and are therefore already included in
# its total. Reported indented, and the parent also gets a derived
# "<parent> (net)" row so the columns still add up to something meaningful.
_NESTED_IN = {
    'decay_event_refill': 'decay_loop',
    # Amplitude generation and code writing happen once, when the decay
    # directory is created, so they sit inside the initial generation pass.
    'decay_me_generate': 'decay_event_generation',
    'decay_me_output': 'decay_event_generation',
    # ... whereas the launcher runs again for every pool refill, so its time is
    # split between decay_event_generation and decay_event_refill and cannot be
    # nested under either.
    'decay_mg7_integrate': 'decay_mg7_launch',
}


def _ordered_phases(seconds):
    known = [p for p in _PHASE_ORDER if p in seconds]
    rest = sorted(p for p in seconds if p not in _PHASE_ORDER)
    return known + rest


# The LHE timers are nested, so they must not simply be added up:
#   next_event_readline_total          EventFile.next_event, the whole read
#     next_event_readline_event_parse    the Event() built inside it
#   event_parse_total                  every Event(), including those above
#     event_parse_particle_block
#       particle_parse_total
#     event_parse_tag_block
#     event_parse_assign_mother
# Only the two roots count, and event_parse_total must have the part already
# covered by next_event_readline_event_parse taken out of it.
_LHE_NESTED = {
    'next_event_readline_event_parse',
    'event_parse_particle_block',
    'particle_parse_total',
    'event_parse_tag_block',
    'event_parse_assign_mother',
}


def lhe_exclusive_seconds(lhe_timers):
    """Wall time actually spent in the LHE parser, without double counting."""
    def secs(key):
        return lhe_timers.get(key, (0.0, 0))[0]
    direct_event_parse = max(
        0.0, secs('event_parse_total') - secs('next_event_readline_event_parse'))
    return secs('next_event_readline_total') + direct_event_parse


def derived_phases(seconds):
    """Return ``seconds`` plus the derived net rows for nested phases."""
    out = dict(seconds)
    for child, parent in _NESTED_IN.items():
        if child in seconds and parent in seconds:
            out['%s (net)' % parent] = seconds[parent] - seconds[child]
    return out


def format_record(record, baseline=None):
    """Render one record (optionally against a baseline) as a text table."""
    seconds = record.get('phase_seconds', {})
    counts = record.get('phase_counts', {})
    total = seconds.get('total') or record.get('wall_seconds') or 0.0
    base_seconds = (baseline or {}).get('phase_seconds', {})

    lines = []
    lines.append('MadSpin benchmark: %s [%s, %s, nevents=%s]'
                 % (record['label'], record['benchmark'], record['spinmode'],
                    record['nevents']))
    lines.append('  wall %.1f s   peak RSS %.0f MiB   %s cores'
                 % (record['wall_seconds'], record['peak_rss_mb'],
                    record['cpu_count']))
    if baseline:
        lines.append('  baseline: %s (wall %.1f s)'
                     % (baseline['label'], baseline['wall_seconds']))

    seconds = derived_phases(seconds)
    base_seconds = derived_phases(base_seconds)

    header = '    %-28s %10s %7s' % ('phase', 'seconds', '%')
    if baseline:
        header += ' %10s %8s' % ('base s', 'speedup')
    lines.append(header)
    for phase in _ordered_phases(seconds):
        val = seconds[phase]
        share = 100. * val / total if total else 0.
        # Nested phases are already counted inside their parent: indent them
        # so the shares are not read as additive.
        name = ('  \\_ ' + phase) if phase in _NESTED_IN else phase
        row = '    %-28s %10.2f %6.1f%%' % (name, val, share)
        if baseline:
            base = base_seconds.get(phase)
            if base is None:
                row += ' %10s %8s' % ('-', '-')
            elif val > 0:
                row += ' %10.2f %7.2fx' % (base, base / val)
            else:
                row += ' %10.2f %8s' % (base, 'inf')
        lines.append(row)

    interesting = {k: v for k, v in counts.items() if v}
    if interesting:
        lines.append('    counts: %s'
                     % ', '.join('%s=%s' % (k, v)
                                 for k, v in sorted(interesting.items())))
    if record.get('efficiency'):
        lines.append('    unweighting efficiency: %.4f' % record['efficiency'])

    lhe = record.get('lhe_timers') or {}
    if lhe:
        lines.append('    LHE parser: %.2f s (%.1f%% of wall)'
                     % (lhe_exclusive_seconds(lhe),
                        100. * lhe_exclusive_seconds(lhe) / total if total else 0.))
        for key, (secs, calls) in sorted(lhe.items(), key=lambda kv: -kv[1][0]):
            marker = '  ' if key in _LHE_NESTED else '* '
            lines.append('      %s%-24s %10.2f s over %d call(s)'
                         % (marker, key, secs, calls))
        lines.append('      (* counted in the total; the rest are nested '
                     'inside a starred timer)')
    return '\n'.join(lines)


def load_records(path):
    if not os.path.exists(path):
        return []
    with open(path) as fp:
        try:
            return json.load(fp)
        except ValueError:
            return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', default='ttbar_full',
                        choices=sorted(BENCHMARKS),
                        help='which benchmark to run (default: ttbar_full)')
    parser.add_argument('--label', default='run',
                        help='name for this measurement, used in the report')
    parser.add_argument('--workdir', default=None,
                        help='where the production sample and run dirs live '
                             '(default: $MADSPIN_BENCH_DIR or ./madspin_bench)')
    parser.add_argument('--nevents', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spinmode', default='PA',
                        help='MadSpin spinmode (default: PA, the shipped default)')
    parser.add_argument('--no-lhe-timers', action='store_true',
                        help='do not set MG_LHE_TIMERS (the timers add a small '
                             'per-particle overhead to the parse path)')
    parser.add_argument('--set', dest='settings', action='append', default=[],
                        metavar='KEY=VALUE',
                        help='extra "set KEY VALUE" line for the MadSpin card '
                             '(repeatable)')
    parser.add_argument('--out', default=None,
                        help='JSON file the records are appended to '
                             '(default: <workdir>/madspin_benchmark_records.json)')
    parser.add_argument('--compare-to', default=None,
                        help='label of a previous record (in --out) to compare '
                             'against; "previous" picks the last one')
    parser.add_argument('--report-only', action='store_true',
                        help='do not run anything, just print the stored records')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    workdir = args.workdir or os.environ.get('MADSPIN_BENCH_DIR') \
        or pjoin(os.getcwd(), 'madspin_bench')
    workdir = os.path.realpath(workdir)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    out_path = args.out or pjoin(workdir, 'madspin_benchmark_records.json')

    records = load_records(out_path)

    if args.report_only:
        if not records:
            print('no records in %s' % out_path)
            return 1
        for rec in records:
            print(format_record(rec))
            print('')
        return 0

    extra = {}
    for item in args.settings:
        if '=' not in item:
            parser.error('--set expects KEY=VALUE, got %r' % item)
        key, val = item.split('=', 1)
        extra[key.strip()] = val.strip()

    record = run_benchmark(
        benchmark=args.benchmark,
        label=args.label,
        workdir=workdir,
        nevents=args.nevents,
        seed=args.seed,
        spinmode=args.spinmode,
        lhe_timers=not args.no_lhe_timers,
        extra_madspin_settings=extra,
    )

    baseline = None
    if args.compare_to:
        candidates = [r for r in records
                      if r['benchmark'] == record['benchmark']]
        if args.compare_to == 'previous':
            baseline = candidates[-1] if candidates else None
        else:
            matching = [r for r in candidates if r['label'] == args.compare_to]
            baseline = matching[-1] if matching else None
        if baseline is None:
            _logger.warning('no stored record labelled %r for benchmark %s',
                            args.compare_to, record['benchmark'])

    records.append(record)
    with open(out_path, 'w') as fp:
        json.dump(records, fp, indent=2, sort_keys=True)

    print('')
    print(format_record(record, baseline))
    print('')
    print('record appended to %s' % out_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
