#!/usr/bin/env python3
"""Build a GitHub Actions job summary for the systematic mg7 cross-section
checks (.github/workflows/check_xsec_processes_mg7.yml).

Reads the per-process JSON records written by
tests/acceptance_tests/test_check_xsec_processes_mg7.py (one file per
process, via $MG7_XSEC_RESULTS_DIR) plus the reference file that defines
which processes should have been tested, and prints a markdown report:
a table of every process (with pass/fail, cross-sections, deviation,
pull and overweight tail), followed by the scraped error message for every
process that did not succeed.

Usage:
    build_xsec_summary.py <reference.json> <results_dir> [--tolerance PCT] \
        >> "$GITHUB_STEP_SUMMARY"
"""
import argparse
import glob
import json
import math
import os

pjoin = os.path.join


def fmt_value(x, sig=4):
    if x is None:
        return '—'
    return '{:.{sig}g}'.format(x, sig=sig)


def fmt_signed(x, decimals=2):
    if x is None or math.isnan(x) or math.isinf(x):
        return '—'
    return '{:+.{d}f}'.format(x, d=decimals)


def slug(section, id_):
    return 'fail-%s-%s' % (section, id_)


def load_results(results_dir):
    """Return {(section, id): record} from every *.json under results_dir."""
    results = {}
    for path in sorted(glob.glob(pjoin(results_dir, '**', '*.json'), recursive=True)):
        try:
            with open(path) as f:
                record = json.load(f)
        except Exception:
            continue
        results[(record.get('section'), record.get('id'))] = record
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', help='path to check_xsec_processes_reference.json')
    parser.add_argument('results_dir', help='directory with the per-process result JSON files')
    parser.add_argument('--tolerance', type=float, default=None,
                         help='tolerance (fraction, e.g. 0.01) used for this run, for display only')
    parser.add_argument('--events', default=None,
                         help='events/process used for this run, for display only')
    args = parser.parse_args()

    with open(args.reference) as f:
        ref = json.load(f)

    results = load_results(args.results_dir)

    rows = []
    failures = []
    n_pass = 0
    n_fail = 0
    for section, entries in ref['sections'].items():
        for entry in entries:
            key = (section, entry['id'])
            record = results.get(key)
            ref_cross = entry['cross']
            ref_error = entry.get('error')

            if record is None:
                status = 'fail'
                got = err = overweight = None
                message = ('No result recorded for this process -- the job '
                            'likely crashed or was cancelled before it '
                            'completed. Check the workflow run log for '
                            '`test_%s_%s_mg7`.' % (section, entry['id']))
            else:
                status = record.get('status', 'fail')
                got = record.get('cross')
                err = record.get('error')
                overweight = record.get('overweight')
                message = record.get('message')

            if status == 'pass':
                n_pass += 1
            else:
                n_fail += 1

            reldev_pct = None
            pull = None
            if got is not None and ref_cross:
                reldev_pct = 100.0 * (got - ref_cross) / ref_cross
            if got is not None and err is not None and ref_error is not None:
                denom = math.sqrt(err ** 2 + ref_error ** 2)
                if denom > 0:
                    pull = (got - ref_cross) / denom

            anchor = slug(section, entry['id'])
            status_cell = '✅' if status == 'pass' else ('[❌](#%s)' % anchor)

            rows.append({
                'status_cell': status_cell,
                'section': section,
                'id': entry['id'],
                'process': entry['process'],
                'cross': fmt_value(got),
                'error': fmt_value(err),
                'ref_cross': fmt_value(ref_cross),
                'ref_error': fmt_value(ref_error),
                'reldev': fmt_signed(reldev_pct),
                'pull': fmt_signed(pull),
                'overweight': (fmt_value(100.0 * overweight, sig=2)
                               if overweight is not None else '—'),
            })

            if status != 'pass':
                failures.append({
                    'anchor': anchor,
                    'section': section,
                    'id': entry['id'],
                    'process': entry['process'],
                    'message': message or '(no error message captured)',
                })

    out = []
    out.append('## Systematic cross-section checks (mg7)')
    out.append('')
    detail = []
    if args.tolerance is not None:
        detail.append('tolerance: %.2f%%' % (100 * args.tolerance))
    if args.events is not None:
        detail.append('events/process: %s' % args.events)
    if detail:
        out.append('_' + ' &middot; '.join(detail) + '_')
        out.append('')
    out.append('**%d/%d processes passed**' % (n_pass, n_pass + n_fail))
    out.append('')
    out.append('| | Section | Process | σ [pb] | ± σ | Ref σ [pb] | ± Ref | Δ [%] | Pull | Overw. [%] |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        out.append('| %s | %s | `%s`<br>%s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['status_cell'], r['section'], r['id'], r['process'],
            r['cross'], r['error'], r['ref_cross'], r['ref_error'],
            r['reldev'], r['pull'], r['overweight']))

    if failures:
        out.append('')
        out.append('## Failure details')
        for fr in failures:
            out.append('')
            out.append('### <a id="%s"></a>%s / `%s`' % (fr['anchor'], fr['section'], fr['id']))
            out.append('')
            out.append('`%s`' % fr['process'])
            out.append('')
            out.append('```')
            out.append(fr['message'])
            out.append('```')

    print('\n'.join(out))


if __name__ == '__main__':
    main()
