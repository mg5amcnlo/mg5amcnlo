#!/usr/bin/env python3
"""Turn work/results.jsonl into the verdict table."""
import json
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, 'work', 'results.jsonl')
PROD = 504.514


def load():
    recs = {}
    with open(RESULTS) as fsock:
        for line in fsock:
            line = line.strip()
            if line:
                rec = json.loads(line)
                recs[rec['tag']] = rec       # last write wins
    return recs


def fmt(value, spec='%.6g'):
    return 'n/a' if value is None else spec % value


def main():
    recs = load()
    rows = []
    header = ('combination', 'exit', 'out?', 'init xsec (pb)', 'BR',
              'n evt', 'mean wgt', 'zero wgt', 'sec')
    for tag in sorted(recs):
        r = recs[tag]
        rows.append((
            tag,
            str(r['exit_code']) + ('/TIMEOUT' if r.get('timeout') else ''),
            'yes' if r['output_exists'] else 'NO',
            fmt(r.get('init_xsec')),
            fmt(r.get('implied_br'), '%.5f'),
            str(r.get('nb_events', '-')),
            fmt(r.get('wgt_mean')),
            str(r.get('n_zero', '-')),
            str(r.get('seconds')),
        ))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda cells: '| ' + ' | '.join(
        str(c).ljust(w) for c, w in zip(cells, widths)) + ' |'
    print(line(header))
    print('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    for row in rows:
        print(line(row))

    print()
    print('fresh vs reuse (identical numbers = the #364 check):')
    for tag in sorted(recs):
        if not tag.endswith('__fresh'):
            continue
        base = tag[:-len('__fresh')]
        other = recs.get(base + '__reuse')
        if not other:
            continue
        f = recs[tag]
        same = (f.get('init_xsec') == other.get('init_xsec')
                and f.get('nb_events') == other.get('nb_events')
                and f.get('wgt_mean') == other.get('wgt_mean'))
        print('  %-45s %s   fresh=%s reuse=%s' % (
            base, 'SAME' if same else 'DIFFERENT',
            fmt(f.get('init_xsec')), fmt(other.get('init_xsec'))))


if __name__ == '__main__':
    main()
