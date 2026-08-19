#!/usr/bin/env python3
"""The two extra checks: generate_events file placement, and fixed_order.

    analyse_extras.py <workdir> <out_dir>

For the ``generate_events`` run it asks the only question that matters: is
there a file at ``Events/<run>_decayed_1/unweighted_events.lhe.gz``, does it
hold decayed events, and does it carry ``XSECUP = 0`` plus the
``<MGPureInterference>`` block?  (MadSpin's ``launch`` path does not overwrite
the production run -- it writes a NEW run directory beside it and says so in
the log.  Looking in ``Events/<run>/`` instead finds the undecayed production
sample and proves nothing.)

For ``fixed_order`` it counts the ``<eventgroup>`` blocks, their members, and
-- per member -- how many particles the event holds.  A decayed
``p p > t t~``(+jet) event has 12 particles; an undecayed one has 4.  That
single number is what the fixed-order finding rests on.
"""

import gzip
import json
import os
import re
import sys

FO = [('fixedorder',      'PA',      'pure_interference ON'),
      ('fo_ctrl_onshell', 'onshell', 'pure_interference OFF'),
      ('fo_ctrl_PA',      'PA',      'pure_interference OFF'),
      ('fo_ctrl_madspin', 'madspin', 'pure_interference OFF')]


def scan_groups(path, max_groups=2000):
    """(n_groups, members per group, particle count of each member position)."""
    groups = 0
    members = []
    cur = 0
    per_position = {}
    pos = 0
    nparts = None
    in_ev = head = False
    with gzip.open(path, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('<eventgroup'):
                if cur:
                    members.append(cur)
                groups, cur, pos = groups + 1, 0, 0
                continue
            if line.startswith('<event'):
                in_ev, head, cur, nparts = True, True, cur + 1, 0
                continue
            if line.startswith('</event'):
                in_ev = False
                per_position.setdefault(pos, []).append(nparts)
                pos += 1
                continue
            if in_ev:
                if head:
                    head = False
                    continue
                if len(line.split()) >= 13:
                    nparts += 1
    if cur:
        members.append(cur)
    return groups, members, per_position


def header_of(path):
    h = []
    with gzip.open(path, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('<event'):
                break
            h.append(line)
    return ''.join(h)


def main():
    work, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    res = {}

    # ------------------------------------------------ 1. generate_events
    ev_dir = os.path.join(work, 'prod', 'Events')
    entry = dict(run_dir_exists=os.path.isdir(ev_dir))
    cand = os.path.join(ev_dir, 'pa_integration_decayed_1',
                        'unweighted_events.lhe.gz')
    prod = os.path.join(ev_dir, 'pa_integration', 'unweighted_events.lhe.gz')
    entry['decayed_path'] = cand
    entry['decayed_exists'] = os.path.exists(cand)
    entry['production_run_untouched'] = os.path.exists(prod)
    if entry['decayed_exists']:
        h = header_of(cand)
        n = sum(1 for line in gzip.open(cand, 'rt', errors='ignore')
                if line.startswith('<event'))
        m = re.search(r'<init>\s*\n[^\n]*\n\s*([-+0-9.eE]+)', h)
        entry.update(
            n_events=n,
            has_pi_block='<MGPureInterference>' in h,
            xsecup=float(m.group(1)) if m else None,
            z=re.search(r'z = S / error\s*:\s*([-+0-9.]+)', h).group(1))
        g, mem, pos = scan_groups(cand)
        entry['particles_first_event'] = pos.get(0, [None])[0]
    res['integration'] = entry

    print('== generate_events placement ==')
    e = res['integration']
    print('  decayed file        :', e['decayed_path'])
    print('  exists              :', e['decayed_exists'])
    print('  production run kept :', e['production_run_untouched'])
    if e['decayed_exists']:
        print('  events / XSECUP     : %d / %s' % (e['n_events'], e['xsecup']))
        print('  <MGPureInterference>:', e['has_pi_block'], ' z =', e['z'])
        print('  particles in event 1: %s (12 = decayed, 4 = not)'
              % e['particles_first_event'])

    # ------------------------------------------------------ 2. fixed_order
    print('\n== fixed_order: are the group members decayed? ==')
    print('%-18s %-8s %-22s %7s %8s %s'
          % ('tag', 'spinmode', 'mode', 'groups', 'members', 'particles per member'))
    for tag, mode, what in FO:
        p = os.path.join(work, tag, 'events_decayed.lhe.gz')
        r = dict(spinmode=mode, what=what, exists=os.path.exists(p))
        if r['exists']:
            g, mem, pos = scan_groups(p)
            # positions beyond the group size are the tail of the scan, not
            # real members -- drop them
            pos = {k: v for k, v in pos.items() if k < max(mem)}
            r.update(groups=g, members_min=min(mem), members_max=max(mem),
                     particles_per_position={
                         k: sorted(set(v))[:4] for k, v in pos.items()})
            print('%-18s %-8s %-22s %7d %8s %s'
                  % (tag, mode, what, g,
                     '%d-%d' % (min(mem), max(mem)),
                     ' | '.join('member %d: %s' % (k, r[
                         'particles_per_position'][k])
                         for k in sorted(pos))))
        else:
            print('%-18s %-8s %-22s   MISSING' % (tag, mode, what))
        res[tag] = r
    print('\n  (12 particles = the decayed t t~(+j) event, 4 = undecayed)')

    with open(os.path.join(out_dir, 'extras.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nwrote', os.path.join(out_dir, 'extras.json'))


if __name__ == '__main__':
    main()
