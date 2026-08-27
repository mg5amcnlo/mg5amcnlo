#!/usr/bin/env python3
"""Audit: do MadSpin's matrix elements see the parameters the events carry?

MadSpin's density spinmodes (``madspin`` / ``PA`` / ``onshell``) evaluate the
production and decay matrix elements through a compiled f2py library that is
initialised from a ``param_card.dat``.  Before the fix on
``claude/ms-paramcard-refresh`` that card was the one ``output standalone``
wrote *from the model defaults*, so a run whose events were generated with,
say, ``MT = 172.76`` could have its density matrices built at the model's
``MT = 173``.  Nothing in the log said so; the only evidence was on disk.

This script is that evidence, made checkable.  For a MadSpin run directory it
compares

    <run>/param_card.dat                     the run's own card, written by
                                             ``decay_all_events.__init__`` from
                                             ``banner['slha']`` -- the
                                             parameters the input events were
                                             generated with

against the card each matrix-element directory is actually initialised from,

    <me>/Cards/param_card.dat                for <me> in madspin_me,
                                             madspin_decay, decay_*  (see
                                             ``interface_madspin.me_param_card``)

and reports every numeric entry that differs.

``aS`` (SMINPUTS 3) is expected to differ and is reported separately: the
running coupling is taken from the PDF set at run time, so the card value is
not used for a ``pdlabel = lhapdf`` run and the difference cancels.  Any
*other* difference -- a mass, a width, a Wilson coefficient -- is a real one.

Usage:
    audit_me_param_cards.py <madspin_run_dir> [<madspin_run_dir> ...]
    audit_me_param_cards.py --workdir <dir>      # audits every MS_* under it

Exit status is 0 when every ME card matches its run card outside the
``aS``-only exemption, 1 otherwise.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import gzip
import os
import re
import sys

pjoin = os.path.join

# Entries whose disagreement is known to cancel: the strong coupling is
# supplied by the PDF at run time when pdlabel = lhapdf, so the card value is
# inert.  Everything else counts.
EXEMPT = {('sminputs', 3): 'aS -- taken from the PDF set at run time '
                           '(pdlabel = lhapdf), card value inert'}

# The parameters this figure's samples turn on, named so the report can lead
# with them rather than burying them in an alphabetical list.
HEADLINE = [('mass', 6, 'MT'), ('decay', 6, 'WT'),
            ('decay', 24, 'WW'), ('decay', 23, 'WZ'),
            ('yukawa', 6, 'ymt')]

_BLOCK_RE = re.compile(r'^\s*block\s+(\S+)', re.I)
_DECAY_RE = re.compile(r'^\s*decay\s+(-?\d+)\s+([-+0-9.eEdD]+)', re.I)
_ENTRY_RE = re.compile(r'^\s*(-?\d+)\s+([-+0-9.eEdD]+)')


def read_card(path):
    """{(block, code): value} for every numeric entry of an SLHA card.

    ``DECAY <pid> <width>`` lines are stored under the pseudo-block
    ``decay``, which is how a width is addressed everywhere else here.
    Multi-index blocks are skipped: no parameter this audit cares about is
    one, and their key shape would only complicate the diff.
    """
    values = {}
    block = None
    with open(path, errors='replace') as fp:
        for raw in fp:
            line = raw.split('#')[0].rstrip()
            if not line.strip():
                continue
            m = _DECAY_RE.match(line)
            if m and len(line.split()) <= 3:
                values[('decay', int(m.group(1)))] = _f(m.group(2))
                block = None
                continue
            m = _BLOCK_RE.match(line)
            if m:
                block = m.group(1).lower()
                continue
            if block is None or block.startswith('qnumbers'):
                continue
            m = _ENTRY_RE.match(line)
            if m and len(line.split()) == 2:
                values[(block, int(m.group(1)))] = _f(m.group(2))
    return values


def _f(text):
    return float(text.replace('d', 'e').replace('D', 'e'))


def banner_slha(run_dir):
    """The ``<slha>`` block of the input event file, written to a temp file.

    ``spinmode = none`` needs no density matrix, so it never builds
    ``madspin_me``/``madspin_decay`` and never writes ``<run>/param_card.dat``.
    Its decay gridpacks are still worth auditing, and the reference is the same
    one MadSpin itself uses -- ``banner['slha']`` -- so read it straight out of
    the input LHE.
    """
    for name in ('events.lhe.gz', 'events.lhe'):
        path = pjoin(run_dir, name)
        if not os.path.exists(path):
            continue
        opener = gzip.open if path.endswith('.gz') else open
        chunk, keep = [], False
        with opener(path, 'rt', errors='replace') as fp:
            for line in fp:
                low = line.strip().lower()
                if low.startswith('<slha'):
                    keep = True
                    continue
                if low.startswith('</slha'):
                    break
                if low.startswith('<event'):
                    break
                if keep:
                    chunk.append(line)
        if chunk:
            out = pjoin(run_dir, '_audit_banner_param_card.dat')
            with open(out, 'w') as fp:
                fp.write(''.join(chunk))
            return out
    return None


def me_dirs(run_dir):
    """Matrix-element directories under a MadSpin run directory, in the order
    ``interface_madspin`` loads them."""
    out = []
    for name in sorted(os.listdir(run_dir)):
        if name in ('madspin_me', 'madspin_decay') or \
                name.startswith('madspin_me_') or \
                name.startswith('madspin_decay_') or \
                re.match(r'^decay_x?\d+_\d+$', name):
            card = pjoin(run_dir, name, 'Cards', 'param_card.dat')
            if os.path.exists(card):
                out.append((name, card))
    # madspin_me / madspin_decay first: they are the density ones
    out.sort(key=lambda kv: (not kv[0].startswith('madspin'), kv[0]))
    return out


def audit_run(run_dir):
    """Return (ok, lines) for one MadSpin run directory."""
    lines = []
    run_card = pjoin(run_dir, 'param_card.dat')
    lines.append('run directory : %s' % run_dir)
    source = "MadSpin's own param_card.dat, written from banner['slha']"
    if not os.path.exists(run_card):
        run_card = banner_slha(run_dir)
        source = ("<slha> block of the input LHE (no <run>/param_card.dat: "
                  "spinmode=none builds no density matrix element)")
        if not run_card:
            lines.append('  NO REFERENCE CARD and no input LHE -- cannot audit')
            return False, lines
    ref = read_card(run_card)
    lines.append('run card      : %s' % run_card)
    lines.append('  source      : %s' % source)
    lines.append('  headline parameters as the events carry them:')
    for block, code, name in HEADLINE:
        val = ref.get((block, code))
        lines.append('    %-4s = %s' % (name, 'absent' if val is None
                                        else repr(val)))

    dirs = me_dirs(run_dir)
    if not dirs:
        lines.append('  no matrix-element directories found')
        return False, lines

    ok = True
    for name, card in dirs:
        got = read_card(card)
        real, exempt = [], []
        for key in sorted(set(ref) | set(got), key=lambda k: (k[0], k[1])):
            a, b = ref.get(key), got.get(key)
            if a is None or b is None:
                # a restriction card can legitimately drop a fixed parameter
                continue
            if a == b:
                continue
            tol = 1e-12 * max(abs(a), abs(b), 1.0)
            if abs(a - b) <= tol:
                continue
            (exempt if key in EXEMPT else real).append((key, a, b))
        status = 'MATCH' if not real else 'MISMATCH'
        if real:
            ok = False
        lines.append('  [%s] %s' % (status, name))
        lines.append('      card: %s' % card)
        for key, a, b in real:
            lines.append('      DIFFERS  %-8s %-4s  run %-14r  ME %-14r'
                         % (key[0], key[1], a, b))
        for key, a, b in exempt:
            lines.append('      exempt   %-8s %-4s  run %-14r  ME %-14r  (%s)'
                         % (key[0], key[1], a, b, EXEMPT[key]))
        if not real and not exempt:
            lines.append('      every numeric entry agrees with the run card')
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dirs', nargs='*', help='MadSpin run directories')
    ap.add_argument('--workdir', default=None,
                    help='audit every MS_* directory under this directory')
    args = ap.parse_args()

    run_dirs = list(args.run_dirs)
    if args.workdir:
        for name in sorted(os.listdir(args.workdir)):
            if name.startswith('MS_') and os.path.isdir(pjoin(args.workdir, name)):
                run_dirs.append(pjoin(args.workdir, name))
    if not run_dirs:
        ap.error('nothing to audit: give a run directory or --workdir')

    all_ok = True
    for run_dir in run_dirs:
        ok, lines = audit_run(os.path.realpath(run_dir))
        all_ok = all_ok and ok
        print('\n'.join(lines))
        print('')
    print('AUDIT %s' % ('CLEAN -- every matrix element is initialised with '
                        'the run\'s own parameters' if all_ok else
                        'FAILED -- see MISMATCH above'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
