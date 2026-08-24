#!/usr/bin/env python3
"""One streaming pass over a showered HepMC2 file, out the far side a small ``.npz``.

The input this was written for is 14 GB of ``IO_GenEvent`` ASCII
(``HepMC::Version 2.06.09``) written by Pythia8 from an MC@NLO ``p p > z z``
sample that MadSpin decayed with ``keep_weight_for_polarization_vector = [0, T]``
on both ``z``.  Nothing here builds an event record: the file is read line by
line, each event is reduced to a fixed row of floats the moment its last
particle has gone past, and the lines are dropped.  Peak memory is one event's
leptons and photons.

What comes out is ``data/weights.npz`` -- every weight and every observable
this study needs, one row per event -- and ``data/meta.json``.  Every figure
and every number downstream runs off those two files; re-reading 14 GB to move
a bin edge is not a thing anyone should have to do.

Four things are established here rather than assumed, because getting any of
them wrong would silently rescale every ratio in the study.

**The ``E`` line layout.**  HepMC2's ``E`` line is
``E <10 fields> N_random <random...> N_weights <weights...>``.  The weight block
is found by *walking forward* through the random-state count, never by counting
back from the end of the line: a file with a non-empty random state would break
the second and not the first, and the shift would land the polarisation weights
on the scale variations without anything looking obviously wrong.

**The ``N`` line names.**  Parsed and compared against the first event's names
on *every* event, not read once and trusted.

**Which weight is the full.**  Not decided here.  ``"0"``, ``"Weight"`` and
``"MUR1.0_MUF1.0"`` are all recorded per event, together with the ``C`` line of
the first and last event and the float64 sums of all 33 weights, so the
question can be settled from the outputs with the evidence in hand.  See
``analyse_pol.py`` and RESULTS.md.

**Which ``z`` decayed to what.**  The present samples' MadSpin card is
``decay z > e+ e-`` and ``decay z > mu+ mu-``, which is exclusive *in intent*.
Whether it is exclusive *in effect* is a different question and it is measured
here rather than assumed, because it decides what the figures mean: which
channel each ``z`` took is read out of the event record, by marking a ``z``'s
end-vertex barcode, extending the marking along the chain of status-44 ``z``
copies, and taking the particles produced at a marked vertex as that ``z``'s
decay products.  (On these three files it comes out at 250 000 of 250 000,
so the card is exclusive in effect too; the earlier samples this script was
written for used ``decay z > light light`` over every fermion but the top and
gave 0.23 %.)  This is truth information and it is used ONLY to categorise
events and to measure the purity of the reconstructed selection -- never to
build the plotted observables, which come from final-state (status 1) particles
alone.

Note on HepMC statuses: this writer maps *final* particles to 1 and everything
else to ``abs(pythia_status)``.  A hard-process lepton that never branches is
therefore status 1, not status 23, and counting status-23 leptons undercounts
the hard leptons by a third.  That is why the vertex chain above exists.

**Gzipped input.**  A path ending in ``.gz`` is streamed through the system
``gzip -dc`` in a child process rather than decompressed to disk: the 4.6 GB
compressed files in this study expand to 14 GB each and there is no reason for
that to exist anywhere but in a pipe.  Decompression then runs on its own core
in parallel with the parsing, which costs about six seconds on top of the plain
pass rather than the fourteen a serial ``gzip`` module read would.

**A different file may carry a different weight set.**  ``"0"`` and ``"Weight"``
are required; ``"MUR1.0_MUF1.0"`` and the four ``ms_pol_*`` are kept when the
``N`` line names them and recorded as absent when it does not.  A MadSpin run
with a different ``spinmode`` is a different reweighting and there is no
guarantee it produced the polarisation weights at all.

Usage::

    extract_hepmc_pol.py INPUT.hepmc[.gz] -o data/weights.npz \
        [--meta meta.json] [--label madspin] [--max-events N]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# The four MadSpin polarisation weights, named as Pythia8 sanitises them (the
# LHE ``initrwgt`` writes ``ms_pol_23:0_23:0``; ':' becomes '.' on the way out).
POL_NAMES = ['ms_pol_23.0_23.0', 'ms_pol_23.0_23.T',
             'ms_pol_23.T_23.0', 'ms_pol_23.T_23.T']
POL_KEYS = ['LL', 'LT', 'TL', 'TT']

# Every weight the .npz keeps a per-event column of.  "0", "Weight" and
# "MUR1.0_MUF1.0" are all candidates for "the full"; all three are kept and the
# choice between them is made downstream, from the evidence.
KEEP_WEIGHTS = ['0', 'Weight', 'MUR1.0_MUF1.0'] + POL_NAMES
# Of those, the two that must be on the N line for the file to be usable at
# all: without them there is no nominal and nothing downstream has a scale.
# Everything else is kept if present and reported as absent if not -- a run
# with a different ``spinmode`` need not have produced the ms_pol_* weights,
# and asserting that it did would turn "this file is different" into a crash.
REQUIRED_WEIGHTS = ['0', 'Weight']
# Float64 for the three nominal candidates (the "0" / "Weight" bit-for-bit
# comparison downstream needs full precision); float32 is plenty for the rest.
WIDE_WEIGHTS = frozenset(('0', 'Weight', 'MUR1.0_MUF1.0'))

# The three samples this study covers.  Written into EVERY meta.json so that
# a reader who opens data/meta.json alone is told that two sibling samples
# exist and where they are, rather than being left with the impression that
# the study is of one file.  ``pol_analysis.EXTRA_SAMPLES`` is the loader's
# copy of the same fact.
SAMPLE_REGISTRY = [
    {'label': 'madspin', 'spinmode': 'madspin (set explicitly in this card)',
     'run': 'run_06_decayed_1', 'npz': 'weights.npz', 'meta': 'meta.json',
     'compressed': True,
     'decay_card': 'decay z > e+ e- / decay z > mu+ mu-'},
    {'label': 'onshell', 'spinmode': 'onshell',
     'run': 'run_08_decayed_1', 'npz': 'weights_onshell.npz',
     'meta': 'meta_onshell.json', 'compressed': True,
     'decay_card': 'decay z > e+ e- / decay z > mu+ mu-'},
    {'label': 'PA', 'spinmode': 'PA',
     'run': 'run_07_decayed_1', 'npz': 'weights_PA.npz',
     'meta': 'meta_PA.json', 'compressed': True,
     'decay_card': 'decay z > e+ e- / decay z > mu+ mu-'},
]

# Final-state PDG ids worth stopping on.  Photons are in because the dressed
# lepton needs them.
WANTED_PDG = frozenset((b'11', b'-11', b'13', b'-13', b'22'))
_Z = b'23'

# The file is read in BINARY mode: 14 GB of ASCII costs a real fraction of the
# pass in UTF-8 decoding alone, and nothing here needs str -- ``float`` and
# ``int`` take bytes and the compared fields are compared against bytes.
_P, _V, _E, _N, _C = ord('P'), ord('V'), ord('E'), ord('N'), ord('C')

DRESS_DR = 0.1          # the run card's own rphreco; see ``dress``
PHOTON_PT_MIN = 0.05    # below this a photon cannot move a dressed mass
STORE_PT_MIN = 3.0      # a lepton softer than this is not stored at all

NAN = float('nan')


def open_stream(path):
    """``(file object, child process or None)`` for a plain or gzipped HepMC.

    A ``.gz`` path is piped through the system ``gzip -dc`` rather than read
    with the ``gzip`` module or unpacked to disk.  Two reasons, in order of
    weight: the 4.6 GB files here expand to 14 GB and no temporary of that size
    should have to exist, and the child does its inflating on its own core
    while this process parses, which is most of the decompression cost hidden.

    The caller must drain the pipe to the end or ``kill`` the child; a reader
    that stops early (``--max-events``) and only closes the pipe leaves ``gzip``
    blocked on a write that will never be read.
    """
    if path.endswith('.gz'):
        proc = subprocess.Popen(['gzip', '-dc', path],
                                stdout=subprocess.PIPE, bufsize=1 << 22)
        return proc.stdout, proc
    return open(path, 'rb', buffering=1 << 22), None


def close_stream(fh, proc):
    if proc is not None:
        proc.kill()             # a no-op once gzip has exited on its own
        fh.close()
        proc.wait()
    else:
        fh.close()


def code_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return '?'


def parse_e_line(line):
    """The weight values off a HepMC2 ``E`` line.

    The layout, from ``HepMC/IO_GenEvent.cc``::

        E  evt_num  n_mpi  scale  alpha_qcd  alpha_qed  signal_process_id
           signal_process_vertex  n_vertices  beam1  beam2
           n_random  <random...>  n_weights  <weights...>

    Ten fields after ``E``, then the random-state block, then the weight block.
    Walked forward through ``n_random`` rather than sliced from the end.
    """
    f = line.split()
    n_rand = int(f[11])
    j = 12 + n_rand
    n_w = int(f[j])
    w = f[j + 1:j + 1 + n_w]
    if len(w) != n_w:
        raise ValueError('E line promises %d weights, carries %d'
                         % (n_w, len(w)))
    return [float(x) for x in w]


def parse_n_line(line):
    """The weight names off an ``N`` line, in order.

    ``N <count> "name" "name" ...``.  Split on the quote character, not on
    whitespace: HepMC2 permits a name containing a space and ``split()`` would
    shred such a name into two columns and shift every column after it.
    """
    _, _, rest = line.partition(b' ')
    cnt, _, rest = rest.strip().partition(b' ')
    names = [x.decode() for x in rest.split(b'"')[1::2]]
    n = int(cnt)
    if len(names) != n:
        raise ValueError('N line promises %d names, carries %d'
                         % (n, len(names)))
    return names


def _phi(p):
    return math.atan2(p[1], p[0])


def _eta(p):
    pt = math.hypot(p[0], p[1])
    if pt == 0.0:
        return math.copysign(1e6, p[2])
    return math.asinh(p[2] / pt)


def dphi(p1, p2):
    """``|Delta phi|`` folded into ``[0, pi]``."""
    d = abs(_phi(p1) - _phi(p2))
    return 2 * math.pi - d if d > math.pi else d


def mass(*ps):
    """Invariant mass of the sum of ``(px, py, pz, E)`` four-vectors."""
    px = py = pz = e = 0.0
    for p in ps:
        px += p[0]
        py += p[1]
        pz += p[2]
        e += p[3]
    m2 = e * e - px * px - py * py - pz * pz
    return math.sqrt(m2) if m2 > 0 else -math.sqrt(-m2)


def select(cands):
    """Highest-pT lepton of one flavour and charge, plus the runner-up's pT.

    The rule this study uses, and why it is the right default here: the sample
    is showered and hadronised, so ``e+`` is not a unique object -- charm and
    bottom decays inside a hadronic ``Z`` make more of them, and the hard one
    can lose momentum to a collinear photon.  Taking the hardest of each
    (flavour, charge) recovers the MadSpin-level lepton whenever the hard one is
    harder than every secondary, which for a ``Z`` decay it essentially always
    is.  The runner-up's pT is returned so that "essentially always" is measured
    rather than asserted: the fraction of events whose second same-flavour
    same-sign lepton clears a few GeV is the fraction where this was a genuine
    choice.
    """
    if not cands:
        return None, 0.0
    if len(cands) > 1:
        cands.sort(key=lambda p: p[0] * p[0] + p[1] * p[1], reverse=True)
        sub = math.hypot(cands[1][0], cands[1][1])
    else:
        sub = 0.0
    return cands[0], sub


def dress(lep, prepared):
    """``lep`` plus every final-state photon within ``DRESS_DR``.

    ``DRESS_DR = 0.1`` is not a free choice: it is ``rphreco`` from this run's
    own run card, the fermion-photon recombination radius the fixed-order
    calculation used to define its leptons.  Dressing in the same cone makes the
    showered lepton the same object the matrix element was integrated for.

    ``prepared`` is the event's photons with their ``(eta, phi)`` already in
    hand -- computed once per event by :func:`observables` and only for the
    events that have a lepton at all, rather than four times over inside here.
    """
    e0, p0 = _eta(lep), _phi(lep)
    px, py, pz, en = lep
    for g, ge, gp in prepared:
        dp = abs(gp - p0)
        if dp > math.pi:
            dp = 2 * math.pi - dp
        if dp > DRESS_DR:
            continue
        de = ge - e0
        if de * de + dp * dp < DRESS_DR * DRESS_DR:
            px += g[0]
            py += g[1]
            pz += g[2]
            en += g[3]
    return (px, py, pz, en)


# Every per-event column, in the order the row is built.
COLS = [
    # observables from the BARE (undressed) leading leptons
    'm_epmup', 'dphi_ee', 'm_ee', 'm_mumu', 'm_4l',
    # the same from the DRESSED leptons
    'm_epmup_dr', 'dphi_ee_dr', 'm_ee_dr', 'm_mumu_dr', 'm_4l_dr',
    # kinematics of the four selected bare leptons, so downstream can cut
    'pt_ep', 'pt_em', 'pt_mup', 'pt_mum',
    'eta_ep', 'eta_em', 'eta_mup', 'eta_mum',
    # dressed pT, for the size of the dressing correction
    'pt_ep_dr', 'pt_em_dr', 'pt_mup_dr', 'pt_mum_dr',
    # the runner-up of each flavour and charge: the ambiguity of the selection
    'sub_pt_ep', 'sub_pt_em', 'sub_pt_mup', 'sub_pt_mum',
]
# Name -> column index, so that :func:`observables` never carries a magic
# number: an off-by-one there would silently write one observable into another
# column and every figure downstream would be of the wrong quantity.
IX = {k: i for i, k in enumerate(COLS)}

ICOLS = ['n_ep', 'n_em', 'n_mup', 'n_mum',    # status-1 leptons above STORE_PT_MIN
         'z1_ch', 'z2_ch']                    # |pdg| of each z's decay products


def observables(ep, em, mup, mum, photons):
    """One event's row of :data:`COLS`, NaN wherever the leptons are not there."""
    row = [NAN] * len(COLS)
    (a, sa), (b, sb) = select(ep), select(em)
    (c, sc), (d, sd) = select(mup), select(mum)
    row[IX['sub_pt_ep']], row[IX['sub_pt_em']] = sa, sb
    row[IX['sub_pt_mup']], row[IX['sub_pt_mum']] = sc, sd
    if not (a or b or c or d):
        return row
    prep = [(g, _eta(g), _phi(g)) for g in photons]
    da = dress(a, prep) if a else None
    db = dress(b, prep) if b else None
    dc = dress(c, prep) if c else None
    dd = dress(d, prep) if d else None
    if a and c:
        row[IX['m_epmup']], row[IX['m_epmup_dr']] = mass(a, c), mass(da, dc)
    if a and b:
        row[IX['dphi_ee']], row[IX['dphi_ee_dr']] = dphi(a, b), dphi(da, db)
        row[IX['m_ee']], row[IX['m_ee_dr']] = mass(a, b), mass(da, db)
    if c and d:
        row[IX['m_mumu']], row[IX['m_mumu_dr']] = mass(c, d), mass(dc, dd)
    if a and b and c and d:
        row[IX['m_4l']] = mass(a, b, c, d)
        row[IX['m_4l_dr']] = mass(da, db, dc, dd)
    for nm, p, q in (('ep', a, da), ('em', b, db),
                     ('mup', c, dc), ('mum', d, dd)):
        if p:
            row[IX['pt_' + nm]] = math.hypot(p[0], p[1])
            row[IX['eta_' + nm]] = _eta(p)
            row[IX['pt_%s_dr' % nm]] = math.hypot(q[0], q[1])
    return row


def run(path, out, max_events=None, report_every=20000, meta_name=None,
        label=None):
    t0 = time.time()
    size = os.path.getsize(path)
    gz = path.endswith('.gz')
    names = None
    kept = None
    keep_idx = None
    c_first = c_last = None
    n_lines = n_bytes = n_seen = 0
    wsum = wsum_abs = None

    rows = []           # float columns
    irows = []          # small integer columns
    wrows = []          # KEEP_WEIGHTS

    wrow = None
    ep, em, mup, mum, photons = [], [], [], [], []
    zvtx = set()
    zdau = []           # (vertex, |pdg|) of every particle born at a z vertex
    cur_z = False
    have = False
    pt2min = STORE_PT_MIN ** 2
    gpt2min = PHOTON_PT_MIN ** 2

    def finish():
        rows.append(observables(ep, em, mup, mum, photons))
        ch = {}
        for v, q in zdau:
            ch.setdefault(v, []).append(q)
        # A z vertex whose only child is another z is a copy, not a decay; the
        # channels are the vertices with two children.
        chans = sorted(min(v) for v in ch.values() if len(v) == 2)
        irows.append([min(len(ep), 127), min(len(em), 127),
                      min(len(mup), 127), min(len(mum), 127),
                      chans[0] if len(chans) > 0 else -1,
                      chans[1] if len(chans) > 1 else -1])
        wrows.append(wrow)

    fh, proc = open_stream(path)
    for line in fh:
        n_lines += 1
        n_bytes += len(line)
        c = line[0]
        if c == _P:
            # maxsplit=9 stops the split once the status field is in hand; the
            # colour-flow tail of the line is never touched.  Fields are
            # P barcode pdg px py pz E m status pol_theta pol_phi end_vtx ...
            f = line.split(b' ', 9)
            q = f[2]
            if q in WANTED_PDG:
                if f[8] == b'1':
                    px, py = float(f[3]), float(f[4])
                    if q == b'22':
                        if px * px + py * py > gpt2min:
                            photons.append((px, py, float(f[5]), float(f[6])))
                    elif px * px + py * py > pt2min:
                        p = (px, py, float(f[5]), float(f[6]))
                        if q == b'-11':
                            ep.append(p)
                        elif q == b'11':
                            em.append(p)
                        elif q == b'-13':
                            mup.append(p)
                        else:
                            mum.append(p)
                if cur_z:
                    zdau.append((cur_vtx, abs(int(q))))
            elif q == _Z:
                # full split only for the handful of z lines per event
                ev = line.split()[11]
                if ev != b'0':
                    zvtx.add(ev)
                if cur_z:
                    zdau.append((cur_vtx, 23))
            elif cur_z:
                zdau.append((cur_vtx, abs(int(q))))
        elif c == _V:
            cur_vtx = line.split(b' ', 2)[1]
            cur_z = cur_vtx in zvtx
        elif c == _E:
            if have:
                finish()
                ep, em, mup, mum, photons = [], [], [], [], []
                zvtx, zdau, cur_z = set(), [], False
                if max_events and n_seen >= max_events:
                    have = False
                    break
            pending = parse_e_line(line)
            have = True
            n_seen += 1
            if report_every and n_seen % report_every == 0:
                el = time.time() - t0
                if gz:
                    # ``size`` is the COMPRESSED size, so it is not the target
                    # ``n_bytes`` is counting towards and no honest ETA can be
                    # formed from the two.  Report what is known instead of a
                    # number that would be wrong by the compression ratio.
                    sys.stderr.write(
                        '  %8d events  %6.2f GB inflated  %6.0f s\n'
                        % (n_seen, n_bytes / 1e9, el))
                else:
                    sys.stderr.write(
                        '  %8d events  %6.2f / %.2f GB  %6.0f s  eta %6.0f s\n'
                        % (n_seen, n_bytes / 1e9, size / 1e9, el,
                           el * (size - n_bytes) / n_bytes))
                sys.stderr.flush()
        elif c == _N:
            nm = parse_n_line(line)
            if names is None:
                names = nm
                missing = [k for k in REQUIRED_WEIGHTS if k not in names]
                if missing:
                    raise SystemExit('weight name(s) not on the N line: %s'
                                     % missing)
                # The rest are kept if the file has them.  A different
                # ``spinmode`` is a different MadSpin run and need not have
                # produced the ms_pol_* weights at all; which ones it did
                # produce goes into meta.json and is reported, not assumed.
                kept = [k for k in KEEP_WEIGHTS if k in names]
                absent = [k for k in KEEP_WEIGHTS if k not in names]
                if absent:
                    sys.stderr.write('  not on the N line, not kept: %s\n'
                                     % ', '.join(absent))
                keep_idx = [names.index(k) for k in kept]
            elif nm != names:
                raise SystemExit('weight names changed at event %d' % n_seen)
            if len(pending) != len(names):
                raise SystemExit('event %d: %d weights for %d names'
                                 % (n_seen, len(pending), len(names)))
            a = np.asarray(pending)
            if wsum is None:
                wsum, wsum_abs = a.copy(), np.abs(a)
            else:
                wsum += a
                wsum_abs += np.abs(a)
            wrow = [pending[i] for i in keep_idx]
        elif c == _C:
            f = line.split()
            v = (float(f[1]), float(f[2]))
            if c_first is None:
                c_first = v
            c_last = v
    if have:
        finish()
    close_stream(fh, proc)
    dt = time.time() - t0

    W = np.asarray(wrows, dtype=np.float64)
    R = np.asarray(rows, dtype=np.float32)
    I = np.asarray(irows, dtype=np.int16)
    payload = {}
    for i, k in enumerate(kept):
        payload['w_' + k] = W[:, i] if k in WIDE_WEIGHTS \
            else W[:, i].astype(np.float32)
    for i, k in enumerate(COLS):
        payload[k] = R[:, i]
    for i, k in enumerate(ICOLS):
        payload[k] = I[:, i]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **payload)

    meta = {
        'label': label or 'madspin',
        'sample_registry': SAMPLE_REGISTRY,
        'input_path': os.path.abspath(path),
        'input_bytes': size,
        'input_gb': round(size / 1e9, 3),
        'gzipped': gz,
        # For a .gz, ``input_gb`` is what is on disk and this is what went
        # through the parser; for a plain file the two are the same number.
        'inflated_bytes': n_bytes,
        'inflated_gb': round(n_bytes / 1e9, 3),
        'hepmc_flavour': 'HepMC2 IO_GenEvent (HepMC::Version 2.06.09)',
        'code_sha': code_sha(),
        'extractor': os.path.basename(__file__),
        'pass_seconds': round(dt, 1),
        'pass_minutes': round(dt / 60.0, 2),
        'lines_read': n_lines,
        'n_events': n_seen,
        'weight_names': names,
        'weight_names_kept': kept,
        'weight_names_absent': [k for k in KEEP_WEIGHTS if k not in kept],
        'has_pol_weights': all(k in kept for k in POL_NAMES),
        'pol_map': dict(zip(POL_KEYS, POL_NAMES)),
        'sum_all_weights': [float(x) for x in wsum],
        'sum_abs_all_weights': [float(x) for x in wsum_abs],
        'C_line_first_event': c_first,
        'C_line_last_event': c_last,
        'columns_float': COLS,
        'columns_int': ICOLS,
        'store_pt_min_gev': STORE_PT_MIN,
        'dress_dr': DRESS_DR,
        'photon_pt_min_gev': PHOTON_PT_MIN,
        'lepton_selection':
            'highest-pT final-state (status 1) lepton per flavour and charge, '
            'among those with pT > %g GeV; dressed variants add every '
            'final-state photon within Delta R < %g (the run card rphreco). '
            'Both bare and dressed observables are stored; the figures use '
            'the dressed ones and RESULTS.md quotes the difference.'
            % (STORE_PT_MIN, DRESS_DR),
        'z_channel_note':
            'z1_ch / z2_ch are the |pdg| of each z\'s decay products, read off '
            'the event record by marking each z\'s end vertex and following '
            'the chain of status-44 copies.  -1 means the channel could not be '
            'read.  Truth information: used to categorise events and to '
            'measure the purity of the reconstructed selection, never to build '
            'a plotted observable.',
    }
    mpath = os.path.join(os.path.dirname(os.path.abspath(out)),
                         meta_name or 'meta.json')
    with open(mpath, 'w') as fp:
        json.dump(meta, fp, indent=2)

    print('read   %.3f GB on disk%s in %.1f s (%.2f min), %d lines'
          % (size / 1e9,
             ' (%.3f GB inflated)' % (n_bytes / 1e9) if gz else '',
             dt, dt / 60, n_lines))
    print('events %d' % n_seen)
    print('wrote  %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))
    print('       %s' % mpath)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--max-events', type=int, default=None)
    ap.add_argument('--report-every', type=int, default=20000)
    ap.add_argument('--meta', default='meta.json',
                    help='name of the meta file written next to --out; give '
                         'each sample its own so a second pass does not '
                         'overwrite the first one\'s')
    ap.add_argument('--label', default=None,
                    help="what this sample is, e.g. the MadSpin spinmode")
    a = ap.parse_args()
    run(a.input, a.out, a.max_events, a.report_every, a.meta, a.label)


if __name__ == '__main__':
    main()
