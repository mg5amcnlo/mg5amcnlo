#!/usr/bin/env python3
################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Tools for the LHE event files written by MG5_aMC@NLO.

Python version of the Fortran utilities of this folder:

  check          check_events.f    consistency checks and statistics
  combine-jobs   combine_jobs.f    combine events_<i>.lhe of split generation jobs
  compare        compare_events.f  compare two files event by event
  merge          merge_events.f    concatenate event files
  plot           plot_events.f     histograms (HwU format) from the events
  select         select_events.f   keep S, H, w>0, w<0 events or a range
  split          split_events.f    split a file into several files
  uncertainties  uncertainties.f   scale and PDF uncertainties of the total
                                   rate (event files) or of histograms (HwU)

Usage: python lhe_utilities.py <command> -h

Gzipped or plain files are read and written by lhe_parser.EventFile of
MG5_aMC (bin/internal/lhe_parser.py in a process directory). The events are
copied verbatim whenever possible; the event records are otherwise read in the
HEPEUP form, keeping the order of the particles of the file.

The weights follow event_norm of the run_card (header of the file): with
'average' or 'bias' they average to the cross section, with 'sum' they sum
to it.
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from array import array

logger = logging.getLogger('lhe_utilities')

HERE = os.path.dirname(os.path.realpath(__file__))

# Tolerances of the Fortran utilities
MOMENTUM_TINY = 5e-3     # momentum conservation and mass shell (relative)
MASS_TINY = 1e-4         # masses compared with the reference masses
UNWEIGHTED_TINY = 1e-5   # |w|/|w_first| - 1 for unweighted events
MAX_PRINTED = 20         # warnings printed on the screen per kind


class LHEUtilitiesError(Exception):
    """Error of the utilities, reported without traceback."""


################################################################################
# MG5_aMC modules
################################################################################
_LHE_PARSER = None


def _mg5_roots():
    roots = [os.path.join(HERE, os.pardir, os.pardir, os.pardir)]
    for var in ('MG5AMC_PATH', 'MG5_PATH'):
        if os.environ.get(var):
            roots.append(os.environ[var])
    return [os.path.realpath(r) for r in roots]


def mg5_module_path(name):
    """Path of madgraph/various/<name>.py (or bin/internal/<name>.py in a
    process directory)."""
    internal = os.path.realpath(os.path.join(HERE, os.pardir, 'bin',
                                             'internal', name + '.py'))
    if os.path.isfile(internal):
        return internal
    for root in _mg5_roots():
        path = os.path.join(root, 'madgraph', 'various', name + '.py')
        if os.path.isfile(path):
            return path
    return None


def lhe_parser():
    """The lhe_parser module of MG5_aMC: from bin/internal in a process
    directory, from the MG5_aMC tree otherwise."""
    global _LHE_PARSER
    if _LHE_PARSER is not None:
        return _LHE_PARSER
    internal = os.path.realpath(os.path.join(HERE, os.pardir, 'bin',
                                             'internal'))
    if os.path.isfile(os.path.join(internal, 'lhe_parser.py')):
        sys.path.insert(0, os.path.dirname(internal))
        import internal.lhe_parser as module
    else:
        for root in _mg5_roots():
            if os.path.isfile(os.path.join(root, 'madgraph', 'various',
                                           'lhe_parser.py')):
                sys.path.insert(0, root)
                import madgraph.various.lhe_parser as module
                break
        else:
            raise LHEUtilitiesError(
                'lhe_parser.py of MG5_aMC not found: run this script from '
                'Template/NLO/Utilities, from the Utilities folder of a '
                'process directory, or set MG5AMC_PATH.')
    _LHE_PARSER = module
    return module


################################################################################
# Files
################################################################################
def lhe_stem(path):
    """Path without the .gz and .lhe extensions."""
    stem = path
    if stem.endswith('.gz'):
        stem = stem[:-3]
    if stem.endswith('.lhe'):
        stem = stem[:-4]
    return stem


def lhe_suffix(path, gzip_output=None):
    gz = path.endswith('.gz') if gzip_output is None else gzip_output
    return '.lhe.gz' if gz else '.lhe'


def open_events(path):
    if not os.path.isfile(path):
        raise LHEUtilitiesError('No such event file: %s' % path)
    ef = lhe_parser().EventFile(path)
    if not ef.banner:
        raise LHEUtilitiesError('%s has no LHE header/init block' % path)
    return ef


def iter_raw(ef):
    """Text of the events (from <event> to </event>), unchanged."""
    if hasattr(ef, '_iter_event_text'):
        for text in ef._iter_event_text():
            yield text
    else:
        old = ef.parsing
        ef.parsing = False
        try:
            for lines in ef:
                yield ''.join(lines) if isinstance(lines, list) else lines
        finally:
            ef.parsing = old


def count_events(path):
    ef = open_events(path)
    try:
        return sum(1 for _ in iter_raw(ef))
    finally:
        ef.close()


class LHEWriter(object):
    """Write an LHE file: header (with its init block), events, end tag."""

    def __init__(self, path, header_text):
        self.path = path
        self.file = lhe_parser().EventFile(path, 'w')
        if not header_text.endswith('\n'):
            header_text += '\n'
        self.file.write(header_text)
        self.nevents = 0

    def write(self, text):
        if not text.endswith('\n'):
            text += '\n'
        self.file.write(text)
        self.nevents += 1

    def close(self):
        self.file.write('</LesHouchesEvents>\n')
        self.file.close()


################################################################################
# Header
################################################################################
_FLOAT = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?'


def to_float(text):
    try:
        return float(text)
    except ValueError:
        return float(text.replace('D', 'E').replace('d', 'e'))


class WeightInfo(object):
    """A weight of <initrwgt>, or a column of a HwU file."""

    def __init__(self, wid, label, group=''):
        self.id = wid
        self.label = ' '.join(label.split())
        self.group = group
        info = parse_weight_label(self.label)
        self.tag = info.get('tag', 0)
        self.dyn = info.get('dyn')
        self.muR = info.get('muR')
        self.muF = info.get('muF')
        self.lhaid = info.get('PDF')
        self.pdfset = info.get('pdfset')
        if self.muR is not None and self.muF is not None:
            self.kind = 'scale'
        elif self.lhaid is not None:
            self.kind = 'pdf'
        else:
            self.kind = 'other'

    def hwu_label(self):
        if self.kind == 'scale':
            dyn = -1 if self.dyn is None else self.dyn
            text = 'dyn=%i muR=%6.3f muF=%6.3f' % (dyn, self.muR, self.muF)
            if self.tag:
                text = 'tag=%i %s' % (self.tag, text)
            return text
        if self.kind == 'pdf':
            if self.pdfset:
                return 'PDF=%i %s' % (self.lhaid, self.pdfset)
            return 'PDF=%i' % self.lhaid
        return ('rwgt %s %s' % (self.id, self.label)).replace('&', 'and').strip()


def parse_weight_label(label):
    """Key-value information of a weight label, as written by MG5_aMC in
    <initrwgt> ('tag= 0 dyn= -1 muR=0.1E+01 muF=0.2E+01', 'PDF= 260001
    NNPDF31_nlo_as_0118') or in the HwU files ('dyn=-1 muR= 1.000 muF= 2.000',
    'muR=1.0 muF=2.0', 'PDF=260001 NNPDF31_nlo_as_0118')."""
    info = {}
    for key in ('tag', 'dyn'):
        m = re.search(r'\b%s\s*=\s*([-+]?\d+)' % key, label, re.I)
        if m:
            info[key] = int(m.group(1))
    for key in ('muR', 'muF'):
        m = re.search(r'\b%s\s*=\s*(%s)' % (key, _FLOAT), label, re.I)
        if m:
            info[key] = to_float(m.group(1))
    m = re.search(r'\bPDF\s*=\s*(\d+)(?:\s+([^\s&]+))?', label)
    if m:
        info['PDF'] = int(m.group(1))
        if m.group(2) and '=' not in m.group(2):
            info['pdfset'] = m.group(2)
    return info


class LHEHeader(object):
    """Information of the header of an LHE file (up to the init block)."""

    def __init__(self, text):
        self.text = text

    def _run_card_value(self, name):
        m = re.search(r'^\s*(\S+)\s*=\s*%s\b' % name, self.text, re.M)
        return m.group(1) if m else None

    @property
    def nevents(self):
        value = self._run_card_value('nevents')
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @property
    def event_norm(self):
        value = (self._run_card_value('event_norm') or 'average').lower()
        if value.startswith('ave'):
            return 'average'
        if value.startswith('bia'):
            return 'bias'
        return value

    @property
    def averaged(self):
        return self.event_norm in ('average', 'bias')

    @property
    def parton_shower(self):
        value = self._run_card_value('parton_shower')
        return value.upper() if value else ''

    @property
    def mcatnlo_delta(self):
        value = self._run_card_value('mcatnlo_delta')
        return bool(value) and value.strip('.').lower().startswith('t')

    def with_nevents(self, nevents):
        """Header text with the number of events of the run_card replaced
        (as copy_header in handling_lhe_events.f)."""
        pattern = re.compile(r'^(\s*)([-+]?\d+)(\s*=\s*nevents\b)', re.M)
        m = pattern.search(self.text)
        if not m:
            return self.text
        width = len(m.group(2))
        value = ('%%%dd' % width) % nevents
        return self.text[:m.start(2)] + value + self.text[m.end(2):]

    def mc_masses(self):
        """Masses of <MonteCarloMasses>, completed as in check_events.f."""
        masses = {}
        m = re.search(r'<MonteCarloMasses>(.*?)</MonteCarloMasses>',
                      self.text, re.S | re.I)
        if not m:
            return masses
        for line in m.group(1).splitlines():
            fields = line.split()
            if len(fields) >= 2:
                try:
                    masses[int(fields[0])] = to_float(fields[1])
                except ValueError:
                    pass
        for pid in (12, 14, 16):
            if pid not in masses and -pid not in masses:
                masses[pid] = 0.
        for pid in list(masses):
            masses.setdefault(-pid, masses[pid])
        return masses

    def init_block(self):
        """(beam line fields, list of process lines fields) of <init>."""
        m = re.search(r'<init>(.*?)</init>', self.text, re.S | re.I)
        if not m:
            return None, []
        lines = [l.split() for l in m.group(1).splitlines()
                 if l.strip() and not l.strip().startswith(('#', '<'))]
        if not lines:
            return None, []
        return lines[0], lines[1:]

    def beam_energies(self):
        beams, _ = self.init_block()
        try:
            return to_float(beams[2]), to_float(beams[3])
        except (TypeError, IndexError, ValueError):
            return None

    def weights(self):
        """Weights of <initrwgt>, in the order of the file."""
        out = []
        m = re.search(r'<initrwgt>(.*?)</initrwgt>', self.text, re.S | re.I)
        if not m:
            return out
        body = m.group(1)
        wpattern = re.compile(
            r'<weight\s+id\s*=\s*[\'"]([^\'"]+)[\'"][^>]*>(.*?)</weight>',
            re.S | re.I)
        gpattern = re.compile(r'<weightgroup([^>]*)>(.*?)</weightgroup>',
                              re.S | re.I)
        pos = 0
        for g in gpattern.finditer(body):
            for w in wpattern.finditer(body[pos:g.start()]):
                out.append(WeightInfo(w.group(1), w.group(2)))
            name = re.search(r'name\s*=\s*[\'"]([^\'"]*)[\'"]', g.group(1))
            name = ' '.join(name.group(1).split()) if name else ''
            for w in wpattern.finditer(g.group(2)):
                out.append(WeightInfo(w.group(1), w.group(2), name))
            pos = g.end()
        for w in wpattern.finditer(body[pos:]):
            out.append(WeightInfo(w.group(1), w.group(2)))
        return out


################################################################################
# Events
################################################################################
_RWGT = re.compile(r'<wgt\s+id\s*=\s*[\'"]([^\'"]+)[\'"]\s*>\s*([^<]*?)\s*</wgt>',
                   re.I)
_SCALES = re.compile(r'<scales\b([^>]*)>', re.I)
_SCALUP = re.compile(r'scalup_(\d+)_(\d+)\s*=\s*[\'"]?\s*(%s)' % _FLOAT, re.I)

AMCATNLO_FIELDS = ('iSorH', 'ifks', 'jfks', 'fksfather', 'ipartner',
                   'scale1', 'scale2', 'jwgtinfo', 'mexternal', 'iwgtnumpartn',
                   'wgtcentral', 'wgtmumin', 'wgtmumax', 'wgtpdfmin',
                   'wgtpdfmax')


class LHEEvent(object):
    """An event in the HEPEUP form (Fortran indices minus one)."""

    __slots__ = ('text', 'nup', 'idprup', 'wgt', 'scale', 'aqed', 'aqcd',
                 'idup', 'istup', 'mothup', 'icolup', 'pup', 'vtimup',
                 'spinup', 'attributes', 'comments', 'extra', '_rwgt')

    def __init__(self, text):
        self.text = text
        self.attributes = {}
        self.comments = []
        self.extra = []
        self._rwgt = None
        lines = text.splitlines()
        i = 0
        while i < len(lines) and '<event' not in lines[i]:
            i += 1
        if i == len(lines):
            raise LHEUtilitiesError('Malformed event:\n%s' % text)
        for key, v1, v2 in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')',
                                      lines[i]):
            self.attributes[key] = v1 or v2
        i += 1
        while i < len(lines) and (not lines[i].strip() or
                                  lines[i].lstrip().startswith('#')):
            if lines[i].strip():
                self.comments.append(lines[i].strip())
            i += 1
        try:
            fields = lines[i].split()
            self.nup = int(fields[0])
            self.idprup = int(fields[1])
            self.wgt, self.scale, self.aqed, self.aqcd = \
                [to_float(x) for x in fields[2:6]]
        except (IndexError, ValueError):
            raise LHEUtilitiesError('Malformed event header line in:\n%s'
                                    % text)
        self.idup, self.istup, self.mothup, self.icolup = [], [], [], []
        self.pup, self.vtimup, self.spinup = [], [], []
        for line in lines[i + 1:i + 1 + self.nup]:
            f = line.split()
            if len(f) < 13:
                raise LHEUtilitiesError('Malformed particle line: %s' % line)
            self.idup.append(int(f[0]))
            self.istup.append(int(f[1]))
            self.mothup.append((int(f[2]), int(f[3])))
            self.icolup.append((int(f[4]), int(f[5])))
            self.pup.append([to_float(x) for x in f[6:11]])
            self.vtimup.append(to_float(f[11]))
            self.spinup.append(to_float(f[12]))
        if len(self.idup) != self.nup:
            raise LHEUtilitiesError('Event with fewer particles than NUP:\n%s'
                                    % text)
        for line in lines[i + 1 + self.nup:]:
            s = line.strip()
            if not s or s.startswith('</event'):
                continue
            if s.startswith('#'):
                self.comments.append(s)
            else:
                self.extra.append(line)

    # --- MG5_aMC information ---------------------------------------------
    def amcatnlo_info(self):
        """Fields of the '#aMCatNLO' line, or None."""
        for c in self.comments:
            if c.startswith('#aMCatNLO'):
                values = c.split()[1:]
                info = {}
                for key, value in zip(AMCATNLO_FIELDS, values):
                    try:
                        info[key] = int(value)
                    except ValueError:
                        try:
                            info[key] = to_float(value)
                        except ValueError:
                            info[key] = value
                return info
        return None

    def rwgt(self):
        """{id: weight} of <rwgt>, ordered as in the event."""
        if self._rwgt is None:
            self._rwgt = collections.OrderedDict()
            extra = '\n'.join(self.extra)
            if '<rwgt' in extra:
                for wid, value in _RWGT.findall(extra):
                    self._rwgt[wid] = to_float(value)
        return self._rwgt

    def dipole_scales(self):
        """{(i, j): scale} of the scalup_i_j entries of <scales> (MC@NLO-Delta),
        with Fortran (1-based) indices; None without such entries."""
        extra = '\n'.join(self.extra)
        m = _SCALES.search(extra)
        if not m:
            return None
        scales = dict(((int(i), int(j)), to_float(v))
                      for i, j, v in _SCALUP.findall(m.group(1)))
        return scales or None

    # --- particles ------------------------------------------------------
    def incoming(self):
        return [i for i in range(self.nup) if self.istup[i] == -1]

    def outgoing(self):
        return [i for i in range(self.nup) if self.istup[i] == 1]

    def external(self):
        """Particles with |ISTUP|=1, in the order of the event."""
        return [i for i in range(self.nup) if abs(self.istup[i]) == 1]

    def particles(self, status=1):
        return [Particle(self, i) for i in range(self.nup)
                if status is None or self.istup[i] == status]


class Particle(object):
    """Read-only view of a particle of an LHEEvent, for the analyses."""

    __slots__ = ('index', 'pdg', 'status', 'px', 'py', 'pz', 'E', 'm',
                 'mothers', 'colours')

    def __init__(self, event, i):
        self.index = i
        self.pdg = event.idup[i]
        self.status = event.istup[i]
        self.px, self.py, self.pz, self.E, self.m = event.pup[i]
        self.mothers = event.mothup[i]
        self.colours = event.icolup[i]

    @property
    def p(self):
        return (self.E, self.px, self.py, self.pz)

    @property
    def pt(self):
        return math.hypot(self.px, self.py)

    @property
    def mass(self):
        return invariant_mass([self])

    @property
    def rapidity(self):
        return rapidity(self.p)

    @property
    def eta(self):
        return pseudorapidity(self.p)

    @property
    def phi(self):
        return math.atan2(self.py, self.px)


def four_momentum(particles):
    """Sum of the four-momenta (E, px, py, pz) of particles."""
    E = px = py = pz = 0.
    for p in particles:
        E += p.E
        px += p.px
        py += p.py
        pz += p.pz
    return (E, px, py, pz)


def invariant_mass(particles):
    E, px, py, pz = four_momentum(particles)
    m2 = E * E - px * px - py * py - pz * pz
    return math.copysign(math.sqrt(abs(m2)), m2)


def transverse_momentum(p):
    return math.hypot(p[1], p[2])


def rapidity(p):
    E, pz = p[0], p[3]
    if E - abs(pz) <= 0.:
        return math.copysign(99., pz)
    return 0.5 * math.log((E + pz) / (E - pz))


def pseudorapidity(p):
    pabs = math.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)
    if pabs - abs(p[3]) <= 0.:
        return math.copysign(99., p[3])
    return 0.5 * math.log((pabs + p[3]) / (pabs - p[3]))


def momentum_messages(event, indices=None, check_mass=True):
    """Checks of momentum conservation and of the mass shell
    (phspncheck_nocms2 of the Fortran utilities): list of messages."""
    if indices is None:
        indices = event.external()
    messages = []
    incoming = [i for i in indices if event.istup[i] == -1]
    outgoing = [i for i in indices if event.istup[i] != -1]
    if incoming:
        for k, name in zip((3, 0, 1, 2), ('E', 'px', 'py', 'pz')):
            xsum = sum(event.pup[i][k] for i in outgoing) - \
                sum(event.pup[i][k] for i in incoming)
            xsuma = sum(abs(event.pup[i][k]) for i in indices)
            xrat = abs(xsum) if xsuma < 1. else abs(xsum) / xsuma
            if xrat > MOMENTUM_TINY:
                messages.append('momentum is not conserved (%s: sum %.6e, '
                                'relative %.3e)' % (name, xsum, xrat))
                break
    if check_mass:
        for i in indices:
            px, py, pz, E, m = event.pup[i]
            m2 = E * E - px * px - py * py - pz * pz
            xm = math.copysign(math.sqrt(abs(m2)), m2)
            den = E if E >= 1. else 1.
            if abs(xm - m) / den > MOMENTUM_TINY and abs(xm - m) > MOMENTUM_TINY:
                messages.append('mass shell violation for particle %d (PDG %d):'
                                ' mass %.6e, computed %.6e'
                                % (i + 1, event.idup[i], m, xm))
    return messages


class Messages(object):
    """Warnings printed up to MAX_PRINTED times per kind, then counted."""

    def __init__(self):
        self.counts = collections.Counter()

    def __call__(self, kind, text):
        self.counts[kind] += 1
        if self.counts[kind] <= MAX_PRINTED:
            logger.warning(text)
        elif self.counts[kind] == MAX_PRINTED + 1:
            logger.warning('further warnings of the kind "%s" are not printed',
                           kind)

    def summary(self):
        for kind, n in sorted(self.counts.items()):
            logger.warning('%d warning(s): %s', n, kind)


################################################################################
# Weights
################################################################################
def rescale_event(text, factor):
    """Event text with the weight of the event and the <rwgt> weights
    multiplied by factor (other lines unchanged)."""
    lines = text.splitlines(True)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('<event'):
            continue
        fields = line.split(None, 3)
        indent = line[:len(line) - len(line.lstrip())]
        eol = line[len(line.rstrip('\r\n')):]
        rest = fields[3].rstrip('\r\n') if len(fields) > 3 else ''
        lines[i] = '%s%s %s %+.10e %s%s' % (indent, fields[0], fields[1],
                                            to_float(fields[2]) * factor,
                                            rest, eol)
        break
    text = ''.join(lines)
    if '<wgt' in text:
        text = _RWGT.sub(lambda m: "<wgt id='%s'> %+.7e </wgt>"
                         % (m.group(1), to_float(m.group(2)) * factor), text)
    return text


################################################################################
# Commands: split, merge, combine-jobs, select
################################################################################
def cmd_split(args):
    ntot = count_events(args.file)
    if ntot == 0:
        raise LHEUtilitiesError('No events in %s' % args.file)
    if args.events_per_file:
        size = args.events_per_file
        sizes = [size] * (ntot // size) + ([ntot % size] if ntot % size else [])
    else:
        nparts = args.nparts
        if nparts is None or nparts < 1:
            raise LHEUtilitiesError('Give the number of files (>0) or '
                                    '--events-per-file')
        if nparts > ntot:
            raise LHEUtilitiesError('Cannot split %d events in %d files'
                                    % (ntot, nparts))
        base, extra = divmod(ntot, nparts)
        sizes = [base + (1 if i < extra else 0) for i in range(nparts)]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.file))
    stem = os.path.join(out_dir, os.path.basename(lhe_stem(args.file)))
    suffix = lhe_suffix(args.file, args.gzip)
    digits = max(3, len(str(len(sizes))))
    ef = open_events(args.file)
    header = LHEHeader(ef.banner)
    events = iter_raw(ef)
    outputs = []
    try:
        for n, size in enumerate(sizes, 1):
            path = '%s_%0*d%s' % (stem, digits, n, suffix)
            writer = LHEWriter(path, header.with_nevents(size))
            for _ in range(size):
                writer.write(next(events))
            writer.close()
            outputs.append((path, size))
    finally:
        ef.close()
    print('%d events written in %d files:' % (ntot, len(outputs)))
    for path, size in outputs:
        print('  %s  (%d events)' % (path, size))
    return 0


def _merge(files, output, rescale=False, force=False, nevts=None):
    """Concatenate event files. With rescale, the weights of the file i are
    multiplied by n_i/sum(n), with n_i the number of its events, or nevts[i]."""
    headers = []
    for path in files:
        ef = open_events(path)
        headers.append(LHEHeader(ef.banner))
        ef.close()
    showers = set(h.parton_shower for h in headers)
    if len(showers) > 1:
        message = ('The files are not relevant to the same parton shower: %s'
                   % ', '.join('%s (%s)' % (f, h.parton_shower or '?')
                               for f, h in zip(files, headers)))
        if not force:
            raise LHEUtilitiesError(message + ' (use --force to merge anyway)')
        logger.warning(message)
    norms = set(h.event_norm for h in headers)
    if len(norms) > 1:
        logger.warning('The files have different event_norm: %s',
                       ', '.join(sorted(norms)))
    counts = [count_events(path) for path in files]
    if nevts is None:
        nevts = counts
    total = sum(counts)
    tot_nevts = float(sum(nevts))
    writer = LHEWriter(output, headers[0].with_nevents(total))
    try:
        for path, nevt in zip(files, nevts):
            factor = nevt / tot_nevts if rescale else 1.
            ef = open_events(path)
            try:
                for text in iter_raw(ef):
                    writer.write(rescale_event(text, factor) if rescale
                                 else text)
            finally:
                ef.close()
    finally:
        writer.close()
    return counts


def cmd_merge(args):
    if args.prefix:
        files = []
        for i in range(1, args.num + 1):
            candidates = ['%s.%03d' % (args.prefix, i), '%s.%03d.gz'
                          % (args.prefix, i)]
            for c in candidates:
                if os.path.isfile(c):
                    files.append(c)
                    break
            else:
                raise LHEUtilitiesError('Missing file %s' % candidates[0])
        output = args.output or '%s.complete' % args.prefix
    else:
        files = args.files
        if len(files) < 1:
            raise LHEUtilitiesError('No input files')
        output = args.output or '%s_merged%s' % (lhe_stem(files[0]),
                                                  lhe_suffix(files[0]))
    if os.path.abspath(output) in [os.path.abspath(f) for f in files]:
        raise LHEUtilitiesError('The output file is one of the inputs')
    counts = _merge(files, output, rescale=args.rescale, force=args.force)
    for path, n in zip(files, counts):
        print('  %-50s %10d events' % (path, n))
    print('%d events written in %s%s' % (sum(counts), output,
          ' (weights rescaled by n_i/n_tot)' if args.rescale else ''))
    return 0


def cmd_combine_jobs(args):
    directory = args.directory
    jobs = []
    i = 1
    while True:
        nfile = os.path.join(directory, 'nevts__%d' % i)
        if not os.path.isfile(nfile):
            break
        with open(nfile) as f:
            nevts = int(f.read().split()[0])
        events = None
        for name in ('events_%d.lhe' % i, 'events_%d.lhe.gz' % i):
            if os.path.isfile(os.path.join(directory, name)):
                events = os.path.join(directory, name)
        if events is None:
            raise LHEUtilitiesError('Missing events_%d.lhe in %s'
                                    % (i, directory))
        jobs.append((events, nevts))
        i += 1
    if not jobs:
        raise LHEUtilitiesError('No nevts__<i> files in %s' % directory)
    output = args.output or os.path.join(directory, 'events.lhe')
    files = [j[0] for j in jobs]
    nevts = [j[1] for j in jobs]
    counts = _merge(files, output, rescale=True, force=args.force,
                    nevts=nevts)
    total = float(sum(nevts))
    for (path, n), count in zip(jobs, counts):
        print('  %-40s %8d events (nevts %d, weight factor %.6f)'
              % (path, count, n, n / total))
        if count != n:
            logger.warning('%s has %d events, nevts__ says %d', path, count, n)
    print('%d events written in %s' % (sum(counts), output))
    if args.delete:
        for path in files:
            os.remove(path)
        print('Removed the events_<i>.lhe files')
    return 0


def select_class(event, nborn):
    """'S' or 'H' (from the #aMCatNLO line or the multiplicity), or None."""
    info = event.amcatnlo_info()
    if info is not None and info.get('iSorH') in (1, 2):
        return 'S' if info['iSorH'] == 1 else 'H'
    if nborn is None:
        return None
    npart = len(event.external())
    if npart == nborn:
        return 'S'
    if npart == nborn + 1:
        return 'H'
    raise LHEUtilitiesError(
        'The Born multiplicity seems incorrect: cannot extract S/H events '
        '(event with %d particles, Born multiplicity %d)' % (npart, nborn))


def infer_nborn(path):
    """Smallest number of initial- and final-state particles."""
    ef = open_events(path)
    try:
        nmin = None
        for text in iter_raw(ef):
            event = LHEEvent(text)
            if event.amcatnlo_info() is not None:
                return None
            n = len(event.external())
            nmin = n if nmin is None else min(nmin, n)
        return nmin
    finally:
        ef.close()


def cmd_select(args):
    keep = args.keep
    if keep == 'range':
        if not args.range:
            raise LHEUtilitiesError('--keep range needs --range FIRST LAST')
        first, last = sorted(args.range)
        if first < 1:
            raise LHEUtilitiesError('Events are numbered from 1')
        name = '%d_to_%d' % (first, last)
    else:
        name = keep
    output = args.output or '%s_%s%s' % (lhe_stem(args.file), name,
                                         lhe_suffix(args.file))
    nborn = args.nborn
    if keep in ('S', 'H') and nborn is None:
        nborn = infer_nborn(args.file)
        if nborn is not None:
            print('No #aMCatNLO information: Born multiplicity taken as the '
                  'smallest multiplicity of the file, %d' % nborn)
    warn = Messages()
    ef = open_events(args.file)
    header = LHEHeader(ef.banner)
    tmp = tempfile.NamedTemporaryFile('w', delete=False, suffix='.lhe',
                                      dir=os.path.dirname(os.path.abspath(output)))
    nread = nkept = 0
    sum_all = sum_kept = 0.
    try:
        for text in iter_raw(ef):
            nread += 1
            if keep == 'range' and nread > last:
                break
            event = LHEEvent(text)
            sum_all += event.wgt
            for message in momentum_messages(event):
                warn('momentum checks', 'event %d: %s' % (nread, message))
            if keep in ('S', 'H'):
                ok = select_class(event, nborn) == keep
            elif keep == 'pos':
                ok = event.wgt >= 0.
            elif keep == 'neg':
                ok = event.wgt < 0.
            else:
                ok = nread >= first
            if ok:
                nkept += 1
                sum_kept += event.wgt
                tmp.write(text if text.endswith('\n') else text + '\n')
        tmp.close()
        if nkept == 0:
            print('No events of the desired type found in the file')
            return 1
        writer = LHEWriter(output, header.with_nevents(nkept))
        with open(tmp.name) as body:
            shutil.copyfileobj(body, writer.file)
        writer.close()
    finally:
        ef.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
    warn.summary()
    print('Number of events kept: %d (of %d read)' % (nkept, nread))
    print('The sum of the weights of the events kept is: %.8e' % sum_kept)
    if keep in ('S', 'H', 'pos', 'neg') and nread:
        norm = nread if header.averaged else 1.
        print('Their contribution to the cross section (event_norm %s): '
              '%.8e of %.8e' % (header.event_norm, sum_kept / norm,
                                sum_all / norm))
    print('Written in %s' % output)
    return 0


################################################################################
# Command: compare
################################################################################
def cmd_compare(args):
    efs = [open_events(args.file1), open_events(args.file2)]
    headers = [LHEHeader(ef.banner) for ef in efs]
    if headers[0].parton_shower != headers[1].parton_shower:
        message = ('Files are not relevant to the same MC: %s, %s'
                   % (headers[0].parton_shower, headers[1].parton_shower))
        if not args.force:
            raise LHEUtilitiesError(message)
        logger.warning(message)
    n1, n2 = headers[0].nevents, headers[1].nevents
    if n1 is not None and n2 is not None and n1 != n2 and not args.force:
        raise LHEUtilitiesError('Files have different number of events: '
                                '%d, %d' % (n1, n2))
    warn = Messages()
    sums = [0., 0.]
    shifts = [0.] * 4
    max_shift = [0.] * 4
    nevt = 0
    pdg_mismatch = 0
    nup_mismatch = 0
    ended = [False, False]
    iters = [iter_raw(ef) for ef in efs]
    try:
        while True:
            texts = []
            for k in (0, 1):
                try:
                    texts.append(next(iters[k]))
                except StopIteration:
                    ended[k] = True
            if any(ended):
                break
            nevt += 1
            ev = [LHEEvent(t) for t in texts]
            for k in (0, 1):
                sums[k] += ev[k].wgt
                for message in momentum_messages(ev[k]):
                    warn('momentum checks', 'file #%d, event %d: %s'
                         % (k + 1, nevt, message))
            if ev[0].nup != ev[1].nup:
                nup_mismatch += 1
                message = 'Mismatch in event #%d: NUP %d, %d' % (
                    nevt, ev[0].nup, ev[1].nup)
                if not args.force:
                    raise LHEUtilitiesError(message)
                warn('NUP mismatch', message)
                continue
            if ev[0].idup != ev[1].idup:
                pdg_mismatch += 1
                warn('different particles', 'event %d: different particles'
                     % nevt)
            for j, k in enumerate((0, 1, 2, 3)):
                ytmp = 0.
                for p1, p2 in zip(ev[0].pup, ev[1].pup):
                    x = p1[k] - p2[k]
                    if abs(p1[k]) >= 1.:
                        x /= p1[k]
                    ytmp += abs(x) / ev[0].nup
                shifts[j] += (ytmp - shifts[j]) / (nevt - nup_mismatch)
                max_shift[j] = max(max_shift[j], ytmp)
    finally:
        for ef in efs:
            ef.close()
    if ended[0] != ended[1]:
        logger.warning('The files have different numbers of events: the '
                       'comparison stopped after %d events', nevt)
    warn.summary()
    print('Events compared: %d' % nevt)
    for k in (0, 1):
        norm = nevt if headers[k].averaged and nevt else 1.
        print('The sum of weights #%d is: %.8e  (cross section, event_norm '
              '%s: %.8e)' % (k + 1, sums[k], headers[k].event_norm,
                             sums[k] / norm))
    if nup_mismatch:
        print('Events with different number of particles: %d' % nup_mismatch)
    if pdg_mismatch:
        print('Events with different particles: %d' % pdg_mismatch)
    print('')
    print('Average momentum shifts (relative if |p|>1 GeV), maximum')
    for j, name in enumerate(('px', 'py', 'pz', 'E')):
        print('  %d (%-2s): %.6e   %.6e' % (j + 1, name, shifts[j],
                                           max_shift[j]))
    return 0


################################################################################
# Histograms (HwU)
################################################################################
class Histogram(object):
    """Histogram with one column per weight; HwU output."""

    def __init__(self, title, nbins, xmin, xmax, nweights=1, logy=True):
        if nbins < 1 or not xmax > xmin:
            raise LHEUtilitiesError('Wrong binning for %s' % title)
        self.title = title
        self.nbins = nbins
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.width = (self.xmax - self.xmin) / nbins
        self.logy = logy
        self.sums = [[0.] * nbins for _ in range(nweights)]
        self.sumw2 = [0.] * nbins

    def fill(self, x, weights):
        if x is None or not (self.xmin <= x < self.xmax):
            return
        ibin = min(int((x - self.xmin) / self.width), self.nbins - 1)
        for col, w in zip(self.sums, weights):
            col[ibin] += w
        self.sumw2[ibin] += weights[0] ** 2

    def hwu(self, labels, norm):
        """HwU text: the histogram of dsigma/dx, with sigma = sum(w)*norm."""
        lines = ['<histogram> %d "%s |X_AXIS@LIN |Y_AXIS@%s |TYPE@NLO"'
                 % (self.nbins, self.title.replace('"', "'"),
                    'LOG' if self.logy else 'LIN')]
        f = norm / self.width
        for i in range(self.nbins):
            lo = self.xmin + i * self.width
            values = [lo, lo + self.width, self.sums[0][i] * f,
                      math.sqrt(self.sumw2[i]) * f]
            values += [col[i] * f for col in self.sums[1:]]
            lines.append(' '.join('%+16.7e' % v for v in values))
        lines.append(r'<\histogram>')
        return '\n'.join(lines)


def write_hwu(path, histograms, labels, norm):
    header = '##& xmin & xmax & central value & dy'
    if labels:
        header += ' & ' + ' & '.join(labels)
    with open(path, 'w') as f:
        f.write(header + '\n\n')
        for h in histograms:
            f.write(h.hwu(labels, norm) + '\n\n')


def nice_ceil(x):
    """Smallest of 1, 2, 2.5, 5 times a power of 10 above x."""
    if x <= 0.:
        return 1.
    p = 10. ** math.floor(math.log10(x))
    for m in (1., 2., 2.5, 5., 10.):
        if m * p >= x * (1. - 1e-12):
            return m * p
    return 10. * p


def unique_labels(infos):
    labels, seen = [], set()
    for w in infos:
        label = w.hwu_label()
        if label in seen:
            label = '%s [%s]' % (label, w.id)
        seen.add(label)
        labels.append(label)
    return labels


################################################################################
# Command: check
################################################################################
CHARGES = {1: -1. / 3, 2: 2. / 3, 3: -1. / 3, 4: 2. / 3, 5: -1. / 3,
           6: 2. / 3, 11: -1., 12: 0., 13: -1., 14: 0., 15: -1., 16: 0.,
           21: 0., 22: 0., 23: 0., 24: 1., 25: 0.,
           111: 0., 211: 1., 130: 0., 310: 0., 321: 1., 311: 0., 221: 0.,
           2212: 1., 2112: 0., 3122: 0., 3112: -1., 3222: 1., 3312: -1.,
           3322: 0., 3334: -1., 411: 1., 421: 0., 431: 1., 511: 0., 521: 1.,
           531: 0., 4122: 1.}
COLOUR_SINGLETS = set([11, 12, 13, 14, 15, 16, 22, 23, 24, 25])
HW6_CODES = {401: 'q qbar -> X', 402: 'q g    -> X', 403: 'qbar q -> X',
             404: 'qbar g -> X', 405: 'g q    -> X', 406: 'g qbar -> X',
             407: 'g g    -> X', 499: 'other'}


def charge(pdg):
    if abs(pdg) in CHARGES:
        return math.copysign(1., pdg) * CHARGES[abs(pdg)]
    return None


def hw6_code(id1, id2):
    """Initial state classification of check_events.f (HW6 conventions)."""
    q1, q2 = 1 <= abs(id1) <= 5, 1 <= abs(id2) <= 5
    if q1 and q2:
        if id1 > 0 and id2 < 0:
            return 401
        if id1 < 0 and id2 > 0:
            return 403
        return 499
    if id1 == 21 and id2 == 21:
        return 407
    if id1 == 21 and q2:
        return 405 if id2 > 0 else 406
    if id2 == 21 and q1:
        return 402 if id1 > 0 else 404
    return 499


def colour_messages(event, indices):
    """Colour flow checks of checkcolconn (check_events.f), for the
    particles with |ISTUP|=1: list of messages."""
    messages = []
    labels = collections.defaultdict(list)   # label -> [(i, 0 col/1 acol)]
    for i in indices:
        pdg = event.idup[i]
        col, acol = event.icolup[i]
        a = abs(pdg)
        if pdg == 21:
            wrong = col == 0 or acol == 0
        elif 1 <= pdg <= 6:
            wrong = col == 0 or acol != 0
        elif -6 <= pdg <= -1:
            wrong = acol == 0 or col != 0
        elif a in COLOUR_SINGLETS or 100 < a < 1000000:
            wrong = col != 0 or acol != 0
        else:
            wrong = False          # BSM particle: colour not known
        if wrong:
            messages.append('wrong colour (%d, %d) for particle %d (PDG %d)'
                            % (col, acol, i + 1, pdg))
        for k, label in enumerate((col, acol)):
            if label:
                labels[label].append((i, k))
    for label, owners in sorted(labels.items()):
        if len(owners) != 2:
            messages.append('colour line %d attached to %d particles'
                            % (label, len(owners)))
            continue
        (i, ki), (j, kj) = owners
        same_side = (event.istup[i] == -1) == (event.istup[j] == -1)
        # Same side: colour to anticolour. Initial to final: same type.
        if (ki != kj) != same_side:
            messages.append('colour line %d connects particles %d and %d '
                            'inconsistently' % (label, i + 1, j + 1))
    return messages


def read_res_file(path):
    """(abs integral, error, integral, error) from res_1.txt."""
    with open(path) as f:
        lines = f.read().splitlines()
    for n, line in enumerate(lines):
        if line.strip().startswith('Total:'):
            try:
                a = lines[n + 1].split()
                b = lines[n + 2].split()
                return (to_float(a[0]), to_float(a[2]), to_float(b[0]),
                        to_float(b[2]))
            except (IndexError, ValueError):
                break
    raise LHEUtilitiesError('File %s has unexpected format' % path)


class DecayStatistics(object):
    """Decay channels of the resonances (status 2) and their correlations
    (setdecmat/findmothers/finddaughters/finalizedecays of check_events.f)."""

    def __init__(self):
        self.channels = collections.OrderedDict()   # slot -> Counter
        self.correlations = collections.defaultdict(collections.Counter)

    def add(self, event):
        seen = collections.Counter()
        decays = []
        for m in range(event.nup):
            if event.istup[m] != 2:
                continue
            daughters = sorted(event.idup[j] for j in range(event.nup)
                               if event.istup[j] > 0 and
                               event.mothup[j][0] == m + 1 and
                               event.mothup[j][1] in (0, m + 1))
            if not daughters:
                continue
            seen[event.idup[m]] += 1
            slot = (event.idup[m], seen[event.idup[m]])
            channel = tuple(daughters)
            self.channels.setdefault(slot, collections.Counter())[channel] += 1
            decays.append((slot, channel))
        for a, ca in decays:
            for b, cb in decays:
                if a != b:
                    self.correlations[(a, b)][(ca, cb)] += 1

    def write(self, path):
        def slot_name(slot):
            return '%d (#%d)' % slot
        slots = list(self.channels)
        order = {}
        with open(path, 'w') as f:
            f.write(' mother     PDG code (occurrence in the event)\n')
            for n, slot in enumerate(slots, 1):
                f.write('   %2d       %s\n' % (n, slot_name(slot)))
            for n, slot in enumerate(slots, 1):
                counter = self.channels[slot]
                total = float(sum(counter.values()))
                order[slot] = [c for c, _ in counter.most_common()]
                f.write('\n-------------------------------------------\n')
                f.write(' mother #%d   %s\n' % (n, slot_name(slot)))
                f.write(' decay channel   decay products              '
                        'events    fraction\n')
                for k, channel in enumerate(order[slot], 1):
                    f.write('   %3d    %-30s %10d  %10.6f\n'
                            % (k, ' '.join('%d' % p for p in channel),
                               counter[channel], counter[channel] / total))
            f.write('\n\n*******************************************\n')
            f.write('   CORRELATIONS\n')
            f.write('*******************************************\n')
            for (a, b), counter in sorted(self.correlations.items()):
                total = float(sum(counter.values()))
                f.write('\n-------------------------------------------\n')
                f.write(' mothers %s and %s\n' % (slot_name(a), slot_name(b)))
                f.write(' (rows: channels of the first, columns: of the '
                        'second)\n')
                for title, row_norm in (('events', None),
                                        ('normalised to total', 'total'),
                                        ('normalised to rows', 'rows')):
                    f.write(' %s\n' % title)
                    f.write('     ' + ''.join('%11d' % (k + 1)
                                              for k in range(len(order[b])))
                            + '\n')
                    for i, ca in enumerate(order[a], 1):
                        row = [counter[(ca, cb)] for cb in order[b]]
                        if row_norm == 'total':
                            row = ['%11.5f' % (x / total) for x in row]
                        elif row_norm == 'rows':
                            s = float(sum(row)) or 1.
                            row = ['%11.5f' % (x / s) for x in row]
                        else:
                            row = ['%11d' % x for x in row]
                        f.write('  %3d' % i + ''.join(row) + '\n')


def cmd_check(args):
    ef = open_events(args.file)
    header = LHEHeader(ef.banner)
    out_dir = args.output_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    nevents_header = header.nevents
    shower = nevents_header is not None and nevents_header < 0
    if shower:
        print('This appears to be a showered LH file: momentum conservation '
              'is not checked')
    averaged = header.averaged
    # Reference masses
    masses = {}
    if args.masses == 'header':
        masses = header.mc_masses()
        if not masses:
            print('No <MonteCarloMasses> in the header: masses taken from the '
                  'events')
    for item in args.mass or []:
        try:
            pid, value = item.split('=')
            masses[int(pid)] = to_float(value)
            masses[-int(pid)] = to_float(value)
        except ValueError:
            raise LHEUtilitiesError('Wrong --mass %s (use PDG=value)' % item)
    weight_infos = header.weights()
    weight_ids = [w.id for w in weight_infos]

    errors = open(os.path.join(out_dir, 'LHEF.errors'), 'w')
    mass_errors = open(os.path.join(out_dir, 'LHEF.mass_errors'), 'w')
    nerr = nmass_err = 0
    warn = Messages()
    decays = DecayStatistics() if args.decays else None

    scal_hist = [Histogram('scalup', 100, 0., 200.),
                 Histogram('scalup', 100, 0., 800.),
                 Histogram('log10 scalup', 40, 0., 4.)]
    dipole_hist = collections.OrderedDict()

    def dipole_fill(name, value, w):
        if name not in dipole_hist:
            dipole_hist[name] = [Histogram(name, 100, 0., 200.),
                                 Histogram(name, 100, 0., 800.),
                                 Histogram('log10 ' + name, 40, 0., 4.)]
        h = dipole_hist[name]
        h[0].fill(value, [w])
        h[1].fill(value, [w])
        if value > 0.:
            h[2].fill(math.log10(value), [w])

    weights = array('d')
    counts = collections.Counter()      # (class, sign)
    npart_counts = collections.Counter()
    processes = collections.Counter()
    unknown_charge = collections.Counter()
    rwgt_sum = collections.OrderedDict((wid, 0.) for wid in weight_ids)
    rwgt_min = {}
    rwgt_max = {}
    old_rwgt = None
    unweighted = True
    first_wgt = None
    add_info = None
    nevt = 0
    npart_ref = {}
    fxfx = header._run_card_value('ickkw') == '3'
    for text in iter_raw(ef):
        nevt += 1
        event = LHEEvent(text)
        w = event.wgt
        weights.append(w)
        if first_wgt is None:
            first_wgt = abs(w)
        elif first_wgt and abs(1. - abs(w) / first_wgt) >= UNWEIGHTED_TINY:
            unweighted = False
        for h in scal_hist[:2]:
            h.fill(event.scale, [w])
        if event.scale > 0.:
            scal_hist[2].fill(math.log10(event.scale), [w])

        info = event.amcatnlo_info()
        if add_info is None:
            add_info = info is not None
        elif add_info != (info is not None):
            raise LHEUtilitiesError('Inconsistency in event file: event %d '
                                    '%s the #aMCatNLO line'
                                    % (nevt, 'without' if add_info else 'with'))
        ext = event.external()
        npart = len(ext)

        # MC@NLO-Delta scales: positive iff the partons are colour connected
        scales = event.dipole_scales()
        if scales is not None:
            wrong = any(not (1 <= i <= event.nup and 1 <= j <= event.nup)
                        or i == j for i, j in scales)
            for j in range(event.nup):
                a = abs(event.idup[j])
                flav = ('light' if a <= 4 else 'bottom' if a == 5 else
                        'top' if a == 6 else 'gluon' if event.idup[j] == 21
                        else None)
                for k in range(event.nup):
                    if j == k:
                        continue
                    cj, ck = event.icolup[j], event.icolup[k]
                    connected = any(x and x in ck for x in cj)
                    s = scales.get((j + 1, k + 1), -1.)
                    if s > 0.:
                        if flav:
                            dipole_fill('scalup ' + flav, s, w)
                        if info and info.get('iSorH') == 2:
                            if j + 1 == info.get('ifks'):
                                dipole_fill('scalup[ifks]', s, w)
                            elif j + 1 == info.get('jfks'):
                                dipole_fill('scalup[jfks]', s, w)
                        wrong = wrong or not connected
                    else:
                        wrong = wrong or connected
            if wrong:
                errors.write('####event: %d\n colour flow and scales not '
                             'consistent\n' % nevt)
                nerr += 1

        # Charge conservation
        chtot = 0.
        unknown = [event.idup[i] for i in ext if charge(event.idup[i]) is None]
        if unknown:
            for pid in unknown:
                unknown_charge[pid] += 1
        else:
            chtot = sum(event.istup[i] * charge(event.idup[i]) for i in ext)
            if abs(chtot) > 1e-8:
                errors.write('####event: %d\n charge is not conserved %g\n'
                             % (nevt, chtot))
                nerr += 1

        # Masses
        for i in ext:
            pid, m = event.idup[i], event.pup[i][4]
            if pid not in masses:
                masses[pid] = m
                if -pid not in masses:
                    masses[-pid] = m
            elif abs(masses[pid] - m) > MASS_TINY:
                mass_errors.write('####event: %d\n Wrong mass shell %.8e for '
                                  'particle %d  Must be: %.8e\n'
                                  % (nevt, m, i + 1, masses[pid]))
                nmass_err += 1

        # Colour connections
        for message in colour_messages(event, ext):
            errors.write('####event: %d\n %s\n' % (nevt, message))
            nerr += 1

        # Processes, S/H
        sign = '+' if w > 0. else '-'
        cls = None
        if add_info:
            isorh = info.get('iSorH')
            cls = {1: 'S', 2: 'H'}.get(isorh)
            if cls is None:
                errors.write('####event: %d\n unknown iSorH %s\n'
                             % (nevt, isorh))
                nerr += 1
            else:
                # Same multiplicity for all S (H) events; with FxFx, for each
                # jet multiplicity (npNLO attribute) only
                key = (cls, event.attributes.get('npNLO'))
                if fxfx and key[1] is None:
                    pass
                elif key not in npart_ref:
                    npart_ref[key] = npart
                elif npart != npart_ref[key]:
                    errors.write('####event: %d\n wrong particle number [%s] '
                                 '%d\n' % (nevt, cls, npart))
                    nerr += 1
            counts[(cls, sign)] += 1
        else:
            npart_counts[(npart, sign)] += 1
        processes[(cls, tuple(event.idup[i] for i in ext))] += 1

        # Weights
        if event.rwgt():
            for wid, value in event.rwgt().items():
                if wid not in rwgt_sum:
                    rwgt_sum[wid] = 0.
                rwgt_sum[wid] += value
                rwgt_min[wid] = min(rwgt_min.get(wid, value), value)
                rwgt_max[wid] = max(rwgt_max.get(wid, value), value)
        if info and info.get('jwgtinfo') == 0 and info.get('wgtcentral'):
            # pre-<rwgt> format: central, mu lower/upper, PDF lower/upper
            if old_rwgt is None:
                old_rwgt = [[0., 0.] for _ in range(5)]
            values = [info.get(k, 0.) for k in ('wgtcentral', 'wgtmumin',
                      'wgtmumax', 'wgtpdfmin', 'wgtpdfmax')]
            if unweighted:
                ratios = values
            else:
                ratios = [values[0] / w if w else 0.]
                ratios += [v / values[0] if values[0] else 0.
                           for v in values[1:]]
            for acc, r in zip(old_rwgt, ratios):
                acc[0] += r
                acc[1] += r * r

        if decays is not None:
            decays.add(event)
        if not shower:
            for message in momentum_messages(event, ext):
                warn('momentum checks', 'event %d: %s' % (nevt, message))
        if nevt % max(10000, (nevents_header or 0) // 10 or 1) == 0:
            print('Processed %d events' % nevt)
    ef.close()
    errors.close()
    mass_errors.close()
    if nevt == 0:
        raise LHEUtilitiesError('No events in %s' % args.file)

    # S/H from the multiplicity when there is no #aMCatNLO line
    if not add_info:
        mults = sorted(set(n for n, _ in npart_counts))
        nborn = args.nborn if args.nborn is not None else mults[0]
        for (n, sign), c in npart_counts.items():
            cls = 'S' if n == nborn else 'H' if n == nborn + 1 else None
            if cls is None:
                raise LHEUtilitiesError(
                    'The Born multiplicity seems incorrect: cannot extract S/H '
                    'events (multiplicities %s, Born %d; use --nborn)'
                    % (mults, nborn))
            counts[(cls, sign)] += c
        classified = collections.Counter()
        for (_, proc), c in processes.items():
            classified[('S' if len(proc) == nborn else 'H', proc)] += c
        processes = classified

    norm = 1. / nevt if averaged else 1.
    sum_wgt = sum(weights) * norm
    sum_abs = sum(abs(x) for x in weights) * norm
    err_wgt = sum_abs / math.sqrt(nevt)

    # Histograms
    hists = list(scal_hist)
    if not unweighted:
        wlim = max(abs(max(weights)), abs(min(weights))) * norm
        for k in (1, 2, 4):
            h = Histogram('weight', 100, -k * wlim, k * wlim)
            for x in weights:
                h.fill(x * norm, [abs(x)])
            hists.append(h)
    for hs in dipole_hist.values():
        hists.extend(hs)
    write_hwu(os.path.join(out_dir, 'check_events.HwU'), hists, [], norm)

    ipos = counts[('S', '+')] + counts[('H', '+')]
    ineg = counts[('S', '-')] + counts[('H', '-')]
    print('')
    if unweighted:
        print('The events appear to be unweighted')
    else:
        print('The events appear to be weighted')
    print('')
    print('The total number of events is: %d' % nevt)
    if nevents_header is not None and abs(nevents_header) != nevt:
        print('  (the header says %d)' % nevents_header)
    print(' of which: %d w>0 %d w<0' % (ipos, ineg))
    print('   ==> %10.4f%% w>0   %10.4f%% w<0' % (100. * ipos / nevt,
                                                  100. * ineg / nevt))
    for cls in ('S', 'H'):
        p, n = counts[(cls, '+')], counts[(cls, '-')]
        print('')
        print('The total number of %s-events is: %d' % (cls, p + n))
        print(' of which: %d w>0 %d w<0' % (p, n))
        if p + n:
            print('   ==> %10.4f%% w>0   %10.4f%% w<0' % (100. * p / (p + n),
                                                          100. * n / (p + n)))
    print('')
    print('Sum of weights is    : %.8e +- %.8e' % (sum_wgt, err_wgt))
    print('Sum of abs weights is: %.8e +- %.8e' % (sum_abs, err_wgt))
    print('  (event_norm %s)' % header.event_norm)

    with open(os.path.join(out_dir, 'res_wgt'), 'w') as f:
        f.write(' Xsec     (check_events) = %.8e +- %.8e\n' % (sum_wgt, err_wgt))
        if args.res:
            xinta, xintaerr, xint, xinterr = read_res_file(args.res)
            f.write(' Xsec     (res_1.txt)    = %.8e +- %.8e\n' % (xint, xinterr))
        f.write('\n Xsec ABS (check_events) = %.8e +- %.8e\n' % (sum_abs, err_wgt))
        if args.res:
            f.write(' Xsec ABS (res_1.txt)    = %.8e +- %.8e\n' % (xinta, xintaerr))
    if args.res:
        toterr = math.sqrt(xinterr ** 2 + err_wgt ** 2)
        if abs(sum_wgt - xint) > toterr:
            nerr += 1
            with open(os.path.join(out_dir, 'LHEF.errors'), 'a') as f:
                f.write('WEIGHTS\nIntegral: %.8e +- %.8e\nWeights:  %.8e +- '
                        '%.8e\n\nSigmas:   %.4f\n' % (xint, xinterr, sum_wgt,
                                                      err_wgt, abs(xint - sum_wgt)
                                                      / toterr))
        print('Integral (%s): %.8e +- %.8e' % (args.res, xint, xinterr))

    nps = [len(proc) for _, proc in processes]
    print('')
    print('Smallest and largest numbers of particles: %d %d'
          % (min(nps), max(nps)))

    # Process statistics
    with open(os.path.join(out_dir, 'LHEF.stats'), 'w') as f:
        f.write(' Statistics for processes (initial state, HW6 conventions)\n')
        for cls in ('S', 'H', None):
            by_code = collections.Counter()
            for (c, proc), n in processes.items():
                if c == cls:
                    by_code[hw6_code(proc[0], proc[1])
                            if len(proc) > 1 else 499] += n
            ncls = sum(by_code.values())
            if cls is None and not ncls:
                continue
            f.write('  %s events: %d\n' % (cls or 'unclassified', ncls))
            for code in sorted(by_code):
                n = by_code[code]
                f.write('   %3d %-14s %10d  %14.8e  %14.8e\n'
                        % (code, HW6_CODES[code], n, n / float(ncls),
                           n / float(nevt)))
            f.write('\n')
        f.write(' Processes (PDG codes of the particles with |ISTUP|=1)\n')
        for (cls, proc), n in processes.most_common():
            f.write('  %s %10d  %14.8e   %s\n'
                    % (cls or '?', n, n / float(nevt),
                       ' '.join(str(p) for p in proc)))

    # Weights
    if rwgt_sum or old_rwgt:
        labels = dict((w.id, w) for w in weight_infos)
        with open(os.path.join(out_dir, 'LHEF.rwgt'), 'w') as f:
            if old_rwgt:
                for name, (s, s2) in zip(('central:  ', 'mu lower: ',
                                          'mu upper: ', 'PDF lower:',
                                          'PDF upper:'), old_rwgt):
                    avg = s / nevt
                    f.write(' %s %.8e +- %.8e\n' % (
                        name, avg, math.sqrt(abs(s2 / nevt - avg ** 2))))
                f.write('\n')
            if rwgt_sum:
                f.write(' Sums of the weights (cross sections, event_norm %s)'
                        ', maximum and minimum\n' % header.event_norm)
                group = None
                for wid, s in rwgt_sum.items():
                    winfo = labels.get(wid)
                    if winfo is not None and winfo.group != group:
                        group = winfo.group
                        f.write('\n %s\n' % group)
                    f.write('  %-8s %16.8e %16.8e %16.8e   %s\n'
                            % (wid, s * norm, rwgt_max.get(wid, 0.) * norm,
                               rwgt_min.get(wid, 0.) * norm,
                               winfo.label if winfo else ''))
                missing = [wid for wid in rwgt_sum if wid not in labels]
                if missing and weight_infos:
                    logger.warning('Weights not described in <initrwgt>: %s',
                                   ', '.join(missing))
    if decays is not None:
        decays.write(os.path.join(out_dir, 'LHEF.decays'))
    if unknown_charge:
        print('')
        print('Charge conservation not checked in the events with particles '
              'of unknown charge: %s'
              % ', '.join('%d (%d times)' % kv for kv in
                          sorted(unknown_charge.items())))
    warn.summary()
    print('')
    print('Total number of errors found: %d' % (nerr + nmass_err))
    print('         Of which mass shell: %d' % nmass_err)
    print('Files written in %s: LHEF.errors, LHEF.mass_errors, LHEF.stats, '
          'res_wgt, check_events.HwU%s%s' % (
              os.path.abspath(out_dir),
              ', LHEF.rwgt' if (rwgt_sum or old_rwgt) else '',
              ', LHEF.decays' if decays is not None else ''))
    return 0


################################################################################
# Command: plot
################################################################################
QCD_PARTONS = set([1, 2, 3, 4, 5, 21])
EW_PARTICLES = set([11, 12, 13, 14, 15, 16, 22, 23, 24, 25])
PARTICLE_NAMES = {1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 11: 'e',
                  12: 've', 13: 'mu', 14: 'vm', 15: 'ta', 16: 'vt', 21: 'g',
                  22: 'a', 23: 'z', 24: 'w', 25: 'h'}


def particle_name(pdg):
    name = PARTICLE_NAMES.get(abs(pdg), str(abs(pdg)))
    if abs(pdg) in (11, 13, 15, 24):
        return name + ('-' if pdg > 0 else '+') if abs(pdg) != 24 else \
            name + ('+' if pdg > 0 else '-')
    if pdg < 0 and abs(pdg) not in (21, 22, 23, 25):
        return name + '~'
    return name


class DefaultAnalysis(object):
    """Observables of the particles of the events (no jet clustering): total
    rate, multiplicity, system of the leptons, photons and EW/Higgs bosons,
    hardest QCD parton, hardest particle of each species."""

    nbins = 50

    def __init__(self, nbins=50):
        self.nbins = nbins
        self.ranges = {}

    def observables(self, event):
        out = collections.OrderedDict()
        final = event.particles(1)
        out['total rate'] = 1.
        out['number of final-state particles'] = len(final)
        ew = [p for p in final if abs(p.pdg) in EW_PARTICLES]
        if ew:
            P = four_momentum(ew)
            out['EW system mass'] = invariant_mass(ew)
            out['EW system pt'] = transverse_momentum(P)
            out['EW system rapidity'] = rapidity(P)
        partons = [p for p in final if abs(p.pdg) in QCD_PARTONS]
        out['number of final-state QCD partons'] = len(partons)
        if partons:
            hardest = max(partons, key=lambda p: p.pt)
            out['hardest parton pt'] = hardest.pt
            out['hardest parton eta'] = hardest.eta
        species = collections.OrderedDict()
        for p in final:
            if abs(p.pdg) in QCD_PARTONS:
                continue
            species.setdefault(particle_name(p.pdg), []).append(p)
        for name, ps in species.items():
            hardest = max(ps, key=lambda p: p.pt)
            out['pt(%s)' % name] = hardest.pt
            out['%s(%s)' % ('y' if hardest.m > 1. else 'eta', name)] = \
                hardest.rapidity if hardest.m > 1. else hardest.eta
        return out

    def set_ranges(self, samples):
        """Binning from the values of the first events."""
        for name, values in samples.items():
            if not values:
                continue
            if name == 'total rate':
                self.ranges[name] = (1, 0.5, 1.5)
            elif name.startswith('number of'):
                top = int(max(values)) + 2
                self.ranges[name] = (top + 1, -0.5, top + 0.5)
            elif name.startswith(('eta', 'y(', 'eta(')) or 'rapidity' in name \
                    or name.endswith('eta'):
                self.ranges[name] = (self.nbins, -5., 5.)
            else:
                values = sorted(values)
                q = values[min(len(values) - 1, int(0.98 * len(values)))]
                self.ranges[name] = (self.nbins, 0., nice_ceil(1.25 * q))

    def book(self, book):
        for name, (nbins, lo, hi) in self.ranges.items():
            book(name, nbins, lo, hi)

    def analyse(self, event, fill):
        for name, value in self.observables(event).items():
            fill(name, value)


def load_user_analysis(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('user_lhe_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('book', 'analyse'):
        if not hasattr(module, name):
            raise LHEUtilitiesError('%s must define %s()' % (path, name))
    return module


def cmd_plot(args):
    ef = open_events(args.file)
    header = LHEHeader(ef.banner)
    weight_infos = [] if args.central_only else header.weights()
    weight_ids = [w.id for w in weight_infos]
    labels = unique_labels(weight_infos)
    nweights = 1 + len(weight_ids)

    if args.analysis:
        analysis = load_user_analysis(args.analysis)
    else:
        analysis = DefaultAnalysis(args.nbins)
        samples = collections.defaultdict(list)
        for n, text in enumerate(iter_raw(ef)):
            if n >= 2000:
                break
            for name, value in analysis.observables(LHEEvent(text)).items():
                samples[name].append(value)
        ef.close()
        analysis.set_ranges(samples)
        ef = open_events(args.file)

    histograms = collections.OrderedDict()

    def book(name, nbins, xmin, xmax, logy=True):
        if name in histograms:
            raise LHEUtilitiesError('Histogram %s booked twice' % name)
        histograms[name] = Histogram(name, int(nbins), xmin, xmax, nweights,
                                     logy)

    analysis.book(book)
    current = [None]
    missing = set()

    def fill(name, value, weight_factor=1.):
        h = histograms.get(name)
        if h is None:
            if name not in missing:
                missing.add(name)
                logger.warning('Histogram %s not booked: ignored', name)
            return
        if weight_factor == 1.:
            h.fill(value, current[0])
        else:
            h.fill(value, [x * weight_factor for x in current[0]])

    warn = Messages()
    nevt = 0
    sum_wgt = 0.
    try:
        for text in iter_raw(ef):
            nevt += 1
            event = LHEEvent(text)
            if weight_ids:
                rwgt = event.rwgt()
                try:
                    current[0] = [event.wgt] + [rwgt[k] for k in weight_ids]
                except KeyError as error:
                    raise LHEUtilitiesError('Event %d has no weight %s'
                                            % (nevt, error))
            else:
                current[0] = [event.wgt]
            sum_wgt += event.wgt
            for message in momentum_messages(event):
                warn('momentum checks', 'event %d: %s' % (nevt, message))
            analysis.analyse(event, fill)
    finally:
        ef.close()
    if nevt == 0:
        raise LHEUtilitiesError('No events in %s' % args.file)
    norm = 1. / nevt if header.averaged else 1.
    output = args.output or lhe_stem(args.file) + '.HwU'
    write_hwu(output, list(histograms.values()), labels, norm)
    warn.summary()
    print('The sum of the weights is: %.8e (event_norm %s)'
          % (sum_wgt * norm, header.event_norm))
    print('%d histograms written in %s' % (len(histograms), output))
    if args.gnuplot:
        histograms_py = mg5_module_path('histograms')
        if histograms_py is None:
            logger.warning('histograms.py of MG5_aMC not found: no gnuplot '
                           'output')
        else:
            # histograms.py writes <out>.HwU too: use another name
            command = [sys.executable, histograms_py, output, '--gnuplot',
                       '--no_open', '--keep_all_weights',
                       '--out=%s_gnuplot' % os.path.splitext(output)[0]]
            print('Running: %s' % ' '.join(command))
            subprocess.call(command)
    return 0


################################################################################
# Uncertainties
################################################################################
def lhapdf_datadirs(lhapdf_config=None):
    dirs = []
    for var in ('LHAPDF_DATA_PATH', 'LHAPATH'):
        dirs += [d for d in os.environ.get(var, '').split(':') if d]
    configs = [lhapdf_config] if lhapdf_config else []
    config_files = [os.path.join(HERE, os.pardir, 'Cards',
                                 'amcatnlo_configuration.txt')]
    config_files += [os.path.join(root, 'input', 'mg5_configuration.txt')
                     for root in _mg5_roots()]
    for path in config_files:
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    m = re.match(r'^\s*(lhapdf|lhapdf_py3)\s*=\s*(\S+)', line)
                    if m and m.group(2) not in ('None', 'none'):
                        configs.append(m.group(2))
    configs.append('lhapdf-config')
    for config in configs:
        try:
            out = subprocess.check_output([config, '--datadir'],
                                          stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            continue
        d = out.decode(errors='ignore').strip()
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def pdf_set_info(setname, lhaid, datadirs):
    """Keys of the .info file of the set (ErrorType, ErrorConfLevel,
    NumMembers), or {} if not found."""
    names = [setname] if setname else []
    if lhaid is not None:
        for d in datadirs:
            index = os.path.join(d, 'pdfsets.index')
            if os.path.isfile(index):
                with open(index) as f:
                    for line in f:
                        fields = line.split()
                        if len(fields) >= 2 and fields[0] == str(lhaid):
                            names.append(fields[1])
    for name in names:
        for d in datadirs:
            path = os.path.join(d, name, name + '.info')
            if os.path.isfile(path):
                info = {'SetName': name}
                with open(path) as f:
                    for line in f:
                        m = re.match(r'^\s*(\w+)\s*:\s*(.*?)\s*$', line)
                        if m:
                            info[m.group(1)] = m.group(2).strip('"\'')
                return info
    return {}


def guess_pdf_error_type(setname, lhaid):
    name = (setname or '').lower()
    if 'nnpdf' in name or '_mc' in name:
        return 'replicas'
    if 'pdf4lhc' in name:
        return 'symmhessian'
    return 'hessian'


def pdf_uncertainty(members, error_type, conf_level=None, target_cl=None):
    """(central, minus, plus) of the PDF uncertainty, as LHAPDF does, for
    members = [values of member 0, member 1, ...], each a list (bins).
    error_type is hessian, symmhessian or replicas, possibly with +as (the
    last two members are then alpha_s variations, ignored)."""
    error_type = error_type.lower()
    n_extra = 0
    while error_type.endswith('+as'):
        error_type = error_type[:-3]
        n_extra += 2
    central = members[0]
    errors = members[1:len(members) - n_extra] if n_extra else members[1:]
    nbins = len(central)
    minus = [0.] * nbins
    plus = [0.] * nbins
    if not errors:
        return central, minus, plus
    for b in range(nbins):
        x0 = central[b]
        if error_type == 'replicas':
            vals = [m[b] for m in errors]
            n = len(vals)
            avg = sum(vals) / n
            var = sum((v - avg) ** 2 for v in vals) / (n - 1) if n > 1 else 0.
            plus[b] = minus[b] = math.sqrt(var)
        elif error_type == 'symmhessian':
            plus[b] = minus[b] = math.sqrt(sum((m[b] - x0) ** 2
                                               for m in errors))
        elif error_type == 'hessian':
            up = down = 0.
            for k in range(0, len(errors) - 1, 2):
                a, c = errors[k][b], errors[k + 1][b]
                up += max(a - x0, c - x0, 0.) ** 2
                down += max(x0 - a, x0 - c, 0.) ** 2
            plus[b], minus[b] = math.sqrt(up), math.sqrt(down)
        else:
            raise LHEUtilitiesError('Unknown PDF error type %s' % error_type)
    if conf_level and target_cl and abs(conf_level - target_cl) > 1e-6 and \
            error_type != 'replicas':
        nd = statistics.NormalDist()
        scale = nd.inv_cdf(0.5 + target_cl / 200.) / \
            nd.inv_cdf(0.5 + conf_level / 200.)
        plus = [x * scale for x in plus]
        minus = [x * scale for x in minus]
    return central, minus, plus


def scale_uncertainty(points, central=None):
    """Envelopes of scale variations. points: {(muR, muF): values (bins)}.
    Returns {name: (central, min, max)} for the full envelope, the 7-point
    one (without the muR/muF = 4, 1/4 points), muR and muF only."""
    if central is None:
        central = points.get((1., 1.))
        if central is None:
            raise LHEUtilitiesError('No muR=muF=1 weight in a scale group')
    out = {}
    selections = {
        'envelope': lambda r, f: True,
        '7-point': lambda r, f: not (abs(math.log(r / f)) > math.log(3.)),
        'muR only': lambda r, f: f == 1.,
        'muF only': lambda r, f: r == 1.,
    }
    for name, sel in selections.items():
        chosen = [v for (r, f), v in points.items() if sel(r, f)]
        if not chosen:
            continue
        nbins = len(central)
        lo = [min([c[b] for c in chosen] + [central[b]]) for b in range(nbins)]
        hi = [max([c[b] for c in chosen] + [central[b]]) for b in range(nbins)]
        out[name] = (central, lo, hi, len(chosen))
    return out


class VariationGroups(object):
    """Scale and PDF groups of a set of weights (WeightInfo)."""

    def __init__(self, infos):
        self.scale = collections.OrderedDict()   # (tag, dyn) -> {(r, f): w}
        self.pdf = collections.OrderedDict()     # key -> [(lhaid, w)]
        for w in infos:
            if w.kind == 'scale':
                key = (w.tag or 0, w.dyn if w.dyn is not None else -1)
                self.scale.setdefault(key, collections.OrderedDict())[
                    (round(w.muR, 6), round(w.muF, 6))] = w
            elif w.kind == 'pdf':
                key = (w.tag or 0, w.pdfset or w.group or '')
                self.pdf.setdefault(key, []).append(w)
        for key in list(self.pdf):
            members = sorted(self.pdf[key], key=lambda w: w.lhaid)
            # Split groups with non-consecutive LHAPDF ids (several sets
            # without names)
            groups = [[members[0]]]
            for w in members[1:]:
                if w.lhaid == groups[-1][-1].lhaid + 1:
                    groups[-1].append(w)
                else:
                    groups.append([w])
            if len(groups) == 1:
                self.pdf[key] = groups[0]
            else:
                del self.pdf[key]
                for g in groups:
                    self.pdf[(key[0], '%s %d' % (key[1], g[0].lhaid))] = g


class UncertaintyEngine(object):
    """Computes the scale and PDF uncertainties for VariationGroups, given a
    function returning the values (bins) of a weight."""

    def __init__(self, args):
        self.args = args
        self._datadirs = None

    def datadirs(self):
        if self._datadirs is None:
            self._datadirs = [self.args.lhapdf_datadir] if \
                self.args.lhapdf_datadir else []
            self._datadirs += lhapdf_datadirs(self.args.lhapdf_config)
        return self._datadirs

    def pdf_type(self, members):
        setname = members[0].pdfset
        lhaid = members[0].lhaid
        if self.args.pdf_error:
            return self.args.pdf_error, None, setname or str(lhaid), 'given'
        info = pdf_set_info(setname, lhaid, self.datadirs())
        if info.get('ErrorType'):
            try:
                cl = float(info.get('ErrorConfLevel', 68.268949))
            except ValueError:
                cl = None
            return info['ErrorType'], cl, info.get('SetName', setname), 'LHAPDF'
        error_type = guess_pdf_error_type(setname, lhaid)
        return error_type, None, setname or str(lhaid), 'guessed'

    def compute(self, groups, values):
        """List of result dictionaries."""
        results = []
        for (tag, dyn), points in groups.scale.items():
            data = dict((k, values(w)) for k, w in points.items())
            if (1., 1.) not in data:
                continue
            res = scale_uncertainty(data)
            if self.args.seven_point and '7-point' in res:
                main = res['7-point']
            else:
                main = res['envelope']
            results.append({'kind': 'scale', 'tag': tag, 'dyn': dyn,
                            'npoints': main[3], 'central': main[0],
                            'min': main[1], 'max': main[2], 'details': res,
                            'ids': [w.id for w in points.values()]})
        for (tag, name), members in groups.pdf.items():
            error_type, cl, setname, source = self.pdf_type(members)
            data = [values(w) for w in members]
            target = self.args.pdf_cl
            central, minus, plus = pdf_uncertainty(data, error_type, cl,
                                                   target)
            results.append({'kind': 'pdf', 'tag': tag, 'set': setname,
                            'lhaid': members[0].lhaid,
                            'nmembers': len(members), 'error_type': error_type,
                            'type_source': source,
                            'conf_level': target if (target and cl) else cl,
                            'central': central,
                            'min': [c - m for c, m in zip(central, minus)],
                            'max': [c + p for c, p in zip(central, plus)]})
        return results


def _percent(value, central):
    return 100. * value / central if central else 0.


def uncertainties_events(args):
    ef = open_events(args.file)
    header = LHEHeader(ef.banner)
    infos = header.weights()
    if not infos:
        raise LHEUtilitiesError('No <initrwgt> weights in %s: no scale or PDF '
                                'variations' % args.file)
    sums = collections.defaultdict(float)
    sumw = 0.
    nevt = 0
    try:
        for text in iter_raw(ef):
            event = LHEEvent(text)
            nevt += 1
            sumw += event.wgt
            for wid, value in event.rwgt().items():
                sums[wid] += value
    finally:
        ef.close()
    if not nevt:
        raise LHEUtilitiesError('No events in %s' % args.file)
    norm = 1. / nevt if header.averaged else 1.
    groups = VariationGroups(infos)
    engine = UncertaintyEngine(args)
    results = engine.compute(groups, lambda w: [sums[w.id] * norm])
    print('Events: %d (event_norm %s)' % (nevt, header.event_norm))
    print('Cross section (central weights): %.6e pb' % (sumw * norm))
    report = {'events': nevt, 'cross_section': sumw * norm, 'scale': [],
              'pdf': []}
    for r in results:
        c, lo, hi = r['central'][0], r['min'][0], r['max'][0]
        if r['kind'] == 'scale':
            print('')
            print('Scale variation, dynamical scale %d%s:'
                  % (r['dyn'], ', orders tag %d' % r['tag'] if r['tag'] else ''))
            for name in ('envelope', '7-point', 'muR only', 'muF only'):
                if name in r['details']:
                    cc, l, h, n = r['details'][name]
                    print('    %-9s (%d points): %.6e pb  %+.1f%% %+.1f%%'
                          % (name, n, cc[0], _percent(h[0] - cc[0], cc[0]),
                             _percent(l[0] - cc[0], cc[0])))
            report['scale'].append({
                'tag': r['tag'], 'dyn': r['dyn'], 'central': c,
                'variations': dict((name, {'min': v[1][0], 'max': v[2][0],
                                           'points': v[3]})
                                   for name, v in r['details'].items())})
        else:
            print('')
            print('PDF variation, %s (%d members, %s (%s)%s)%s:'
                  % (r['set'], r['nmembers'], r['error_type'],
                     r['type_source'],
                     ', %.0f%% CL' % r['conf_level'] if r['conf_level'] else '',
                     ', orders tag %d' % r['tag'] if r['tag'] else ''))
            print('    %.6e pb  %+.1f%% %+.1f%%'
                  % (c, _percent(hi - c, c), _percent(lo - c, c)))
            if r['type_source'] == 'guessed':
                print('    (error type guessed from the name: give '
                      '--pdf-error or --lhapdf-config to be sure)')
            report['pdf'].append({'set': r['set'], 'lhaid': r['lhaid'],
                                  'members': r['nmembers'],
                                  'error_type': r['error_type'],
                                  'type_source': r['type_source'],
                                  'conf_level': r['conf_level'],
                                  'central': c, 'min': lo, 'max': hi})
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=1)
        print('\nWritten in %s' % args.output)
    return 0


def read_hwu(path):
    """(labels, [(title, rows)]) of a HwU file; rows: lists of floats."""
    labels = None
    histos = []
    current = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('##'):
                labels = [x.strip() for x in s.lstrip('#').split('&')
                          if x.strip()]
            elif s.startswith('<histogram>'):
                m = re.match(r'<histogram>\s*(\d+)\s*"(.*)"', s)
                current = (m.group(2) if m else s, [])
            elif s.startswith('<\\histogram>') or s.startswith('</histogram>'):
                if current is not None:
                    histos.append(current)
                current = None
            elif current is not None and s:
                current[1].append([to_float(x) for x in s.split()])
    if labels is None:
        raise LHEUtilitiesError('%s is not a HwU file (no ## header)' % path)
    return labels, histos


def uncertainties_hwu(args):
    labels, histos = read_hwu(args.file)
    if labels[:4] != ['xmin', 'xmax', 'central value', 'dy']:
        raise LHEUtilitiesError('Unexpected HwU header in %s' % args.file)
    infos = []
    columns = {}
    for k, label in enumerate(labels[4:], 4):
        if '@aux' in label:
            continue
        w = WeightInfo(str(k), label)
        if w.kind in ('scale', 'pdf'):
            infos.append(w)
            columns[w.id] = k
    if not infos:
        raise LHEUtilitiesError('No scale or PDF weights in %s' % args.file)
    groups = VariationGroups(infos)
    engine = UncertaintyEngine(args)
    output = args.output or os.path.splitext(args.file)[0] + \
        '_uncertainties.dat'
    with open(output, 'w') as f:
        first = True
        for title, rows in histos:
            results = engine.compute(
                groups, lambda w: [row[columns[w.id]] for row in rows])
            if first:
                for r in results:
                    if r['kind'] == 'pdf':
                        print('PDF set %s: %d members, %s (%s)'
                              % (r['set'], r['nmembers'], r['error_type'],
                                 r['type_source']))
                first = False
            f.write('# %s\n' % title.split('|')[0].strip())
            names = ['xmin', 'xmax', 'central', 'dy']
            for r in results:
                if r['kind'] == 'scale':
                    tag = 'scale(dyn=%d%s)' % (r['dyn'], ',tag=%d' % r['tag']
                                               if r['tag'] else '')
                else:
                    tag = 'pdf(%s)' % r['set']
                names += [tag + '_min', tag + '_max']
            f.write('# ' + ' '.join(names) + '\n')
            for b, row in enumerate(rows):
                values = row[:4]
                for r in results:
                    values += [r['min'][b], r['max'][b]]
                f.write(' '.join('%+16.7e' % v for v in values) + '\n')
            f.write('\n\n')
    print('%d histograms written in %s' % (len(histos), output))
    return 0


def cmd_uncertainties(args):
    if args.file.endswith(('.HwU', '.hwu')):
        return uncertainties_hwu(args)
    if args.file.endswith('.top'):
        raise LHEUtilitiesError(
            'Topdrawer files are not supported: use the HwU output of the '
            'analyses (or uncertainties.f)')
    return uncertainties_events(args)


################################################################################
# Command line
################################################################################
def build_parser():
    parser = argparse.ArgumentParser(
        prog='lhe_utilities.py',
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n\n'.join(__doc__.split('\n\n')[1:]))
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='debug messages')
    sub = parser.add_subparsers(dest='command', metavar='command')
    sub.required = True

    p = sub.add_parser('check', help='consistency checks and statistics '
                       '(check_events.f)', description=
                       'Checks the events: S/H and sign statistics, sum of '
                       'the weights, momentum and charge conservation, mass '
                       'shells, colour flows (and the MC@NLO-Delta scales), '
                       'processes, resonance decays. Writes LHEF.errors, '
                       'LHEF.mass_errors, LHEF.stats, LHEF.rwgt, LHEF.decays, '
                       'res_wgt and check_events.HwU.')
    p.add_argument('file')
    p.add_argument('--res', metavar='RES_FILE',
                   help='compare the sum of the weights with the integral of '
                   'this file (res_1.txt of SubProcesses)')
    p.add_argument('--masses', choices=('header', 'events'), default='header',
                   help='reference masses of quarks, leptons and gluons: '
                   '<MonteCarloMasses> of the header (default) or the first '
                   'occurrence in the events (always used for other '
                   'particles)')
    p.add_argument('--mass', action='append', metavar='PDG=MASS',
                   help='reference mass of a particle (repeatable)')
    p.add_argument('--decays', action='store_true',
                   help='statistics of the decays of the resonances')
    p.add_argument('--nborn', type=int,
                   help='Born multiplicity (initial and final state, no '
                   'resonances), for events without the #aMCatNLO line; '
                   'default: smallest multiplicity')
    p.add_argument('-d', '--output-dir', default='.',
                   help='directory of the output files (default: .)')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('combine-jobs', help='combine the events of split '
                       'generation jobs (combine_jobs.f)', description=
                       'Concatenates events_<i>.lhe, with weights multiplied by '
                       'nevts_i/sum(nevts), nevts_i read from nevts__<i>. The '
                       '<rwgt> weights are rescaled as well.')
    p.add_argument('directory', nargs='?', default='.')
    p.add_argument('-o', '--output', help='default: <directory>/events.lhe')
    p.add_argument('--delete', action='store_true',
                   help='remove events_<i>.lhe afterwards (as combine_jobs.f)')
    p.add_argument('--force', action='store_true',
                   help='combine files for different showers')
    p.set_defaults(func=cmd_combine_jobs)

    p = sub.add_parser('compare', help='compare two event files '
                       '(compare_events.f)', description=
                       'Compares two files with the same events (e.g. before '
                       'and after a reshuffling): sums of the weights and '
                       'average relative momentum shifts.')
    p.add_argument('file1')
    p.add_argument('file2')
    p.add_argument('--force', action='store_true',
                   help='compare even if the showers, numbers of events or '
                   'of particles differ')
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('merge', help='concatenate event files '
                       '(merge_events.f)', description=
                       'Concatenates event files with the header and init '
                       'block of the first one (with the total number of '
                       'events). The weights are not changed, unless '
                       '--rescale is given: then those of each file are '
                       'multiplied by n_i/n_tot, as appropriate for files '
                       'whose weights sum to the cross section '
                       '(event_norm=sum).')
    p.add_argument('files', nargs='*')
    p.add_argument('--prefix', help='merge PREFIX.001 ... PREFIX.<num> '
                   '(output PREFIX.complete), as merge_events.f')
    p.add_argument('--num', type=int, default=0,
                   help='number of files with --prefix')
    p.add_argument('-o', '--output')
    p.add_argument('--rescale', action='store_true',
                   help='multiply the weights of each file by n_i/n_tot')
    p.add_argument('--force', action='store_true',
                   help='merge files for different showers')
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser('plot', help='histograms of the events '
                       '(plot_events.f)', description=
                       'Histograms of the events in HwU format, with a column '
                       'for each weight of <rwgt> (so that histograms.py and '
                       'the uncertainties command give the scale and PDF '
                       'bands). Without --analysis, a set of observables of '
                       'the particles is used. An analysis is a python file '
                       'with\n'
                       '  def book(book): book(name, nbins, xmin, xmax)\n'
                       '  def analyse(event, fill): fill(name, value)\n'
                       'where event.particles(status=1) gives the particles '
                       '(pdg, px, py, pz, E, m, pt, eta, rapidity, phi).',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('file')
    p.add_argument('-o', '--output', help='HwU file (default: <file>.HwU)')
    p.add_argument('--analysis', help='python analysis file')
    p.add_argument('--nbins', type=int, default=50,
                   help='bins of the default observables')
    p.add_argument('--central-only', action='store_true',
                   help='only the central weight')
    p.add_argument('--gnuplot', action='store_true',
                   help='also run histograms.py for gnuplot output')
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser('select', help='keep some of the events '
                       '(select_events.f)', description=
                       'Keeps the S or H events, the events with w>=0 or w<0, '
                       'or a range of events. The output is <file>_<S|H|pos|'
                       'neg|first_to_last>.lhe[.gz] by default.')
    p.add_argument('file')
    p.add_argument('--keep', required=True,
                   choices=('S', 'H', 'pos', 'neg', 'range'))
    p.add_argument('--range', nargs=2, type=int, metavar=('FIRST', 'LAST'),
                   help='events to keep with --keep range (from 1)')
    p.add_argument('--nborn', type=int,
                   help='Born multiplicity for S/H without #aMCatNLO line '
                   '(default: smallest multiplicity)')
    p.add_argument('-o', '--output')
    p.set_defaults(func=cmd_select)

    p = sub.add_parser('split', help='split an event file '
                       '(split_events.f)', description=
                       'Splits a file into NPARTS files <file>_001.lhe, ... '
                       '(event numbers differing by at most one), or into '
                       'files of --events-per-file events. Each file has the '
                       'header and init block of the input, with its number of '
                       'events; the weights are unchanged.')
    p.add_argument('file')
    p.add_argument('nparts', type=int, nargs='?')
    p.add_argument('--events-per-file', type=int)
    p.add_argument('-d', '--output-dir')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--gzip', dest='gzip', action='store_true', default=None,
                   help='gzip the output (default: as the input)')
    g.add_argument('--no-gzip', dest='gzip', action='store_false')
    p.set_defaults(func=cmd_split)

    p = sub.add_parser('uncertainties', help='scale and PDF uncertainties '
                       '(uncertainties.f)', description=
                       'Scale and PDF uncertainties from the <rwgt> weights of '
                       'an event file (total rate), or from the weights of the '
                       'histograms of a HwU file (written in '
                       '<file>_uncertainties.dat). Scale: envelope of the '
                       'variations of each dynamical scale. PDF: as LHAPDF, '
                       'with the error type of the .info file of the set '
                       '(found with lhapdf-config), or --pdf-error.')
    p.add_argument('file', help='event file or .HwU file')
    p.add_argument('-o', '--output', help='output: JSON summary for event '
                   'files, table for HwU files')
    p.add_argument('--seven-point', action='store_true',
                   help='7-point scale envelope (no muR/muF=4,1/4) for the '
                   'HwU table')
    p.add_argument('--pdf-error', choices=('hessian', 'symmhessian',
                                           'replicas', 'hessian+as',
                                           'symmhessian+as', 'replicas+as'),
                   help='PDF error type (default: from LHAPDF)')
    p.add_argument('--pdf-cl', type=float,
                   help='rescale Hessian PDF uncertainties to this confidence '
                   'level (e.g. 68.27); default: that of the set')
    p.add_argument('--lhapdf-config', help='lhapdf-config of the LHAPDF '
                   'installation')
    p.add_argument('--lhapdf-datadir', help='LHAPDF data directory')
    p.set_defaults(func=cmd_uncertainties)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        logging.getLogger('madgraph').setLevel(logging.WARNING)
    try:
        return args.func(args)
    except LHEUtilitiesError as error:
        logger.error(str(error))
        return 1


if __name__ == '__main__':
    sys.exit(main())
