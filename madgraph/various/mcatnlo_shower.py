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
"""Consistency layer between MC@NLO short-distance generation and the shower.

The MC@NLO subtraction terms are computed for a specific parton shower, with
specific settings (recoil scheme, starting scales, kernels, dead zones, photon
conversion bound, ...). The shower that later reads the events must use the
same choices, otherwise the NLO accuracy is lost.

This module provides

* MatchingContract: the choices made at generation time that the shower must
  respect. It is written in the <MCatNLOMatching> block of the event-file
  header, so that showering is consistent even if the cards were changed
  after generation, or if the events are showered elsewhere.

* scan_events: a light-weight inspection of the beginning of LHE files, used
  to obtain the Born coloured multiplicity and the resonances whose decays
  are part of the NLO calculation.
"""

from __future__ import absolute_import, division

import gzip
import hashlib
import io
import logging
import os
import re

pjoin = os.path.join
logger = logging.getLogger('madgraph.mcatnlo_shower')


class ShowerInterfaceError(Exception):
    """Inconsistent or unsupported shower configuration."""


CONTRACT_TAG = 'MCatNLOMatching'
CONTRACT_VERSION = 1

# Number of events inspected at the beginning of each event file (each
# integration channel) when scanning; every file is inspected.
SCAN_EVENTS_PER_FILE = 2000


################################################################################
# Small helpers
################################################################################

def _as_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ['t', 'true', '.true.', 'on', '1', 'yes']:
        return True
    if text in ['f', 'false', '.false.', 'off', '0', 'no', '']:
        return False
    raise ValueError('Not a logical value: %s' % value)


def _fortran_float(text):
    return float(str(text).strip().lower().replace('d', 'e'))


def read_fks_param(path, name, default=None):
    """Value following the line '#name' in an FKS_params.dat card."""
    if not path or not os.path.isfile(path):
        return default
    want = False
    with open(path) as card:
        for line in card:
            stripped = line.strip()
            if not stripped:
                continue
            if want:
                if stripped.startswith(('!', '#')):
                    return default
                return stripped.split()[0]
            if stripped.split()[0] == '#' + name:
                want = True
    return default


def pythia8_mmaxgamma(fks_params):
    """#Pythia8MMaxGamma of an FKS_params.dat card (10 GeV if absent), with
    the validation of FKSParams.f90: a number in [0.001, 5000] GeV."""
    text = read_fks_param(fks_params, 'Pythia8MMaxGamma', None)
    if text is None:
        if fks_params and os.path.isfile(fks_params) and \
                any(l.split()[:1] == ['#Pythia8MMaxGamma'] for l in open(fks_params)):
            raise ShowerInterfaceError('#Pythia8MMaxGamma has no value in %s'
                                       % fks_params)
        return 10.
    try:
        value = _fortran_float(text)
    except ValueError:
        raise ShowerInterfaceError('Invalid #Pythia8MMaxGamma in %s: %s'
                                   % (fks_params, text))
    if not 0.001 <= value <= 5000.:
        raise ShowerInterfaceError('#Pythia8MMaxGamma in %s must be between '
                                   '0.001 and 5000 GeV: %s' % (fks_params, text))
    return value


def file_md5(path):
    if not path or not os.path.isfile(path):
        return ''
    digest = hashlib.md5()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b''):
            digest.update(chunk)
    return digest.hexdigest()


# Electric charges (in units of e) of SM particles, for the QED case.
_SM_CHARGES = {1: -1. / 3, 2: 2. / 3, 3: -1. / 3, 4: 2. / 3, 5: -1. / 3,
               6: 2. / 3, 11: -1., 13: -1., 15: -1., 24: 1., 37: 1.}


def sm_charge(pdg):
    charge = _SM_CHARGES.get(abs(pdg), 0.)
    return charge if pdg > 0 else -charge


################################################################################
# Event scanning
################################################################################

class EventScan(object):
    """Summary of the beginning of one or more LHE files."""

    def __init__(self):
        self.nevents = 0
        self.nevents_S = 0
        self.born_coloured = set()     # production-level coloured, S events
        self.real_coloured = set()     # the same for H events
        self.coloured_decays = set()   # resonances with coloured daughters
        self.charged_decays = set()    # resonances with charged daughters
        self.has_npnlo = None          # all events carry npNLO?

    def born_multiplicity(self):
        """Unique Born coloured multiplicity, or -1 if mixed or unknown."""
        if len(self.born_coloured) == 1:
            return list(self.born_coloured)[0]
        return -1

    def update(self, other):
        self.nevents += other.nevents
        self.nevents_S += other.nevents_S
        self.born_coloured |= other.born_coloured
        self.real_coloured |= other.real_coloured
        self.coloured_decays |= other.coloured_decays
        self.charged_decays |= other.charged_decays
        if other.has_npnlo is not None:
            self.has_npnlo = other.has_npnlo if self.has_npnlo is None \
                else (self.has_npnlo and other.has_npnlo)


def _open_text(path):
    if path.endswith('.gz'):
        return io.TextIOWrapper(gzip.open(path, 'rb'), errors='replace')
    with open(path, 'rb') as stream:
        magic = stream.read(2)
    if magic == b'\x1f\x8b':
        return io.TextIOWrapper(gzip.open(path, 'rb'), errors='replace')
    return open(path, errors='replace')


_npnlo_re = re.compile(r'''npNLO\s*=\s*['"]?\s*(-?\d+)''')


def scan_events(paths, max_per_file=SCAN_EVENTS_PER_FILE):
    """Inspect the first max_per_file events of every LHE file.

    Every file (integration channel) is inspected, but only its beginning,
    so that the cost does not grow with the size of the sample. This is a
    heuristic: a resonance present in a tiny fraction of the events of a
    channel may be missed; it only matters for the guard against
    matrix-element corrections in decays corrected at NLO."""
    if isinstance(paths, str):
        paths = [paths]
    total = EventScan()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with _open_text(path) as stream:
            total.update(_scan_stream(stream, max_per_file))
    return total


def _scan_stream(stream, limit):
    scan = EventScan()
    in_event = False
    block = []
    tag = ''
    for line in stream:
        if not in_event:
            stripped = line.lstrip()
            if stripped.startswith('<event'):
                in_event = True
                tag = stripped
                block = []
            elif stripped.startswith('</LesHouchesEvents'):
                break
            continue
        if line.lstrip().startswith('</event>'):
            in_event = False
            _scan_event(scan, tag, block)
            if scan.nevents >= limit:
                break
            continue
        block.append(line)
    return scan


def _scan_event(scan, tag, lines):
    if not lines:
        return
    try:
        nup = int(lines[0].split()[0])
        particles = []
        for line in lines[1:1 + nup]:
            data = line.split()
            particles.append((int(data[0]), int(data[1]), int(data[2]),
                              int(data[4]), int(data[5])))
    except (ValueError, IndexError):
        return
    isorh = 0
    for line in lines[1 + nup:]:
        if line.lstrip().startswith('#aMCatNLO'):
            try:
                isorh = int(line.split()[1])
            except (ValueError, IndexError):
                isorh = 0
            break
    scan.nevents += 1
    has_npnlo = bool(_npnlo_re.search(tag))
    scan.has_npnlo = has_npnlo if scan.has_npnlo is None \
        else (scan.has_npnlo and has_npnlo)

    nin = sum(1 for p in particles if p[1] == -1)
    coloured = 0
    for i, (pdg, status, mother, col1, col2) in enumerate(particles):
        if status in (1, 2) and 1 <= mother <= max(nin, 1) and (col1 or col2):
            coloured += 1
        if status != 2:
            continue
        for daughter in particles:
            if daughter[2] != i + 1:
                continue
            if daughter[3] or daughter[4]:
                scan.coloured_decays.add(abs(pdg))
            if sm_charge(daughter[0]) != 0.:
                scan.charged_decays.add(abs(pdg))
    if isorh == 1:
        scan.nevents_S += 1
        scan.born_coloured.add(coloured)
    elif isorh == 2:
        scan.real_coloured.add(coloured)


################################################################################
# The matching contract
################################################################################

class MatchingContract(dict):
    """Generation-time choices that the shower has to reproduce."""

    # name: (type, default)
    fields = [
        ('contract_version', int, CONTRACT_VERSION),
        ('parton_shower', str, ''),
        ('mcatnlo_delta', bool, False),
        ('ickkw', int, 0),
        ('ptj', float, 0.),
        ('shower_scale_factor', float, 1.),
        ('lpp1', int, 1),
        ('lpp2', int, 1),
        ('qed_matching', bool, False),
        ('pythia8_mmaxgamma', float, 10.),
        ('fxfx_pythia8_finite_terms', bool, False),
        ('born_coloured_partons', int, -1),
        ('nlo_resonance_decays', list, []),
        ('npnlo_attribute', bool, False),
        ('scanned_events', int, 0),
        ('fks_params_md5', str, ''),
        ('legacy', bool, False),
    ]

    def __init__(self, **values):
        dict.__init__(self)
        for name, kind, default in self.fields:
            self[name] = list(default) if kind is list else default
        for name, value in values.items():
            self.set(name, value)

    def set(self, name, value):
        kinds = dict((n, k) for n, k, _ in self.fields)
        if name not in kinds:
            # Unknown entries from newer versions are kept as strings.
            self[name] = str(value)
            return
        kind = kinds[name]
        if kind is bool:
            value = _as_bool(value)
        elif kind is int:
            value = int(float(str(value).replace('d', 'e')))
        elif kind is float:
            value = _fortran_float(value)
        elif kind is list:
            if isinstance(value, str):
                value = [] if value.strip().lower() in ['', 'none'] else \
                    [int(v) for v in re.split(r'[\s,]+', value.strip()) if v]
            value = sorted(set(int(v) for v in value))
        else:
            value = str(value).strip()
        self[name] = value

    # ------------------------------------------------------------------
    def to_text(self):
        lines = ['# Choices made in the MC@NLO short-distance generation.',
                 '# The parton shower must reproduce them; see README.shower.']
        for name, kind, _ in self.fields:
            value = self[name]
            if kind is bool:
                value = 'T' if value else 'F'
            elif kind is list:
                value = ','.join(str(v) for v in value) if value else 'none'
            lines.append('%s = %s' % (name, value))
        for name in sorted(k for k in self if k not in dict(
                (n, 0) for n, _, _ in self.fields)):
            lines.append('%s = %s' % (name, self[name]))
        return '\n'.join(lines) + '\n'

    def to_block(self):
        return '<%s>\n%s</%s>\n' % (CONTRACT_TAG, self.to_text(), CONTRACT_TAG)

    @classmethod
    def from_text(cls, text):
        contract = cls()
        for line in text.split('\n'):
            line = line.split('#', 1)[0].strip()
            if not line or '=' not in line:
                continue
            name, value = [x.strip() for x in line.split('=', 1)]
            contract.set(name.lower(), value)
        return contract

    @classmethod
    def from_banner(cls, banner):
        """Contract stored in a banner, or None for older event files."""
        text = dict.get(banner, CONTRACT_TAG.lower()) if banner else None
        if not text:
            return None
        return cls.from_text(text)

    # ------------------------------------------------------------------
    def nlo_decays_with_mec_risk(self):
        """Resonance decays corrected at NLO (decay MECs would double count)."""
        return list(self['nlo_resonance_decays'])

    def summary(self):
        text = '%s' % self['parton_shower']
        if self['mcatnlo_delta']:
            text += ', MC@NLO-Delta'
        if self['ickkw'] == 3:
            text += ', FxFx'
        if self['qed_matching']:
            text += ', QED matching'
        return text


def contract_from_generation(me_dir, run_card, proc_characteristics,
                             event_files=(), fks_params=None):
    """Build the contract from the cards used for the generation and the
    first events of the produced (unshowered) event files."""
    if fks_params is None:
        fks_params = pjoin(me_dir, 'Cards', 'FKS_params.dat')
    splittings = []
    if proc_characteristics and 'splitting_types' in proc_characteristics:
        splittings = proc_characteristics['splitting_types']
    qed = 'QED' in splittings
    scan = scan_events(list(event_files))
    decays = set(scan.coloured_decays)
    if qed:
        decays |= scan.charged_decays

    def value(name, default):
        try:
            return run_card[name]
        except (KeyError, TypeError):
            return default

    mmax = pythia8_mmaxgamma(fks_params)
    finite = read_fks_param(fks_params, 'FxFxPythia8FiniteTerms', '.false.')
    shower = str(value('parton_shower', '')).upper()
    ickkw = int(value('ickkw', 0))
    try:
        finite = _as_bool(finite)
    except ValueError:
        finite = False
    if ickkw == 3 and shower == 'PYTHIA8':
        finite = True   # forced in the Fortran code for FxFx with Pythia8

    contract = MatchingContract(
        parton_shower=shower,
        mcatnlo_delta=bool(value('mcatnlo_delta', False)),
        ickkw=ickkw,
        ptj=value('ptj', 0.),
        shower_scale_factor=value('shower_scale_factor', 1.),
        lpp1=value('lpp1', 1),
        lpp2=value('lpp2', 1),
        qed_matching=qed,
        pythia8_mmaxgamma=mmax,
        fxfx_pythia8_finite_terms=finite,
        born_coloured_partons=scan.born_multiplicity() if scan.nevents_S else -1,
        nlo_resonance_decays=sorted(decays),
        npnlo_attribute=bool(scan.has_npnlo),
        scanned_events=scan.nevents,
        fks_params_md5=file_md5(fks_params),
    )
    return contract


def legacy_contract(banner, me_dir, event_file=None):
    """Reconstruct a contract for event files written before contracts
    existed. Values that were not recorded are taken from the current cards,
    which may have changed since the generation; the caller warns."""
    run_card = banner.charge_card('run_card')
    proc = {}
    path = pjoin(me_dir, 'SubProcesses', 'proc_characteristics') if me_dir else None
    if path and os.path.isfile(path):
        for line in open(path):
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = [x.strip() for x in line.split('=', 1)]
                if name == 'splitting_types':
                    proc[name] = re.findall(r"'(\w+)'", value)
    has_madspin = bool(dict.get(banner, 'madspin'))
    files = [event_file] if event_file else []
    contract = contract_from_generation(
        me_dir, run_card, proc, files,
        fks_params=pjoin(me_dir, 'Cards', 'FKS_params.dat') if me_dir else None)
    contract['legacy'] = True
    contract['fks_params_md5'] = ''
    if has_madspin:
        # MadSpin decays are at LO and appear as resonances as well: they
        # cannot be told apart from the NLO ones.
        contract['nlo_resonance_decays'] = []
        contract['madspin_in_legacy_file'] = 'T'
    return contract


def add_contract_to_banner_file(path, contract):
    """Insert (or replace) the contract block in a banner/header file."""
    with open(path) as stream:
        text = stream.read()
    block = contract.to_block()
    pattern = re.compile(r'<%s>.*?</%s>\n?' % (CONTRACT_TAG, CONTRACT_TAG),
                         re.DOTALL | re.IGNORECASE)
    if pattern.search(text):
        text = pattern.sub(lambda m: block, text, count=1)
    elif '</header>' in text:
        text = text.replace('</header>', block + '</header>', 1)
    else:
        raise ShowerInterfaceError('No <header> in %s' % path)
    with open(path, 'w') as stream:
        stream.write(text)


################################################################################
# Pythia8 installation
################################################################################

class Pythia8Installation(object):
    """What an installed Pythia8 provides: version, data files, settings,
    and the HepMC/LHAPDF interfaces it was configured with."""

    _setting_re = re.compile(
        r'<(flag|flagfix|mode|modeopen|modepick|modefix|parm|parmfix|word|'
        r'wordfix|fvec|mvec|pvec|wvec)\s+name\s*=\s*"([^"]+)"', re.IGNORECASE)

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path)) if path else ''
        if not self.path or not os.path.isdir(self.path):
            raise ShowerInterfaceError(
                'pythia8_path is not set to a valid Pythia8 installation: %s.\n'
                'Set it with "set pythia8_path /path/to/pythia8" in MG5_aMC, or '
                'in input/mg5_configuration.txt.' % path)
        if os.path.isdir(pjoin(self.path, 'xmldoc')) and \
                not os.path.isdir(pjoin(self.path, 'share', 'Pythia8')):
            raise ShowerInterfaceError(
                'Pythia 8.1 (%s) is no longer supported. Please install '
                'Pythia 8.2 or later (8.3 recommended).' % self.path)
        self.config = pjoin(self.path, 'bin', 'pythia8-config')
        if not os.path.isfile(self.config):
            raise ShowerInterfaceError('No bin/pythia8-config in %s' % self.path)
        if not os.access(self.config, os.X_OK):
            try:
                os.chmod(self.config, os.stat(self.config).st_mode | 0o111)
            except OSError:
                pass
        self.datadir = pjoin(self.path, 'share', 'Pythia8')
        self.xmldoc = pjoin(self.datadir, 'xmldoc')
        if not os.path.isfile(pjoin(self.xmldoc, 'Version.xml')):
            raise ShowerInterfaceError('Cannot find the Pythia8 xmldoc '
                                       'directory in %s' % self.datadir)
        self.version = self._read_version()
        self._settings = None
        self._config_cache = {}

    def _read_version(self):
        text = open(pjoin(self.xmldoc, 'Version.xml')).read()
        match = re.search(r'name\s*=\s*"Pythia:versionNumber"\s+default\s*=\s*"([\d.]+)"',
                          text)
        if not match:
            raise ShowerInterfaceError('Cannot determine the Pythia8 version '
                                       'from %s' % pjoin(self.xmldoc, 'Version.xml'))
        return match.group(1)

    @property
    def version_tuple(self):
        major, minor = self.version.split('.')[:2]
        return (int(major), int(minor))

    def run_config(self, *flags):
        key = flags
        if key not in self._config_cache:
            import subprocess
            try:
                output = subprocess.check_output(
                    [self.config] + list(flags), stderr=subprocess.STDOUT)
                self._config_cache[key] = output.decode(errors='ignore').strip()
            except (OSError, subprocess.CalledProcessError):
                self._config_cache[key] = ''
        return self._config_cache[key]

    def settings(self):
        """{lower-case name: (kind, name)} of all settings of this version."""
        if self._settings is None:
            settings = {}
            for name in os.listdir(self.xmldoc):
                if not name.endswith('.xml'):
                    continue
                with open(pjoin(self.xmldoc, name), errors='replace') as stream:
                    for kind, setting in self._setting_re.findall(stream.read()):
                        settings[setting.lower()] = (kind.lower(), setting)
            self._settings = settings
        return self._settings

    def has_setting(self, name):
        if re.match(r'^-?\d+:', name):
            return True  # particle data
        return name.lower() in self.settings()

    def hepmc(self, hepmc_path=None):
        """(major version, compile/link flags) of the HepMC library to use for
        writing events, preferring HepMC3. (0, '') if none is available."""
        include = pjoin(self.path, 'include', 'Pythia8Plugins')
        for major in (3, 2):
            if not os.path.isfile(pjoin(include, 'HepMC%d.h' % major)):
                continue
            if self.run_config('--with-hepmc%d' % major) == 'true':
                flags = self.run_config('--hepmc%d' % major)
                if flags and '-I ' not in flags + ' ':
                    return major, flags
        # Fall back on the HepMC installation of the MG5_aMC configuration.
        if hepmc_path and os.path.isdir(hepmc_path):
            libdir = pjoin(hepmc_path, 'lib')
            if os.path.isfile(pjoin(include, 'HepMC3.h')) and \
                    os.path.isdir(pjoin(hepmc_path, 'include', 'HepMC3')):
                return 3, '-I%s -L%s -Wl,-rpath,%s -lHepMC3' % (
                    pjoin(hepmc_path, 'include'), libdir, libdir)
            if os.path.isdir(pjoin(hepmc_path, 'include', 'HepMC')):
                return 2, '-I%s -L%s -Wl,-rpath,%s -lHepMC' % (
                    pjoin(hepmc_path, 'include'), libdir, libdir)
        return 0, ''

    def has_lhapdf6(self):
        if self.run_config('--with-lhapdf6') == 'true':
            return True
        for ext in ('so', 'dylib'):
            if os.path.isfile(pjoin(self.path, 'lib', 'libpythia8lhapdf6.%s' % ext)):
                return True
        return False

    def cxx_compiler(self):
        """C++ compiler used for Pythia8, if it is available here."""
        import shutil
        makefile = pjoin(self.datadir, 'examples', 'Makefile.inc')
        if os.path.isfile(makefile):
            for line in open(makefile, errors='replace'):
                if line.startswith('CXX='):
                    compiler = line.split('=', 1)[1].strip()
                    if compiler and shutil.which(compiler.split()[0]):
                        return compiler
        for compiler in (os.environ.get('CXX', ''), 'c++', 'g++', 'clang++'):
            if compiler and shutil.which(compiler.split()[0]):
                return compiler
        raise ShowerInterfaceError('No C++ compiler found to build the '
                                   'Pythia8 driver')


################################################################################
# Pythia8 settings
################################################################################

LOCKED, FREE, INTERNAL = 'locked', 'free', 'internal'


def _py8_value(value):
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float):
        return repr(value)
    return str(value)


class Pythia8Setting(object):
    def __init__(self, key, value, tier, comment='', reason=''):
        self.key = key
        self.value = _py8_value(value)
        self.tier = tier
        self.comment = comment
        self.reason = reason

    def line(self):
        text = '%s = %s' % (self.key, self.value)
        if self.comment:
            text = '%-50s ! %s' % (text, self.comment)
        return text


class Pythia8Settings(object):
    """Ordered Pythia8 settings, divided into tiers.

    INTERNAL: required by the driver and the event file (not user settable);
    LOCKED:   required by the MC@NLO subtraction, verified by the driver after
              initialisation;
    FREE:     defaults that may be changed with pythia8_options."""

    sections = ['main', 'beams', 'matching', 'mec', 'qed', 'merging',
                'couplings', 'pdf', 'masses', 'nonperturbative', 'decays',
                'random']
    titles = {
        'main': 'Run control (set by MG5_aMC)',
        'beams': 'Beams and event input',
        'matching': 'MC@NLO matching: starting scales and recoil scheme',
        'mec': 'Matrix-element corrections',
        'qed': 'QED shower',
        'merging': 'FxFx merging',
        'couplings': 'Shower couplings',
        'pdf': 'Parton distributions',
        'masses': 'Masses and widths',
        'nonperturbative': 'Underlying event and hadronisation',
        'decays': 'Decays and stable particles',
        'random': 'Random numbers',
    }

    def __init__(self):
        self.entries = dict((section, []) for section in self.sections)
        self.index = {}
        self.extra_lines = []        # decay settings such as '23:onMode = off'
        self.user = []               # user (pythia8_options) settings
        self.overridden = []         # locked settings overridden on request

    def set(self, section, key, value, tier, comment='', reason=''):
        lower = key.lower()
        if lower in self.index:
            old_section, old = self.index[lower]
            self.entries[old_section].remove(old)
        setting = Pythia8Setting(key, value, tier, comment, reason)
        self.entries[section].append(setting)
        self.index[lower] = (section, setting)
        return setting

    def lock(self, section, key, value, comment='', reason=''):
        return self.set(section, key, value, LOCKED, comment, reason)

    def free(self, section, key, value, comment=''):
        return self.set(section, key, value, FREE, comment)

    def internal(self, section, key, value, comment=''):
        return self.set(section, key, value, INTERNAL, comment)

    def get(self, key):
        entry = self.index.get(key.lower())
        return entry[1] if entry else None

    def tier(self, key):
        entry = self.get(key)
        return entry.tier if entry else None

    def locked(self):
        return [s for section in self.sections for s in self.entries[section]
                if s.tier == LOCKED]

    def apply_user_options(self, options, allow_unsafe=False):
        """Apply pythia8_options. Changing a locked or internal setting needs
        allow_unsafe; the list of such changes is returned."""
        refused, overridden = [], []
        for key, value in options.items():
            current = self.get(key)
            if current is not None and current.tier in (LOCKED, INTERNAL) \
                    and _same_value(current.value, value):
                continue
            if current is not None and current.tier == INTERNAL:
                refused.append((current, value))
            elif current is not None and current.tier == LOCKED:
                if allow_unsafe:
                    overridden.append((current, value))
                else:
                    refused.append((current, value))
            else:
                self.user.append(Pythia8Setting(key, value, FREE))
        if refused:
            text = []
            for setting, value in refused:
                reason = setting.reason or setting.comment
                text.append('  %s = %s (required: %s%s)' % (
                    setting.key, value, setting.value,
                    '; %s' % reason if reason else ''))
            internal = [s for s, _ in refused if s.tier == INTERNAL]
            raise ShowerInterfaceError(
                'The shower card changes Pythia8 settings that the MC@NLO '
                'matching requires:\n%s\n%s' % ('\n'.join(text),
                'These settings are set by MG5_aMC and cannot be changed.'
                if internal and len(internal) == len(refused) else
                'Changing them breaks the NLO accuracy of the showered events. '
                'To do it anyway, set allow_unsafe_shower_options = T in the '
                'shower card.'))
        for setting, value in overridden:
            setting.value = _py8_value(value)
            setting.tier = FREE
            setting.comment = 'UNSAFE: overridden by pythia8_options'
            self.overridden.append(setting)
        return overridden

    def render(self, header=''):
        """The command file. Free defaults and user settings come before the
        locked settings, so that a tune selected by the user cannot reset
        them; overridden locked settings come last."""
        lines = []
        if header:
            lines.extend('! ' + l if l else '!' for l in header.split('\n'))
        def block(title, settings):
            if settings:
                lines.append('')
                lines.append('! %s' % title)
                lines.extend(s.line() for s in settings)
        for section in self.sections:
            block(self.titles[section] + ' [MG5_aMC default]',
                  [s for s in self.entries[section] if s.tier == FREE
                   and s not in self.overridden])
        block('User settings (pythia8_options)', self.user)
        if self.extra_lines:
            lines.append('')
            lines.append('! Decays (DM_* in the shower card)')
            lines.extend(self.extra_lines)
        for section in self.sections:
            block(self.titles[section] + ' [set by MG5_aMC]',
                  [s for s in self.entries[section] if s.tier == INTERNAL])
        for section in self.sections:
            block(self.titles[section] + ' [required by MC@NLO, checked at run time]',
                  [s for s in self.entries[section] if s.tier == LOCKED])
        block('Required settings overridden on request '
              '(allow_unsafe_shower_options): NLO accuracy not guaranteed',
              self.overridden)
        return '\n'.join(lines) + '\n'

    def render_locked(self):
        lines = ['! Settings required by the MC@NLO subtraction. They are',
                 '! verified by the shower driver after initialisation.']
        lines.extend('%s = %s' % (s.key, s.value) for s in self.locked())
        return '\n'.join(lines) + '\n'


def _same_value(a, b):
    a, b = str(a).strip(), str(b).strip()
    if a.lower() == b.lower():
        return True
    flags = {'on': True, 'true': True, 'yes': True, '1': True,
             'off': False, 'false': False, 'no': False, '0': False}
    if a.lower() in flags and b.lower() in flags:
        return flags[a.lower()] == flags[b.lower()]
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1., abs(float(a)))
    except ValueError:
        return False


def pythia8_settings(contract, card, run, installation):
    """Settings for showering MC@NLO events with Pythia8.

    contract      MatchingContract of the event file
    card          ShowerCard
    run           dict with nevents (per job), nevents_total (per job),
                  event_norm, mc_masses {pdg: mass}, masses {pdg: mass},
                  widths {pdg: width}, pdf ('internal'|'lhapdf6', name,
                  extrapolate), seed
    installation  Pythia8Installation
    """
    s = Pythia8Settings()
    has = installation.has_setting
    notes = []

    # --- run control ---------------------------------------------------
    nevents = int(run['nevents'])
    s.internal('main', 'Main:numberOfEvents', int(run['nevents_total']),
               'events in the LHE file (per job)')
    s.internal('main', 'Main:spareMode1', nevents, 'events to shower (per job)')
    s.internal('main', 'Main:spareWord1', run['event_norm'],
               'normalisation of the event weights')
    s.free('main', 'Main:timesAllowErrors',
           int(float(card['maxerrs']) * nevents) + 1, 'allowed failed events')
    s.free('main', 'Next:numberShowEvent', int(card['maxprint']),
           'events listed in the log')
    s.free('main', 'Init:showChangedParticleData', False)
    s.free('main', 'Stat:showProcessLevel', True)
    s.free('main', 'Stat:showErrors', True)
    s.free('main', 'Check:epTolErr', 0.001, 'momentum-conservation tolerance')

    # --- beams -----------------------------------------------------------
    s.internal('beams', 'Beams:frameType', 4, 'beams and events from the LHE file')
    s.internal('beams', 'Beams:LHEF', 'events.lhe')
    lpp = (contract['lpp1'], contract['lpp2'])
    if abs(lpp[0]) == 2 and abs(lpp[1]) == 2:
        reason = 'photon-photon collisions from ultraperipheral beams'
        s.lock('beams', 'Check:event', False, reason=reason)
        s.lock('beams', 'BeamRemnants:unresolvedHadron', 3, reason=reason)
        s.lock('beams', 'PartonLevel:Remnants', False, reason=reason)
        s.lock('beams', 'PartonLevel:ISR', False, 'no ISR in UPC', reason=reason)
    if 0 in lpp:
        # Lepton beams without a structure function: there is no initial-
        # state radiation off them in the short-distance calculation.
        s.lock('beams', 'PDF:lepton', False, 'lepton beams without ISR',
               reason='the leptons were treated as pointlike (lpp=0)')

    # --- matching: scales and recoil -------------------------------------
    why = 'assumed by the PYTHIA8 MC counterterms'
    for shower in ('TimeShower', 'SpaceShower'):
        s.lock('matching', '%s:pTmaxMatch' % shower, 1,
               'start the shower at SCALUP', reason=why)
        s.lock('matching', '%s:pTmaxFudge' % shower, 1.0, reason=why)
    s.lock('matching', 'TimeShower:globalRecoil', True, reason=why)
    s.lock('matching', 'TimeShower:nMaxGlobalRecoil', 1, reason=why)
    s.lock('matching', 'TimeShower:globalRecoilMode', 2,
           'global recoil only for the first FSR emission of S events', reason=why)
    s.lock('matching', 'TimeShower:nMaxGlobalBranch', 1, reason=why)
    n_born = contract['born_coloured_partons']
    if contract['ickkw'] == 3 or contract['npnlo_attribute']:
        n_born = -1    # set event by event from the npNLO attribute
    s.lock('matching', 'TimeShower:nPartonsInBorn', n_born,
           'coloured Born partons, to tell S from H events', reason=why)
    s.lock('matching', 'TimeShower:limitPTmaxGlobal', True, reason=why)
    s.lock('matching', 'TimeShower:dampenBeamRecoil', False, reason=why)
    s.lock('matching', 'TimeShower:weightGluonToQuark', 1, reason=why)
    s.lock('matching', 'TimeShower:recoilDeadCone', True,
           'gluon recoil weight included in the MC counterterms', reason=why)
    s.lock('matching', 'SpaceShower:rapidityOrder', False, reason=why)
    s.lock('matching', 'SpaceShower:dipoleRecoil', False, reason=why)
    if contract['mcatnlo_delta']:
        why_delta = 'MC@NLO-Delta reads the dipole starting scales from the LHE file'
        s.lock('matching', 'Beams:setProductionScalesFromLHEF', True, reason=why_delta)
        s.lock('matching', 'Beams:setDipoleShowerStartingScalesFromLHEF', True,
               reason=why_delta)
    elif contract['ickkw'] != 3:
        s.lock('matching', 'Beams:setProductionScalesFromLHEF', False, reason=why)

    # --- matrix-element corrections (arXiv:2308.06389) --------------------
    why_mec = ('production MECs would double count the NLO real emission '
               '(arXiv:2308.06389)')
    s.lock('mec', 'SpaceShower:MEcorrections', False, reason=why_mec)
    s.lock('mec', 'TimeShower:MEextended', False, reason=why_mec)
    decays = contract.nlo_decays_with_mec_risk()
    if decays:
        s.lock('mec', 'TimeShower:MEcorrections', False,
               'the decays of %s are part of the NLO calculation'
               % ', '.join(str(d) for d in decays),
               reason='MECs in decays that are corrected at NLO double count')
        if card['time_shower_me_corrections']:
            notes.append('Matrix-element corrections in resonance decays are '
                         'switched off: the decays of %s are part of the NLO '
                         'calculation.' % ', '.join(str(d) for d in decays))
    else:
        s.free('mec', 'TimeShower:MEcorrections',
               bool(card['time_shower_me_corrections']),
               'MECs in resonance decays only (MEextended = off)')
    s.free('mec', 'TimeShower:MEafterFirst',
           bool(card['time_shower_me_after_first']))

    # --- QED -------------------------------------------------------------
    qed = bool(card['qed_shower'])
    if contract['qed_matching']:
        why_qed = 'the NLO QED corrections are matched to the QED shower'
        for key in ('TimeShower:QEDshowerByQ', 'TimeShower:QEDshowerByL',
                    'TimeShower:QEDshowerByGamma', 'TimeShower:QEDshowerByOther',
                    'SpaceShower:QEDshowerByQ', 'SpaceShower:QEDshowerByL'):
            s.lock('qed', key, True, reason=why_qed)
        s.lock('qed', 'TimeShower:mMaxGamma', contract['pythia8_mmaxgamma'],
               'photon-conversion bound of the MC subtraction (FKS_params.dat)',
               reason=why_qed)
    else:
        # qed_shower switches off all photon emissions and splittings:
        # quarks (top included), leptons, W bosons and other charged
        # resonances (QEDshowerByOther), and gamma -> f fbar.
        for key in ('TimeShower:QEDshowerByQ', 'TimeShower:QEDshowerByL',
                    'TimeShower:QEDshowerByOther', 'TimeShower:QEDshowerByGamma',
                    'SpaceShower:QEDshowerByQ', 'SpaceShower:QEDshowerByL'):
            s.free('qed', key, qed, 'qed_shower in the shower card')
        s.free('qed', 'TimeShower:mMaxGamma', contract['pythia8_mmaxgamma'],
               'as in the MC subtraction (FKS_params.dat)')

    # --- FxFx ------------------------------------------------------------
    if contract['ickkw'] == 3:
        why_fxfx = 'FxFx merging, as in the short-distance generation'
        s.lock('merging', 'JetMatching:doFxFx', True, reason=why_fxfx)
        s.lock('merging', 'JetMatching:merge', True, reason=why_fxfx)
        s.lock('merging', 'JetMatching:qCutME', contract['ptj'],
               'ptj of the run card', reason=why_fxfx)
        s.lock('merging', 'JetMatching:coneRadius', 1.0, reason=why_fxfx)
        s.lock('merging', 'JetMatching:etaJetMax', 1000.0, reason=why_fxfx)
        s.lock('merging', 'JetMatching:scheme', 1, reason=why_fxfx)
        s.lock('merging', 'JetMatching:setMad', False, reason=why_fxfx)
        s.lock('merging', 'Beams:setProductionScalesFromLHEF', True,
               'identifies the weak jets', reason=why_fxfx)
        qcut = float(card['qcut'])
        if qcut <= float(contract['ptj']):
            raise ShowerInterfaceError(
                'FxFx merging: Qcut in the shower card (%s) must be larger than '
                'ptj in the run card (%s).' % (card['qcut'], contract['ptj']))
        s.free('merging', 'JetMatching:qCut', qcut, 'merging scale (Qcut)')
        s.free('merging', 'JetMatching:nJetMax', int(card['njmax']),
               'maximal multiplicity (njmax)')
    else:
        s.lock('merging', 'JetMatching:doFxFx', False, reason='no FxFx merging')

    # --- couplings: free -------------------------------------------------
    for shower in ('TimeShower', 'SpaceShower'):
        s.free('couplings', '%s:alphaSvalue' % shower, 0.118)
        s.free('couplings', '%s:alphaSorder' % shower, 1)
        s.free('couplings', '%s:alphaSuseCMW' % shower, False)
        s.free('couplings', '%s:alphaEMorder' % shower, 0)

    # --- PDFs: free ------------------------------------------------------
    pdf_kind, pdf_name, extrapolate = run['pdf']
    if pdf_kind == 'lhapdf6':
        s.free('pdf', 'PDF:pSet', 'LHAPDF6:%s' % pdf_name)
        s.free('pdf', 'PDF:extrapolate', bool(extrapolate))
    else:
        s.free('pdf', 'PDF:pSet', 13, 'internal NNPDF2.3 QCD+QED LO')

    # --- masses ----------------------------------------------------------
    why_mass = 'the events were put on these mass shells'
    for pdg, mass in sorted(run['mc_masses'].items()):
        if pdg == 21:
            continue
        s.lock('masses', '%d:m0' % pdg, float(mass), reason=why_mass)
    for pdg, mass in sorted(run.get('masses', {}).items()):
        s.lock('masses', '%d:m0' % pdg, float(mass),
               reason='as in the short-distance calculation')
    for pdg, width in sorted(run.get('widths', {}).items()):
        s.free('masses', '%d:mWidth' % pdg, float(width))
    if float(card['b_mass']) >= 0. and 5 in run['mc_masses'] and \
            abs(float(card['b_mass']) - float(run['mc_masses'][5])) > 1e-9:
        raise ShowerInterfaceError(
            'b_mass = %s in the shower card differs from the b mass %s used to '
            'put the events on the shower mass shell (MCmasses_PYTHIA8.inc). '
            'To use it anyway, set b_mass = -1 and pass {\'5:m0\': %s} in '
            'pythia8_options with allow_unsafe_shower_options = T.'
            % (card['b_mass'], run['mc_masses'][5], card['b_mass']))

    # --- non-perturbative ------------------------------------------------
    s.free('nonperturbative', 'PartonLevel:MPI', bool(card['ue_enabled']),
           'underlying event (ue_enabled)')
    s.free('nonperturbative', 'HadronLevel:Hadronize', bool(card['hadronize']))
    s.free('nonperturbative', 'BeamRemnants:primordialKT', bool(card['primordialkt']))

    # --- decays and stable particles -------------------------------------
    stable = [('pi_stable', [111]),
              ('b_stable', [511, 521, 531, 541, 551, 553, 5112, 5122, 5132,
                            5222, 5232, 5332]),
              ('wp_stable', [24]), ('wm_stable', [24]), ('z_stable', [23]),
              ('h_stable', [25]), ('tap_stable', [-15]), ('tam_stable', [15]),
              ('mup_stable', [-13]), ('mum_stable', [13])]
    done = set()
    for name, pdgs in stable:
        if card[name]:
            for pdg in pdgs:
                if pdg not in done:
                    done.add(pdg)
                    s.free('decays', '%d:mayDecay' % pdg, False, name)
    for i in range(1, 100):
        line = card['dm_%d' % i]
        if line:
            s.extra_lines.append(str(line).strip())

    # --- random numbers --------------------------------------------------
    s.free('random', 'Random:setSeed', True)
    s.free('random', 'Random:seed', int(run['seed']),
           'offset for each split job')

    # --- gating on the installed version ---------------------------------
    missing = [x for x in s.locked() if not has(x.key)]
    if missing:
        raise ShowerInterfaceError(
            'Pythia8 %s in %s does not provide settings required for the '
            'MC@NLO matching:\n  %s\nPlease use a more recent Pythia8 version.'
            % (installation.version, installation.path,
               '\n  '.join(m.key for m in missing)))
    for section in s.sections:
        for setting in list(s.entries[section]):
            if setting.tier != LOCKED and not has(setting.key):
                # A default that this version does not know: drop it.
                s.entries[section].remove(setting)
                del s.index[setting.key.lower()]
                notes.append('Pythia8 %s does not know %s; not set.'
                             % (installation.version, setting.key))
    if pdf_kind == 'lhapdf6' and not installation.has_lhapdf6() and \
            run.get('pdf_fallback'):
        s.free('pdf', 'PDF:pSet', 13, 'internal NNPDF2.3 QCD+QED LO')
        s.entries['pdf'] = [x for x in s.entries['pdf'] if x.key != 'PDF:extrapolate']
        s.index.pop('pdf:extrapolate', None)
        notes.append('Pythia8 %s was not configured with LHAPDF6: the shower '
                     'uses its internal PDF instead of %s.'
                     % (installation.version, pdf_name))
    elif pdf_kind == 'lhapdf6' and not installation.has_lhapdf6():
        raise ShowerInterfaceError(
            'The shower PDF %s needs the LHAPDF6 interface of Pythia8, but '
            '%s was not configured with LHAPDF6. Reconfigure Pythia8 with '
            '--with-lhapdf6, or set pdfcode = 0 in the shower card (internal '
            'Pythia8 PDF).' % (pdf_name, installation.path))

    # --- user options ----------------------------------------------------
    from_card = []
    if card['space_shower_me_corrections']:
        from_card.append(('SpaceShower:MEcorrections', 'on'))
    if card['time_shower_me_extended']:
        from_card.append(('TimeShower:MEextended', 'on'))
    if contract['qed_matching'] and not qed:
        notes.append('qed_shower = F is ignored: the NLO QED corrections are '
                     'matched to the QED shower, which is therefore on.')
    options = dict(from_card)
    options.update(card['pythia8_options'] or {})
    s.apply_user_options(options, allow_unsafe=bool(
        card['allow_unsafe_shower_options']))
    for setting in s.overridden:
        notes.append('UNSAFE: %s = %s overrides a setting required by the '
                     'MC@NLO matching.' % (setting.key, setting.value))
    for setting in s.user:
        if not has(setting.key) and not re.match(r'^-?\d+:', setting.key):
            raise ShowerInterfaceError(
                'pythia8_options: %s is not a setting of Pythia8 %s.'
                % (setting.key, installation.version))
    s.notes = notes
    return s


################################################################################
# Building the Pythia8 driver
################################################################################

def _hash_tree(digest, directory, suffixes):
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        path = pjoin(directory, name)
        if os.path.isfile(path) and name.endswith(suffixes):
            digest.update(name.encode())
            with open(path, 'rb') as stream:
                digest.update(stream.read())


def fortran_compiler(me_dir, default='gfortran'):
    """The Fortran compiler configured for the process directory."""
    path = pjoin(me_dir, 'Source', 'make_opts')
    if os.path.isfile(path):
        for line in open(path):
            if line.startswith('DEFAULT_F_COMPILER'):
                value = line.split('=', 1)[1].strip()
                if value:
                    return value
    return default


def build_pythia8_driver(mcatnlo_dir, installation, analysis=(), hepmc=(0, ''),
                         fc='gfortran', extra_cxxflags='', extra_ldflags='',
                         log=None, jobs=1):
    """Build (or reuse) the Pythia8 driver and return its path.

    Executables are cached in MCatNLO/build/PYTHIA8_<hash>, the hash covering
    the sources, the analysis, the Pythia8 installation and the flags, so that
    showering the same configuration again does not recompile anything."""
    import subprocess
    import shutil
    analysis = [a for a in analysis if a]
    if not analysis and not hepmc[0]:
        raise ShowerInterfaceError(
            'Pythia8 events can only be written in HepMC format, but %s was '
            'not configured with HepMC (pythia8-config --with-hepmc3 / '
            '--with-hepmc2). Reconfigure Pythia8 with HepMC, set hepmc_path in '
            'the MG5_aMC configuration, or give an analysis (ANALYSE) in the '
            'shower card.' % installation.path)
    if analysis:
        hepmc = (0, '')   # the analysis reads the event record directly
    cxx = installation.cxx_compiler()
    config = [installation.path, installation.version, cxx, fc,
              str(hepmc[0]), hepmc[1], ' '.join(analysis), extra_cxxflags,
              extra_ldflags, installation.run_config('--cxxflags'),
              installation.run_config('--libs')]
    digest = hashlib.sha1('\n'.join(config).encode())
    for sub, suffixes in [('pythia8', ('.cc', '.h', 'Makefile')),
                          ('include', ('.h', '.inc', '.INC', '.hh')),
                          ('PY8Analyzer', ('.f', '.cc', '.inc')),
                          ('srcCommon', ('.f', '.cc'))]:
        _hash_tree(digest, pjoin(mcatnlo_dir, sub), suffixes)
    builddir = pjoin(mcatnlo_dir, 'build', 'PYTHIA8_%s' % digest.hexdigest()[:12])
    exe = pjoin(builddir, 'Pythia8.exe')
    if os.path.isfile(exe) and os.access(exe, os.X_OK):
        logger.info('Using the Pythia8 driver compiled previously (%s)'
                    % os.path.relpath(builddir, mcatnlo_dir))
        return exe

    tmpdir = builddir + '.tmp'
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    command = ['make', '-f', pjoin(mcatnlo_dir, 'pythia8', 'Makefile'),
               'PYTHIA8_CONFIG=%s' % installation.config,
               'CXX=%s' % cxx, 'FC=%s' % fc,
               'HEPMC=%d' % hepmc[0], 'HEPMC_FLAGS=%s' % hepmc[1],
               'ANALYSIS=%s' % ' '.join(analysis),
               'EXTRA_CXXFLAGS=%s' % extra_cxxflags,
               'EXTRA_LDFLAGS=%s' % extra_ldflags]
    if jobs and int(jobs) > 1:
        command.insert(1, '-j%d' % int(jobs))
    with open(log, 'a') if log else open(os.devnull, 'w') as output:
        output.write('Building the Pythia8 driver in %s\n%s\n'
                     % (builddir, ' '.join("'%s'" % c for c in command)))
        output.flush()
        status = subprocess.call(command, cwd=tmpdir, stdout=output,
                                 stderr=subprocess.STDOUT)
    if status or not os.path.isfile(pjoin(tmpdir, 'Pythia8.exe')):
        raise ShowerInterfaceError(
            'Compilation of the Pythia8 driver failed. See %s.' % log)
    if os.path.isdir(builddir):
        shutil.rmtree(builddir)
    os.rename(tmpdir, builddir)
    return exe


def lhapdf_set_name(lhapdf_config, lhaid):
    """Pythia8 name of the LHAPDF6 set member lhaid: 'set' for the central
    member, 'set/member' otherwise. None if the set is unknown.

    In pdfsets.index each line gives the id of the first member of a set
    (the third column is the data version, not the number of members)."""
    import subprocess
    try:
        datadir = subprocess.check_output([lhapdf_config, '--datadir'],
                                          stderr=subprocess.DEVNULL)
        datadir = datadir.decode(errors='ignore').strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    index = pjoin(datadir, 'pdfsets.index')
    if not os.path.isfile(index):
        return None
    best = None
    for line in open(index):
        data = line.split()
        if len(data) < 2:
            continue
        try:
            first = int(data[0])
        except ValueError:
            continue
        if first <= lhaid and (best is None or first > best[0]):
            best = (first, data[1])
    if best is None:
        return None
    member = lhaid - best[0]
    # The number of members, when the set is installed.
    for directory in [datadir] + os.environ.get('LHAPDF_DATA_PATH', '').split(':'):
        info = pjoin(directory, best[1], '%s.info' % best[1])
        if directory and os.path.isfile(info):
            match = re.search(r'^NumMembers:\s*(\d+)', open(info).read(), re.M)
            if match and member >= int(match.group(1)):
                return None
            break
    return best[1] if member == 0 else '%s/%d' % (best[1], member)


def check_shower_log(path):
    """Interpret the summary lines written by the Pythia8 driver."""
    info = {'contract': None, 'showered': None, 'failed': None}
    if not os.path.isfile(path):
        return info
    with open(path, errors='replace') as stream:
        for line in stream:
            if line.startswith('MG5_MCATNLO_CONTRACT OK'):
                info['contract'] = True
            elif line.startswith('MG5_MCATNLO_CONTRACT FAILED'):
                info['contract'] = False
            elif line.startswith('MG5_MCATNLO_SUMMARY'):
                for item in line.split()[1:]:
                    if '=' in item:
                        name, value = item.split('=', 1)
                        info[name] = value
    return info
