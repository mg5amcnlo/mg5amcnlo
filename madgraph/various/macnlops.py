"""First-emission matching from a common, signed MC@NLO input.

Ordinary MC@NLO retains its counterterms and bogus-Sudakov damping. Sampled
fixed-flow sources carry the native sector, Born diagram and colour flow.
Their Delta veto uses the complete undamped local H, including the G term,
over the S density beta*K_sh: the channel's sigma_S/sigma_B times the
leading kernel at the shower's own emission coupling. Legacy globally summed
Delta sources still use conditional damping replay.
Any part that cannot be transferred with a bounded probability remains a
signed H residual. The scalar/gated source is an explicit legacy option.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass
import gzip
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading

try:
    from . import collect_events, macnlops_components
except ImportError:  # Direct invocation of an exported bin/internal/macnlops.py.
    import collect_events
    import macnlops_components


class MatchingError(RuntimeError):
    """A sample cannot satisfy the matching contract."""


ORDINARY_SOURCE = 'ordinary-mcatnlo-v1'
RESOLVED_SOURCE = 'resolved-convex-v1'
DELTA_SOURCE = 'mcatnlo-delta-v1'
SOURCE_NAMES = {ORDINARY_SOURCE: 'ordinary', RESOLVED_SOURCE: 'resolved',
                DELTA_SOURCE: 'delta'}
SOURCE_CONSTRUCTIONS = {name: construction for construction, name in SOURCE_NAMES.items()}
# v2 normalises the S veto to beta*K_sh; v1 divided by the bare K.
UNDAMPED_DELTA_POLICY = 'component-undamped-delta-v2'
LEGACY_UNDAMPED_DELTA_POLICY = 'component-undamped-delta-v1'
HARD_METADATA_FIELDS = (
    'nup', 'kind', 'owner', 'native_sector', 'native_provider',
    'native_context', 'native_history', 'born_flow', 'mc_kl_sum',
    'fixed_flow', 'fks_sum', 'flavour_coalesced', 'max_bcol', 'group_size')
_model_mass_cache = {}
_model_mass_lock = threading.Lock()
_CHECKSUM_CACHE_LIMIT = 4096
_checksum_cache = OrderedDict()
_checksum_cache_lock = threading.Lock()


def hard_model_masses(process_dir):
    """Resolve dependent masses from the actual exported, restricted UFO.

    A MASS entry for a dependent parameter can be rounded or inactive. The
    native Fortran model and shower must instead share its evaluated value.
    UFO imports change module state, so serialize the first import per model.
    """
    process_dir = Path(process_dir).resolve()
    model_dir = process_dir / 'bin/internal/ufomodel'
    card = process_dir / 'Cards/param_card.dat'
    if not model_dir.is_dir():
        raise MatchingError('Missing exported UFO for the hard-model mass contract: %s' % model_dir)
    key = (str(model_dir), sha256(card), sha256(model_dir / 'parameters.py'),
           sha256(model_dir / 'particles.py'))
    with _model_mass_lock:
        if key not in _model_mass_cache:
            # Different outputs all name their UFO "ufomodel". Importing a
            # second one in the same interpreter aliases its Python modules.
            script = '''import json, math, sys
request = json.load(sys.stdin)
sys.path[:] = request['paths']
from models import import_ufo, model_reader
model = model_reader.ModelReader(import_ufo.import_model(request['model']))
model.set_parameters_and_couplings(param_card=request['card'])
masses = {}
for particle in model['particles']:
    value = complex(model['parameter_dict'][particle['mass']])
    if not (math.isfinite(value.real) and math.isfinite(value.imag)) or abs(value.imag) > 1.e-12 * max(1., abs(value.real)):
        raise ValueError('Invalid hard-model mass for PDG %d' % particle['pdg_code'])
    masses[abs(particle['pdg_code'])] = abs(value.real)
print('HARD_MODEL_MASSES_V1 ' + json.dumps(masses))
'''
            search_paths = list(sys.path)
            configuration = process_dir / 'Cards/amcatnlo_configuration.txt'
            if configuration.is_file():
                match = re.search(r'^\s*mg5_path\s*=\s*([^#\n]+)', configuration.read_text(), re.M)
                if match and (Path(match[1].strip()) / 'models/model_reader.py').is_file():
                    search_paths.insert(0, match[1].strip())
            result = subprocess.run([sys.executable, '-c', script],
                input=json.dumps(dict(paths=search_paths, model=str(model_dir), card=str(card))),
                capture_output=True, text=True)
            records = [line for line in result.stdout.splitlines()
                       if line.startswith('HARD_MODEL_MASSES_V1 ')]
            if result.returncode or len(records) != 1:
                raise MatchingError('Cannot resolve the exported hard-model masses: ' +
                                    result.stderr[-3000:])
            masses = json.loads(records[0].split(' ', 1)[1])
            _model_mass_cache[key] = {int(pdg): mass for pdg, mass in masses.items()}
        return dict(_model_mass_cache[key])


def source_construction(card):
    """Keep source physics separate from common-input preparation."""
    # RunCardNLO's get is an alias of __getitem__, rather than dict.get.
    def flag(key):
        value = card[key] if key in card else False
        if isinstance(value, str):
            value = value.strip().lower()
            if value not in ('true', '.true.', 't', 'false', '.false.', 'f'):
                raise MatchingError('Invalid %s value' % key)
            value = value in ('true', '.true.', 't')
        if type(value) is not bool:
            raise MatchingError('Invalid %s value' % key)
        return value
    resolved, delta = flag('macnlops_resolved_source'), flag('mcatnlo_delta')
    if resolved and delta:
        raise MatchingError('MC@NLO-Delta requires macnlops_resolved_source=False')
    return DELTA_SOURCE if delta else RESOLVED_SOURCE if resolved else ORDINARY_SOURCE


def hard_source_name(construction):
    try:
        return SOURCE_NAMES[construction]
    except KeyError as exc:
        raise MatchingError('Unknown MAcNLOPS source construction') from exc


def matching_prescription(construction, partition):
    if partition == 'sampled':
        if construction == RESOLVED_SOURCE:
            raise MatchingError('The legacy resolved source requires a global H sum')
        return (UNDAMPED_DELTA_POLICY if construction == DELTA_SOURCE
                else 'component-source-v1')
    if partition == 'global_sum':
        return 'global-source-v1'
    raise MatchingError('Unknown native source partition')


@dataclass(frozen=True)
class Densities:
    """R, the effective transfer R-H, and the physical leading kernel K_g.

    In global Delta mode, subtraction includes the compensating real term.
    In sampled Delta mode it is the undamped local G-completed counterterm.
    kernel_alpha_s is the coupling of K's emission vertex, when reported.
    """
    real: float
    subtraction: float
    shower: float
    kernel_alpha_s: float = None

    def transfers(self):
        """Whether a negative hard density meets positive shower support."""
        return self.subtraction > self.real and self.shower > 0

    def acceptance(self, signed_residuals=False, kernel_scale=1.):
        """Bounded S acceptance, using only positive signed shower support.

        Signed-residual mode admits signed PDF luminosities. A nonpositive
        K transfers no negative H measure; no absolute-valued veto is used.
        There is no Sudakov in the denominator. kernel_scale rescales only
        the S denominator, to the S events' actual normalisation and shower
        coupling; the transfer min(u,K) removed from H is unchanged.
        """
        if not (math.isfinite(kernel_scale) and kernel_scale > 0):
            raise MatchingError('Invalid S-veto kernel normalisation')
        if not all(math.isfinite(x) for x in
                   (self.real, self.subtraction, self.shower)):
            raise MatchingError('Nonfinite real, subtraction or shower density')
        if not signed_residuals and (self.real < 0 or self.shower < 0):
            raise MatchingError('Real and shower densities must be nonnegative')
        if not math.isfinite(self.subtraction - self.real):
            raise MatchingError('Nonfinite reconstructed hard density')
        negative_hard = max(self.subtraction - self.real, 0.0)
        if negative_hard == 0:
            return 1.0
        if self.shower <= 0:
            # A signed PDF luminosity can make K negative. Rejecting that S
            # measure cannot compensate removal of a negative H measure.
            # Leave both unchanged, just as when no shower support exists.
            if signed_residuals:
                return 1.0
            raise MatchingError('Negative hard density outside shower support')
        probability = negative_hard / self.shower
        # R=0,D=K can give 1+O(eps) after independent history summation.
        # Only machine-roundoff overshoots are identified with that endpoint.
        if probability > 1.0 + 32 * math.ulp(1.0) and not signed_residuals:
            raise MatchingError('Negative hard density exceeds shower density: '
                                'H-/K = %.17g' % probability)
        # min(u/K,1) = min(u,K)/K; a scaled denominator can still saturate.
        probability = min(min(probability, 1.0) / kernel_scale, 1.0)
        return 1.0 - probability

    def hard_acceptance(self, signed_residuals=False):
        """Keep the compensating signed H residual when cancellation is capped."""
        self.acceptance(signed_residuals=signed_residuals)
        negative_hard = max(self.subtraction - self.real, 0.0)
        if negative_hard == 0:
            raise MatchingError('Negative H event disagrees with reconstructed hard density')
        if signed_residuals:
            return max(1.0 - max(self.shower, 0.0) / negative_hard, 0.0)
        return 0.0


def shower_kernel_scale(beta, record, density):
    """beta*alpha_s^PS/alpha_s^K for one eligible emitted S proposal.

    The eligible S density is approximately beta*Delta_PS*K_sh, where beta is
    the channel's sigma_S/sigma_B and K_sh replaces K's emission coupling by
    the shower's at this emission. With the Delta source the H removal is
    Delta_src*min(u,K), so the S veto min(u,K)/(beta*K_sh) balances it where
    the source and shower Sudakovs agree. Both factors are 1+O(alpha_s).
    """
    alpha_shower, alpha_kernel = record.get('alpha_s_emission'), density.kernel_alpha_s
    if alpha_shower is None or alpha_kernel is None:
        raise MatchingError('The S veto needs the shower and kernel emission couplings; '
                            'rebuild the matching driver and evaluator')
    values = (beta, alpha_shower, alpha_kernel)
    if not all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values):
        raise MatchingError('Invalid S-veto normalisation: beta %r, alpha_s %r over %r' % values)
    return beta * alpha_shower / alpha_kernel


def channel_normalisation(filename):
    """sigma_S/sigma_B of the integration channel that generated an event file.

    MINT writes both integrals to the channel's res_1, which every split
    event-generation job of that channel links.
    """
    path = Path(filename).parent / 'res_1'
    values = {}
    try:
        for line in path.read_text().splitlines():
            match = re.match(r'\s*Final result \[(S|Born)\]:\s*(\S+)\s+\+/-\s+(\S+)', line)
            if match:
                values[match[1]] = (number(match[2]), number(match[3]))
    except (OSError, ValueError) as exc:
        raise MatchingError('Cannot read the channel integrals in %s' % path) from exc
    if set(values) != {'S', 'Born'}:
        raise MatchingError('%s lacks the channel S and Born integrals; regenerate the '
                            'source with the current integrator' % path)
    (sigma_s, error_s), (sigma_b, error_b) = values['S'], values['Born']
    if not all(math.isfinite(x) for x in (sigma_s, error_s, sigma_b, error_b)) or \
            sigma_s <= 0 or sigma_b <= 0:
        raise MatchingError('Nonpositive channel S or Born cross section in %s' % path)
    return dict(path=str(path), sigma_S=sigma_s, sigma_S_error=error_s,
                sigma_B=sigma_b, sigma_B_error=error_b, beta=sigma_s / sigma_b)


def open_lhe(path, mode='rt'):
    return gzip.open(path, mode) if str(path).endswith('.gz') else open(path, mode)


class LHEStream:
    """Stream event blocks verbatim, including event attributes and extra XML."""

    def __init__(self, path):
        self.file = open_lhe(path)
        self.header = ''
        self.pending = None
        for line in self.file:
            if re.match(r'\s*<event(?:\s|>)', line):
                self.pending = line
                break
            if re.match(r'\s*</LesHouchesEvents>', line):
                break
            self.header += line
        if '<init>' not in self.header:
            self.file.close()
            raise MatchingError('Missing LHE init block in %s' % path)

    def __iter__(self):
        while self.pending is not None:
            lines = [self.pending]
            self.pending = None
            for line in self.file:
                lines.append(line)
                if re.match(r'\s*</event>', line):
                    break
            else:
                raise MatchingError('Truncated LHE event')
            yield ''.join(lines)
            for line in self.file:
                if re.match(r'\s*<event(?:\s|>)', line):
                    self.pending = line
                    break
                if re.match(r'\s*</LesHouchesEvents>', line):
                    return

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def xml_attributes(text, context='continuation XML'):
    """Parse producer-owned attributes strictly, including duplicate checks."""
    attributes, rest = {}, text
    pattern = re.compile(r'''\s*([a-z_][a-z_0-9]*)\s*=\s*(["'])(.*?)\2''', re.S)
    while rest.strip():
        item = pattern.match(rest)
        if item is None or item[1] in attributes:
            raise MatchingError('Malformed or duplicate %s attribute' % context)
        attributes[item[1]] = item[3]
        rest = rest[item.end():]
    return attributes


def xml_packet(block, name, context):
    """Read one optional annotation, rejecting duplicate or incomplete tags."""
    tag = re.escape(name)
    starts = list(re.finditer(r'<%s\b' % tag, block))
    closes = list(re.finditer(r'</%s\s*>' % tag, block))
    if not starts and not closes:
        return None
    match = re.search(r'<%s\b([^>]*)>(.*?)</%s\s*>' % (tag, tag), block, re.S)
    if len(starts) != 1 or len(closes) != 1 or match is None:
        raise MatchingError('Malformed or duplicate %s' % context)
    return xml_attributes(match[1], context), match[2]


@dataclass(frozen=True)
class HardEvent:
    """The native external point, stored before the shower-mass transport.

    This is source provenance even when the containing event is an emitted
    S endpoint. It must not replace the real endpoint in that case.
    """
    source: str
    partition: str
    metadata: tuple
    particles: tuple
    masses: tuple
    colours: tuple

    @classmethod
    def from_block(cls, block, kind):
        packet = xml_packet(block, 'macnlops_hard_event', 'hard-event annotation')
        if packet is None:
            return None
        attributes, contents = packet
        if (set(attributes) != {'version', 'frame', 'source', 'partition'} or
                attributes['version'] != '1' or attributes['frame'] != 'lab' or
                attributes['source'] not in SOURCE_NAMES.values() or
                attributes['partition'] not in ('global_sum', 'sampled')):
            raise MatchingError('Unsupported hard-event annotation contract')
        rows = contents.strip().splitlines()
        try:
            metadata = tuple(map(int, rows[0].split()))
            if len(metadata) != len(HARD_METADATA_FIELDS):
                raise ValueError('metadata field count')
            nlegs, hard_kind = metadata[:2]
            if (not 3 <= nlegs <= 500 or len(rows) != nlegs + 1 or
                    hard_kind != (1 if kind == 'S' else 2) or
                    any(x < 0 for i, x in enumerate(metadata)
                        if i >= 2 and i != 7) or
                    any(x not in (0, 1) for x in metadata[8:12]) or
                    metadata[12] < 1 or metadata[13] < 0):
                raise ValueError('metadata values')
            if (attributes['partition'] == 'global_sum' and
                    ((metadata[8], metadata[9], metadata[11]) != (1, 0, 1) or
                     metadata[13] < 1)):
                raise ValueError('global partition flags')
            particles, masses, colours = [], [], []
            for row in rows[1:]:
                fields = row.split()
                if len(fields) != 13:
                    raise ValueError('particle field count')
                ids = tuple(map(int, fields[:6]))
                values = tuple(map(number, fields[6:]))
                if (ids[0] == 0 or ids[1] not in (-1, 1) or
                        any(x < 0 for x in ids[2:]) or
                        any(x > nlegs for x in ids[2:4]) or
                        not all(math.isfinite(x) for x in values) or
                        values[3] <= 0 or values[4] < 0):
                    raise ValueError('particle values')
                particles.append((ids[0], ids[1],
                                  (values[3], values[0], values[1], values[2])))
                masses.append(values[4])
                colours.append(ids[4:6])
            if (tuple(status for _, status, _ in particles[:2]) != (-1, -1) or
                    any(status != 1 for _, status, _ in particles[2:])):
                raise ValueError('incoming particle count')
        except (IndexError, ValueError, OverflowError) as exc:
            raise MatchingError('Invalid hard-event annotation: %s' % exc) from exc
        return cls(attributes['source'], attributes['partition'], metadata,
                   tuple(particles), tuple(masses), tuple(colours))


@dataclass(frozen=True)
class Component:
    """A sampled native component; S sector is provenance, not its emitter."""
    registry: str
    sector: int
    config: int
    graph: int
    flow: int
    flow_probability: float
    insertion: float
    source_factor: float
    diagram_symmetry: float

    @classmethod
    def from_block(cls, block):
        packet = xml_packet(block, 'macnlops_component', 'sampled-component annotation')
        if packet is None:
            return None
        attributes, contents = packet
        if (set(attributes) != {'version', 'registry'} or attributes['version'] != '1' or
                re.fullmatch(r'[0-9a-f]{64}', attributes['registry']) is None):
            raise MatchingError('Invalid sampled-component registry or version')
        rows = contents.strip().splitlines()
        try:
            if len(rows) != 2:
                raise ValueError('row count')
            labels = tuple(map(int, rows[0].split()))
            values = tuple(map(number, rows[1].split()))
            if (len(labels) != 4 or min(labels) < 1 or len(values) != 4 or
                    not all(math.isfinite(value) for value in values) or
                    not 0 < values[0] <= 1 or not 0 <= values[1] <= 1 or
                    not 0 <= values[2] <= 1 or values[3] <= 0):
                raise ValueError('component labels or weights')
        except (ValueError, OverflowError) as exc:
            raise MatchingError('Invalid sampled component: %s' % exc) from exc
        return cls(attributes['registry'], *labels, *values)


@dataclass(frozen=True)
class DeltaContext:
    """The conditional source H damping, indexed by an immutable registry."""
    owner: int
    insertion: float
    registry: str
    probabilities: tuple

    @classmethod
    def from_block(cls, block):
        packet = xml_packet(block, 'macnlops_delta_context', 'Delta context')
        if packet is None:
            return None
        attributes, contents = packet
        try:
            if (set(attributes) != {'version', 'owner', 'insertion', 'registry'} or
                    attributes['version'] != '1' or
                    re.fullmatch('[0-9a-f]{64}', attributes['registry']) is None):
                raise ValueError('unsupported contract or registry')
            owner, insertion = int(attributes['owner']), number(attributes['insertion'])
            if owner < 1 or not math.isfinite(insertion) or not 0 <= insertion <= 1:
                raise ValueError('invalid owner or insertion variate')
            probabilities, keys = [], set()
            for row in contents.strip().splitlines():
                fields = row.split()
                if len(fields) != 4:
                    raise ValueError('invalid probability field count')
                key, probability = tuple(map(int, fields[:3])), number(fields[3])
                if (min(key) < 1 or key in keys or not math.isfinite(probability) or
                        not 0 <= probability <= 1):
                    raise ValueError('invalid or duplicate probability')
                keys.add(key)
                probabilities.append(key + (probability,))
            if len(probabilities) > 8192:
                raise ValueError('probability table exceeds the native replay capacity')
        except (KeyError, ValueError, OverflowError) as exc:
            raise MatchingError('Invalid Delta context: %s' % exc) from exc
        return cls(owner, insertion, attributes['registry'], tuple(probabilities))


@dataclass(frozen=True)
class Continuation:
    """Physical shower starts of the written first-emission endpoint."""
    state: str
    scale: float
    source_scalup: float
    cutoff_isr: float
    cutoff_fsr: float

    @classmethod
    def from_block(cls, block):
        starts = list(re.finditer(r'<macnlops_continuation\b', block))
        if not starts and not re.search(r'</macnlops_continuation\b', block):
            return None
        markers = list(re.finditer(r'<macnlops_continuation\b([^<>]*?)/>', block, re.S))
        if (len(starts) != 1 or len(markers) != 1 or
                re.search(r'</macnlops_continuation\b', block)):
            raise MatchingError('Duplicate or malformed continuation marker')
        attributes = xml_attributes(markers[0][1])
        if (set(attributes) != {'version', 'state', 'scale', 'source_scalup',
                               'cutoff_isr', 'cutoff_fsr'} or
                attributes['version'] != '1' or
                attributes['state'] not in ('emitted', 'exhausted', 'unshowered')):
            raise MatchingError('Unsupported continuation contract')
        try:
            values = tuple(number(attributes[key]) for key in
                           ('scale', 'source_scalup', 'cutoff_isr', 'cutoff_fsr'))
        except (ValueError, OverflowError) as exc:
            raise MatchingError('Invalid continuation scale') from exc
        if (not all(math.isfinite(x) for x in values) or values[0] <= 0 or
                (values[1] < 0 and values[1] != -1) or min(values[2:]) <= 0):
            raise MatchingError('Invalid continuation scale or cutoff')
        return cls(attributes['state'], *values)


@dataclass
class Event:
    block: str

    def __post_init__(self):
        lines = self.block.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if re.match(r'\s*<event(?:\s|>)', line)) + 1
        while not lines[index].strip() or lines[index].lstrip().startswith('#'):
            index += 1
        fields = lines[index].split()
        self.nlegs, self.process = map(int, fields[:2])
        self.weight, self.scale, self.aqed, self.aqcd = map(number, fields[2:6])
        if not all(math.isfinite(x) for x in
                   (self.weight, self.scale, self.aqed, self.aqcd)):
            raise MatchingError('Nonfinite LHE event header')
        self.particles = []
        self.particle_masses = []
        self.particle_colours = []
        for line in lines[index + 1:index + 1 + self.nlegs]:
            fields = line.split()
            if len(fields) != 13:
                raise MatchingError('Invalid LHE particle record')
            self.particles.append((int(fields[0]), int(fields[1]),
                                   tuple(number(fields[i]) for i in (9, 6, 7, 8))))
            mass = number(fields[10])
            if not math.isfinite(mass) or mass < 0:
                raise MatchingError('Invalid declared LHE particle mass')
            self.particle_masses.append(mass)
            self.particle_colours.append(tuple(map(int, fields[4:6])))
        tag = re.search(r'^\s*#aMCatNLO\s+([12])\s', self.block, re.M)
        if tag is None:
            raise MatchingError('Missing MC@NLO S/H event tag')
        self.kind = 'S' if tag[1] == '1' else 'H'
        self.hard = HardEvent.from_block(self.block, self.kind)
        self.component = Component.from_block(self.block)
        if self.component is not None:
            if (self.hard is None or self.hard.partition != 'sampled' or
                    self.hard.metadata[8:10] != (0, 1)):
                raise MatchingError('Component packet requires a native fixed-flow source')
            if self.hard.source == 'resolved':
                raise MatchingError('The legacy resolved source requires a global H sum')
            if (self.component.sector != self.hard.metadata[2] or
                    self.component.flow != abs(self.hard.metadata[7])):
                raise MatchingError('Component packet disagrees with source owner or colour flow')
            if (self.hard.metadata[11] != 1 or self.hard.metadata[13] < 1):
                raise MatchingError('Sampled source requires a qualified complete flavour group')
            if self.kind == 'H' and self.hard.metadata[3] != self.component.sector:
                raise MatchingError('Sampled H requires a complete native flavour group')
            if self.kind == 'S' and (self.component.source_factor != 1. or
                                     self.component.insertion != 0.):
                raise MatchingError('S component packet must not pretend to be a source-H Delta context')
        self.delta = DeltaContext.from_block(self.block)
        if self.delta is not None:
            if self.hard is None or self.hard.source != 'delta':
                raise MatchingError('Delta context requires a Delta hard-event annotation')
            if self.delta.owner != self.hard.metadata[2]:
                raise MatchingError('Delta context owner disagrees with the source annotation')
        self.endpoint_origin = None
        endpoint_starts = list(re.finditer(r'<macnlops_endpoint\b', self.block))
        endpoint = list(re.finditer(r'<macnlops_endpoint\b[^>]*>', self.block))
        if endpoint_starts or re.search(r'</macnlops_endpoint\b', self.block):
            if (len(endpoint_starts) != 1 or len(endpoint) != 1 or
                    re.search(r'</macnlops_endpoint\b', self.block)):
                raise MatchingError('Duplicate or truncated endpoint annotation')
            match = re.fullmatch(r'<macnlops_endpoint\s+version="1"\s+'
                                 r'origin="(shower|hard_annotation)"\s+'
                                 r'stage="first"\s*/>', endpoint[0][0])
            if match is None or self.hard is None:
                raise MatchingError('Invalid endpoint annotation')
            self.endpoint_origin = match[1]
        self.continuation = Continuation.from_block(self.block)
        self.source_scale = (self.continuation.source_scalup if self.continuation
                             else self.scale)
        if self.continuation is not None:
            self.check_continuation()
        if self.hard is not None and self.endpoint_origin != 'shower':
            # Resonances in the ordinary LHE body are bookkeeping entries;
            # the annotation contains only native external ME legs.
            written = Counter((pdg, status) for pdg, status, _ in self.particles
                              if status in (-1, 1))
            hard = Counter((pdg, status) for pdg, status, _ in self.hard.particles)
            if written != hard:
                raise MatchingError('Hard-event flavours disagree with the source event')

    def require_continuation(self):
        """Return the contract already validated when this event was parsed."""
        if self.continuation is None:
            raise MatchingError('First-emission LHE lacks shower-continuation scales; '
                                'regenerate it with the updated writer')
        return self.continuation

    def check_continuation(self):
        continuation = self.require_continuation()
        # Standard Delta H sources use SCALUP=-1: their dipole matrix
        # supplies the physical starts. Keep that sentinel as provenance;
        # the written endpoint still has strictly positive current scales.
        if (continuation.source_scalup < 0 and
                (self.kind != 'H' or self.hard is None or self.hard.source != 'delta')):
            raise MatchingError('Negative source SCALUP requires a Delta H endpoint')
        if ((self.kind == 'H') != (continuation.state == 'unshowered') or
                not math.isclose(self.scale, continuation.scale, rel_tol=2.e-14)):
            raise MatchingError('Continuation state/SCALUP conflicts with the written endpoint')
        if (continuation.state == 'exhausted' and
                self.scale >= min(continuation.cutoff_isr, continuation.cutoff_fsr)):
            raise MatchingError('Exhausted S event can restart above its shower cutoff')
        starts = list(re.finditer(r'<scales\b', self.block))
        packets = list(re.finditer(r'<scales\b([^<>]*?)(?:/>|>\s*</scales\s*>)',
                                   self.block, re.S))
        if len(starts) != 1 or len(packets) != 1:
            raise MatchingError('Continuation requires one active current-event scales packet')
        attributes = xml_attributes(packets[0][1])
        legs = set()
        self.continuation_leg_scales, self.continuation_dipole_scales = {}, {}
        for key, value in attributes.items():
            if key in ('muf', 'mur', 'mups'):
                continue
            match = re.fullmatch(r'scalup_([1-9][0-9]*)(?:_([1-9][0-9]*))?', key)
            try:
                scale = number(value)
            except (ValueError, OverflowError) as exc:
                raise MatchingError('Invalid active continuation scale') from exc
            zero_source_pair = (match is not None and match[2] is not None and
                                continuation.state == 'unshowered' and scale == 0)
            if (match is None or not math.isfinite(scale) or (scale <= 0 and not zero_source_pair) or
                    scale > self.scale * (1 + 2.e-14)):
                raise MatchingError('Active continuation scale exceeds its ceiling or is invalid')
            indices = [int(index) for index in match.groups() if index is not None]
            if (max(indices) > self.nlegs or len(set(indices)) != len(indices) or
                    any(self.particles[index-1][1] not in (-1, 1) for index in indices)):
                raise MatchingError('Continuation scale refers to an invalid current-event leg')
            if len(indices) == 1:
                legs.add(indices[0])
                self.continuation_leg_scales[indices[0]] = scale
            else:
                self.continuation_dipole_scales[tuple(indices)] = scale
        external = {index for index, (_, status, _) in enumerate(self.particles, 1)
                    if status in (-1, 1)}
        if legs != external:
            raise MatchingError('Continuation lacks a starting scale for a current external leg')

    def check_proposal_metadata(self, record, require_continuation=False):
        """Check that the native first-step record describes this written event."""
        if record['event_kind'] != self.kind:
            raise MatchingError('One-step shower S/H tag mismatch')
        if 'source_weight' in record and record['source_weight'] != self.weight:
            raise MatchingError('One-step shower changed an input event weight')
        if self.component is not None:
            required = ('particles', 'source_hard_particles', 'source_particles',
                        'source_partition', 'source_weight', 'beam_xmax')
            if any(key not in record for key in required):
                raise MatchingError('Sampled proposal lacks complete source/endpoint metadata')
            self._check_record_particles(record['particles'], self.particles,
                self.particle_masses, self.particle_colours, 'written endpoint')
            self._check_record_particles(record['source_hard_particles'],
                self.hard.particles, self.hard.masses, self.hard.colours, 'source hard point')
            self._check_record_particles(record['source_particles'],
                self.hard.particles, self.hard.masses, self.hard.colours, 'showered source point')
            expected = dict(zip(HARD_METADATA_FIELDS, self.hard.metadata))
            expected.update(version=1, frame='lab', source=self.hard.source,
                            partition=self.hard.partition)
            partition = record['source_partition']
            if (not isinstance(partition, dict) or
                    any(partition.get(key) != value for key, value in expected.items()) or
                    any(type(partition.get(key)) is not int
                        for key in HARD_METADATA_FIELDS + ('version',))):
                raise MatchingError('One-step metadata disagrees with source partition')
        continuation = (self.require_continuation() if require_continuation
                        else self.continuation)
        if continuation is None:
            if self.component is not None:
                raise MatchingError('Sampled proposal lacks first-emission continuation metadata')
            return
        expected_state = ('unshowered' if self.kind == 'H' else
                          'emitted' if record['emitted'] else 'exhausted')
        if continuation.state != expected_state:
            raise MatchingError('Continuation state disagrees with native emission metadata')
        if (expected_state == 'emitted' and not math.isclose(
                self.scale, record['pt_evol'], rel_tol=2.e-14)):
            raise MatchingError('Continuation SCALUP differs from the first-emission scale')
        expected_metadata = dict(
            continuation_state=expected_state,
            continuation_scalup=self.scale,
            continuation_leg_scales=[list(item) for item in sorted(
                self.continuation_leg_scales.items())],
            continuation_dipole_scales=[list(pair) + [scale] for pair, scale in sorted(
                self.continuation_dipole_scales.items())],
            source_scalup=self.source_scale,
            shower_cutoffs=[continuation.cutoff_isr, continuation.cutoff_fsr])
        if self.component is not None and any(key not in record for key in expected_metadata):
            raise MatchingError('Sampled proposal lacks complete continuation metadata')
        for key, value in expected_metadata.items():
            if key in record and record[key] != value:
                raise MatchingError('Written continuation disagrees with native %s' % key)

    @staticmethod
    def _check_record_particles(recorded, particles, masses, colours, label):
        """Bind discrete labels and round-tripped numerical coordinates."""
        try:
            if not isinstance(recorded, list) or len(recorded) != len(particles):
                raise ValueError('particle count')
            for row, (pdg, status, momentum), mass, colour in zip(
                    recorded, particles, masses, colours):
                if (not isinstance(row, dict) or type(row.get('id')) is not int or
                        type(row.get('status')) is not int or
                        (row['id'], row['status']) != (pdg, status) or
                        not isinstance(row.get('colors'), (list, tuple)) or
                        tuple(row['colors']) != tuple(colour) or
                        any(type(value) is not int for value in row['colors'])):
                    raise ValueError('particle labels')
                point = row['p']
                if (len(point) != 4 or not all(math.isfinite(value) for value in point) or
                        not math.isfinite(row['mass'])):
                    raise ValueError('particle coordinates')
                scale = max(1., *(abs(value) for value in momentum))
                if (any(not math.isclose(value, other, rel_tol=5.e-14,
                                         abs_tol=5.e-14 * scale)
                        for value, other in zip(point, momentum)) or
                        not math.isclose(row['mass'], mass, rel_tol=5.e-14, abs_tol=5.e-14)):
                    raise ValueError('particle coordinates')
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MatchingError('One-step metadata disagrees with %s: %s' % (label, exc)) from exc

    def density_particles(self):
        if (self.hard is not None and self.kind == 'H' and
                self.endpoint_origin != 'shower'):
            return self.hard.particles, self.hard.masses
        return self.particles, self.particle_masses

    def external(self):
        particles, _ = self.density_particles()
        incoming = [(pdg, p) for pdg, status, p in particles if status == -1]
        final = [(pdg, p) for pdg, status, p in particles if status == 1]
        if len(incoming) != 2:
            raise MatchingError('Matching requires two incoming hard particles')
        return incoming + final


def check_hard_mass_coordinates(event, model_masses):
    """Do not mistake a shower-mass transport for hard-density coordinates.

    A supported inverse transport must run before this check. This check
    never projects or rewrites the supplied event.
    """
    particles, masses = event.density_particles()
    for (pdg, status, _), mass in zip(particles, masses):
        if status not in (-1, 1):
            continue
        model_mass = abs(model_masses.get(abs(pdg), 0.))
        if not math.isclose(mass, model_mass, rel_tol=1.e-10, abs_tol=1.e-10):
            raise MatchingError(
                'LHE PDG %d has shower mass %.17g, hard model mass %.17g; '
                'an inverse write-time transport and its density Jacobian are required '
                '(the input event has not been changed)' % (pdg, mass, model_mass))


def require_lhef_scales(header):
    """Stock Pythia imports event starting-scale attributes only in LHEF 2/3."""
    if re.search(r'''<LesHouchesEvents\b[^>]*\bversion\s*=\s*(["'])[23](?:\.0)?\1''',
                 header) is None:
        raise MatchingError('MAcNLOPS continuation scales require LHEF version 2 or 3')


def hard_source_contract(filename, construction):
    """Qualify every original source event before requesting any cancellation.

    Sampled and globally summed source signs require different evaluators.
    The qualified component packet selects the local construction explicitly.
    """
    expected = hard_source_name(construction)
    counts = Counter()
    partitions = set()
    with LHEStream(filename) as stream:
        require_lhef_scales(stream.header)
        for sequence, block in enumerate(stream, 1):
            event = Event(block)
            if event.hard is None:
                raise MatchingError('Source event %d lacks its pre-shuffle hard annotation' % sequence)
            if event.endpoint_origin is not None:
                raise MatchingError('Input event %d is already a shower endpoint' % sequence)
            if event.hard.source != expected:
                raise MatchingError('Source event %d has a different counterterm construction' % sequence)
            if event.hard.partition == 'sampled' and event.component is None:
                raise MatchingError('Source event %d has a sampled H partition; its '
                                    'qualified conditional component packet is required' % sequence)
            if (expected == 'delta' and event.kind == 'H' and event.delta is None and
                    event.hard.partition == 'global_sum'):
                raise MatchingError('Source H event %d lacks its conditional Delta replay context; '
                                    'generate a fresh annotated MC@NLO-Delta source' % sequence)
            counts[event.kind] += 1
            partitions.add(event.hard.partition)
    if not counts:
        raise MatchingError('The source contains no events')
    if len(partitions) != 1:
        raise MatchingError('Mixed sampled and globally summed source partitions')
    return dict(schema='macnlops-hard-source-contract-v1',
                source=expected, partition=partitions.pop(), counts=dict(counts))


def matched_lhe_header(header):
    """Recognize matched samples without reusing their source run-card flag."""
    matches = list(re.finditer(r'<macnlops\b([^>]*)>', header))
    if not matches:
        return False
    if len(matches) != 1:
        raise MatchingError('Duplicate MAcNLOPS sample header')
    attributes = xml_attributes(matches[0][1])
    if attributes.get('stage') != 'first-emission':
        raise MatchingError('Unsupported MAcNLOPS sample stage')
    if attributes.get('continuation') != 'lhe-v1':
        raise MatchingError('This older matched LHE lacks corrected continuation scales; '
                            'regenerate the first-emission sample')
    return True


def continuation_contract(filename):
    """Validate the hand-off before a separate normal shower consumes it."""
    counts, cutoffs = Counter(), None
    with LHEStream(filename) as stream:
        require_lhef_scales(stream.header)
        if not matched_lhe_header(stream.header):
            raise MatchingError('Continuation needs an accepted first-emission sample')
        profile = read_shower_profile(stream.header)
        if profile is None:
            raise MatchingError('This older matched LHE lacks its measured shower profile; '
                                'regenerate the first-emission sample')
        for block in stream:
            event = Event(block)
            continuation = event.require_continuation()
            current = continuation.cutoff_isr, continuation.cutoff_fsr
            if cutoffs is not None and current != cutoffs:
                raise MatchingError('Continuation shower cutoffs differ within the sample')
            cutoffs = current
            counts[event.kind] += 1
            counts[continuation.state] += 1
    if cutoffs is None:
        raise MatchingError('The accepted first-emission sample is empty')
    if list(cutoffs) != profile['support']['shower_cutoffs']:
        raise MatchingError('Continuation scales and recorded shower profile have different cutoffs')
    return dict(schema='macnlops-lhe-continuation-v1', events=counts['S'] + counts['H'],
                counts=dict(counts), shower_cutoffs=list(cutoffs), shower_profile=profile)


def shower_setting_key(key):
    """Physics settings to carry between stages; exclude paths, seeds and I/O."""
    return (key.startswith(('SpaceShower:', 'TimeShower:', 'PDF:')) or
            key in ('StandardModel:alphaSnfmax', 'PartonShowers:model'))


def continuation_physics(profile):
    """Render the measured first-step physics independently of current Cards.

    Pythia's settings dump rounds floating-point values. The validated profile
    supplies the exact coupling, masses and cutoffs; stage-specific recoil and
    scale-import choices are applied subsequently by the normal shower script.
    """
    try:
        if (profile['schema'] != 'macnlops-generation-profile-v1' or
                profile['construction'] not in (ORDINARY_SOURCE, RESOLVED_SOURCE, DELTA_SOURCE) or
                profile['shower_backend'] != 'stock-pythia8'):
            raise ValueError('profile identity')
        support = profile['support']
        settings = dict(profile['effective_shower_settings'])
        for key, value in settings.items():
            if (not shower_setting_key(key) or not isinstance(value, str) or
                    not value or any(char in value for char in '\r\n!#')):
                raise ValueError('physics setting')
        required = ('SpaceShower:alphaSvalue', 'SpaceShower:alphaSorder',
                    'TimeShower:alphaSvalue', 'TimeShower:alphaSorder',
                    'SpaceShower:nQuarkIn', 'TimeShower:nGluonToQuark',
                    'SpaceShower:pT0Ref', 'StandardModel:alphaSnfmax', 'PDF:pSet')
        if any(key not in settings for key in required):
            raise ValueError('incomplete effective settings')
        if (settings['PDF:pSet'] != profile['pdf'] or
                number(settings['SpaceShower:pT0Ref']) != 0.):
            raise ValueError('PDF or ISR regularization')
        alpha, order = profile['alpha_s_mz'], profile['alpha_s_order']
        if not math.isfinite(alpha) or alpha <= 0 or type(order) is not int or order not in range(4):
            raise ValueError('coupling')
        for family in ('SpaceShower', 'TimeShower'):
            # Check the rounded native dump before restoring the exact input.
            if (abs(number(settings[family + ':alphaSvalue']) - alpha) > 1.e-5 or
                    int(settings[family + ':alphaSorder']) != order):
                raise ValueError('effective coupling differs from generation')
            settings[family + ':alphaSvalue'] = '%.17e' % alpha
            settings[family + ':alphaSorder'] = str(order)
        for key, field in (('StandardModel:alphaSnfmax', 'alpha_s_nfmax'),
                           ('SpaceShower:nQuarkIn', 'n_quark_in'),
                           ('TimeShower:nGluonToQuark', 'n_gluon_to_quark')):
            if int(settings[key]) != support[field]:
                raise ValueError('flavour policy')
        cutoffs = support['shower_cutoffs']
        if len(cutoffs) != 2 or any(not math.isfinite(x) or x <= 0 for x in cutoffs):
            raise ValueError('shower cutoffs')
        settings['SpaceShower:pTmin'] = '%.17e' % cutoffs[0]
        settings['TimeShower:pTmin'] = '%.17e' % cutoffs[1]
        masses = support['particle_masses']
        if not masses or any(int(pdg) <= 0 or not math.isfinite(mass) or mass < 0
                             for pdg, mass in masses.items()):
            raise ValueError('particle masses')
        for key, value in support['qcd_only_off_switches'].items():
            if not re.fullmatch(r'[A-Za-z0-9]+:[A-Za-z0-9]+', key) or value is not False:
                raise ValueError('parton-level profile')
            settings[key] = 'off'
        if (not profile['shower_libraries'] or any(not re.fullmatch(r'[0-9a-f]{64}', value)
                for value in list(profile['shower_libraries'].values()) +
                [profile['pdf_info_sha256'], profile['pdf_member_sha256']])):
            raise ValueError('library/PDF fingerprints')
        backend = profile['lhapdf_library_sha256']
        if backend is not None and not re.fullmatch(r'[0-9a-f]{64}', backend):
            raise ValueError('LHAPDF backend fingerprint')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise MatchingError('Invalid recorded MAcNLOPS shower profile: %s' % exc) from exc
    # The native consumer uses forceParm: ordinary readString would clamp zero
    # to the stock XML minimum, introducing an ISR damping absent at stage one.
    del settings['SpaceShower:pT0Ref']
    lines = ['! Physics measured for the accepted MAcNLOPS sample.',
             '! SpaceShower:pT0Ref = 0 is restored by the native continuation consumer.']
    lines.extend('%s = %s' % item for item in sorted(settings.items()))
    lines.extend('%d:m0 = %.17e' % (int(pdg), mass)
                 for pdg, mass in sorted(masses.items(), key=lambda item: int(item[0])))
    return '\n'.join(lines) + '\n'


def lhapdf_backend_hash(environment):
    """Identify the shared backend selected by the native library search path."""
    for folder in environment.get('LD_LIBRARY_PATH', '').split(os.pathsep):
        if folder and (Path(folder) / 'libLHAPDF.so').is_file():
            return sha256(Path(folder) / 'libLHAPDF.so')
    return None


def recorded_shower_profile(profile, effective_settings, environment=None):
    """Make a portable physics packet from the validated run and native dump."""
    record = {key: value for key, value in profile.items() if key != 'native_build'}
    record['lhapdf_library_sha256'] = lhapdf_backend_hash(environment or {})
    record['effective_shower_settings'] = {}
    for line in Path(effective_settings).read_text().splitlines():
        if '=' not in line or line.lstrip().startswith(('!', '#')):
            continue
        key, value = (item.strip() for item in line.split('=', 1))
        if shower_setting_key(key):
            record['effective_shower_settings'][key] = value
    continuation_physics(record)
    return record


def with_shower_profile(header, profile):
    if '<macnlops_shower_profile' in header:
        raise MatchingError('Duplicate MAcNLOPS shower profile')
    continuation_physics(profile)
    payload = json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    packet = '<macnlops_shower_profile sha256="%s">%s</macnlops_shower_profile>\n' % (
        digest, html.escape(payload, quote=False))
    if '</header>' not in header:
        raise MatchingError('Missing LHE header')
    return header.replace('</header>', packet + '</header>', 1)


def read_shower_profile(header):
    packet = xml_packet(header, 'macnlops_shower_profile', 'MAcNLOPS shower profile')
    if packet is None:
        return None
    attributes, contents = packet
    payload = html.unescape(contents)
    if attributes.get('sha256') != hashlib.sha256(payload.encode('utf-8')).hexdigest():
        raise MatchingError('MAcNLOPS shower profile fingerprint mismatch')
    try:
        profile = json.loads(payload)
    except ValueError as exc:
        raise MatchingError('Invalid MAcNLOPS shower profile JSON') from exc
    continuation_physics(profile)
    return profile


def write_continuation_settings(process_dir, pythia8, filename, contract, output):
    """Check installed physics inputs, then restore the accepted file's profile."""
    profile = contract['shower_profile']
    if shower_library_hashes(pythia8) != profile['shower_libraries']:
        raise MatchingError('Continuation Pythia libraries differ from the first-emission run')
    with LHEStream(filename) as stream:
        source_card = re.search(r'<MGRunCard>(.*?)</MGRunCard>', stream.header, re.S | re.I)
    if source_card is None:
        raise MatchingError('Accepted LHE has no source run card for its PDF')
    pdf, info = pdf_name(Path(process_dir), run_card_values(source_card[1]), with_info=True)
    if (pdf != profile['pdf'] or sha256(info) != profile['pdf_info_sha256'] or
            pdf_member_hash(pdf, info) != profile['pdf_member_sha256']):
        raise MatchingError('Continuation PDF differs from the first-emission run')
    environment = shower_environment(pythia8, info)
    if lhapdf_backend_hash(environment) != profile['lhapdf_library_sha256']:
        raise MatchingError('Continuation LHAPDF library differs from the first-emission run')
    Path(output).write_text(continuation_physics(profile))
    return environment


class NativeEvaluator:
    """Route real points to their qualified global or sampled densities."""

    def __init__(self, process_dir, log_dir, environment=None, auxiliary_seed=1,
                 partition='global_sum'):
        if partition not in ('global_sum', 'sampled'):
            raise MatchingError('Unknown native source partition')
        self.partition = partition
        self.process_dir = Path(process_dir).resolve()
        self.log_dir = Path(log_dir)
        self.routes = []
        self.workers = {}
        self.configuration = None
        self.environment = environment
        self.auxiliary_seed = int(auxiliary_seed)
        self.registries = {}
        self.delta_law_certificates = {}
        self._route_cache = {}
        self._component_basis_cache = {}
        self.construction = source_construction(run_card_values(
            (self.process_dir / 'Cards/run_card.dat').read_text()))
        matching_prescription(self.construction, self.partition)
        self.model_masses = hard_model_masses(self.process_dir)
        for filename in sorted((self.process_dir / 'SubProcesses').glob('P*/leshouche_info.dat')):
            if self.construction == DELTA_SOURCE or self.partition == 'sampled':
                registry_file = filename.with_name('mc_histories.json')
                if not registry_file.is_file():
                    raise MatchingError('Missing Delta history registry; export a fresh process: %s' % registry_file)
                try:
                    record = json.loads(registry_file.read_text())
                    fingerprint = hashlib.sha256(json.dumps(record['registry'], sort_keys=True,
                        separators=(',', ':')).encode('utf-8')).hexdigest()
                    owners = [int(table['sector']) for table in record['registry']['tables']]
                    if (record.get('schema') != 'macnlops-history-registry-v1' or
                            record.get('sha256') != fingerprint or
                            type(record['registry']['flavour_complete']) is not bool or
                            any(owner < 1 for owner in owners) or
                            len(set(owners)) != len(owners)):
                        raise ValueError('registry identity or owners')
                    record['contexts_sha256'] = hashlib.sha256(json.dumps(
                        record['registry']['contexts'], sort_keys=True,
                        separators=(',', ':')).encode('utf-8')).hexdigest()
                except (KeyError, TypeError, ValueError) as exc:
                    raise MatchingError('Invalid Delta history registry: %s' % registry_file) from exc
                self.registries[filename.parent] = record
            for line in filename.read_text().splitlines():
                fields = line.split()
                if fields and fields[0] == 'I':
                    self.routes.append((filename.parent, int(fields[1]),
                                        tuple(map(int, fields[3:]))))
        if not self.routes:
            raise MatchingError('No generated real-flavour tables in %s' % process_dir)
        self._registry_routes = {}
        for directory, record in self.registries.items():
            self._registry_routes.setdefault(record['sha256'], []).append(directory)

    def configure(self, beam_xmax, shower_cutoffs):
        values = tuple(beam_xmax) + tuple(shower_cutoffs)
        if len(values) != 4 or not all(math.isfinite(x) for x in values):
            raise MatchingError('Invalid one-step shower support configuration')
        if self.configuration is not None and values != self.configuration:
            raise MatchingError('Shower support changed within a source file')
        self.configuration = values

    @staticmethod
    def _flavour_permutation(ids, labels):
        """Map compatible flavour rows, preserving the order of identical legs."""
        available, permutation = list(range(2, len(ids))), [0, 1]
        for pdg in labels[2:]:
            position = next(i for i in available if ids[i] == pdg)
            available.remove(position)
            permutation.append(position)
        return tuple(permutation)

    @staticmethod
    def _momenta(external):
        """Canonical point serialization, also used to seed auxiliary draws."""
        return ''.join('%d %s\n' % (pdg, ' '.join('%.17e' % x for x in p))
                       for pdg, p in external)

    def _route(self, external, delta=None):
        ids = tuple(pdg for pdg, _ in external)
        if self.construction == DELTA_SOURCE and delta is None:
            self._certify_delta_auxiliary_law(ids)
        # Generated flavour tables are fixed for this evaluator. Cache only
        # their discrete route/permutation, never momenta or density values.
        # Replay must remain keyed by the exact source registry and owner.
        key = (ids, None if delta is None else (delta.registry, delta.owner))
        cached = self._route_cache.get(key)
        if cached is not None:
            directory, sector, permutation = cached
            return directory, sector, [external[i] for i in permutation]
        final_flavours = Counter(ids[2:])
        for directory, sector, labels in self.routes:
            if self.construction == DELTA_SOURCE:
                registry = self.registries[directory]
                if not registry['registry']['flavour_complete']:
                    continue
                owners = {table['sector'] for table in registry['registry']['tables']}
                if sector not in owners:
                    continue
                if delta is not None:
                    # Replay indices and momenta refer to precisely this source
                    # owner.  A same-flavour spectator permutation is insufficient.
                    if (registry['sha256'] != delta.registry or sector != delta.owner or
                            labels != ids):
                        continue
                    self._route_cache[key] = directory, sector, tuple(range(len(ids)))
                    return directory, sector, external
            if labels[:2] != ids[:2] or Counter(labels[2:]) != final_flavours:
                continue
            permutation = self._flavour_permutation(ids, labels)
            self._route_cache[key] = directory, sector, permutation
            return directory, sector, [external[i] for i in permutation]
        if delta is not None:
            raise MatchingError('No exact Delta replay registry/owner/flavour route for PDGs %s' % (ids,))
        raise MatchingError('No real matrix-element context for PDGs %s' % (ids,))

    def _certify_delta_auxiliary_law(self, ids):
        """Prove that choosing a compatible owner preserves the joint H law.

        Full label maps must agree, not merely flavour coverage or mean H.
        Each native context/sector fixes its recoil groups and all-flow scale
        matrix; identical mapped histories differ only in the ordering of IID
        scalar-Q draws.  There is still just one common insertion uniform.
        Unsupported owner mixtures fail instead of silently changing the law.
        """
        if ids in self.delta_law_certificates:
            return
        signatures = []
        for directory, owner, labels in self.routes:
            registry = self.registries[directory]['registry']
            tables = [table for table in registry['tables'] if table['sector'] == owner]
            if (not registry['flavour_complete'] or not tables or labels[:2] != ids[:2] or
                    Counter(labels[2:]) != Counter(ids[2:])):
                continue
            permutation = self._flavour_permutation(ids, labels)
            outer = registry['outer']
            sectors = [sector for sector in outer['sectors'] if sector['sector'] == owner]
            processes = [process for sector in sectors for process in sector['processes']
                         if tuple(leg[0] for leg in process[1]) == labels]
            if len(processes) != 1:
                raise MatchingError('Delta auxiliary law needs a unique physical source flavour')
            process = processes[0]
            legs = [None] * len(ids)
            for leg, position in zip(process[1], permutation):
                legs[position] = leg
            histories = []
            for history in tables[0]['histories']:
                mapped = tuple(permutation[index - 1] for index in history['permutation'])
                histories.append((history['context'], history['sector'], mapped))
            if not histories or len(set(histories)) != len(histories):
                raise MatchingError('Delta auxiliary law has missing or duplicate native draws')
            signature = json.dumps(dict(process=[process[0], legs],
                orders=outer['orders'], amp_orders=outer['amp_orders'],
                histories=sorted(histories),
                contexts=self.registries[directory]['contexts_sha256']),
                sort_keys=True, separators=(',', ':'))
            signatures.append(signature)
        if signatures and any(value != signatures[0] for value in signatures[1:]):
            raise MatchingError('Delta source owners have different joint auxiliary laws; '
                                'their source-partition mixture must be sampled explicitly')
        if signatures:
            self.delta_law_certificates[ids] = hashlib.sha256(signatures[0].encode()).hexdigest()

    def _request(self, event, request_id):
        """Construct the native protocol without starting or building a worker."""
        delta_mode = self.construction == DELTA_SOURCE
        replay = event.delta if delta_mode and event.kind == 'H' else None
        if delta_mode and event.kind == 'H':
            if replay is None:
                raise MatchingError('Source H event lacks its conditional Delta replay context')
            if event.endpoint_origin == 'shower':
                raise MatchingError('Delta H replay requires its original hard point')
        directory, sector, external = self._route(event.external(), delta=replay)
        version = 2 if delta_mode else 1
        request = 'MACNLOPS_POINT_V%d %d %d %d\n' % (version, request_id, sector, len(external))
        momenta = self._momenta(external)
        request += momenta
        if replay is not None:
            request += 'DELTA_REPLAY %d %.17e %s\n' % (
                len(replay.probabilities), replay.insertion, replay.registry)
            request += ''.join('%d %d %d %.17e\n' % row for row in replay.probabilities)
        elif delta_mode:
            # A fresh auxiliary H context, independent of the actual S shower
            # start, is necessary for the negative conditional H measure.
            identity = '%d:%d:%s\n%s' % (self.auxiliary_seed, request_id,
                                        self.registries[directory]['sha256'], momenta)
            seed = 1 + int.from_bytes(hashlib.sha256(identity.encode('ascii')).digest()[:8],
                                      'big') % 899999999
            request += 'DELTA_SAMPLE %d\n' % seed
        return directory, request + 'END_POINT\n'

    def _component_requests(self, event, request_id):
        """Build local requests in the exact source Born provider and basis."""
        component = event.component
        if component is None or event.hard.partition != 'sampled':
            raise MatchingError('Local matching requires a qualified component packet')
        directories = self._registry_routes.get(component.registry, ())
        if len(directories) != 1:
            raise MatchingError('No unique native Born provider for the component registry')
        directory = directories[0]
        registry = self.registries[directory]['registry']
        external = event.external()
        if event.kind == 'H':
            if event.endpoint_origin == 'shower':
                raise MatchingError('H component requires its original hard point')
            if event.weight < 0 and component.source_factor == 0.:
                raise MatchingError('Negative H event has a zero source multiplier')
            ids = tuple(pdg for pdg, _ in external)
            if (directory, component.sector, ids) not in self.routes:
                raise MatchingError('H component registry/sector/ordered flavours disagree')
            choices = [(component.sector, tuple(range(len(external))), None, 'source')]
        else:
            record = getattr(event, 'shower_record', None)
            if record is None or not record.get('emitted'):
                raise MatchingError('Local S evaluation requires the actual first-branch record')
            try:
                candidates = macnlops_components.branch_candidates(
                    record, registry, source_directory=directory.name)
            except macnlops_components.MappingError as exc:
                raise MatchingError('Cannot identify sampled shower component: %s' % exc) from exc
            choices = [(choice.sector, choice.permutation, choice.born_permutation,
                        choice.orientation) for choice in candidates]
        mode = int(self.construction == DELTA_SOURCE)
        requests = []
        for sector, permutation, born_permutation, orientation in choices:
            config, graph, flow = component.config, component.graph, component.flow
            if born_permutation is not None:
                key = (directory, tuple(born_permutation), config, graph, flow)
                if key not in self._component_basis_cache:
                    try:
                        self._component_basis_cache[key] = (
                            macnlops_components.transport_component_basis(
                                registry, born_permutation, config, graph, flow))
                    except macnlops_components.MappingError as exc:
                        raise MatchingError('Cannot transport the source Born basis: %s' % exc) from exc
                config, graph, flow = self._component_basis_cache[key]
            ordered = [external[index] for index in permutation]
            request = 'MACNLOPS_POINT_V3 %d %d %d\n' % (request_id, sector, len(ordered))
            request += self._momenta(ordered)
            request += 'COMPONENT_LOCAL_V1 %d %d %d %d %.17e\n' % (
                config, graph, flow, mode, component.diagram_symmetry)
            if born_permutation is not None:
                # The hard annotation supplies pre-shuffle Born momenta and
                # oriented colours. The native worker independently checks
                # this actual state against its inverse map and selected flow.
                request += 'COMPONENT_BORN_V1 %d\n' % len(born_permutation)
                for index in born_permutation:
                    pdg, _, momentum = event.hard.particles[index]
                    colours = event.hard.colours[index]
                    request += '%d %d %d %s\n' % (pdg, *colours, ' '.join(
                        '%.17e' % x for x in (*momentum, event.hard.masses[index])))
            requests.append((directory, request + 'END_POINT\n', dict(
                registry=component.registry, sector=sector, config=config,
                graph=graph, flow=flow, orientation=orientation,
                source_basis=[component.config, component.graph, component.flow],
                born_permutation=None if born_permutation is None else list(born_permutation),
                permutation=list(permutation),
                prescription='undamped-delta' if mode else 'ordinary-source')))
        if not requests:
            raise MatchingError('No qualified local component for the observed branching')
        return requests

    def _evaluate_component(self, event, request_id):
        candidates = self._component_requests(event, request_id)
        densities = [self._query(directory, request, request_id)
                     for directory, request, _ in candidates]
        for density in densities:
            density.acceptance(signed_residuals=True)  # Validate R,D,K before marking.
        total_kernel = math.fsum(density.shower for density in densities)
        if not math.isfinite(total_kernel):
            raise MatchingError('Nonfinite sum of component shower kernels')
        # Equivalent daughter orientations share a signed luminosity. Their
        # common sign cancels in the auxiliary marking law, but never in the
        # physical veto: a nonpositive K transfers no negative H weight.
        positive = any(density.shower > 0. for density in densities)
        negative = any(density.shower < 0. for density in densities)
        if positive and negative:
            raise MatchingError('Equivalent shower orientations have incompatible PDF signs')
        kernel_sign = -1. if negative else 1.
        marking_kernels = [kernel_sign * density.shower for density in densities]
        marking_total = kernel_sign * total_kernel
        selected, marking = 0, 1.
        if len(candidates) > 1 and marking_total > 0.:
            # This auxiliary labels equivalent daughter orientations only.
            # It is separate from both the native shower and the event veto.
            identity = '%d:%d:%s' % (self.auxiliary_seed, request_id,
                                     ''.join(request for _, request, _ in candidates))
            # Exactly representable in [0,1); a 256-bit integer converted to
            # float can round to one and select a zero-kernel first entry.
            uniform = (int.from_bytes(hashlib.sha256(identity.encode('ascii')).digest()[:8],
                                      'big') >> 11) * 2.**-53
            threshold, cumulative = uniform * marking_total, 0.
            selected = max(index for index, kernel in enumerate(marking_kernels)
                           if kernel > 0.)
            for index, kernel in enumerate(marking_kernels):
                cumulative += kernel
                if threshold < cumulative:
                    selected = index
                    break
            marking = marking_kernels[selected] / marking_total
        # If every component has K=0, the bounded veto retains S for every
        # possible mark. Keep one evaluated density and expose zero support.
        event.matching_component = dict(candidates[selected][2],
            candidates=len(candidates), marking_probability=marking,
            total_kernel=total_kernel, kernel_sign=int(kernel_sign),
            candidate_kernels=[density.shower for density in densities])
        return densities[selected]

    def evaluate(self, event, request_id):
        if self.configuration is None:
            raise MatchingError('Missing actual beam bounds and cutoffs from the one-step shower')
        if event.hard is None and self.construction != RESOLVED_SOURCE:
            raise MatchingError('Common-source matching requires the saved pre-shuffle '
                                'hard point and its source-partition annotation')
        if (event.hard is not None and event.hard.source !=
                hard_source_name(self.construction)):
            raise MatchingError('Hard-event source construction disagrees with the evaluator')
        if event.hard is not None and event.hard.partition != self.partition:
            raise MatchingError('Sampled H source partitions require their conditional '
                                'density; a global H density cannot determine this veto')
        check_hard_mass_coordinates(event, self.model_masses)
        if self.partition == 'sampled':
            return self._evaluate_component(event, request_id)
        directory, request = self._request(event, request_id)
        return self._query(directory, request, request_id)

    def _query(self, directory, request, request_id):
        """One native request, with no change to its source or proposal laws."""
        if directory not in self.workers:
            log = open(self.log_dir / (directory.name + '.log'), 'w')
            try:
                subprocess.run(['make', 'macnlops_evaluator'], cwd=directory,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=self.environment)
                worker = subprocess.Popen([str(directory / 'macnlops_evaluator')],
                    cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, text=True, bufsize=1, env=self.environment)
            except BaseException:
                log.close()
                raise
            self.workers[directory] = worker, log
            worker.stdin.write('MACNLOPS_CONFIG_V1 ' + ' '.join(
                '%.17e' % x for x in self.configuration) + '\n')
            worker.stdin.flush()
        worker, log = self.workers[directory]
        worker.stdin.write(request)
        worker.stdin.flush()
        kernel_alpha_s = None
        for line in worker.stdout:
            if line.startswith('MACNLOPS_KERNEL_COUPLING_V1 '):
                fields = line.split()
                if len(fields) != 3 or int(fields[1]) != request_id:
                    raise MatchingError('Invalid native kernel coupling record: ' + line.strip())
                kernel_alpha_s = number(fields[2])
                continue
            if not line.startswith('MACNLOPS_DENSITY_V1 '):
                log.write(line)
                continue
            fields = line.split()
            if int(fields[1]) != request_id or int(fields[2]) != 0:
                if int(fields[2]) == 23:
                    raise MatchingError('Pythia heavy-flavour threshold conversion needs its '
                                        'own density and inverse map (event %d)' % request_id)
                if int(fields[2]) == 24:
                    raise MatchingError('Hard-model mass-shell mismatch or unavailable '
                                        'massive emitted-parton map (event %d)' % request_id)
                raise MatchingError('Native density evaluator failed: ' + line.strip())
            if len(fields) != 6:
                raise MatchingError('Invalid native density record: ' + line.strip())
            return Densities(*(number(x) for x in fields[3:]), kernel_alpha_s=kernel_alpha_s)
        raise MatchingError('Native density evaluator exited for %s; see %s' %
                            (directory.name, log.name))

    def close(self):
        for worker, log in self.workers.values():
            if worker.poll() is None:
                worker.stdin.close()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait()
            worker.stdout.close()
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def sample_sum_variance(trials, weight, square):
    """Variance of a signed sum, including rejected trials with zero weight."""
    return (max((trials * square - weight * weight) / (trials - 1), 0.0)
            if trials > 1 else 0.0)


def sample_header(header, counts, process_weights, process_squares, process_maxima,
                  process_variances=None, continuation=False, matching_policy=None):
    """Describe the retained sum-normalised sample in its LHE init block."""
    match = re.search(r'(<init>)(.*?)(</init>)', header, re.S)
    rows = [line.split() for line in match[2].splitlines() if line.strip()]
    nprocess = int(rows[0][-1])
    if len(rows) != nprocess + 1:
        raise MatchingError('Invalid source LHE init block')
    total_trials = counts['input_S'] + counts['input_H']
    for row in rows[1:]:
        process = int(row[3])
        weight = process_weights[process]
        # Finite-sample uncertainty in the sum over independent source trials.
        variance = (sample_sum_variance(total_trials, weight, process_squares[process])
                    if process_variances is None else process_variances[process])
        row[:3] = ['%.17e' % weight, '%.17e' % math.sqrt(variance),
                   '%.17e' % process_maxima[process]]
    replacement = '<init>\n' + '\n'.join(' '.join(row) for row in rows) + '\n</init>'
    header = header[:match.start()] + replacement + header[match.end():]
    if continuation:
        retained = counts['accepted_S'] + counts['accepted_H']
        def retained_card(match):
            body = re.sub(r'(?m)^\s*[^\n!#=]+\s*=\s*nevents\b[^\n]*',
                          '\n%d = nevents ! retained first-emission events' % retained, match[1])
            return '<MGRunCard>' + body + '</MGRunCard>'
        header = re.sub(r'<MGRunCard>(.*?)</MGRunCard>', retained_card, header, flags=re.S | re.I)
    policy_attribute = (' prescription="%s"' % html.escape(matching_policy, quote=True)
                        if matching_policy is not None else '')
    annotation = ('<macnlops normalization="sum" stage="first-emission"%s%s>\n'
                  'Input events: %d\nRetained events: %d\n'
                  'Cross section is the sum of retained input weights.\n'
                  '</macnlops>\n') % (' continuation="lhe-v1"' if continuation else '',
                  policy_attribute, total_trials,
                  counts['accepted_S'] + counts['accepted_H'])
    if '</header>' in header:
        header = header.replace('</header>', annotation + '</header>', 1)
    else:
        raise MatchingError('Missing LHE header')
    return header


def collect_matched_events(inputs, reports, output, seed):
    """Shuffle all surviving events with the ordinary MG event collector.

    Input files are independent, additive channel samples. Their original
    trial counts include rejected events; channel variances add in quadrature.
    Events themselves are copied verbatim and never reweighted.
    """
    output = Path(output)
    if output.exists():
        raise MatchingError('Output already exists: %s' % output)
    if not inputs or len(inputs) != len(reports):
        raise MatchingError('Collection requires one report per matching input')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, weights, squares, maxima, variances = (Counter() for _ in range(5))
    process_rows = {}
    beam_fields = None
    header = None
    shower_profiles = []
    matching_policies = []
    continuation_modes = set()
    with tempfile.TemporaryDirectory(prefix='.macnlops_collect_', dir=output.parent) as temporary:
        temporary = Path(temporary)
        plain_inputs = []
        for index, (filename, report) in enumerate(zip(inputs, reports)):
            plain = temporary / ('channel_%06d.lhe' % index)
            # The collector indexes plain files with mmap. Preserve raw event
            # bytes while decompressing; the parsing pass only computes totals.
            with open_lhe(filename, 'rb') as source, plain.open('wb') as target:
                shutil.copyfileobj(source, target)
            local_counts, local_weights, local_squares = Counter(), Counter(), Counter()
            signed_weight = absolute_weight = 0.0
            with LHEStream(plain) as source:
                if header is None:
                    header = source.header
                markers = list(re.finditer(r'<macnlops\b([^>]*)>', source.header))
                if len(markers) > 1:
                    raise MatchingError('Duplicate MAcNLOPS sample header')
                policy = (xml_attributes(markers[0][1]).get('prescription')
                          if markers else None)
                if policy not in (None, 'global-source-v1', 'component-source-v1',
                                  LEGACY_UNDAMPED_DELTA_POLICY, UNDAMPED_DELTA_POLICY):
                    raise MatchingError('Unknown MAcNLOPS veto prescription')
                matching_policies.append(policy)
                if policy != report.get('veto_prescription'):
                    raise MatchingError('Matching report/header prescriptions disagree')
                if matching_policies[-1] != matching_policies[0]:
                    raise MatchingError('Cannot combine different MAcNLOPS veto prescriptions')
                shower_profiles.append(read_shower_profile(source.header))
                if shower_profiles[-1] != shower_profiles[0]:
                    raise MatchingError('Cannot combine different first-emission shower profiles')
                init = re.search(r'<init>(.*?)</init>', source.header, re.S)
                try:
                    rows = [line.split('#', 1)[0].split() for line in init[1].splitlines()
                            if line.split('#', 1)[0].strip()]
                    if len(rows[0]) != 10 or len(rows) != int(rows[0][-1]) + 1:
                        raise ValueError('beam/process row count')
                    beam_signature = [number(x) for x in rows[0][:-1]]
                    declared = set()
                    for row in rows[1:]:
                        if len(row) != 4 or int(row[3]) in declared:
                            raise ValueError('process table')
                        process = int(row[3])
                        declared.add(process)
                        process_rows.setdefault(process, row)
                except (TypeError, IndexError, ValueError) as exc:
                    raise MatchingError('Invalid LHE init block in %s' % filename) from exc
                if beam_fields is None:
                    beam_fields = rows[0]
                elif beam_signature != [number(x) for x in beam_fields[:-1]]:
                    raise MatchingError('Incompatible LHE beams/PDFs/weight convention in %s' % filename)
                for block in source:
                    event = Event(block)
                    if event.hard is not None:
                        if event.hard.partition == 'sampled' and event.component is None:
                            raise MatchingError('Sampled collection requires a qualified component packet')
                        construction = SOURCE_CONSTRUCTIONS[event.hard.source]
                        expected_policy = matching_prescription(construction, event.hard.partition)
                        if (shower_profiles[-1] is not None and
                                shower_profiles[-1]['construction'] != construction):
                            raise MatchingError('Event body/shower-profile source construction disagrees')
                        legacy_global = policy is None and event.hard.partition == 'global_sum'
                        # The body cannot tell the v1 and v2 S-veto denominators apart.
                        legacy_delta = (policy == LEGACY_UNDAMPED_DELTA_POLICY and
                                        expected_policy == UNDAMPED_DELTA_POLICY)
                        if not (legacy_global or legacy_delta) and policy != expected_policy:
                            raise MatchingError('Event body/header matching prescriptions disagree')
                        for field, expected in (('matching_partition', event.hard.partition),
                                                ('source_construction', construction)):
                            if field in report and report[field] != expected:
                                raise MatchingError('Event body/report source provenance disagrees')
                    elif policy is not None:
                        raise MatchingError('Matching prescription requires event source provenance')
                    continuation_modes.add(event.continuation is not None)
                    if event.process not in declared:
                        raise MatchingError('Event process missing from LHE init block in %s' % filename)
                    local_counts[event.kind] += 1
                    local_weights[event.process] += event.weight
                    local_squares[event.process] += event.weight * event.weight
                    maxima[event.process] = max(maxima[event.process], abs(event.weight))
                    signed_weight += event.weight
                    absolute_weight += abs(event.weight)
            if 'weight_sums' in report:
                expected_weight = report['weight_sums'].get('accepted', 0.0)
                if (not math.isfinite(expected_weight) or
                        abs(signed_weight - expected_weight) > 1.e-12 * max(absolute_weight, 1.0)):
                    raise MatchingError('Matching report/event weight sums disagree in %s' % filename)
            original = report['counts']
            for kind in ('S', 'H'):
                trials = original.get('input_' + kind, 0)
                if (type(trials) is not int or trials < local_counts[kind] or
                        original.get('accepted_' + kind, 0) != local_counts[kind]):
                    raise MatchingError('Matching report/event counts disagree in %s' % filename)
                counts['input_' + kind] += trials
                counts['accepted_' + kind] += local_counts[kind]
            trials = original.get('input_S', 0) + original.get('input_H', 0)
            for process in declared:
                weight, square = local_weights[process], local_squares[process]
                weights[process] += weight
                squares[process] += square
                variances[process] += sample_sum_variance(trials, weight, square)
            plain_inputs.append(plain)

        # collect_events takes <init> from its FIRST input, not its template.
        # Supply a zero-event first input carrying the complete process table
        # and totals, including processes whose channels retained no events.
        beam_fields[-1] = str(len(process_rows))
        init = '<init>\n' + ' '.join(beam_fields) + '\n' + '\n'.join(
            ' '.join(process_rows[process]) for process in sorted(process_rows)) + '\n</init>'
        header = re.sub(r'<init>.*?</init>', lambda unused: init, header, count=1, flags=re.S)
        header = re.sub(r'<macnlops\b[^>]*>.*?</macnlops>\s*', '', header, flags=re.S)
        if len(continuation_modes) > 1:
            raise MatchingError('Cannot combine legacy endpoints with corrected continuation endpoints')
        continuation = (continuation_modes == {True} or
                        (not continuation_modes and all(report.get('lhe_continuation', False)
                                                        for report in reports)))
        header = sample_header(header, counts, weights, squares, maxima,
                               process_variances=variances, continuation=continuation,
                               matching_policy=matching_policies[0])
        header = re.sub(r'(<macnlops\b[^>]*)(>)',
                        lambda match: match[1] + ' shuffle_seed="%d">' % seed, header, count=1)
        # The standard collector rewrites the complete iseed line. Keep the
        # run-card delimiters on separate lines even for compact XML headers.
        header = re.sub(r'(</?MGRunCard\b[^>]*>)', r'\n\1\n', header, flags=re.I)
        preamble = temporary / 'combined_header.lhe'
        preamble.write_text(header + '</LesHouchesEvents>\n')
        staged = temporary / ('events.lhe.gz' if str(output).endswith('.gz') else 'events.lhe')
        try:
            collect_events.collect_events(output=staged, header_template=preamble,
                input_files=[preamble] + plain_inputs, seed=seed, subset=None,
                mode='auto', prefer_pigz=True, gzip_level=6)
        except (OSError, RuntimeError) as exc:
            raise MatchingError('Collecting MAcNLOPS events failed: %s' % exc) from exc
        os.replace(staged, output)
    return dict(output=str(output), shuffle_seed=seed, inputs=[str(path) for path in inputs],
                lhe_continuation=continuation,
                veto_prescription=matching_policies[0],
                matching_seeds=[report.get('seed') for report in reports], counts=dict(counts),
                sumw=sum(weights.values()), sumw2=sum(squares.values()),
                normalization='sum; unchanged surviving input weights',
                cross_section_error_convention='quadrature of independent-channel finite-sample errors; original trials include rejections')


def matching_scale(record, profile):
    """Boundary of the *actual* Born history, before the first shower step.

    Below this boundary the source transfers R to S. Native heavy-flavour
    threshold conversions are kept, without projecting their massive sister
    onto the massless real matrix element.
    """
    incoming = [abs(p['id']) for p in record['source_particles'] if p['status'] == -1]
    if len(incoming) != 2:
        raise MatchingError('Missing Born incoming flavours in shower metadata')
    scale = max(profile['shower_cutoffs'])
    for pdg in incoming:
        if pdg in (4, 5):
            scale = max(scale, math.sqrt(2.) * profile['heavy_mass_floors'][str(pdg)])
    pt = record['pt_evol']
    if not math.isfinite(pt) or pt < 0:
        raise MatchingError('Invalid first-emission evolution scale')
    return scale


def reject_proposals(proposals, metadata, output, evaluator, seed=1, signed_residuals=False,
                     profile=None, require_continuation=False, shower_profile=None,
                     matching_policy=None, beta=None):
    """Make one decision per original event and preserve its absolute weight.

    Output is published only after all density and synchronisation checks pass.
    No accepted-count normalisation is performed.  The caller must supply sum
    normalised MC@NLO weights, which remain additive after event rejection.
    With beta, the channel's sigma_S/sigma_B, S vetoes use beta*K_sh.
    """
    rng = random.Random(seed)
    output = Path(output)
    if output.exists():
        raise MatchingError('Output already exists: %s' % output)
    counts = Counter()
    weight_sums = Counter()
    continuation_modes = set()
    process_weights, process_squares, process_maxima = Counter(), Counter(), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.gz' if str(output).endswith('.gz') else '.lhe', dir=output.parent)
    os.close(handle)
    audit_output = output.with_name(output.name + '.decisions.jsonl')
    if audit_output.exists():
        Path(temporary).unlink()
        raise MatchingError('Decision audit already exists: %s' % audit_output)
    audit_temporary = Path(temporary + '.jsonl')
    try:
        with LHEStream(proposals) as source, open(metadata) as meta, \
                tempfile.TemporaryFile(mode='w+t') as retained, open(audit_temporary, 'w') as audit:
            for sequence, block in enumerate(source, 1):
                line = meta.readline()
                if not line:
                    raise MatchingError('Missing one-step shower metadata')
                record = json.loads(line)
                if record['sequence'] != sequence:
                    raise MatchingError('One-step shower sequence mismatch')
                event = Event(block)
                event.shower_record = record
                continuation_modes.add(event.continuation is not None)
                event.check_proposal_metadata(record, require_continuation)
                if 'beam_xmax' in record and 'shower_cutoffs' in record:
                    if profile is not None:
                        for key in ('beam_xmax', 'shower_cutoffs'):
                            if record[key] != profile[key]:
                                raise MatchingError('Shower %s differs from generation profile' % key)
                    evaluator.configure(record['beam_xmax'], record['shower_cutoffs'])
                kind = event.kind
                counts['input_' + kind] += 1
                if event.weight < 0:
                    counts['negative_input_' + kind] += 1
                weight_sums['input'] += event.weight
                density = None
                probability = 1.0
                gate_scale = None
                kernel_scale = None
                if kind == 'H' and event.weight < 0:
                    density = evaluator.evaluate(event, sequence)
                    probability = density.hard_acceptance(signed_residuals=signed_residuals)
                elif kind == 'S' and record['emitted']:
                    if profile is not None:
                        gate_scale = matching_scale(record, profile)
                    if gate_scale is not None and record['pt_evol'] <= gate_scale:
                        counts['below_matching_boundary_S'] += 1
                    else:
                        density = evaluator.evaluate(event, sequence)
                        if beta is not None and density.transfers():
                            kernel_scale = shower_kernel_scale(beta, record, density)
                        probability = density.acceptance(signed_residuals=signed_residuals,
                                                         kernel_scale=kernel_scale or 1.)
                accepted = rng.random() < probability
                counts[('accepted_' if accepted else 'rejected_') + kind] += 1
                if kind == 'S' and not record['emitted']:
                    counts['no_emission_S'] += 1
                if accepted:
                    if event.weight < 0:
                        counts['negative_accepted_' + kind] += 1
                    weight_sums['accepted'] += event.weight
                    process_weights[event.process] += event.weight
                    process_squares[event.process] += event.weight * event.weight
                    process_maxima[event.process] = max(process_maxima[event.process], abs(event.weight))
                    retained.write(block)
                audit.write(json.dumps(dict(sequence=sequence, kind=kind,
                    accepted=accepted, acceptance=probability,
                    matching_scale=gate_scale, kernel_scale=kernel_scale,
                    component=getattr(event, 'matching_component', None),
                    densities=None if density is None else density.__dict__)) + '\n')
            if meta.readline():
                raise MatchingError('Extra one-step shower metadata')
            if len(continuation_modes) > 1:
                raise MatchingError('Mixed legacy and corrected continuation proposals')
            continuation = require_continuation or continuation_modes == {True}
            retained.seek(0)
            with open_lhe(temporary, 'wt') as target:
                header = (with_shower_profile(source.header, shower_profile)
                          if shower_profile is not None else source.header)
                target.write(sample_header(header, counts, process_weights,
                                           process_squares, process_maxima,
                                           continuation=continuation,
                                           matching_policy=matching_policy))
                shutil.copyfileobj(retained, target)
                target.write('</LesHouchesEvents>\n')
        os.replace(temporary, output)
        os.replace(audit_temporary, audit_output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)
        raise
    return dict(counts=dict(counts), weight_sums=dict(weight_sums),
                seed=seed, decisions=str(audit_output), signed_residuals=signed_residuals,
                veto_prescription=matching_policy, s_veto_beta=beta,
                lhe_continuation=continuation)


def _checksum_signature(info):
    """Identity and mutation metadata of the actual file, following symlinks."""
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _assert_checksum_unchanged(path, signature):
    try:
        current = _checksum_signature(os.stat(path))
    except OSError as exc:
        raise MatchingError('File changed while hashing: %s' % path) from exc
    if current != signature:
        raise MatchingError('File changed while hashing: %s' % path)


def _read_checksum(path, decompressed, signature):
    """Hash one stable opened file; never publish a digest after a mutation."""
    digest, size = hashlib.sha256(), 0
    opener = gzip.open if decompressed else open
    with opener(path, 'rb') as stream:
        if _checksum_signature(os.fstat(stream.fileno())) != signature:
            raise MatchingError('File changed while hashing: %s' % path)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
        if _checksum_signature(os.fstat(stream.fileno())) != signature:
            raise MatchingError('File changed while hashing: %s' % path)
    _assert_checksum_unchanged(path, signature)
    return digest.hexdigest(), size


def _cached_checksum(path, decompressed=False):
    """Reuse stable regular-file digests without weakening normal mutation checks.

    The content identity includes ctime, so restoring mtime after a rewrite
    cannot reuse an old digest. Hard links and symlinks to the same inode
    share entries, while raw and gzip-decoded content remain distinct.
    Metadata is checked again even on a cache hit. Nonregular/zero-size files
    are not cached; this also excludes common dynamic procfs-style inputs.
    """
    path = os.fspath(path)
    info = os.stat(path)
    signature = _checksum_signature(info)
    cacheable = stat.S_ISREG(info.st_mode) and info.st_size > 0
    key = (decompressed, signature)
    if cacheable:
        with _checksum_cache_lock:
            cached = _checksum_cache.get(key)
            if cached is not None:
                _checksum_cache.move_to_end(key)
        if cached is not None:
            _assert_checksum_unchanged(path, signature)
            return cached
    value, size = _read_checksum(path, decompressed, signature)
    # Some virtual files advertise a fixed nonzero size unrelated to their
    # content. A raw read of such an input must not become a reusable entry.
    if cacheable and (decompressed or size == info.st_size):
        with _checksum_cache_lock:
            _checksum_cache[key] = value
            _checksum_cache.move_to_end(key)
            while len(_checksum_cache) > _CHECKSUM_CACHE_LIMIT:
                _checksum_cache.popitem(last=False)
    return value


def sha256(path):
    """SHA256 of the exact file bytes."""
    return _cached_checksum(path)


def lhe_sha256(path):
    """SHA256 of LHE bytes, decoding only the same .gz suffix as open_lhe."""
    return _cached_checksum(path, decompressed=str(path).endswith('.gz'))


def source_files(process_dir):
    """Files defining the generated hard density and its subtraction."""
    patterns = ('Cards/run_card.dat', 'Cards/param_card.dat',
                'Cards/macnlops_profile.json', 'Cards/macnlops_shower.cmnd',
                'MCatNLO/srcPythia8/MacNLOPS.cc',
                'SubProcesses/FKS_params.dat', 'SubProcesses/*.f',
                'SubProcesses/*.f90', 'SubProcesses/*.inc',
                'SubProcesses/P*/*.f', 'SubProcesses/P*/*.f90',
                'SubProcesses/P*/*.inc', 'SubProcesses/P*/mc_histories.json',
                'Source/BornSupport/registry.json',
                'Source/BornSupport/**/*.f', 'Source/BornSupport/**/*.f90',
                'Source/MODEL/*.f', 'Source/MODEL/*.inc')
    return {str(path.relative_to(process_dir)): sha256(path)
            for pattern in patterns for path in sorted(process_dir.glob(pattern))
            if path.is_file() and path.name != 'driver_macnlops.f'}


def write_source_manifest(process_dir, run_name, inputs):
    """Record the exact generation state alongside every source event file."""
    process_dir = Path(process_dir).resolve()
    manifest = dict(schema='macnlops-source-v2', run=run_name,
                    sources=source_files(process_dir))
    for filename in inputs:
        filename = Path(filename)
        record = dict(manifest, lhe_sha256=lhe_sha256(filename))
        Path(str(filename) + '.macnlops.json').write_text(
            json.dumps(record, indent=2, sort_keys=True) + '\n')


def check_manifest(filename, process_dir):
    candidate = Path(str(filename) + '.macnlops.json')
    if not candidate.exists():
        # run_mcatnlo may have decompressed the collected source file.
        candidate = Path(str(filename) + '.gz.macnlops.json')
    if not candidate.exists():
        raise MatchingError('Missing generation manifest for %s; generate a fresh macnlops=True sample' % filename)
    manifest = json.loads(candidate.read_text())
    if manifest.get('schema') != 'macnlops-source-v2' or manifest.get('lhe_sha256') != lhe_sha256(filename):
        raise MatchingError('Generation manifest does not describe the input event file')
    mismatch = [name for name, digest in manifest['sources'].items()
                if Path(name).name != 'driver_macnlops.f' and
                (not (process_dir / name).is_file() or sha256(process_dir / name) != digest)]
    if mismatch:
        raise MatchingError('Generation sources/cards changed since event generation: ' + ', '.join(mismatch[:8]))
    return manifest


def source_header(filename, process_dir, manifest):
    """Restore the original banner for individual generator channel files."""
    with LHEStream(filename) as source:
        header = source.header
    if '<MGRunCard>' in header:
        return header
    collected = process_dir / 'Events' / manifest['run'] / 'events.lhe.gz'
    if not collected.exists():
        collected = collected.with_suffix('')
    combined_manifest = check_manifest(collected, process_dir)
    if combined_manifest['sources'] != manifest['sources']:
        raise MatchingError('Channel and collected source have different generation states')
    with LHEStream(collected) as source:
        banner = re.search(r'<header>.*?</header>', source.header, re.S)
    if not banner:
        raise MatchingError('Missing original collected source banner')
    header, replacements = re.subn(r'<header>.*?</header>', lambda _: banner[0], header, count=1, flags=re.S)
    if not replacements:
        raise MatchingError('Missing channel LHE header')
    return header


def run_card_values(text):
    values = {}
    for line in text.splitlines():
        line = line.split('!', 1)[0].split('#', 1)[0]
        match = re.match(r'\s*(.*?)\s*=\s*([A-Za-z][A-Za-z_0-9]*)\s*$', line)
        if match:
            values[match[2].lower()] = match[1].strip().strip("'\"").lower()
    return values


def check_source(header, process_dir):
    """The veto numerator must come from the same cards as the hard sample."""
    card = re.search(r'<MGRunCard>(.*?)</MGRunCard>', header, re.S | re.I)
    if card is None:
        raise MatchingError('Input must contain its original MGRunCard')
    source = run_card_values(card[1])
    current = run_card_values((process_dir / 'Cards/run_card.dat').read_text())
    # Run names, seeds, integration targets and worker counts do not enter D.
    operational = {'nevents', 'iseed', 'req_acc', 'req_acc_fo', 'nevt_job',
                   'npoints_fo_grid', 'niters_fo_grid', 'npoints_fo', 'niters_fo',
                   'run_tag'}
    if all(source.get(key) in ('true', '.true.', 't') for key in
           ('fixed_ren_scale', 'fixed_fac_scale')):
        # MG rewrites the inactive dynamic choice to its fixed-scale sentinel.
        operational.add('dynamical_scale_choice')
    mismatch = [key for key in source if key not in operational and
                key in current and source[key] != current[key]]
    if mismatch:
        raise MatchingError('Generation cards differ from input for: ' + ', '.join(mismatch))
    if source.get('event_norm') != 'sum':
        raise MatchingError('Generate with event_norm=sum: rejection must retain absolute weights')
    if source.get('store_hard_momenta', 'false') not in ('true', '.true.', 't'):
        raise MatchingError('Generate with store_hard_momenta=True to retain the '
                            'pre-shuffle hard point and source-partition provenance')
    if source_construction(source) != source_construction(current):
        raise MatchingError('Source and evaluator counterterm constructions differ')
    if source.get('parton_shower', '').upper() != 'PYTHIA8':
        raise MatchingError('Input must use PYTHIA8 MC subtraction')
    if source.get('ickkw', '0') != '0':
        raise MatchingError('Merged samples need their merging veto and are not ordinary MC@NLO inputs')
    slha = re.search(r'<slha>(.*?)</slha>', header, re.S | re.I)
    if slha is None:
        raise MatchingError('Missing source parameter card')
    def physics_lines(text):
        return [re.sub(r'\s+', ' ', line.split('#')[0].strip().lower())
                for line in text.splitlines() if line.split('#')[0].strip()]
    if physics_lines(slha[1]) != physics_lines((process_dir / 'Cards/param_card.dat').read_text()):
        raise MatchingError('Source and evaluator parameter cards differ')
    return source


def pdf_name(process_dir, card, with_info=False):
    """Use the source MSbar LHAPDF set, without a scheme transformation."""
    if card.get('pdlabel') != 'lhapdf':
        raise MatchingError('Use pdlabel=lhapdf so the identical MSbar PDF can be selected in Pythia')
    try:
        pdf_id = int(card['lhaid'])
    except (KeyError, ValueError) as exc:
        raise MatchingError('A single central LHAPDF member is required') from exc
    paths = os.environ.get('LHAPDF_DATA_PATH', '').split(os.pathsep)
    config = process_dir / 'Cards/amcatnlo_configuration.txt'
    if config.exists():
        match = re.search(r'^\s*lhapdf\s*=\s*(\S+)', config.read_text(), re.M)
        if match:
            result = subprocess.run([match[1], '--datadir'], check=True,
                                    capture_output=True, text=True)
            paths.append(result.stdout.strip())
    for folder in paths:
        if not folder:
            continue
        index = Path(folder) / 'pdfsets.index'
        if not index.is_file():
            continue
        entries = []
        for line in index.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit():
                entries.append((int(fields[0]), fields[1]))
        entries.sort(reverse=True)
        for base, name in entries:
            if base > pdf_id:
                continue
            info = Path(folder) / name / (name + '.info')
            if not info.exists():
                break
            content = info.read_text()
            members = re.search(r'^NumMembers:\s*(\d+)', content, re.M)
            if members and pdf_id - base < int(members[1]):
                pdf = 'LHAPDF6:%s/%d' % (name, pdf_id - base)
                return (pdf, info) if with_info else pdf
            break
    raise MatchingError('Cannot resolve LHAPDF ID %d in the configured data directories' % pdf_id)


def shower_parameters(process_dir, pdf_info, output, custom=None):
    """Use the hard model's external masses and the PDF's coupling input."""
    text = pdf_info.read_text()
    alpha = re.search(r'^AlphaS_MZ:\s*([\d.Ee+-]+)', text, re.M)
    order = re.search(r'^AlphaS_OrderQCD:\s*(\d+)', text, re.M)
    if not alpha or not order:
        raise MatchingError('PDF metadata must specify AlphaS_MZ and AlphaS_OrderQCD')
    lines = []
    if custom:
        lines.append(custom.read_text())
    for pdg, mass in sorted(hard_model_masses(process_dir).items()):
        lines.append('%d:m0 = %.17e' % (pdg, mass))
    # These define the numerator's shower support and the external map.
    # User overrides are applied first; the matching profile is checked in C++.
    lines += ['SpaceShower:pTmin = 0.5', 'TimeShower:pTmin = 0.5']
    output.write_text('\n'.join(lines) + '\n')
    return number(alpha[1]), int(order[1]) + 1


def native_library_environment(pythia8):
    """Use the LHAPDF ABI against which this Pythia plugin was compiled.

    The hard generator can link a different static LHAPDF build. Its inherited
    shared-library search path must not redirect the native shower backend.
    """
    import shlex
    prefix = Path(pythia8).resolve()
    environment = os.environ.copy()
    directories = [str(prefix / 'lib')]
    plugin = prefix / 'lib/libpythia8lhapdf6.so'
    if plugin.is_file():
        candidates = set()
        for name in ('Makefile.inc', 'config.mk'):
            config = prefix / name
            if not config.is_file():
                continue
            match = re.search(r'^LHAPDF6_LIB\s*=\s*(.*)$', config.read_text(), re.M)
            if match:
                for token in shlex.split(match[1]):
                    if token.startswith('-L') and (Path(token[2:]) / 'libLHAPDF.so').is_file():
                        candidates.add(str(Path(token[2:]).resolve()))
        if not candidates:
            # Installed libraries retain the compiler's recorded ABI path.
            for library in (plugin,):
                if not library.is_file():
                    continue
                result = subprocess.run(['readelf', '-d', str(library)], check=True,
                                        capture_output=True, text=True)
                for value in re.findall(r'\((?:RUNPATH|RPATH)\).*?\[(.*?)\]', result.stdout):
                    candidates.update(str(Path(item).resolve()) for item in value.split(':')
                                      if item and (Path(item) / 'libLHAPDF.so').is_file())
        if len(candidates) != 1:
            raise MatchingError('Cannot identify a unique configured Pythia LHAPDF shared backend')
        directories.extend(sorted(candidates))
    inherited = environment.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(
        path for path in directories + inherited if path))
    return environment


def shower_environment(pythia8, pdf_info):
    environment = native_library_environment(pythia8)
    environment['LHAPDF_DATA_PATH'] = str(pdf_info.parent.parent) + os.pathsep + environment.get('LHAPDF_DATA_PATH', '')
    environment['PYTHIA8DATA'] = str(Path(pythia8).resolve() / 'share/Pythia8/xmldoc')
    return environment


def pdf_member_hash(pdf, info):
    member = int(pdf.rsplit('/', 1)[1])
    return sha256(info.with_name('%s_%04d.dat' % (info.stem, member)))


def shower_library_hashes(pythia8):
    directory = Path(pythia8).resolve() / 'lib'
    libraries = [p for p in directory.glob('libpythia8*')
                 if p.is_file() and p.suffix in ('.so', '.dylib', '.a')]
    if not libraries:
        raise MatchingError('No Pythia libraries in %s' % directory)
    return {p.name: sha256(p) for p in sorted(libraries)}


def build_shower(process_dir, pythia8, log, pdf=None):
    """Build the matching driver against an ordinary Pythia installation.

    The deterministic sidecar ``str(log) + '.build.json'`` pins the complete
    runtime input. It must accompany generation/runtime resume checks.
    """
    process_dir, prefix, log = Path(process_dir).resolve(), Path(pythia8).resolve(), Path(log)
    directory = process_dir / 'MCatNLO/srcPythia8'
    target = 'MacNLOPS'
    environment = native_library_environment(prefix)
    command = ['make', target, 'PYTHIA8LOCATION=' + str(prefix)]
    backend = None
    for item in environment['LD_LIBRARY_PATH'].split(os.pathsep)[1:]:
        candidate = Path(item) / 'libLHAPDF.so'
        if candidate.is_file():
            backend = candidate.resolve()
            break
    pdf_info = None
    if pdf is not None:
        card = run_card_values((process_dir / 'Cards/run_card.dat').read_text())
        actual_pdf, pdf_info = pdf_name(process_dir, card, with_info=True)
        if pdf != actual_pdf:
            raise MatchingError('Shower PDF differs from the source run card')
    with log.open('w') as stream:
        subprocess.run(command, cwd=directory, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, env=environment)
    driver = directory / (target + '.exe')
    pins = {name: sha256(directory / name) for name in
            ('MacNLOPS.cc', 'Makefile') if (directory / name).is_file()}
    libraries = shower_library_hashes(prefix)
    linked = subprocess.run(['ldd', str(driver)], capture_output=True, text=True,
                            check=True, env=environment).stdout
    dependencies = {}
    for name, path in re.findall(r'^\s*(\S+)\s+=>\s+(/\S+)', linked, re.M):
        if name.startswith(('libpythia8', 'libLHAPDF')):
            dependencies[name] = dict(path=path, sha256=sha256(path))
    record = dict(schema='macnlops-native-build-v1', command=command,
                  native_environment={key: environment[key] for key in ('LD_LIBRARY_PATH',)},
                  configured_lhapdf_backend=(dict(path=str(backend), sha256=sha256(backend))
                                             if backend else None),
                  cwd=str(directory), target=target, driver=str(driver),
                  driver_sha256=sha256(driver), sources=pins, pythia_prefix=str(prefix),
                  libraries=libraries, dynamic_dependencies=dependencies,
                  shower_backend='stock-pythia8',
                  pdf=pdf, pdf_info=(dict(path=str(pdf_info), sha256=sha256(pdf_info),
                       member_sha256=pdf_member_hash(pdf, pdf_info)) if pdf_info else None))
    Path(str(log) + '.build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return driver


def probe_shower(driver, output, beam_ids, energies, pdf, alpha, order,
                 settings, environment, log, flavour_count=5, mcatnlo_delta=False):
    command = [str(driver), '--probe-config', str(output),
               '--beam-id-a', str(beam_ids[0]), '--beam-id-b', str(beam_ids[1]),
               '--beam-energy-a', str(energies[0]), '--beam-energy-b', str(energies[1]),
               '--pdf', pdf, '--alpha-s-mz', str(alpha), '--alpha-s-order', str(order),
               '--settings', str(settings),
               '--flavour-count', str(flavour_count)]
    if mcatnlo_delta:
        command.append('--mcatnlo-delta')
    with open(log, 'w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env=environment)
    result = json.loads(output.read_text())
    if result.get('schema') != 'macnlops-shower-config-v1':
        raise MatchingError('Unknown native Pythia matching profile')
    return result


def prepare_generation(process_dir, run_card, pythia8):
    """Measure shower support before integration and export it to Fortran.

    The profile is a generation input. Rejection never retrofits a different
    threshold into a previously generated hard sample.
    """
    process_dir = Path(process_dir).resolve()
    construction = source_construction(run_card)
    if not pythia8:
        raise MatchingError('macnlops=True requires pythia8_path before generation')
    if any(abs(run_card[key]) != 1 for key in ('lpp1', 'lpp2')):
        raise MatchingError('The validated MAcNLOPS profile requires proton or antiproton beams')
    ids = run_card['lhaid']
    if not isinstance(ids, (list, tuple)):
        ids = [ids]
    if len(ids) != 1:
        raise MatchingError('A single central LHAPDF member is required')
    pdf, info = pdf_name(process_dir, dict(pdlabel=run_card['pdlabel'],
                                          lhaid=str(ids[0])), with_info=True)
    flavours = re.search(r'^NumFlavors:\s*(\d+)', info.read_text(), re.M)
    if flavours and int(flavours[1]) != run_card['macnlops_flavour_count']:
        raise MatchingError('PDF flavour count differs from macnlops_flavour_count')
    settings = process_dir / 'Cards/macnlops_shower.cmnd'
    custom = process_dir / 'Cards/macnlops_custom.cmnd'
    alpha, order = shower_parameters(process_dir, info, settings,
                                     custom if custom.exists() else None)
    driver = build_shower(process_dir, pythia8, process_dir / 'macnlops_build.log', pdf=pdf)
    native_build = Path(str(process_dir / 'macnlops_build.log') + '.build.json')
    probe = process_dir / 'Cards/macnlops_probe.json'
    support = probe_shower(driver, probe,
        [2212 * run_card[key] for key in ('lpp1', 'lpp2')],
        [run_card[key] for key in ('ebeam1', 'ebeam2')], pdf, alpha, order, settings,
        shower_environment(pythia8, info), process_dir / 'macnlops_probe.log',
        flavour_count=run_card['macnlops_flavour_count'],
        mcatnlo_delta=construction == DELTA_SOURCE)
    profile = dict(schema='macnlops-generation-profile-v1', shower_backend='stock-pythia8',
                   construction=construction, support=support,
                   shower_libraries=shower_library_hashes(pythia8),
                   pdf=pdf, pdf_info_sha256=sha256(info),
                   pdf_member_sha256=pdf_member_hash(pdf, info),
                   settings_sha256=sha256(settings), alpha_s_mz=alpha, alpha_s_order=order)
    if native_build.is_file():
        profile['native_build'] = dict(path=str(native_build), sha256=sha256(native_build))
    (process_dir / 'Cards/macnlops_profile.json').write_text(
        json.dumps(profile, indent=2, sort_keys=True) + '\n')
    # RunCardNLO also restores these from the profile when Source/makefile
    # rewrites run_card.inc from the card on disk.
    for key, value in run_card.macnlops_profile_values(support).items():
        run_card.set(key, value, user=False)
    return profile


def generation_profile(process_dir):
    path = process_dir / 'Cards/macnlops_profile.json'
    if not path.is_file():
        raise MatchingError('Missing measured generation profile; generate a fresh macnlops=True sample')
    profile = json.loads(path.read_text())
    if (profile.get('schema') != 'macnlops-generation-profile-v1' or
            profile.get('construction') not in (ORDINARY_SOURCE, RESOLVED_SOURCE, DELTA_SOURCE)):
        raise MatchingError('Unknown MAcNLOPS source construction')
    card = run_card_values((Path(process_dir) / 'Cards/run_card.dat').read_text())
    if profile['construction'] != source_construction(card):
        raise MatchingError('Generation profile and run-card source constructions differ; generate a fresh common input')
    return profile


def run_generated_sample(process_dir, filename, pythia8, seed):
    """Dispatch a normal matched launch to the dedicated first-step runner."""
    process_dir, filename = Path(process_dir).resolve(), Path(filename).resolve()
    if not filename.exists() and Path(str(filename) + '.gz').exists():
        filename = Path(str(filename) + '.gz')
    manifest = check_manifest(filename, process_dir)
    inputs = []
    listing = process_dir / 'SubProcesses/nevents_unweighted'
    if listing.is_file():
        for line in listing.read_text().splitlines():
            fields = line.split()
            if not fields or int(fields[1]) <= 0:
                continue
            candidate = process_dir / 'SubProcesses' / fields[0]
            record = check_manifest(candidate, process_dir)
            if record['run'] != manifest['run'] or record['sources'] != manifest['sources']:
                # An earlier collected run may outlive the current raw files.
                inputs = []
                break
            inputs.append(candidate)
    if not inputs:
        inputs = [filename]
    output = filename.parent / 'macnlops'
    index = 1
    while output.exists():
        index += 1
        output = filename.parent / ('macnlops_%d' % index)
    seed = 1 + int(seed) % (899999998 - len(inputs))
    main(['--process-dir', str(process_dir), '--pythia8', str(pythia8),
          '--output-dir', str(output), '--seed', str(seed),
          *(str(path) for path in inputs)])
    return output / 'events.lhe.gz'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process-dir', required=True, type=Path,
                        help='fresh NLO output containing the generation cards and evaluator')
    parser.add_argument('--pythia8', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    residual = parser.add_mutually_exclusive_group()
    residual.add_argument('--signed-residuals', dest='signed_residuals', action='store_true',
                          help='retain negative H density that a bounded S veto cannot cancel')
    residual.add_argument('--strict-positive-h', dest='signed_residuals', action='store_false',
                          help='fail if any negative H density cannot be removed completely')
    parser.set_defaults(signed_residuals=None)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    return run_files(args)


def run_files(args):
    process_dir = args.process_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not 0 < args.seed < 900000000 - len(args.inputs):
        raise MatchingError('Seed must be in Pythia range 1..899999999')
    output_dir.mkdir(parents=True, exist_ok=True)
    collected = output_dir / 'events.lhe.gz'
    if collected.exists():
        raise MatchingError('Output already exists: %s' % collected)
    names = [path.name.removesuffix('.gz').removesuffix('.lhe') for path in args.inputs]
    if len(set(names)) != len(names):
        names = ['__'.join(path.resolve().relative_to(process_dir).parts).removesuffix('.gz').removesuffix('.lhe')
                 if path.resolve().is_relative_to(process_dir) else
                 hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12] + '__' + name
                 for path, name in zip(args.inputs, names)]
    if len(set(names)) != len(names):
        raise MatchingError('The same input file was supplied more than once')
    # Build only against the requested installation; no HepMC dependency.
    driver = build_shower(process_dir, args.pythia8, output_dir / 'build_shower.log')
    summary, reports = [], []
    for offset, (filename, name) in enumerate(zip(args.inputs, names)):
        job = output_dir / name
        job.mkdir(exist_ok=False)
        try:
            report = run_file(filename, process_dir, job, driver, args, args.seed + offset)
            print('%s: %s' % (job / 'events.lhe.gz', json.dumps(report['counts'], sort_keys=True)))
            summary.append(dict(input=str(filename), status='complete', output=str(job / 'events.lhe.gz')))
            reports.append(report)
        except (MatchingError, subprocess.CalledProcessError, OSError) as exc:
            report = dict(input=str(filename), status='failed', error=str(exc))
            (job / 'failure.json').write_text(json.dumps(report, indent=2) + '\n')
            summary.append(report)
            print('%s: %s' % (filename, exc))
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    failed = sum(record['status'] == 'failed' for record in summary)
    if failed:
        raise MatchingError('%d of %d source files failed; see %s' %
                            (failed, len(summary), output_dir / 'summary.json'))
    try:
        collection = collect_matched_events([Path(record['output']) for record in summary],
                                           reports, collected, args.seed)
    except (MatchingError, OSError) as exc:
        summary.append(dict(stage='collection', status='failed', error=str(exc)))
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        raise
    (output_dir / 'collection.json').write_text(json.dumps(collection, indent=2, sort_keys=True) + '\n')
    for record in summary:
        record['collected_output'] = str(collected)
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Collected MAcNLOPS events: %s' % collected)
    return collected


def run_file(filename, process_dir, job, driver, args, seed):
    manifest = check_manifest(filename, process_dir)
    header = source_header(filename, process_dir, manifest)
    card = check_source(header, process_dir)
    central_pdf, pdf_info = pdf_name(process_dir, card, with_info=True)
    profile = generation_profile(process_dir)
    contract = hard_source_contract(filename, profile['construction'])
    policy = matching_prescription(profile['construction'], contract['partition'])
    # Each event file belongs to one integration channel; fail before showering.
    normalisation = (channel_normalisation(filename) if policy == UNDAMPED_DELTA_POLICY
                     else None)
    signed_residuals = getattr(args, 'signed_residuals', None)
    if signed_residuals is None:
        signed_residuals = profile['construction'] != RESOLVED_SOURCE
    if shower_library_hashes(args.pythia8) != profile['shower_libraries']:
        raise MatchingError('Pythia libraries differ from those used before generation')
    if (central_pdf != profile['pdf'] or sha256(pdf_info) != profile['pdf_info_sha256'] or
            pdf_member_hash(central_pdf, pdf_info) != profile['pdf_member_sha256']):
        raise MatchingError('The source MSbar PDF metadata changed since generation')
    environment = shower_environment(args.pythia8, pdf_info)
    settings = job / 'shower_input.cmnd'
    if args.settings:
        shower_parameters(process_dir, pdf_info, settings, args.settings)
    else:
        shutil.copyfile(process_dir / 'Cards/macnlops_shower.cmnd', settings)
    if sha256(settings) != profile['settings_sha256']:
        raise MatchingError('Shower settings differ from those used before generation')
    alpha_s, alpha_order = profile['alpha_s_mz'], profile['alpha_s_order']
    support = probe_shower(driver, job / 'probe.json', profile['support']['beam_ids'],
        profile['support']['beam_energies_lab'], central_pdf, alpha_s, alpha_order,
        settings, environment, job / 'probe.log',
        flavour_count=profile['support']['flavour_count'],
        mcatnlo_delta=profile['construction'] == DELTA_SOURCE)
    if support != profile['support']:
        raise MatchingError('Actual Pythia profile differs from the generation profile')
    proposal = job / 'proposals.lhe'
    metadata = job / 'shower.jsonl'
    # Decompress independently of Pythia's optional gzip build support.
    with tempfile.TemporaryDirectory(prefix='macnlops_input_', dir=job) as temporary:
        plain = Path(temporary) / 'input.lhe'
        with LHEStream(filename) as original, open(plain, 'w') as target:
            target.write(header)
            for block in original:
                target.write(block)
            target.write('</LesHouchesEvents>\n')
        command = [str(driver), '--input', str(plain), '--output', str(proposal),
                   '--metadata', str(metadata), '--seed', str(seed),
                   '--pdf', central_pdf, '--alpha-s-mz', str(alpha_s),
                   '--alpha-s-order', str(alpha_order), '--settings', str(settings),
                   '--flavour-count', str(support['flavour_count'])]
        if profile['construction'] == DELTA_SOURCE:
            command.append('--mcatnlo-delta')
        with open(job / 'shower.log', 'w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    shower_profile = recorded_shower_profile(profile, Path(str(metadata) + '.settings.cmnd'), environment)
    with NativeEvaluator(process_dir, job, environment=environment, auxiliary_seed=seed,
                         partition=contract['partition']) as evaluator:
        report = reject_proposals(proposal, metadata, job / 'events.lhe.gz', evaluator,
                                  seed=seed, profile=support,
                                  signed_residuals=signed_residuals, require_continuation=True,
                                  shower_profile=shower_profile, matching_policy=policy,
                                  beta=normalisation['beta'] if normalisation else None)
        report['shower_support'] = evaluator.configuration
        report['channel_normalisation'] = normalisation
        if profile['construction'] == DELTA_SOURCE and contract['partition'] == 'global_sum':
            report['delta_auxiliary'] = dict(seed=seed,
                sampling='independent conditional source-H law; bounded C/K_g',
                owner_law_certificates=[dict(pdgs=list(ids), sha256=digest)
                    for ids, digest in sorted(evaluator.delta_law_certificates.items())])
    report.update(input=str(filename.resolve()), input_sha256=sha256(filename),
                  hard_source_contract=contract,
                  matching_partition=contract['partition'],
                  source_construction=profile['construction'],
                  pdf=central_pdf, shower=str(args.pythia8.resolve()),
                  normalization='sum; retained input event weights',
                  cards={name: sha256(process_dir / 'Cards' / name)
                         for name in ('run_card.dat', 'param_card.dat')},
                  fks_parameters=sha256(process_dir / 'SubProcesses/FKS_params.dat'),
                  pdf_info_sha256=sha256(pdf_info),
                  effective_settings_sha256=sha256(Path(str(metadata) + '.settings.cmnd')))
    (job / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


if __name__ == '__main__':
    try:
        main()
    except (MatchingError, subprocess.CalledProcessError, OSError) as error:
        print('MAcNLOPS: ' + str(error), file=sys.stderr)
        sys.exit(1)
