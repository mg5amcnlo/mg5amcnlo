################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
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
"""Which program reweights the events: the classic reweighting module
(madgraph/interface/reweight_interface.py, f2py-compiled matrix elements) or
MadtRex (madtrex/, MG7 outputs only).

Both read the same Cards/reweight_card.dat. This module looks at that card
and at the process directory and decides which backend runs it:

- classify_card() lists what the card asks for: a "change backend" line, and
  the directives only one of the backends understands (with line numbers);
- probe_environment() describes the output (mg7/madevent/aMC@NLO, its run
  card's output_format) and which backends can run here;
- select_backend() combines the two with the backend the user asked for
  (command line, launch-question switch, card) into a Decision, or raises
  ReweightBackendError with every reason the request cannot be met.

This module is pure Python: it imports neither madspace nor madtrex (whether
MadtRex is installed is a file-system check), so it is safe to use when
neither is installed.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

BACKENDS = ('classic', 'madtrex', 'auto')

# The card line choosing the backend: "change backend classic|madtrex|auto".
_BACKEND_RE = re.compile(r'^\s*change\s+backend(?:\s+(?P<value>\S+))?\s*(?:#.*)?$', re.I)

# "change mode" values: LO is understood by both backends, pdf only by
# MadtRex, the NLO ones only by the classic module.
SHARED_MODES = frozenset(['lo'])
MADTREX_MODES = frozenset(['pdf', 'parameter'])
CLASSIC_MODES = frozenset(['nlo', 'nlo_tree', 'lo+nlo'])

# "change" options both backends understand (with the classic name and
# default); "mode" depends on its value, see above.
SHARED_CHANGE_OPTIONS = frozenset(['backend', 'model', 'process', 'helicity', 'mode'])

_MG_ROOT = Path(__file__).resolve().parents[2]


class ReweightBackendError(Exception):
    """The requested reweighting cannot run. reasons lists every reason."""

    def __init__(self, message, reasons=()):
        self.reasons = list(reasons)
        if self.reasons:
            message = '%s:\n%s' % (message, '\n'.join('  - %s' % r for r in self.reasons))
        super().__init__(message)


@dataclass
class Directive:
    """One reweight-card line only one of the backends understands."""
    line: int  # 1-based line number in the card
    text: str  # the line, without its comment
    why: str   # what makes it backend-specific

    def __str__(self):
        return 'line %d: %r (%s)' % (self.line, self.text, self.why)


@dataclass
class CardTraits:
    """What a reweight card asks for (see classify_card())."""
    # value of the card's "change backend" line, if any, and its line number
    backend: 'str | None' = None
    backend_line: 'int | None' = None
    classic_only: list = field(default_factory=list)  # list[Directive]
    madtrex_only: list = field(default_factory=list)  # list[Directive]
    # a density-matrix card (DensityInterface); also listed in classic_only
    density: bool = False
    # which MadtRex runner reads the card: "param" (matrix-element
    # reweighting), "pdf" (PDF/scale reweighting) or "generic" (sections
    # selected with "change mode LO|pdf")
    madtrex_kind: str = 'param'
    launches: int = 0


@dataclass
class Env:
    """The process directory and the installed backends (see
    probe_environment()); tests build it by hand."""
    # "mg7", "madevent", "aMC@NLO", or None when unknown
    output_type: 'str | None' = 'mg7'
    # the events to reweight are NLO events (an aMC@NLO output)
    nlo_events: bool = False
    # [run] output_format of an mg7 run_card.toml (compact_npy, lhe_npy, lhe)
    output_format: 'str | None' = None
    # the classic module can run (MadGraph + f2py), and why not otherwise
    classic_available: bool = True
    classic_reason: str = ''
    # madtrex/install/madtrex exists
    madtrex_installed: bool = False
    # why the installed madtrex cannot be used although it is there (built
    # against another madspace), or ''
    madtrex_problem: str = ''


@dataclass
class Decision:
    backend: str            # "classic" or "madtrex"
    card_kind: str          # MadtRex: param/pdf/generic; classic: "classic" or "density"
    reasons: list = field(default_factory=list)   # why this backend
    warnings: list = field(default_factory=list)  # e.g. disagreeing explicit requests
    source: str = 'auto'    # "auto", or which request chose the backend
    plugin: 'str | None' = None  # a classic reweighting plugin

    def describe(self, card_name='reweight_card.dat'):
        """One log line, e.g. "Reweighting with madtrex (auto: MG7 output,
        madtrex installed, no classic-only directives in reweight_card.dat)"."""
        reasons = ', '.join(r.replace('{card}', card_name) for r in self.reasons)
        return 'Reweighting with %s (%s%s)' % (
            self.backend, self.source, ': %s' % reasons if reasons else '')


def _madtrex_card_vocabulary():
    """The MadtRex-only reweight-card names, from the MadtRex card parsers
    (pure Python modules, importing neither madspace nor madtrex):
    ("change" options, "set NAME VALUE" names, the PDF-reweighting ones
    among both)."""
    from madgraph.iolibs.template_files.mg7 import param_reweight_card as param
    from madgraph.iolibs.template_files.mg7 import pdf_reweight_card as pdf
    old_driver = set(param.DRIVER_OPTION_NAMES) - {'helicity_reweighting', 'use_helicity'}
    change = {'fptype'} | old_driver | set(pdf.CARD_OPTION_NAMES)
    # the threading options never had a "set" spelling
    set_names = (set(pdf.PDF_SET_NAMES) | set(pdf.SCALE_NAMES)
                 | (change - param.THREAD_OPTION_NAMES)
                 | {'helicity_reweighting', 'use_helicity'})
    pdf_only = set(pdf.PDF_SET_NAMES) | set(pdf.SCALE_NAMES) | {'alpha_s_power',
                                                                 'scale_variation_output'}
    return frozenset(change), frozenset(set_names), frozenset(pdf_only)


def _classic_change_options():
    """("change" options only the classic module understands, the density
    ones among them)."""
    from madgraph.interface.reweight_interface import (REWEIGHT_CHANGE_OPTIONS,
                                                       DENSITY_CHANGE_OPTIONS)
    return (frozenset(REWEIGHT_CHANGE_OPTIONS | DENSITY_CHANGE_OPTIONS) - SHARED_CHANGE_OPTIONS,
            frozenset(DENSITY_CHANGE_OPTIONS))


def _check_backend_value(value, where):
    value = (value or '').lower()
    if value not in BACKENDS:
        raise ReweightBackendError(
            "%s: unknown reweighting backend %r; expected one of %s"
            % (where, value, ', '.join(BACKENDS)))
    return value


def classify_card(text):
    """CardTraits of the reweight card text: its "change backend" line, the
    lines only the classic module or only MadtRex understands, and which
    MadtRex card reader it needs. Lines neither backend knows are left to the
    backend that runs the card. Raises ReweightBackendError for an unknown
    or contradictory "change backend"."""
    madtrex_change, madtrex_set, pdf_names = _madtrex_card_vocabulary()
    classic_change, density_change = _classic_change_options()
    traits = CardTraits()
    has_mode_line = False
    has_pdf_names = False

    for number, raw in enumerate(text.splitlines(), 1):
        line = strip_comment(raw).strip()
        if not line:
            continue
        words = line.split()
        keyword = words[0].lower()

        if keyword == 'launch':
            traits.launches += 1
            continue

        if keyword == 'change' and len(words) > 1:
            name = words[1].lower()
            if name == 'backend':
                value = _check_backend_value(words[2] if len(words) > 2 else '',
                                             'reweight card line %d' % number)
                if traits.backend is not None and traits.backend != value:
                    raise ReweightBackendError(
                        "reweight card asks for two backends: '%s' (line %d) and "
                        "'%s' (line %d)" % (traits.backend, traits.backend_line,
                                            value, number))
                traits.backend, traits.backend_line = value, number
            elif name == 'mode':
                has_mode_line = True
                value = words[2].lower() if len(words) > 2 else ''
                if value in CLASSIC_MODES:
                    traits.classic_only.append(Directive(
                        number, line, 'NLO reweighting needs the classic module'))
                elif value in MADTREX_MODES:
                    traits.madtrex_only.append(Directive(
                        number, line, 'PDF reweighting needs MadtRex'))
                    has_pdf_names = has_pdf_names or value == 'pdf'
            elif name == 'process' and words[-1] == '--add':
                traits.classic_only.append(Directive(
                    number, line, 'several processes per model need the classic module'))
            elif name in classic_change:
                if name in density_change:
                    traits.density = True
                    why = 'density-matrix reweighting needs the classic module'
                else:
                    why = "'change %s' is a classic reweighting option" % name
                traits.classic_only.append(Directive(number, line, why))
            elif name in madtrex_change:
                traits.madtrex_only.append(Directive(
                    number, line, "'change %s' is a MadtRex option" % name))
                has_pdf_names = has_pdf_names or name in pdf_names
            continue

        if keyword == 'set' and len(words) == 3:
            # "set NAME VALUE"; "set BLOCK ID VALUE" has four words
            name = words[1].lower()
            if name == 'mode':
                has_mode_line = True
                traits.madtrex_only.append(Directive(
                    number, line, "'set mode' selects a MadtRex card section"))
            elif name in madtrex_set:
                why = ('a PDF/scale reweighting setting (MadtRex)' if name in pdf_names
                       else "'set %s' is a MadtRex option" % name)
                traits.madtrex_only.append(Directive(number, line, why))
                has_pdf_names = has_pdf_names or name in pdf_names
            continue

        if keyword not in ('set', 'change', 'launch', 'import') and \
                (os.sep in words[0] or words[0].endswith('.dat')):
            traits.classic_only.append(Directive(
                number, line, 'a param_card given by path needs the classic module'))

    if has_mode_line:
        traits.madtrex_kind = 'generic'
    elif has_pdf_names:
        traits.madtrex_kind = 'pdf'
    return traits


def strip_comment(line):
    """line without its comment ("#", or "!" at the start or after
    whitespace), as the MadtRex card readers strip it
    (generic_reweight_card.strip_comment)."""
    from madgraph.iolibs.template_files.mg7 import generic_reweight_card
    return generic_reweight_card.strip_comment(line)


def strip_backend_line(text):
    """text with its "change backend" lines commented out (the backends'
    own card readers need not know it; the line numbers stay the same)."""
    lines = text.splitlines(True)
    for i, line in enumerate(lines):
        if _BACKEND_RE.match(line):
            lines[i] = '# ' + line
    return ''.join(lines)


def _read_output_format(me_dir):
    """[run] output_format of Cards/run_card.toml, or None."""
    path = os.path.join(me_dir, 'Cards', 'run_card.toml')
    try:
        import tomllib
        with open(path, 'rb') as f:
            card = tomllib.load(f)
    except Exception:
        # tomllib is Python 3.11+; fall back to the one line we need
        try:
            with open(path) as f:
                text = f.read()
        except OSError:
            return None
        match = re.search(r'''^\s*output_format\s*=\s*["']([^"']*)["']''', text, re.M)
        return match.group(1) if match else None
    value = card.get('run', {}).get('output_format')
    if value is None:
        # MadGraph always writes it out; RunCardMG7's default otherwise
        return 'lhe_npy'
    return value


def output_type(me_dir):
    """"mg7", "madevent", "aMC@NLO" or None for the process directory me_dir."""
    if os.path.exists(os.path.join(me_dir, 'Cards', 'run_card.toml')):
        return 'mg7'
    if os.path.exists(os.path.join(me_dir, 'bin', 'internal', 'amcatnlo_run_interface.py')) or \
            os.path.exists(os.path.join(me_dir, 'Cards', 'FKS_params.dat')):
        return 'aMC@NLO'
    if os.path.exists(os.path.join(me_dir, 'bin', 'madevent')):
        return 'madevent'
    return None


def madtrex_is_installed(install_dir=None):
    """MadtRex is installed (madtrex/install/madtrex exists); nothing is
    imported."""
    install_dir = Path(install_dir) if install_dir else _MG_ROOT / 'madtrex' / 'install'
    return (install_dir / 'madtrex').is_dir()


def madtrex_build_problem(install_dir=None, madspace_dir=None):
    """Why the installed madtrex cannot be used with the madspace MadGraph
    uses (it was built against other madspace headers), or ''. Reads
    madtrex/_compat.py directly: nothing compiled is imported."""
    install_dir = Path(install_dir) if install_dir else _MG_ROOT / 'madtrex' / 'install'
    madspace_dir = (Path(madspace_dir) if madspace_dir
                    else _MG_ROOT / 'madspace' / 'install' / 'madspace')
    compat_path = install_dir / 'madtrex' / '_compat.py'
    if not compat_path.is_file() or not madspace_dir.is_dir():
        return ''
    import importlib.util
    spec = importlib.util.spec_from_file_location('_madtrex_compat', compat_path)
    compat = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(compat)
        return compat.build_problem(str(install_dir / 'madtrex'), str(madspace_dir)) or ''
    except Exception:
        return ''


def probe_environment(me_dir, options=None, madtrex_install_dir=None, has_f2py=None):
    """Env of the process directory me_dir. options are the MadGraph run
    options (mg5_path, f2py_compiler); has_f2py overrides the f2py lookup
    (misc.has_f2py)."""
    options = options or {}
    kind = output_type(me_dir)
    env = Env(output_type=kind, nlo_events=(kind == 'aMC@NLO'))
    if kind == 'mg7':
        env.output_format = _read_output_format(me_dir)

    # as AskRun.check_available_module: MadGraph and f2py. This module is
    # part of MadGraph, so MadGraph is there unless options say otherwise.
    mg5_path = options.get('mg5_path', str(_MG_ROOT))
    if has_f2py is None:
        from madgraph.various import misc
        has_f2py = misc.has_f2py() or bool(options.get('f2py_compiler'))
    if not mg5_path:
        env.classic_available = False
        env.classic_reason = 'the classic module needs MadGraph (mg5_path is not set)'
    elif not has_f2py:
        env.classic_available = False
        env.classic_reason = 'the classic module needs f2py, which was not found'
    env.madtrex_installed = madtrex_is_installed(madtrex_install_dir)
    if env.madtrex_installed:
        env.madtrex_problem = madtrex_build_problem(madtrex_install_dir)
    return env


def _madtrex_problems(env):
    """Why MadtRex cannot reweight these events (empty when it can)."""
    problems = []
    if env.output_type != 'mg7':
        problems.append('MadtRex needs an MG7 output (this is %s)'
                        % (env.output_type or 'not a known output'))
    if env.nlo_events:
        problems.append('MadtRex does not reweight NLO events')
    if env.output_format == 'compact_npy':
        problems.append("MadtRex needs run_card [run] output_format 'lhe' or 'lhe_npy': "
                        "'compact_npy' drops the flavour/helicity information it matches "
                        "events with")
    if not env.madtrex_installed:
        problems.append('MadtRex is not installed (python madtrex/install.py, or '
                        "'install madtrex' in MadGraph)")
    elif env.madtrex_problem:
        problems.append(env.madtrex_problem)
    return problems


def _resolve_request(requested, traits):
    """(backend, source, warnings) from the explicit requests, highest
    precedence first, then the card's "change backend"; backend is "auto"
    when nothing explicit was asked."""
    if requested is None:
        requested = []
    elif isinstance(requested, str):
        requested = [('requested', requested)]
    explicit = []
    for source, value in requested:
        if value is None:
            continue
        value = _check_backend_value(value, source)
        if value != 'auto':
            explicit.append((source, value))
    if traits.backend not in (None, 'auto'):
        explicit.append(("'change backend' (reweight_card.dat line %d)" % traits.backend_line,
                         traits.backend))
    if not explicit:
        return 'auto', 'auto', []
    source, value = explicit[0]
    warnings = ['%s asks for the %s backend but %s asks for %s: using %s'
                % (other, other_value, source, value, value)
                for other, other_value in explicit[1:] if other_value != value]
    return value, source, warnings


def _classic_kind(traits):
    return 'density' if traits.density else 'classic'


def select_backend(requested, traits, env, plugin=None):
    """The Decision for reweighting the card described by traits in env.

    requested is "classic"/"madtrex"/"auto"/None, or a list of (source,
    value) pairs, highest precedence first (the --reweight-backend option,
    then the launch-question switch); the card's own "change backend" comes
    after them. When explicit requests disagree, the highest one wins and
    Decision.warnings names both. plugin is a reweighting plugin selected in
    the launch question (classic only).

    Raises ReweightBackendError listing every reason when the card cannot
    be run: a forced backend that cannot run it, a card mixing classic-only
    and MadtRex-only directives, or no usable backend."""
    decision = _select(requested, traits, env, plugin)
    decision.plugin = plugin
    return decision


def _select(requested, traits, env, plugin):
    backend, source, warnings = _resolve_request(requested, traits)

    if backend == 'classic':
        problems = []
        if not env.classic_available:
            problems.append(env.classic_reason or 'the classic module is not available')
        problems += ['%s: not understood by the classic module' % d for d in traits.madtrex_only]
        if problems:
            raise ReweightBackendError('cannot reweight with the classic module (%s)' % source,
                                       problems)
        return Decision('classic', _classic_kind(traits), [], warnings, source)

    if backend == 'madtrex':
        problems = _madtrex_problems(env)
        if plugin:
            problems.append('reweighting plugin %r needs the classic module' % plugin)
        problems += ['%s: needs the classic module' % d for d in traits.classic_only]
        if problems:
            raise ReweightBackendError('cannot reweight with MadtRex (%s)' % source, problems)
        return Decision('madtrex', traits.madtrex_kind, [], warnings, source)

    # auto
    if traits.classic_only and traits.madtrex_only:
        raise ReweightBackendError(
            'the reweight card mixes directives only the classic module understands '
            'with directives only MadtRex understands',
            ['classic only, %s' % d for d in traits.classic_only]
            + ['MadtRex only, %s' % d for d in traits.madtrex_only])

    if traits.classic_only or plugin:
        if plugin:
            why = 'reweighting plugin %r' % plugin
        elif traits.density:
            why = 'density-matrix reweighting in {card}'
        else:
            why = 'classic-only directives in {card} (line %s)' % ', '.join(
                str(d.line) for d in traits.classic_only)
        if not env.classic_available:
            raise ReweightBackendError('cannot reweight: %s needs the classic module'
                                       % why.replace('{card}', 'reweight_card.dat'),
                                       [env.classic_reason])
        return Decision('classic', _classic_kind(traits), [why], warnings)

    if traits.madtrex_only:
        problems = _madtrex_problems(env)
        if problems:
            raise ReweightBackendError(
                'cannot reweight: reweight_card.dat has MadtRex-only directives (line %s)'
                % ', '.join(str(d.line) for d in traits.madtrex_only), problems)
        return Decision('madtrex', traits.madtrex_kind,
                        ['MadtRex-only directives in {card}'], warnings)

    madtrex_problems = _madtrex_problems(env)
    if env.output_type != 'mg7' or env.nlo_events:
        reasons = [p for p in madtrex_problems if 'MG7' in p or 'NLO' in p]
        if not env.classic_available:
            raise ReweightBackendError('cannot reweight: no backend can run here',
                                       [env.classic_reason] + reasons)
        return Decision('classic', _classic_kind(traits), reasons, warnings)

    if not madtrex_problems:
        return Decision('madtrex', traits.madtrex_kind,
                        ['MG7 output', 'madtrex installed',
                         'no classic-only directives in {card}'], warnings)

    if env.classic_available:
        return Decision('classic', _classic_kind(traits),
                        ['MadtRex unusable: %s' % '; '.join(madtrex_problems)], warnings)

    raise ReweightBackendError('cannot reweight: no backend can run here',
                               [env.classic_reason] + madtrex_problems)


def available_backends(env, plugin=None):
    """The backends that could reweight in env for a card with no
    backend-specific directives: a subset of ("classic", "madtrex")."""
    out = []
    if env.classic_available:
        out.append('classic')
    if not _madtrex_problems(env) and not plugin:
        out.append('madtrex')
    return out
