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
"""Run-wide citation tracking.

A piece of code (Python, Fortran or C++) records that it relies on a given
reference by *citing its key* -- a short INSPIRE-HEP texkey such as
``Alwall:2014hca``.  The full bibliographic information lives only once, in the
companion ``citations.bib`` file; the running code never carries BibTeX text
around, it only ever emits a key (plus an optional free-text *context* saying
what the reference was used for).

Mechanism
---------
Every process keeps an in-memory set of the (key, context) pairs it has already
seen and, on a *new* pair, appends one line to a per-process file::

    $MG5_CITATION_DIR/cite.<host>.<pid>.log      # one "key<TAB>context" per line

Per-process file names mean there is never a concurrent-write race, on any
filesystem.  If ``$MG5_CITATION_DIR`` is unset or empty the call is a no-op.

At the end of a run the orchestrator collects every ``cite.*.log`` in the
directory, unions the keys, looks them up in ``citations.bib`` and writes two
user-facing deliverables next to the run:

* ``citations.bib`` -- ready to drop into a paper;
* ``citations.md``  -- a context-driven summary of what was used and for what.

The Fortran (``mg5_citation.f``) and C++ (``mg5_citation.cc``) helpers write the
exact same ``cite.<host>.<pid>.log`` format, so they are collected by the same
code path.
"""

from __future__ import absolute_import
import logging
import os
import re
import socket
import glob

logger = logging.getLogger('madgraph.citation')

ENV_VAR = 'MG5_CITATION_DIR'

# location of the curated bibliography shipped with the code
_ROOT = os.path.dirname(os.path.realpath(__file__))
DEFAULT_BIB = os.path.join(_ROOT, 'citations.bib')


def _safe(text):
    """Make a string safe to use inside a file name."""
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(text))


class CitationLogger(object):
    """Per-process recorder of cited references.

    Records are de-duplicated in memory and appended to
    ``$MG5_CITATION_DIR/cite.<host>.<pid>.log``.  When the environment variable
    is not set the recorder is silently inactive, so library code can call
    :meth:`cite` unconditionally.
    """

    def __init__(self):
        self._seen = set()
        self._path = None
        self._resolved = False

    def _logfile(self):
        """Return the per-process log path, or None if tracking is disabled."""
        if not self._resolved:
            self._resolved = True
            directory = os.environ.get(ENV_VAR)
            if directory:
                try:
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    name = 'cite.%s.%d.log' % (_safe(socket.gethostname()),
                                               os.getpid())
                    self._path = os.path.join(directory, name)
                except (OSError, IOError) as error:
                    logger.debug('citation tracking disabled: %s', error)
                    self._path = None
        return self._path

    def cite(self, key, context=''):
        """Record that reference ``key`` was used (optionally for ``context``).

        Safe to call many times and from anywhere: duplicates are dropped and
        any failure to write is swallowed -- citation tracking must never break
        a physics run.
        """
        if not key:
            return
        record = (key, context)
        if record in self._seen:
            return
        self._seen.add(record)
        path = self._logfile()
        if not path:
            return
        try:
            with open(path, 'a') as fsock:
                fsock.write('%s\t%s\n' % (key, context))
        except (OSError, IOError) as error:
            logger.debug('could not record citation %s: %s', key, error)


# module-level singleton + convenience wrapper -------------------------------

_LOGGER = CitationLogger()


def cite(key, context=''):
    """Record a citation from the current (Python) process. See CitationLogger."""
    _LOGGER.cite(key, context)


# ---------------------------------------------------------------------------
# Collection and rendering (run orchestrator side)
# ---------------------------------------------------------------------------

def _accumulate(lines, collected):
    """Parse ``key<TAB>context`` *lines* into the *collected* mapping."""
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        key, _, context = line.partition('\t')
        key = key.strip()
        context = context.strip()
        if not key:
            continue
        contexts = collected.setdefault(key, [])
        if context and context not in contexts:
            contexts.append(context)
    return collected


def collect(directory):
    """Union all ``cite.*.log`` files in *directory*.

    Returns an ordered ``{key: [context, ...]}`` mapping.  Order of first
    appearance is preserved for both keys and contexts so the rendered output is
    stable.  Empty contexts are dropped from the list (but the key is kept).
    """
    collected = {}
    if not directory or not os.path.isdir(directory):
        return collected
    for logfile in sorted(glob.glob(os.path.join(directory, 'cite.*.log'))):
        try:
            with open(logfile) as fsock:
                lines = fsock.readlines()
        except (OSError, IOError):
            continue
        _accumulate(lines, collected)
    return collected


def collect_file(path):
    """Parse a single ``key<TAB>context`` log file into ``{key: [context]}``."""
    collected = {}
    if not path or not os.path.isfile(path):
        return collected
    try:
        with open(path) as fsock:
            _accumulate(fsock.readlines(), collected)
    except (OSError, IOError):
        pass
    return collected


def generation_pairs(model_name='', is_ufo=True):
    """Return the ``(key, context)`` citations implied by *generating* code with
    the framework: the MadGraph5_aMC@NLO paper, the model description format
    (UFO, or HELAS for legacy v4 models) and the ALOHA/HELAS helicity routines.

    Keeping the key list here (rather than inline at the call site) makes it the
    single, CI-validated source for the generation-time references.
    """
    pairs = [('Alwall:2014hca', 'MadGraph5_aMC@NLO framework')]
    if is_ufo:
        ctx = 'UFO model format'
        if model_name:
            ctx = 'UFO model format (model: %s)' % model_name
        pairs.append(('Degrande:2011ua', ctx))
        pairs.append(('deAquino:2011ub', 'ALOHA helicity amplitude routines'))
    pairs.append(('Murayama:1992gi', 'HELAS helicity amplitudes'))
    return pairs


def optional_generation_pairs(gauge=None, polarization=False, taudecay=False):
    """Return the ``(key, context)`` citations implied by optional generation
    choices: the gauge (axial / Feynman-diagram), polarized matrix elements and
    the TauDecay add-on.  Single, CI-validated source for these keys.
    """
    pairs = []
    if gauge == 'axial':
        pairs.append(('Hagiwara:2020tbx', 'axial (parton-shower) gauge'))
    elif gauge == 'FD':
        pairs.append(('Hagiwara:2024xdh', 'Feynman-diagram (FD) gauge'))
    if polarization:
        pairs.append(('BuarqueFranzosi:2019boy', 'polarized matrix elements'))
    if taudecay:
        pairs.append(('Hagiwara:2012vz', 'polarized tau decays (TauDecay)'))
    return pairs


def write_log(path, pairs, append=False):
    """Write ``(key, context)`` *pairs* to a citation log in the standard
    ``key<TAB>context`` format.

    Used at code-generation time to persist, into the generated directory, the
    citations that are known up front (framework, model, ALOHA/HELAS, UFO
    format).  Duplicate pairs are dropped.  Overwrites by default; pass
    ``append=True`` to add to an existing log.
    """
    seen = set()
    with open(path, 'a' if append else 'w') as fsock:
        for key, context in pairs:
            if not key:
                continue
            context = context or ''
            if (key, context) in seen:
                continue
            seen.add((key, context))
            fsock.write('%s\t%s\n' % (key, context))


class Bibliography(object):
    """A lightweight read-only view on a BibTeX file, indexed by key."""

    _entry_re = re.compile(r'@(\w+)\s*\{\s*([^,\s]+)\s*,', re.IGNORECASE)
    _title_re = re.compile(r'title\s*=\s*[{"](.+?)["}]\s*,?\s*$',
                           re.IGNORECASE | re.DOTALL)

    def __init__(self, path=None):
        self.path = path or DEFAULT_BIB
        self._entries = {}   # key -> raw bibtex entry text
        self._titles = {}    # key -> title (best effort)
        if os.path.exists(self.path):
            self._parse()
        else:
            logger.warning('citation database not found: %s', self.path)

    def _parse(self):
        with open(self.path) as fsock:
            text = fsock.read()
        for match in self._entry_re.finditer(text):
            key = match.group(2)
            entry = self._extract_entry(text, match.start())
            self._entries[key] = entry
            self._titles[key] = self._extract_title(entry)

    @staticmethod
    def _extract_entry(text, start):
        """Return the full ``@type{...}`` entry beginning at *start*."""
        depth = 0
        i = text.index('{', start)
        for pos in range(i, len(text)):
            char = text[pos]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:pos + 1]
        return text[start:]

    @classmethod
    def _extract_title(cls, entry):
        match = re.search(r'title\s*=\s*[{"]', entry, re.IGNORECASE)
        if not match:
            return ''
        # balance braces/quotes starting at the opening delimiter
        start = match.end() - 1
        opener = entry[start]
        if opener == '{':
            depth = 0
            for pos in range(start, len(entry)):
                if entry[pos] == '{':
                    depth += 1
                elif entry[pos] == '}':
                    depth -= 1
                    if depth == 0:
                        title = entry[start + 1:pos]
                        break
            else:
                return ''
        else:  # quoted
            end = entry.index('"', start + 1)
            title = entry[start + 1:end]
        return ' '.join(title.replace('{', '').replace('}', '').split())

    def __contains__(self, key):
        return key in self._entries

    def entry(self, key):
        """Raw BibTeX entry for *key*, or None if unknown."""
        return self._entries.get(key)

    def title(self, key):
        """Human-readable title for *key* (best effort, possibly empty)."""
        return self._titles.get(key, '')


def write_bibtex(collected, out_path, bib=None):
    """Write a ready-to-use ``.bib`` with the entries for the cited keys.

    Returns the list of keys that were *not* found in the database (so the
    caller can warn the user / fail CI).
    """
    bib = bib or Bibliography()
    missing = []
    chunks = []
    for key in collected:
        entry = bib.entry(key)
        if entry is None:
            missing.append(key)
            continue
        chunks.append(entry.strip())
    with open(out_path, 'w') as fsock:
        fsock.write('%% Bibliography automatically generated by '
                    'MadGraph5_aMC@NLO.\n')
        fsock.write('%% It contains every reference relevant to this run.\n\n')
        fsock.write('\n\n'.join(chunks))
        fsock.write('\n')
    if missing:
        logger.warning('no BibTeX entry for cited key(s): %s',
                       ', '.join(missing))
    return missing


def write_report(collected, out_path, bib=None, run_name=None):
    """Write a context-driven Markdown summary of what was used and for what.

    The report is organised by *context*: each context becomes a section listing
    the references that were used for it.  Keys cited without any context are
    gathered under a generic section.
    """
    bib = bib or Bibliography()

    # invert: context -> [keys]   (preserving first-seen order)
    by_context = {}
    no_context = []
    for key, contexts in collected.items():
        if not contexts:
            no_context.append(key)
        for context in contexts:
            by_context.setdefault(context, [])
            if key not in by_context[context]:
                by_context[context].append(key)

    def render_key(key):
        title = bib.title(key)
        if title:
            return '- %s  \n  *(`%s`)*' % (title, key)
        return '- `%s`' % key

    lines = []
    header = 'References for this MadGraph5_aMC@NLO run'
    if run_name:
        header = 'References for run `%s`' % run_name
    lines.append('# %s' % header)
    lines.append('')
    lines.append('This run relied on the tools and methods listed below. '
                 'Please cite the corresponding references; a ready-to-use '
                 'BibTeX file is provided alongside this summary '
                 '(`citations.bib`).')
    lines.append('')

    for context in by_context:
        lines.append('## %s' % context)
        lines.append('')
        for key in by_context[context]:
            lines.append(render_key(key))
        lines.append('')

    if no_context:
        lines.append('## Other references')
        lines.append('')
        for key in no_context:
            lines.append(render_key(key))
        lines.append('')

    with open(out_path, 'w') as fsock:
        fsock.write('\n'.join(lines))


def render(collected, output_dir, run_name=None, bib_path=None):
    """Write ``citations.bib`` + ``citations.md`` for an already-collected
    ``{key: [context]}`` mapping into *output_dir*.

    Returns ``(out_bib, out_md)`` on success, or ``None`` if *collected* is empty.
    """
    if not collected:
        return None
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    bib = Bibliography(bib_path)
    out_bib = os.path.join(output_dir, 'citations.bib')
    out_md = os.path.join(output_dir, 'citations.md')
    write_bibtex(collected, out_bib, bib=bib)
    write_report(collected, out_md, bib=bib, run_name=run_name)
    return out_bib, out_md


def finalize(citation_dir, output_dir=None, run_name=None, bib_path=None):
    """Collect a run's citations and write the two user-facing deliverables.

    * ``citation_dir`` -- the ``$MG5_CITATION_DIR`` where ``cite.*.log`` live.
    * ``output_dir``   -- where to write ``citations.bib`` / ``citations.md``
      (defaults to ``citation_dir``).

    Returns ``(out_bib, out_md)`` on success, or ``None`` if nothing was cited.
    """
    return render(collect(citation_dir), output_dir or citation_dir,
                  run_name=run_name, bib_path=bib_path)
