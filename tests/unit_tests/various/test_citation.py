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
from __future__ import absolute_import
import os
import re
import shutil
import tempfile
import tests.unit_tests as unittest

import madgraph.various.citation as citation
pjoin = os.path.join


class TestCitation(unittest.TestCase):
    """Check the run-wide citation tracking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.citation_dir = pjoin(self.tmpdir, 'Events', 'run_01', 'citations')
        os.makedirs(self.citation_dir)
        # point a fresh recorder at the temporary directory
        self._old_env = os.environ.get(citation.ENV_VAR)
        os.environ[citation.ENV_VAR] = self.citation_dir
        self.logger = citation.CitationLogger()

    def tearDown(self):
        if self._old_env is None:
            os.environ.pop(citation.ENV_VAR, None)
        else:
            os.environ[citation.ENV_VAR] = self._old_env
        shutil.rmtree(self.tmpdir)

    def _write_extra_log(self, name, lines):
        with open(pjoin(self.citation_dir, name), 'w') as fsock:
            fsock.write('\n'.join(lines) + '\n')

    def test_cite_writes_logfile(self):
        self.logger.cite('Alwall:2014hca', 'core')
        logs = [f for f in os.listdir(self.citation_dir)
                if f.startswith('cite.') and f.endswith('.log')]
        self.assertEqual(len(logs), 1)
        content = open(pjoin(self.citation_dir, logs[0])).read()
        self.assertEqual(content, 'Alwall:2014hca\tcore\n')

    def test_cite_deduplicates(self):
        self.logger.cite('Alwall:2014hca', 'core')
        self.logger.cite('Alwall:2014hca', 'core')
        logs = [f for f in os.listdir(self.citation_dir) if f.startswith('cite.')]
        content = open(pjoin(self.citation_dir, logs[0])).read()
        self.assertEqual(content.count('Alwall:2014hca'), 1)

    def test_cite_noop_without_env(self):
        os.environ.pop(citation.ENV_VAR, None)
        local = citation.CitationLogger()
        local.cite('Alwall:2014hca', 'core')  # must not raise
        self.assertEqual(os.listdir(self.citation_dir), [])

    def test_collect_unions_and_keeps_contexts(self):
        self.logger.cite('Alwall:2014hca', 'core')
        self._write_extra_log('cite.other.999.log', [
            'Alwall:2014hca\tcore',          # duplicate across processes
            'Frederix:2012ps\tFxFx merging',
            '',                              # blank line tolerated
            'deAquino:2011ub',               # no context
        ])
        collected = citation.collect(self.citation_dir)
        self.assertEqual(set(collected),
                         {'Alwall:2014hca', 'Frederix:2012ps', 'deAquino:2011ub'})
        self.assertEqual(collected['Alwall:2014hca'], ['core'])
        self.assertEqual(collected['deAquino:2011ub'], [])

    def test_collect_empty_dir(self):
        self.assertEqual(citation.collect(self.citation_dir), {})
        self.assertEqual(citation.collect('/does/not/exist'), {})

    def test_bibliography_parsing(self):
        bib = citation.Bibliography()
        self.assertIn('Alwall:2014hca', bib)
        self.assertIn('automated computation', bib.title('Alwall:2014hca'))
        self.assertIsNotNone(bib.entry('Alwall:2014hca'))
        self.assertIsNone(bib.entry('Nope:0000zz'))

    def test_write_bibtex_reports_missing(self):
        collected = {'Alwall:2014hca': ['core'], 'Bogus:9999xx': ['x']}
        out = pjoin(self.tmpdir, 'out.bib')
        missing = citation.write_bibtex(collected, out)
        self.assertEqual(missing, ['Bogus:9999xx'])
        text = open(out).read()
        self.assertIn('@article{Alwall:2014hca', text)
        self.assertNotIn('Bogus:9999xx', text)

    def test_finalize_produces_two_files(self):
        self.logger.cite('Alwall:2014hca', 'core matrix-element generation')
        self.logger.cite('Artoisenet:2012st', 'spin-correlated decays (MadSpin)')
        result = citation.finalize(self.citation_dir,
                                   output_dir=os.path.dirname(self.citation_dir),
                                   run_name='run_01')
        self.assertIsNotNone(result)
        out_bib, out_md = result
        self.assertTrue(os.path.exists(out_bib))
        self.assertTrue(os.path.exists(out_md))
        md = open(out_md).read()
        self.assertIn('run_01', md)
        self.assertIn('spin-correlated decays (MadSpin)', md)
        self.assertIn('Alwall:2014hca', md)

    def test_finalize_no_citation_returns_none(self):
        self.assertIsNone(citation.finalize(self.citation_dir))

    def test_write_log_and_collect_file(self):
        path = pjoin(self.tmpdir, 'citations.log')
        citation.write_log(path, [('Alwall:2014hca', 'framework'),
                                  ('Alwall:2014hca', 'framework'),  # dup dropped
                                  ('deAquino:2011ub', 'ALOHA')])
        collected = citation.collect_file(path)
        self.assertEqual(collected['Alwall:2014hca'], ['framework'])
        self.assertEqual(set(collected), {'Alwall:2014hca', 'deAquino:2011ub'})

    def test_generation_pairs_render_to_dir(self):
        # simulate what madgraph_interface.write_generation_citations does
        pairs = citation.generation_pairs('sm', is_ufo=True)
        gen_log = pjoin(self.tmpdir, 'citations.log')
        citation.write_log(gen_log, pairs)
        result = citation.render(citation.collect_file(gen_log), self.tmpdir)
        self.assertIsNotNone(result)
        md = open(result[1]).read()
        self.assertIn('UFO model format (model: sm)', md)
        self.assertIn('MadGraph5_aMC@NLO framework', md)

    def test_generation_log_seeds_a_run(self):
        # generation log copied into the run dir is collected by the run
        with open(pjoin(self.citation_dir, 'cite.generation.log'), 'w') as f:
            f.write('Degrande:2011ua\tUFO model format\n')
        self.logger.cite('Alwall:2014hca', 'core matrix-element generation')
        collected = citation.collect(self.citation_dir)
        self.assertIn('Degrande:2011ua', collected)
        self.assertIn('Alwall:2014hca', collected)


class TestCitationDatabaseIntegrity(unittest.TestCase):
    """Guard against malformed entries in the shipped citations.bib."""

    def test_database_loads(self):
        bib = citation.Bibliography()
        # every parsed entry must expose a non-empty title
        self.assertTrue(bib._entries)
        for key in bib._entries:
            self.assertTrue(bib.entry(key).strip().startswith('@'),
                            'entry %s is malformed' % key)

    def test_generation_pairs_keys_known(self):
        """Every key returned by generation_pairs (both UFO and v4 branches)
        must exist in citations.bib."""
        bib = citation.Bibliography()
        keys = set()
        for is_ufo in (True, False):
            keys.update(k for k, _ in citation.generation_pairs('sm', is_ufo))
        missing = sorted(k for k in keys if k not in bib)
        self.assertFalse(missing, 'generation keys missing from bib: %s' % missing)

    def test_optional_generation_pairs_keys_known(self):
        """Every key optional_generation_pairs can emit (gauge axial/FD,
        polarization, taudecay) must exist in citations.bib."""
        bib = citation.Bibliography()
        keys = set()
        for gauge in ('axial', 'FD', 'unitary', 'Feynman'):
            for pol in (True, False):
                for tau in (True, False):
                    keys.update(k for k, _ in citation.optional_generation_pairs(
                        gauge=gauge, polarization=pol, taudecay=tau))
        missing = sorted(k for k in keys if k not in bib)
        self.assertFalse(missing, 'optional keys missing from bib: %s' % missing)

    def test_every_cited_key_is_known(self):
        """Every key cited anywhere in the source tree (Python, Fortran, C++)
        must exist in citations.bib, otherwise it would be silently dropped from
        the generated bibliography."""
        try:
            from madgraph import MG5DIR as root
        except ImportError:
            root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # match cite("KEY" / cite('KEY' / call cite('KEY' / mg5::cite("KEY"
        pattern = re.compile(r"""cite\s*\(\s*['"]([A-Za-z0-9_:.+\-]+)['"]""")
        scan_exts = ('.py', '.f', '.f90', '.cc', '.cpp', '.cxx', '.h', '.hh')
        scan_dirs = [pjoin(root, 'madgraph'), pjoin(root, 'Template')]

        bib = citation.Bibliography()
        unknown = {}
        for base in scan_dirs:
            if not os.path.isdir(base):
                continue
            for dirpath, _, filenames in os.walk(base):
                if os.sep + 'tests' + os.sep in dirpath + os.sep:
                    continue
                for filename in filenames:
                    # skip the infrastructure itself (its docstrings hold examples)
                    if filename in ('citation.py', 'mg5_citation.f',
                                    'mg5_citation.cc'):
                        continue
                    if not filename.endswith(scan_exts):
                        continue
                    path = pjoin(dirpath, filename)
                    try:
                        with open(path, encoding='utf-8', errors='ignore') as fsock:
                            text = fsock.read()
                    except (OSError, IOError):
                        continue
                    for key in pattern.findall(text):
                        if key not in bib:
                            unknown.setdefault(key, []).append(
                                os.path.relpath(path, root))
        self.assertFalse(unknown,
            'cited key(s) missing from citations.bib: %s' % unknown)


if __name__ == '__main__':
    unittest.unittest.main()
