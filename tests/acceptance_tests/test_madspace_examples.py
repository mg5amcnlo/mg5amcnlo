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
"""Runs the Python examples embedded in the MadSpace user documentation.

Each ``docs/source/madspace/examples/*.rst`` page carries its example as one or
more ``.. code-block:: python`` directives, with prose in between. This module
extracts those blocks (in document order), concatenates them into one script
and runs it as a subprocess, so the documentation cannot silently drift from a
working example. There is one test method per example page.

Every test method self-skips when its dependencies (madspace, an LHAPDF set, a
GPU, ...) are unavailable, so it can be run on a laptop that only has a subset
installed. CI instead selects test methods by name, one per job, so that a
missing dependency there is a real failure rather than a silent skip --
``test_manager.py`` treats a skipped test the same as a failed one.

Run locally with e.g.::

    ./tests/test_manager.py test_madspace_example_simple_mappings -pA -t0 -l INFO
"""

from __future__ import absolute_import

import atexit
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from madgraph import MG5DIR

pjoin = os.path.join

# Harmless everywhere; avoids a common "OMP: Error #15" abort on setups (e.g.
# conda + Homebrew on macOS) where more than one OpenMP runtime ends up linked
# into the process. Set at import time, before any torch import in this
# process (has_torch()/has_madnis() below import it just to check), not only
# in the subprocess env run_doc_example() builds.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

_DOCS_DIR = pjoin(MG5DIR, 'docs', 'source')
_EXAMPLES_DIR = pjoin(_DOCS_DIR, 'madspace', 'examples')
_MADSPACE_INSTALL = pjoin(MG5DIR, 'madspace', 'install')

_CODE_BLOCK_RE = re.compile(r'^(?P<indent>[ \t]*)\.\.[ \t]+code-block::[ \t]*python[ \t]*$')


def extract_python_blocks(rst_path):
    """Return the bodies of every ``.. code-block:: python`` directive in
    *rst_path*, in document order, dedented. A block whose directive carries
    ``:class: no-run`` is skipped, so a page can show a snippet it does not
    want executed."""
    with open(rst_path) as f:
        lines = f.read().splitlines()

    blocks = []
    i = 0
    while i < len(lines):
        m = _CODE_BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        base_indent = len(m.group('indent'))
        i += 1

        options = []
        while i < len(lines) and lines[i].strip().startswith(':'):
            options.append(lines[i].strip())
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        body = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                body.append('')
                i += 1
                continue
            if len(line) - len(line.lstrip()) <= base_indent:
                break
            body.append(line)
            i += 1
        while body and not body[-1].strip():
            body.pop()

        if not any(':class: no-run' in o for o in options):
            blocks.append(textwrap.dedent('\n'.join(body)))

    return blocks


def extract_named_block(rst_path, name):
    """Return the dedented body of the code block whose directive carries
    ``:name: <name>``, or None if there is none."""
    with open(rst_path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ':name: %s' % name:
            continue
        j = i
        while not lines[j].lstrip().startswith('.. code-block::'):
            j -= 1
        base_indent = len(lines[j]) - len(lines[j].lstrip())
        k = i + 1
        while k < len(lines) and lines[k].strip().startswith(':'):
            k += 1
        body = []
        for line in lines[k:]:
            if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                break
            body.append(line)
        return textwrap.dedent('\n'.join(body)).strip()
    return None


def has_madspace():
    try:
        import madspace
        return hasattr(madspace, 'PhaseSpaceMapping')
    except ImportError:
        return False


def has_cuda_backend():
    try:
        import madspace
        madspace.cuda_device()
        return True
    except Exception:
        return False


def has_pdf_set(name):
    try:
        import lhapdf
    except ImportError:
        return False
    return any(os.path.isdir(pjoin(p, name)) for p in lhapdf.paths())


def has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def has_madnis():
    try:
        import madnis.integrator  # noqa: F401
        return True
    except ImportError:
        return False


def run_doc_example(test, page_name, cwd=None, examples_dir=_EXAMPLES_DIR):
    """Extract the python blocks of ``examples/<page_name>.rst``, concatenate
    them into one script, and run it as a subprocess. Fails the test if the
    script raises or if the page has no python blocks at all (a moved or
    renamed page must fail loudly, not silently pass).

    *cwd* lets an example that depends on a generated process directory (see
    ``ggttg_process_dir`` below) run with that directory as its working
    directory, matching what the page's shell commands build. Without it, a
    fresh empty temporary directory is used and cleaned up afterwards."""
    rst_path = pjoin(examples_dir, page_name + '.rst')
    if not os.path.isfile(rst_path):
        test.fail('documentation page not found: %s' % rst_path)

    blocks = extract_python_blocks(rst_path)
    test.assertTrue(blocks, 'no python code-block found in %s' % rst_path)
    script = '\n'.join(blocks)

    env = dict(os.environ)
    if os.path.isdir(_MADSPACE_INSTALL):
        env['PYTHONPATH'] = os.pathsep.join(
            [_MADSPACE_INSTALL] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
        )
    # Harmless everywhere; avoids a common "OMP: Error #15" abort on setups
    # (e.g. conda + Homebrew on macOS) where more than one OpenMP runtime is
    # linked into the process, which torch-based examples otherwise hit.
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

    def _run(directory):
        script_path = pjoin(directory, page_name.replace('-', '_') + '.py')
        with open(script_path, 'w') as f:
            f.write(script)
        return subprocess.run(
            [sys.executable, script_path],
            cwd=directory, env=env, capture_output=True, text=True,
        )

    if cwd is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _run(tmp_dir)
    else:
        result = _run(cwd)

    test.assertEqual(
        result.returncode, 0,
        'example %s exited with code %s\n--- stdout ---\n%s\n--- stderr ---\n%s'
        % (page_name, result.returncode, result.stdout, result.stderr),
    )


_GGTTG_PROCESS_CACHE = {}


def ggttg_process_dir():
    """Generate and compile ``g g > t t~ g`` as an mg7 output, once per test
    process, and return the directory that contains ``PROC_ggttg/`` -- the
    same relative name the matrix-element examples show being created by
    ``output mg7 PROC_ggttg``. Returns ``None`` if the MadGraph interface or
    a C++ compiler is unavailable, so dependent tests can self-skip."""
    if 'dir' in _GGTTG_PROCESS_CACHE:
        return _GGTTG_PROCESS_CACHE['dir']

    try:
        import madgraph.interface.master_interface as mg_interface
    except ImportError:
        _GGTTG_PROCESS_CACHE['dir'] = None
        return None

    scratch = tempfile.mkdtemp(prefix='madspace_doc_ggttg_')
    atexit.register(shutil.rmtree, scratch, ignore_errors=True)
    proc_dir = pjoin(scratch, 'PROC_ggttg')

    mg = mg_interface.MasterCmd()
    mg.no_notification()
    for c in ['set automatic_html_opening False --no_save',
              'import model sm',
              'generate g g > t t~ g']:
        mg.exec_cmd(c)
    mg.exec_cmd('output mg7 %s' % proc_dir)

    # BACKEND=scalar matches the docs pages, which hardcode the resulting
    # library name -- a plain 'make' would pick a machine-dependent SIMD
    # backend instead.
    make = subprocess.run(
        ['make', 'BACKEND=scalar'],
        cwd=pjoin(proc_dir, 'SubProcesses'), capture_output=True, text=True,
    )
    if make.returncode != 0:
        _GGTTG_PROCESS_CACHE['dir'] = None
        _GGTTG_PROCESS_CACHE['make_error'] = make.stdout + make.stderr
        return None

    _GGTTG_PROCESS_CACHE['dir'] = scratch
    return scratch


class TestMadSpaceExamples(unittest.TestCase):

    def test_madspace_example_simple_mappings(self):
        """docs/source/madspace/examples/simple-mappings.rst -- constructing
        and calling a Rambo/Chili PhaseSpaceMapping. Needs madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'simple-mappings')

    def test_madspace_example_cuts(self):
        """docs/source/madspace/examples/cuts.rst -- building fiducial cuts
        and folding them into a PhaseSpaceMapping. Needs madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'cuts')

    def test_madspace_example_diagram_mapping(self):
        """docs/source/madspace/examples/diagram-mapping.rst -- a mapping
        for one Feynman diagram, t-channel plus resonant s-channel. Needs
        madspace + numpy."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'diagram-mapping')

    def test_madspace_example_integration_order(self):
        """docs/source/madspace/examples/integration-order.rst -- multiple
        s-channel on-shell configurations and the propagator integration
        order, plus printing a Topology. Needs madspace only."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        run_doc_example(self, 'integration-order')

    def test_madspace_example_matrix_element(self):
        """docs/source/madspace/examples/matrix-element.rst -- loading a
        MadGraph-generated matrix element for g g > t t~ g through the UMAMI
        interface. Needs madspace, numpy and a C++ compiler; generates and
        compiles the process itself."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        scratch = ggttg_process_dir()
        if scratch is None:
            self.skipTest('could not generate/compile g g > t t~ g: %s'
                          % _GGTTG_PROCESS_CACHE.get('make_error', 'mg7 unavailable'))
        run_doc_example(self, 'matrix-element', cwd=scratch)

    def test_madspace_example_gpu(self):
        """docs/source/madspace/examples/gpu.rst -- sampling the same
        PhaseSpaceMapping on a CUDA device with PyTorch. Needs a CUDA-enabled
        madspace build and a GPU; not run in CI, which is CPU-only."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_cuda_backend():
            self.skipTest('CUDA backend unavailable')
        run_doc_example(self, 'gpu')

    def test_madspace_example_pdf(self):
        """docs/source/madspace/examples/pdf.rst -- the built-in PDF
        interpolator (PdfGrid/PartonDensity/AlphaSGrid/RunningCoupling) on
        the NNPDF40_lo_as_01180 set. Needs madspace, numpy, lhapdf and that
        PDF set installed."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_pdf_set('NNPDF40_lo_as_01180'):
            self.skipTest('NNPDF40_lo_as_01180 LHAPDF data not found '
                          '(set $LHAPDF_DATA_PATH)')
        run_doc_example(self, 'pdf')

    def test_madspace_example_integrator(self):
        """docs/source/madspace/examples/integrator.rst -- a hand-rolled,
        non-adaptive integrator for g g > t t~ g: PhaseSpaceMapping +
        DifferentialCrossSection (matrix element, PDF, HT/2 scale). Needs
        madspace, numpy, lhapdf/NNPDF40_lo_as_01180 and a C++ compiler."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_pdf_set('NNPDF40_lo_as_01180'):
            self.skipTest('NNPDF40_lo_as_01180 LHAPDF data not found '
                          '(set $LHAPDF_DATA_PATH)')
        scratch = ggttg_process_dir()
        if scratch is None:
            self.skipTest('could not generate/compile g g > t t~ g: %s'
                          % _GGTTG_PROCESS_CACHE.get('make_error', 'mg7 unavailable'))
        run_doc_example(self, 'integrator', cwd=scratch)

    def test_madspace_example_integrator_madnis(self):
        """docs/source/madspace/examples/integrator-madnis.rst -- the same
        integrand as the previous example, integrated with the external
        madnis package's neural importance sampling instead of plain
        sampling. Needs madspace, torch, the external madnis package,
        lhapdf/NNPDF40_lo_as_01180 and a C++ compiler."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_torch():
            self.skipTest('torch unavailable')
        if not has_madnis():
            self.skipTest('external madnis package unavailable')
        if not has_pdf_set('NNPDF40_lo_as_01180'):
            self.skipTest('NNPDF40_lo_as_01180 LHAPDF data not found '
                          '(set $LHAPDF_DATA_PATH)')
        scratch = ggttg_process_dir()
        if scratch is None:
            self.skipTest('could not generate/compile g g > t t~ g: %s'
                          % _GGTTG_PROCESS_CACHE.get('make_error', 'mg7 unavailable'))
        run_doc_example(self, 'integrator-madnis', cwd=scratch)

    def test_madspace_example_flow_training(self):
        """docs/source/madspace/examples/flow-training.rst -- fusing RNG,
        the built-in Flow, the mapping and the matrix element into one
        FunctionBuilder graph, trained with a plain torch optimizer. Needs
        madspace, torch, lhapdf/NNPDF40_lo_as_01180 and a C++ compiler (no
        external madnis package)."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_torch():
            self.skipTest('torch unavailable')
        if not has_pdf_set('NNPDF40_lo_as_01180'):
            self.skipTest('NNPDF40_lo_as_01180 LHAPDF data not found '
                          '(set $LHAPDF_DATA_PATH)')
        scratch = ggttg_process_dir()
        if scratch is None:
            self.skipTest('could not generate/compile g g > t t~ g: %s'
                          % _GGTTG_PROCESS_CACHE.get('make_error', 'mg7 unavailable'))
        run_doc_example(self, 'flow-training', cwd=scratch)


class TestGridpackExamples(unittest.TestCase):

    def test_gridpack_example_event_stream(self):
        """docs/source/event_stream.rst -- creating a fix_max_weight gridpack
        for p p > e+ e- with the MadGraph commands of the page, then filling
        an m_ee histogram from its bin/event_stream.py. Needs madspace, numpy,
        a C++ compiler and lhapdf/NNPDF40_lo_as_01180."""
        if not has_madspace():
            self.skipTest('madspace unavailable')
        if not has_pdf_set('NNPDF40_lo_as_01180'):
            self.skipTest('NNPDF40_lo_as_01180 LHAPDF data not found '
                          '(set $LHAPDF_DATA_PATH)')
        rst_path = pjoin(_DOCS_DIR, 'event_stream.rst')
        commands = extract_named_block(rst_path, 'event-stream-mg5-commands')
        self.assertTrue(commands, 'MadGraph commands not found in %s' % rst_path)

        with tempfile.TemporaryDirectory(prefix='doc_event_stream_') as scratch:
            script = pjoin(scratch, 'mg5_commands.txt')
            with open(script, 'w') as f:
                f.write('set automatic_html_opening False --no_save\n')
                f.write(commands + '\n')
            env = dict(os.environ)
            if os.path.isdir(_MADSPACE_INSTALL):
                env['PYTHONPATH'] = os.pathsep.join(
                    [_MADSPACE_INSTALL]
                    + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
                )
            result = subprocess.run(
                [sys.executable, pjoin(MG5DIR, 'bin', 'madgraph'), '-f', script],
                cwd=scratch, env=env, capture_output=True, text=True,
            )
            self.assertEqual(
                result.returncode, 0,
                'MadGraph commands failed\n--- stdout ---\n%s\n--- stderr ---\n%s'
                % (result.stdout[-5000:], result.stderr[-5000:]),
            )
            run_doc_example(self, 'event_stream', cwd=scratch, examples_dir=_DOCS_DIR)


if __name__ == '__main__':
    unittest.main()
