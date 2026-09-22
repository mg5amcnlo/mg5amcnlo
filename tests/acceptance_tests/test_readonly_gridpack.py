################################################################################
#
# Copyright (c) 2024 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Acceptance test for the *concurrent read-only gridpack* mode.

A gridpack whose ``madevent`` tree has been restored to its default state and
made read-only (``restore_data default`` + ``chmod -R 555 madevent``) must be
runnable simultaneously by several processes, each from its own empty working
directory, without any of them writing into the shared (read-only) gridpack.

This exercises the read-only code paths in
``madevent_interface.GridPackCmd`` / ``gen_ximprove`` / ``combine_runs`` /
``sum_html`` -- historically several base-class helpers (make_all_html_results,
update_html, write_multijob/reset_multijob, CombineRuns) wrote into or read
from ``me_dir`` unconditionally, which broke concurrent read-only use.
"""
from __future__ import absolute_import
import os
import subprocess
import sys
import tempfile

pjoin = os.path.join
_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, pjoin(_file_path, '..', '..'))

import tests.unit_tests as unittest
from madgraph import MG5DIR
import madgraph.various.banner as banner
import madgraph.various.lhe_parser as lhe_parser


class TestReadOnlyGridpack(unittest.TestCase):
    """Build a small LO gridpack, freeze it read-only, run it concurrently."""

    # a fast, PDF-free process that still goes through the full gridpack
    # survey/refine/combine machinery. Event counts are kept small on purpose:
    # the point is to exercise the read-only code paths (refine4grid ->
    # make_all_html_results / write_multijob / CombineRuns), not to accumulate
    # statistics, and these still run for any non-zero request.
    process = 'e+ e- > mu+ mu-'
    nb_worker = 3
    # events the gridpack grid is built for
    build_nevents = 200
    # events each concurrent worker asks the frozen gridpack for
    run_nevents = 100

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='ro_gridpack_')

    def tearDown(self):
        # the frozen gridpack is chmod 555 -> make everything writable first
        for root, dirs, files in os.walk(self.tmpdir):
            try:
                os.chmod(root, 0o755)
            except OSError:
                pass
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _build_gridpack(self):
        """Generate ``self.process`` and build+extract a gridpack. Returns the
        directory that holds ``run.sh`` and ``madevent/``."""
        medir = pjoin(self.tmpdir, 'PROC')
        script = pjoin(self.tmpdir, 'mg5_script.dat')
        with open(script, 'w') as fp:
            fp.write('\n'.join([
                'set automatic_html_opening False --no_save',
                'generate %s' % self.process,
                'output %s' % medir,
            ]) + '\n')

        mlog = pjoin(self.tmpdir, 'mg5.log')
        with open(mlog, 'w') as logf:
            ret = subprocess.call(
                [sys.executable, pjoin(MG5DIR, 'bin', 'mg5_aMC'), '-f', script],
                stdout=logf, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'mg5_aMC output failed (see %s)' % mlog)
        self.assertTrue(os.path.isdir(medir), 'process directory not created')

        # turn the run into a (small) gridpack run
        rc = banner.RunCard(pjoin(medir, 'Cards', 'run_card.dat'))
        rc['gridpack'] = True
        rc['nevents'] = self.build_nevents
        rc.write(pjoin(medir, 'Cards', 'run_card.dat'))

        glog = pjoin(self.tmpdir, 'gen.log')
        with open(glog, 'w') as logf:
            ret = subprocess.call(
                [pjoin(medir, 'bin', 'generate_events'), '-f'],
                stdout=logf, stderr=subprocess.STDOUT)
        self.assertEqual(ret, 0, 'gridpack build failed (see %s)' % glog)
        tar = pjoin(medir, 'run_01_gridpack.tar.gz')
        self.assertTrue(os.path.exists(tar),
                        'gridpack tarball not produced (see %s)' % glog)

        # extract the gridpack (gives run.sh + madevent/)
        gpdir = pjoin(self.tmpdir, 'GP')
        os.makedirs(gpdir)
        subprocess.check_call(['tar', '-xzpf', tar], cwd=gpdir)
        self.assertTrue(os.path.exists(pjoin(gpdir, 'run.sh')),
                        'run.sh missing after gridpack extraction')
        self.assertTrue(os.path.isdir(pjoin(gpdir, 'madevent')),
                        'madevent/ missing after gridpack extraction')
        return gpdir

    def _freeze(self, gpdir):
        """The supported concurrent-gridpack recipe: restore the pristine grid
        then make the madevent tree read-only."""
        me = pjoin(gpdir, 'madevent')
        restore = pjoin(me, 'bin', 'internal', 'restore_data')
        if os.path.exists(restore):
            subprocess.call([restore, 'default'], cwd=me)
        subprocess.check_call(['chmod', '-R', '555', 'madevent'], cwd=gpdir)

    # ------------------------------------------------------------------
    def test_concurrent_readonly_gridpack(self):
        """N workers run the frozen gridpack simultaneously; each must produce
        events from its own directory, and none may write into the shared
        read-only gridpack."""
        gpdir = self._build_gridpack()
        self._freeze(gpdir)

        run_sh = pjoin(gpdir, 'run.sh')
        procs, rundirs = [], []
        for i in range(self.nb_worker):
            rundir = pjoin(self.tmpdir, 'run_%d' % i)
            os.makedirs(rundir)
            rundirs.append(rundir)
            logf = open(pjoin(rundir, 'run.log'), 'w')
            # single-core generation from an empty dir, distinct seed per worker
            p = subprocess.Popen([run_sh, str(self.run_nevents), str(1001 + i)],
                                 cwd=rundir, stdout=logf, stderr=subprocess.STDOUT)
            procs.append((p, logf))
        for p, logf in procs:
            p.wait()
            logf.close()

        # every worker must have produced events, none crashed
        counts = []
        for rundir in rundirs:
            evt = pjoin(rundir, 'events.lhe.gz')
            self.assertTrue(
                os.path.exists(evt),
                'read-only gridpack worker produced no events.lhe.gz; '
                'run.sh/gridrun output:\n%s'
                % open(pjoin(rundir, 'run.log')).read()[-3000:])
            nb = sum(1 for _ in lhe_parser.EventFile(evt))
            self.assertGreater(nb, 0, 'no events written in %s' % evt)
            counts.append(nb)

        # the read-only gridpack must not have been polluted with a run: no
        # GridRun_* dirs or events should have leaked into the shared madevent.
        leaked = []
        me_events = pjoin(gpdir, 'madevent', 'Events')
        if os.path.isdir(me_events):
            leaked = [d for d in os.listdir(me_events) if d.startswith('GridRun_')]
        self.assertEqual(leaked, [],
                         'read-only gridpack was written into: %s' % leaked)
