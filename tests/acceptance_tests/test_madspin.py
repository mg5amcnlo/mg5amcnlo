from __future__ import division
from __future__ import absolute_import
import subprocess
import unittest
import os
import re
import shutil
import sys
import logging
import time
import tempfile
import math
import glob
import gzip

logger = logging.getLogger('test_cmd')

import tests.unit_tests.iolibs.test_file_writers as test_file_writers

import madgraph.interface.master_interface as MGCmd
import madgraph.interface.madevent_interface as MECmd
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.iolibs.files as files

import madgraph.various.misc as misc
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser
import madgraph.various.banner as banner

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
_pickle_path =os.path.join(_file_path, 'input_files')

from madgraph import MG4DIR, MG5DIR, MadGraph5Error, InvalidCmd

pjoin = os.path.join

#===============================================================================
# TestCmd
#===============================================================================
class TestMadSpin(unittest.TestCase):
    """test that we can launch everything from a single file"""

    def setUp(self):
        
        self.debuging = unittest.debug
        if self.debuging:
            self.path = pjoin(MG5DIR, 'MS_TEST')
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            os.mkdir(self.path) 
        else:
            self.path = tempfile.mkdtemp(prefix='ms_test_mg5')
        self.run_dir = pjoin(self.path, 'MGPROC') 
        
    
    def tearDown(self):

        if not self.debuging:
            shutil.rmtree(self.path)
        self.assertFalse(self.debuging)

    def _get_single_decayed_lhe_path(self):
        decayed_events = glob.glob(pjoin(self.run_dir, 'Events', '*decayed*', '*.lhe.gz'))
        self.assertTrue(decayed_events)
        return decayed_events[0]


    def test_hepmc_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test.hepmc.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        set cross_section {0:1.0}
        set new_wgt BR
        set input_format hepmc
        import ./test.hepmc.gz
        import model %s/tests/input_files/DM_pion %s/tests/input_files/DM_pion/param_pion.dat
        decay k0 > xr xr a
        launch
        """ % (MG5DIR, MG5DIR)
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)
        self.assertTrue(os.path.exists(pjoin(self.path, 'test_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_decayed.lhe.gz'))
        self.assertEqual(10, len(lhe))
        
        nb_dec = 0
        nb_photon = 0
        for event in lhe:
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 130:
                    self.assertEqual(particle.status,2)
                    nb_dec +=1
                if particle.pdg ==22:
                    nb_photon += 1
                    
        self.assertEqual(nb_dec, 116)
        self.assertEqual(nb_photon, 116)

    def test_lhe_none_decay(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test_spinmode_none.lhe.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        set spinmode none
        import ./test_spinmode_none.lhe.gz
        decay z > mu+ mu-
        launch
        """
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz'))
        self.assertEqual(100, len(lhe))
        
        nb_dec = 0
        nb_muon = 0
        for event in lhe:
            muon_in = 0
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pdg == 23:
                    self.assertEqual(particle.status,2)
                    nb_dec += 1
                if particle.pdg == 13:
                    nb_muon += 1
                    muon_in +=1
            self.assertEqual(muon_in, 1)
        self.assertEqual(nb_dec, 189)
        self.assertEqual(nb_muon, 100)

    def test_madspin_spin_only(self):
        """ """
        
        cwd = os.getcwd()
        
        files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'test_spinmode_none.lhe.gz'), self.path)


        fsock = open(pjoin(self.path, 'test_hepmc'),'w')
        text = """
        import ./test_spinmode_none.lhe.gz
        set onlyhelicity True
        launch
        """
        
        fsock.write(text)
        fsock.close()

        import subprocess
        if logging.getLogger('madgraph').level <= 20:
            stdout=None
            stderr=None
        else:
            devnull =open(os.devnull,'w')
            stdout=devnull
            stderr=devnull

        subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         pjoin(self.path, 'test_hepmc')],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz')))
        lhe = lhe_parser.EventFile(pjoin(self.path, 'test_spinmode_none_decayed.lhe.gz'))
        self.assertEqual(100, len(lhe))
        
        nb_dec = 0
        nb_notdec = 0 
        nb_muon = 0
        pol = {0:0, -1:0,1:0, 9:9}
        for event in lhe:
            muon_in = 0
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.status == 2:
                    self.assertEqual(particle.helicity, 9)
                    nb_dec +=1
                    continue
                if particle.pdg == 23:
                    if particle.status == 1:
                        nb_notdec += 1    
                    else: 
                        nb_dec += 1
                if particle.pdg == 13:
                    nb_muon += 1
                    muon_in +=1
                self.assertIn(int(particle.helicity), pol)
                pol[int(particle.helicity)] +=1
            self.assertEqual(muon_in, 0)
        
        self.assertEqual(nb_notdec, 100)
        self.assertEqual(nb_dec, 89)
        self.assertEqual(nb_muon, 0)
        import math
        self.assertLess(abs(pol[1]-pol[-1]), 2 * math.sqrt(pol[1]))
        self.assertLess(pol[0], pol[-1])

    def test_madspin_mixed_flavor_decay_log_summary(self):
        """Check MadSpin log summary for mixed lepton/quark decay definition."""

        cmd_path = pjoin(self.path, 'test_madspin_mixed_flavor.cmd')
        log_path = pjoin(self.path, 'test_madspin_mixed_flavor.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
define l+ = e+ mu+ u d~
define l- = e- mu- u~ d
generate u u~ > z g
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 10000
set iseed 1
decay w+ > j j
decay w- > j j
decay z > l+ l-
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'mg5_aMC'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        with open(log_path) as log_file:
            log = log_file.read()
            misc.sprint(log)

        # MadSpin's default spinmode is "PA" (pole approximation), which
        # routes through the density code path. The density path emits a
        # single summary line of the form:
        #   CRITICAL: MadSpin unweight efficiency: 0.3697
        #            (10000 written / 27048 trials, 2.70 trials/event)
        # rather than the legacy multi-line onshell summary
        # ("Total number of events written", "Average number of trial
        # points per production event", "Branching ratio to allowed
        # decays", ...).
        #
        # Parse the metrics that are still meaningful here individually and
        # leniently, so the test does not break on minor wording/spacing
        # changes of that line:
        #   - the "unweight efficiency" marker must be present (MadSpin ran),
        #   - the number of written events (deterministic: == nevents),
        #   - the trials/event ratio (sampling dependent -> sanity range only).
        self.assertIsNotNone(
            re.search(r'MadSpin\s+unweight\s+efficiency', log),
            msg='MadSpin density-mode summary line not found in log')

        written = re.search(r'([0-9]+)\s+written\b', log)
        self.assertIsNotNone(written,
            msg='MadSpin written-event count not found in log')
        n_written = int(written.group(1))
        self.assertEqual(n_written, 10000,
            msg='Expected 10000 written events, got %d' % n_written)

        # trials/event is sampling dependent (~2.70 in density mode, ~4.98 on
        # the legacy onshell path): only sanity-check it is present and in a
        # physically reasonable range rather than pinning the exact value.
        trials = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*trials\s*/\s*event', log)
        if trials is not None:
            trials_per_event = float(trials.group(1))
            self.assertGreater(trials_per_event, 1.0)
            self.assertLess(trials_per_event, 20.0)

        # The legacy onshell-mode summary lines (Branching ratio to allowed
        # decays / Number of events with weights larger than max_weight /
        # Number of subprocesses / Number of failures when restoring the
        # Monte Carlo masses) are not emitted by the density code path,
        # so those assertions are intentionally not ported here.

        with gzip.open(self._get_single_decayed_lhe_path(), 'rt') as lhe_file:
            banner_lines = []
            for line in lhe_file:
                if '<event' in line:
                    break
                banner_lines.append(line)
        banner_text = ''.join(banner_lines)
        self.assertNotRegex(banner_text, r'(?mi)^\s*81\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*82\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*83\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+81\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+82\s+[0-9eE.+-]+\s+# added\s*$')
        self.assertNotRegex(banner_text, r'(?mi)^\s*decay\s+83\s+[0-9eE.+-]+\s+# added\s*$')

    @unittest.expectedFailure
    def test_madspin_mixed_flavor_decay_log_summary_mg7(self):
        """TODO (mg7 + MadSpin): same check as
        test_madspin_mixed_flavor_decay_log_summary but with the current
        default 'mg7' (madspace/madnis) exporter instead of Fortran madevent.

        This is *expected to fail for now*: the mg7 launch does not run the
        MadSpin density flow and does not emit the
        'MadSpin unweight efficiency: ...' summary line (it currently runs the
        madnis pipeline instead, which here does not produce decayed events in
        the bounded time). It is kept as an @expectedFailure so it is tracked
        in CI: once mg7 + MadSpin is supported it will report an *unexpected
        success*, which is the signal to wire mg7 into the MadSpin flow and
        drop this decorator.
        """
        cmd_path = pjoin(self.path, 'test_madspin_mixed_flavor_mg7.cmd')
        log_path = pjoin(self.path, 'test_madspin_mixed_flavor_mg7.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
define l+ = e+ mu+ u d~
define l- = e- mu- u~ d
generate u u~ > z g
output mg7 %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 500
set iseed 1
decay w+ > j j
decay w- > j j
decay z > l+ l-
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        # Bounded and with stdin closed so an (expected) failing mg7 run never
        # blocks on the launch card menu -- it does not understand the
        # madevent-style madspin=ON/shower=OFF switches -- and a TimeoutExpired
        # is itself the expected failure.
        with open(log_path, 'w') as log_file:
            try:
                return_code = subprocess.call(
                    [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'mg5_aMC'), cmd_path],
                    cwd=pjoin(_file_path, os.path.pardir),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file, stderr=subprocess.STDOUT, timeout=240)
            except subprocess.TimeoutExpired:
                self.fail('mg7 + MadSpin run timed out (TODO: not supported yet)')
        self.assertEqual(return_code, 0)

        with open(log_path) as log_file:
            log = log_file.read()
        self.assertIsNotNone(
            re.search(r'MadSpin\s+unweight\s+efficiency', log),
            msg='mg7 + MadSpin: density-mode summary line not found in log')

    def test_madspin_wplus_all_all_flavor_balance(self):
        """`w+ > all all` should populate e/mu decay modes with similar rates."""

        cmd_path = pjoin(self.path, 'test_madspin_wplus_all_all.cmd')
        log_path = pjoin(self.path, 'test_madspin_wplus_all_all.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
generate p p > w+ g
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 2000
set iseed 1
decay w+ > all all
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'mg5_aMC'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        counts={}
        for event in lhe_parser.EventFile(self._get_single_decayed_lhe_path()):
            for particle in event:
                if particle.status == 1:
                    if particle.pdg in counts:
                        counts[particle.pdg] += 1
                    else:
                        counts[particle.pdg] = 1

        misc.sprint(counts)
        self.assertNotIn(81, counts)
        self.assertNotIn(82, counts)
        self.assertNotIn(83, counts)
        self.assertNotIn(-81, counts)
        self.assertNotIn(-82, counts)
        self.assertNotIn(-83, counts)    
        self.assertGreater(counts[-11], 0)
        self.assertEqual(counts[-11],counts[12])
        self.assertGreater(counts[-13], 0)
        self.assertEqual(counts[-13],counts[14])
        self.assertGreater(counts[-15], 0) 
        self.assertEqual(counts[-15],counts[16])
        self.assertGreater(counts[4], 0)
        self.assertEqual(counts[4], counts[-3])
        self.assertGreater(counts[2], 0)
        self.assertEqual(counts[2], counts[-1])


        lepton_total = counts[-11] + counts[-13]
        self.assertLess(abs(counts[-11] - counts[-13]), 4* math.sqrt(counts[-11]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
        self.assertLess(abs(counts[2] - counts[4]), 4* math.sqrt(counts[4]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
                
    def test_madspin_wplus_all_all_flavor_balance_2to1(self):
        """`w+ > all all` should populate e/mu decay modes with similar rates."""

        cmd_path = pjoin(self.path, 'test_madspin_wplus_all_all.cmd')
        log_path = pjoin(self.path, 'test_madspin_wplus_all_all.log')
        command = """import model sm
set automatic_html_opening False --no_save
set notification_center False --no_save
generate p p > w+
output madevent %(path)s
launch
madspin=ON
shower=OFF
analysis=OFF
set nevents 2000
set iseed 1
decay w+ > all all
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'mg5_aMC'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        counts={}
        for event in lhe_parser.EventFile(self._get_single_decayed_lhe_path()):
            for particle in event:
                if particle.status == 1:
                    if particle.pdg in counts:
                        counts[particle.pdg] += 1
                    else:
                        counts[particle.pdg] = 1

        misc.sprint(counts)
        self.assertNotIn(81, counts)
        self.assertNotIn(82, counts)
        self.assertNotIn(83, counts)
        self.assertNotIn(-81, counts)
        self.assertNotIn(-82, counts)
        self.assertNotIn(-83, counts)    
        self.assertGreater(counts[-11], 0)
        self.assertEqual(counts[-11],counts[12])
        self.assertGreater(counts[-13], 0)
        self.assertEqual(counts[-13],counts[14])
        self.assertGreater(counts[-15], 0) 
        self.assertEqual(counts[-15],counts[16])
        self.assertGreater(counts[4], 0)
        self.assertEqual(counts[4], counts[-3])
        self.assertGreater(counts[2], 0)
        self.assertEqual(counts[2], counts[-1])


        lepton_total = counts[-11] + counts[-13]
        self.assertLess(abs(counts[-11] - counts[-13]), 4* math.sqrt(counts[-11]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
        self.assertLess(abs(counts[2] - counts[4]), 4* math.sqrt(counts[4]),
            msg='Expected electron/muon counts to be comparable, got %s' % counts)
               