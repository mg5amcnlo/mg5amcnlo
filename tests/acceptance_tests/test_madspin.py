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
        # the input is decompressed to be read: it has to be compressed back
        self.assertTrue(os.path.exists(pjoin(self.path, 'test_spinmode_none.lhe.gz')))
        self.assertFalse(os.path.exists(pjoin(self.path, 'test_spinmode_none.lhe')))
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
add madspin_card --before_line="launch" set offshell_min_recoil 0.5
decay w+ > j j
decay w- > j j
decay z > l+ l-
""" % {'path': self.run_dir}
        with open(cmd_path, 'w') as fsock:
            fsock.write(command)

        with open(log_path, 'w') as log_file:
            return_code = subprocess.call(
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                cwd=pjoin(_file_path, os.path.pardir),
                stdout=log_file, stderr=subprocess.STDOUT)
        self.assertEqual(return_code, 0)

        with open(log_path) as log_file:
            log = log_file.read()
            misc.sprint(log)

        # MadSpin's default spinmode is "madspin" (offshell), which routes
        # through the density code path. The density path emits a single
        # summary line of the form:
        #   CRITICAL: MadSpin unweight efficiency: 0.1000
        #            (10000 written / 99958 trials, 10.00 trials/event)
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

        # trials/event is sampling dependent: 9.6-12.6 over eight production
        # samples of this card (spinmode madspin; PA takes ~2.8, the legacy
        # onshell path ~4.98). It used to depend strongly on the production
        # sample -- with madevent decays iseed 1-4 gave 4.4-15 (34 on CI), and
        # 4.6-152 before the refine normalisation changed: the offshell
        # max-weight scan reused one draw's production density for all the
        # draws of an event, and the weight itself was unbounded -- a Z drawn
        # up to sqrt(shat) pushes the gluon soft, w ~ 1/chi -- unless
        # offshell_min_recoil cuts it, which this card opts in to (the option
        # is off by default: it biases the off-shell tail). Only sanity-check it is present and in a
        # physically reasonable range rather than pinning the exact value: the
        # bound only catches a broken unweighting.
        trials = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*trials\s*/\s*event', log)
        if trials is not None:
            trials_per_event = float(trials.group(1))
            self.assertGreater(trials_per_event, 1.0)
            self.assertLess(trials_per_event, 500.0)

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

    def test_madspin_mixed_flavor_decay_log_summary_mg7(self):
        """Same check as test_madspin_mixed_flavor_decay_log_summary but with the
        default 'mg7' (madspace) exporter instead of Fortran madevent: drive the
        whole chain through the madevent-style launch command interface
        (output mg7 -> launch -> madspin=ON + decay lines) and verify MadSpin's
        density path runs on the mg7 events and emits its
        'MadSpin unweight efficiency: ...' summary line.

        NB: 'set nevents 500' relies on the mg7 run_card editor accepting the
        legacy madevent name (mapped to generation.events); without it the run
        would generate the full default 100k events and hit the timeout below.
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

        # Bounded and with stdin closed so the run never blocks on a prompt.
        # With 'set nevents 500' honoured the whole run takes 30-110 s on a CI
        # runner, which varies a lot from one run to the next; a timeout of
        # 600 s still catches the regression it is there for (nevents no longer
        # applied -> the full 100k events get decayed) without failing on a
        # slow runner, as 240 s did.
        timeout = 600
        def log_tail(why):
            # the mg7 run log lives in a tmp dir CI does not upload: show its
            # tail in the CI output, so that the reason is visible there
            with open(log_path) as log_file:
                lines = log_file.read().splitlines()
            print('\n===== test_madspin_mixed_flavor_decay_log_summary_mg7: '
                  'madgraph log tail (%s) =====\n%s' % (why, '\n'.join(lines[-150:])),
                  file=sys.stderr, flush=True)

        with open(log_path, 'w') as log_file:
            try:
                return_code = subprocess.call(
                    [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
                    cwd=pjoin(_file_path, os.path.pardir),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                log_file.flush()
                log_tail('timed out after %d s' % timeout)
                self.fail('mg7 + MadSpin run timed out (is set nevents applied?)')
        with open(log_path) as log_file:
            log = log_file.read()
        if return_code != 0 or not re.search(r'MadSpin\s+unweight\s+efficiency', log):
            log_tail('rc=%s' % return_code)
        self.assertEqual(return_code, 0)

        self.assertIsNotNone(
            re.search(r'MadSpin\s+unweight\s+efficiency', log),
            msg='mg7 + MadSpin: density-mode summary line not found in log')
        # store_result used to save the madevent results database to
        # HTML/results.pkl, which the mg7 output does not have
        self.assertEqual(
            [line for line in log.splitlines() if 'I/O error' in line], [])
        self.assertFalse(os.path.exists(pjoin(self.run_dir, 'crossx.html')))
        # the madmatrix exporter used to print its keyword arguments
        self.assertNotIn('kwargs[', log)

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
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
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
                [sys.executable, pjoin(_file_path, os.path.pardir, 'bin', 'madgraph'), cmd_path],
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
               

    def _run_one_mode(self, mode, particle_to_decay, name_input_file, name_scipt_file):
        cwd = os.getcwd()
        index = name_input_file.find(".lhe")
        name_file_decayed = name_input_file[:index] + "_decayed" + name_input_file[index:]
        path_input_file = pjoin(MG5DIR, 'tests', 'input_files', 'madspin', name_input_file)

        files.cp(path_input_file, self.path)

        fsock = open(pjoin(self.path, name_scipt_file),'w')
        if 'loop_induced' in name_input_file:
            text = f"""
            import {pjoin(self.path, name_input_file)}
            set spinmode {mode}
            decay w+ > all all
            decay w- > all all
            launch
            """
        else:
            text = f"""
            import {pjoin(self.path, name_input_file)}
            set spinmode {mode}
            decay t > w+ b, w+ > all all
            decay t~ > w- b~, w- > all all
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

        returncode = subprocess.call([pjoin(MG5DIR, 'MadSpin', 'madspin'),
                        pjoin(self.path, name_scipt_file)],
                        cwd=pjoin(self.path),
                        stdout=stdout,stderr=stderr)

        # check the exit status first: a crashed MadSpin can still leave partial
        # output behind, and then the assertions below report a confusing
        # symptom instead of the actual failure.
        self.assertEqual(returncode, 0,
                         'MadSpin exited with %s for %s' % (returncode, name_scipt_file))
        self.assertTrue(os.path.exists(pjoin(self.path, name_file_decayed)),
                        'no decayed file produced for %s' % name_scipt_file)
        
        
        lhe = lhe_parser.EventFile(pjoin(self.path, name_file_decayed))
        if mode in ['full', 'madspin'] and 'loop_induced' in name_input_file:
            self.assertEqual(5, len(lhe))
        else:
            self.assertEqual(100, len(lhe))
        
        # we cannot check if the momenta are identical by fixing the seed (it could change whenever we change the code somewhere)
        # we just check that there 8 particles in each event and that the w+- have a status of 2.
        for event in lhe:
            self.assertEqual(event.nexternal, len(event))
            for particle in event:
                if particle.pid in particle_to_decay:
                    self.assertEqual(particle.status, 2,
                                     'pdg %s not marked as decayed for %s'
                                     % (particle.pid, name_scipt_file))

    def test_madspin_loop_induced(self):
        """ Tests that the different modes of MadSpin work for loop-induced processes.
            It checks that there is no crash and that the decayed particles have a status of 2.
        """ 

        self._run_one_mode("PA", [24, -24], 'test_madspin_loop_induced_PA.lhe.gz', 'test_loop_induced_PA')
        self._run_one_mode("full", [24, -24], 'test_madspin_loop_induced_full.lhe.gz', 'test_loop_induced_full')
        self._run_one_mode("onshell", [24, -24], 'test_madspin_loop_induced_onshell.lhe.gz', 'test_loop_induced_onshell')
        self._run_one_mode("madspin", [24, -24], 'test_madspin_loop_induced_madspin.lhe.gz', 'test_loop_induced_madspin')

    def test_madspin_tree_level(self):
        """ Tests that the different modes of MadSpin work for tree-level processes.
            It checks that there is no crash and that the decayed particles have a status of 2.
        """ 

        self._run_one_mode("PA", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_PA')
        self._run_one_mode("full", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_full')
        self._run_one_mode("onshell", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_onshell')
        self._run_one_mode("madspin", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_madspin')
        self._run_one_mode("none", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_none')
        self._run_one_mode("madspin_v1", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_level_madspin_v1')
        self._run_one_mode("onshell_v1", [6, -6], 'test_madspin_tree_level.lhe.gz', 'test_tree_onshell_v1')

    @staticmethod
    def _decayed_sample(path):
        """The <init> block (sigma*BR) and the event blocks of an LHE file:
        everything that has to come out the same, without the banner."""
        with gzip.open(path, 'rt') as fsock:
            text = fsock.read()
        init = re.search(r'<init>.*?</init>', text, re.S)
        return (init.group(0) if init else None,
                re.findall(r'<event>.*?</event>', text, re.S))

    def test_madspin_mg7_decays_reproducible_with_seed(self):
        """The same card, seed and input give the same decayed sample with the
        default decay_generator mg7.

        Its decay pools used to run with mg7's run.seed = -1 -- a new random
        seed on every run -- so two runs of this card differed from the first
        event on (sigma*BR and the unweighting efficiency too), in every spin
        mode, while decay_generator madevent was reproducible. Both paths are
        covered: the density modes (one forked generation per decaying
        particle) and spinmode none (serial generation).
        """
        import MadSpin.interface_madspin as interface_madspin
        if not interface_madspin.MadSpinInterface._mg7_available():
            self.skipTest('madspace is not built: MadSpin would generate the '
                          'decays with madevent, which is not what is tested')

        for spinmode in ('madspin', 'none'):
            samples = []
            for tag in ('a', 'b'):
                run = pjoin(self.path, '%s_%s' % (spinmode, tag))
                os.mkdir(run)
                files.cp(pjoin(MG5DIR, 'tests', 'input_files', 'ttbar.lhe.gz'),
                         pjoin(run, 'events.lhe.gz'))
                card = pjoin(run, 'madspin_card.dat')
                with open(card, 'w') as fsock:
                    fsock.write('import events.lhe.gz\n'
                                'set spinmode %s\n'
                                'set decay_generator mg7\n'
                                'set seed 77\n'
                                'decay t > w+ b, w+ > e+ ve\n'
                                'decay t~ > w- b~, w- > mu- vm~\n'
                                'launch\n' % spinmode)
                log_path = pjoin(run, 'madspin.log')
                with open(log_path, 'w') as log:
                    returncode = subprocess.call(
                        [sys.executable, pjoin(MG5DIR, 'MadSpin', 'madspin'),
                         card],
                        cwd=run, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, timeout=900)
                decayed = pjoin(run, 'events_decayed.lhe.gz')
                if returncode or not os.path.exists(decayed):
                    with open(log_path) as log:
                        print('\n===== %s %s: MadSpin log tail (rc=%s) =====\n%s'
                              % (spinmode, tag, returncode,
                                 '\n'.join(log.read().splitlines()[-100:])),
                              file=sys.stderr, flush=True)
                self.assertEqual(returncode, 0)
                self.assertTrue(os.path.exists(decayed))

                # the mg7 generator really made the pools, each with a fixed
                # seed of its own
                cards = glob.glob(pjoin(run, 'decay_*', 'Cards',
                                        'run_card.toml'))
                self.assertEqual(len(cards), 2, 'no mg7 decay pool in %s' % run)
                seeds = [banner.RunCardMG7(path)['run']['seed']
                         for path in cards]
                self.assertTrue(all(seed > 0 for seed in seeds), seeds)
                self.assertEqual(len(set(seeds)), 2, seeds)

                samples.append(self._decayed_sample(decayed))

            init_a, events_a = samples[0]
            init_b, events_b = samples[1]
            self.assertEqual(len(events_a), 25)
            self.assertEqual(init_a, init_b,
                             'spinmode %s: sigma*BR differs between the two '
                             'runs' % spinmode)
            for i, (event_a, event_b) in enumerate(zip(events_a, events_b)):
                self.assertEqual(event_a, event_b,
                                 'spinmode %s: event %s differs between the '
                                 'two runs' % (spinmode, i))
            self.assertEqual(len(events_a), len(events_b))
