"""Exercise split-shower scripts, collection and cleanup with small analyses."""

import gzip
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock

from madgraph import MG5DIR
from madgraph.interface import amcatnlo_run_interface as interface
from madgraph.various import cluster, histograms
from madgraph.various.shower_card import ShowerCard


class TestShowerCollection(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mg5_shower_')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = self.root / 'Events' / 'run_01'
        self.events.mkdir(parents=True)
        self.mcatnlo = self.root / 'MCatNLO'
        self.mcatnlo.mkdir()
        shutil.copy(Path(MG5DIR) / 'Template/NLO/MCatNLO/shower_template.sh', self.mcatnlo)
        internal = self.root / 'bin' / 'internal'
        internal.mkdir(parents=True)
        (internal / 'histograms.py').symlink_to(Path(MG5DIR) / 'madgraph/various/histograms.py')
        self.input = self.events / 'events.lhe'
        self.write_events()

        # The executable reads the actual split input and writes two different
        # histogram files. Only physics compilation is replaced by a fixture;
        # submission, shell scripts, tar handling and HwU combination are real.
        executable = self.mcatnlo / 'analysis_fixture'
        executable.write_text('''#!/usr/bin/env python3
from pathlib import Path
import re
import sys

input_card = Path('MCATNLO_HERWIG6_input')
mode = (input_card.read_text().strip() if input_card.is_file()
        else Path('Pythia8.cmd').read_text().splitlines()[0])
events = re.findall(r'<event id="(\\d+)">', Path('events.lhe').read_text())
print('Events: ' + ','.join(events), flush=True)
if mode == 'FAIL':
    print('Analysis failed', flush=True)
    sys.exit(17)
job = int(Path.cwd().name[4:]) if Path.cwd().name.startswith('run_') else 1
if Path.cwd().name.startswith('run_'):
    assert Path('events.lhe').is_symlink(), 'Input should be linked, not copied'
if mode == 'HEP':
    Path('events.lhe.hep').write_text('Showered events: ' + ','.join(events))
elif mode == 'TOP':
    Path('analysis.top').write_text('Histogram job %d\\n' % job)
elif mode in ('HWU', 'FXFX'):
    if mode == 'FXFX':
        print('Les Houches User Process(es) 9999 | 4 4 %d | %se-9 1e-10 |' % (job, job))
    for name, factor in [('a', 1), ('b', 10)]:
        Path(name + '.HwU').write_text(
            '##& xmin & xmax & central value & dy\\n\\n'
            '<histogram> 1 "Analysis ' + name + ' |X_AXIS@LIN |Y_AXIS@LIN |TYPE@NLO"\\n'
            ' 0 1 %s 0.1\\n<\\\\histogram>\\n' % (job * factor))
''')
        executable.chmod(0o755)
        compiler = self.mcatnlo / 'MCatNLO_MadFKS.inputs'
        compiler.write_text('#!/usr/bin/env bash\n'
                            'cp analysis_fixture MCATNLO_HERWIG6_EXE\n'
                            'echo "Compiler stdout"\n'
                            'echo "Compiler stderr" >&2\n')
        compiler.chmod(0o755)

        self.cmd = object.__new__(interface.aMCatNLOCmd)
        self.cmd.stop_for_runweb = True
        self.cmd.me_dir = str(self.root)
        self.cmd.run_name = 'run_01'
        self.cmd.run_card = {'ickkw': 0}
        self.cmd.shower_card = ShowerCard()
        self.cmd.shower_card['nsplit_jobs'] = 3
        self.cmd.shower_card['extralibs'] = ''
        self.cmd.shower_card['analyse'] = 'analysis.o HwU.o'
        self.cmd.options = {'lhapdf': 'lhapdf-config'}
        self.cmd.cluster_mode = 0
        self.cmd.cluster = cluster.MultiCore(nb_core=3)
        self.addCleanup(self.cmd.cluster.remove)
        self.cmd.update_status = mock.Mock()
        self.cmd.results = mock.Mock()
        self.cmd.do_plot = mock.Mock()
        self.cmd.banner_to_mcatnlo = mock.Mock()
        self.render = mock.patch.object(interface.common_run, 'render_HwU_plot')
        self.render.start()
        self.addCleanup(self.render.stop)

    def write_events(self, norm='sum'):
        banner = ('<LesHouchesEvents version="3.0">\n<header>\n<MGRunCard>\n'
                  "12 = nevents\n'HERWIG6' = parton_shower\n'%s' = event_norm\n" % norm +
                  '</MGRunCard>\n</header>\n<init>\n'
                  ' 2212 2212 6500 6500 0 0 0 0 -4 1\n 1 0 1 1\n</init>\n')
        self.original = (banner + ''.join(
            '<event id="%d">\n1 1 -1 100 .007 .118\n'
            '21 1 0 0 501 502 0 0 0 100 0 0 9\n</event>\n' % i
            for i in range(12)) + '</LesHouchesEvents>\n')
        self.input.write_text(self.original)

    def run_shower(self, mode='HWU'):
        (self.mcatnlo / 'MCATNLO_HERWIG6_input').write_text(mode)
        self.cmd.run_mcatnlo(str(self.input), {})

    def assert_clean(self, invocation=1, shower='HERWIG6'):
        self.assertEqual(list(self.events.glob('events.lhe_[0-9]*')), [])
        rundir = self.mcatnlo / ('RUN_%s_%d' % (shower, invocation))
        self.assertEqual(list(rundir.glob('**/*.lhe')), [])
        self.assertEqual(list(rundir.glob('**/*.lhe.gz')), [])
        with gzip.open(str(self.input) + '.gz', 'rt') as stream:
            self.assertEqual(stream.read(), self.original)
        log = self.events / ('mcatnlo_%s_%d.log' % (shower, invocation))
        self.assertIn('Compiler stdout', log.read_text())
        self.assertIn('Compiler stderr', log.read_text())
        self.assertFalse((self.root / 'mcatnlo.log').exists())
        self.assertTrue((rundir / 'mcatnlo_run.log').is_file())
        return rundir

    def test_split_hwu_combines_each_file_and_cleans_up(self):
        self.run_shower()
        self.assert_clean()
        for index, factor in enumerate((1, 10)):
            output = self.events / ('plot_HERWIG6_1_%d.HwU' % index)
            hwu = histograms.HwUList(str(output))
            self.assertEqual(len(hwu), 1)
            self.assertAlmostEqual(hwu[0].bins[0].wgts['central'], 6 * factor)
            with tarfile.open(self.events / ('plot_HERWIG6_1_%d.tar.gz' % index)) as archive:
                self.assertEqual(len(archive.getnames()), 3)
        for job in range(1, 4):
            log = self.events / ('mcatnlo_run_HERWIG6_1__%d.log' % job)
            self.assertIn('Events: ' + ','.join(str(i) for i in range((job-1)*4, job*4)),
                          log.read_text())

    def test_numeric_split_order_and_logs_on_multicore(self):
        self.cmd.cluster_mode = 2
        self.cmd.shower_card['nsplit_jobs'] = 12
        self.cmd.shower_card['analyse'] = 'analysis.o'
        self.cmd.shower_card['combine_td'] = False
        self.run_shower('TOP')
        self.assert_clean()
        for index in range(1, 13):
            output = self.events / ('plot_HERWIG6_1_0__%d.top' % index)
            self.assertEqual(output.read_text(), 'Histogram job %d\n' % index)
            log = self.events / ('mcatnlo_run_HERWIG6_1__%d.log' % index)
            self.assertEqual(log.read_text().strip(), 'Events: %d' % (index-1))

    def test_cluster_scratch_staging_cleans_inputs_and_delivers_logs(self):
        scratch = self.root / 'scratch'
        scratch.mkdir()
        self.cmd.cluster_mode = 2
        self.cmd.cluster.temp_dir = str(scratch)
        self.cmd.shower_card['analyse'] = 'analysis.o'
        self.cmd.shower_card['combine_td'] = False
        self.run_shower('TOP')
        self.assert_clean()
        self.assertEqual(list(scratch.iterdir()), [])

    def test_split_pythia_logs_seeds_and_fxfx_summary(self):
        pythia = self.root / 'pythia'
        (pythia / 'bin').mkdir(parents=True)
        (pythia / 'share/Pythia8/xmldoc').mkdir(parents=True)
        (pythia / 'share/Pythia8/tunes').mkdir()
        config = pythia / 'bin/pythia8-config'
        config.write_text('#!/bin/sh\nexit 0\n')
        config.chmod(0o755)
        self.cmd.options['pythia8_path'] = str(pythia)
        self.cmd.run_card['ickkw'] = 3
        self.input.write_text(self.input.read_text().replace('HERWIG6', 'PYTHIA8'))
        self.original = self.input.read_text()
        compiler = self.mcatnlo / 'MCatNLO_MadFKS.inputs'
        compiler.write_text('#!/bin/sh\ncp analysis_fixture Pythia8.exe\n'
                            'printf "FXFX\\nRandom:seed = 123\\n" > Pythia8.cmd\n'
                            'echo "Compiler stdout"\necho "Compiler stderr" >&2\n')
        with mock.patch.object(interface, 'check_pythia8_fxfx_plugin'):
            self.cmd.run_mcatnlo(str(self.input), {})
        rundir = self.assert_clean(shower='PYTHIA8')
        for index in range(1, 4):
            self.assertIn('Random:seed = %d' % (122 + index),
                          (rundir / ('run_%d/Pythia8.cmd' % index)).read_text())
        details = dict(call.args for call in self.cmd.results.add_detail.call_args_list)
        self.assertAlmostEqual(details['cross_pythia'], 2.)
        self.assertAlmostEqual(details['error_pythia'], 0.1 / 3**0.5)
        self.assertEqual(details['nb_event_pythia'], 6)

    def test_repeated_runs_keep_logs_and_remove_obsolete_splits(self):
        (self.events / 'events.lhe_99.lhe').write_text('stale input')
        (self.events / 'events.lhe_99.lhe.gz').write_text('stale compressed input')
        unrelated = self.events / 'events.lhe_notes'
        unrelated.write_text('keep this file')
        self.cmd.shower_card['analyse'] = ''
        self.run_shower('HEP')
        self.assert_clean()
        self.assertTrue(unrelated.is_file())
        logs = {p.name: p.read_text() for p in self.events.glob('*.log')}
        self.cmd.shower_card['nsplit_jobs'] = 1
        self.run_shower('HEP')
        self.assert_clean(invocation=2)
        for name, contents in logs.items():
            self.assertEqual((self.events / name).read_text(), contents)
        self.assertIn('Events: 0,1,2,3,4,5,6,7,8,9,10,11',
                      (self.events / 'mcatnlo_run_HERWIG6_2.log').read_text())
        self.assertTrue((self.events / 'events_HERWIG6_1.hep.gz').is_file())

    def test_average_normalisation_and_single_job_hwu(self):
        self.write_events(norm='average')
        self.run_shower()
        hwu = histograms.HwUList(str(self.events / 'plot_HERWIG6_1_0.HwU'))
        self.assertAlmostEqual(hwu[0].bins[0].wgts['central'], 2.)
        self.cmd.shower_card['nsplit_jobs'] = 1
        self.run_shower()
        self.assert_clean(invocation=2)
        hwu = histograms.HwUList(str(self.events / 'plot_HERWIG6_2_0.HwU'))
        self.assertAlmostEqual(hwu[0].bins[0].wgts['central'], 1.)

    def test_failed_analysis_preserves_input_and_archives_log(self):
        with self.assertRaisesRegex(interface.aMCatNLOError, 'status 17'):
            self.run_shower('FAIL')
        log = self.events / 'mcatnlo_run_HERWIG6_1__1.log'
        self.assertIn('Analysis failed', log.read_text())
        self.assertTrue(self.input.is_file())
        self.assertTrue((self.events / 'events.lhe_0.lhe').is_file())

    def test_analysis_with_no_histograms_still_archives_logs_and_cleans_up(self):
        self.run_shower('EMPTY')
        self.assert_clean()
        self.assertTrue((self.events / 'mcatnlo_run_HERWIG6_1.log').is_file())
