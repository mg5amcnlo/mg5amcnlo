"""Python side of the per-channel MAcNLOPS first emission (MacNLOPSMatch.cc).

No process export, compilation, integration or shower is run here.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from madgraph.various import collect_events
from madgraph.various import macnlops as mc


ROOT = Path(__file__).resolve().parents[3]

PREAMBLE = """<LesHouchesEvents version="3.0">
  <!--
  <macnlops_first_emission version="1" prescription="component-undamped-delta-v3" weight="macnlops" rejected="%(rejected)s" shower_alpha_s="0.13650000000000001" shower_alpha_s_order="1" pythia="8.318" cutoff_isr="0.50000000000000011" cutoff_fsr="0.53000000000000003"/>
  <initrwgt>
    <weightgroup name='macnlops' combine='none'>
      <weight id='macnlops'> macnlops </weight>
    </weightgroup>
  </initrwgt>
  -->
  <header>
  </header>
  <init>
   2212   2212 0.65000000E+04 0.65000000E+04 -1 -1   331700   331700 -3   2
 0.10000000E+02 0.10000000E+00 0.00000000E+00      1
 0.20000000E+02 0.20000000E+00 0.00000000E+00      2
  </init>
"""


def event(process, weight, macnlops):
    return """  <event>
  4 %d %s 0.10000000E+03 0.75467711E-02 0.11800000E+00
       21 -1    0    0  501  502  0.0  0.0  0.5E+02  0.5E+02  0.0 0.0 9.0
       21 -1    0    0  502  501  0.0  0.0 -0.5E+02  0.5E+02  0.0 0.0 9.0
       11  1    1    2    0    0  0.5E+02  0.0  0.0  0.5E+02  0.0 0.0 9.0
      -11  1    1    2    0    0 -0.5E+02  0.0  0.0  0.5E+02  0.0 0.0 9.0
#aMCatNLO 1  0  0  0  0 0.0D+00 0.0D+00   0  0  0 0.0D+00 0.0D+00 0.0D+00 0.0D+00 0.0D+00
  <rwgt>
   <wgt id='macnlops'> %s </wgt>
  </rwgt>
  </event>
""" % (process, weight, macnlops)


class TestFirstEmissionPython(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='macnlops_first_emission_')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def test_table_coupling_is_shared_by_python_c_and_the_backward_correction(self):
        native = (ROOT / 'Template/NLO/MCatNLO/srcPythia8/MacNLOPSMatch.cc').read_text()
        self.assertEqual(float(re.search(r'tableAlphaS = ([0-9.]+);', native)[1]),
                         mc.TABLE_ALPHA_S)
        self.assertEqual(int(re.search(r'tableAlphaSOrder = (\d+);', native)[1]),
                         mc.TABLE_ALPHA_S_ORDER)
        card = (ROOT / 'Template/NLO/SubProcesses/FKS_params.dat').read_text()
        self.assertEqual(float(re.search(r'#DeltaBackwardAlphaS\n([0-9.]+)d0', card)[1]),
                         mc.TABLE_ALPHA_S)
        self.assertEqual(int(re.search(r'#DeltaBackwardAlphaSOrder\n(\d+)', card)[1]),
                         mc.TABLE_ALPHA_S_ORDER)
        self.assertEqual(native.count(mc.FIRST_EMISSION_POLICY), 1)

    def gggg_registry(self):
        from models import import_ufo
        from madgraph.core import base_objects, diagram_generation, helas_objects
        from madgraph.iolibs.born_support import component_basis_metadata
        model = import_ufo.import_model('sm')
        legs = base_objects.LegList([
            base_objects.Leg({'id': 21, 'state': i > 1, 'number': i + 1}) for i in range(4)])
        born = helas_objects.HelasMatrixElement(diagram_generation.Amplitude(
            base_objects.Process({'legs': legs, 'model': model, 'orders': {'QED': 0}})))
        gluons = [[21, i >= 2, [], False, None] for i in range(5)]
        sectors = [dict(sector=1, real=1, real_identity=6,
                        fks=dict(i=5, j=3, ij=3, splitting_type=['QCD']),
                        processes=[['amplitude', gluons]]),
                   dict(sector=2, real=1, real_identity=6,
                        fks=dict(i=5, j=1, ij=1, splitting_type=['QCD']),
                        processes=[['amplitude', gluons]]),
                   dict(sector=3, real=1, real_identity=6,
                        fks=dict(i=5, j=4, ij=4, splitting_type=['QED']),
                        processes=[['amplitude', gluons]])]
        registry = dict(outer=dict(directory='P0_gg_gg', nexternal=5, born_pdgs=[[21] * 4],
                                   component_basis=component_basis_metadata(born),
                                   sectors=sectors),
                        tables=[], contexts=[], flavour_complete=True)
        fingerprint = hashlib.sha256(json.dumps(registry, sort_keys=True,
            separators=(',', ':')).encode('utf-8')).hexdigest()
        directory = self.work / 'P0_gg_gg'
        directory.mkdir()
        (directory / 'mc_histories.json').write_text(json.dumps(dict(
            schema='macnlops-history-registry-v1', sha256=fingerprint, registry=registry)))
        return directory, registry, fingerprint

    def test_route_table_exports_sectors_and_certified_born_transports(self):
        directory, registry, fingerprint = self.gggg_registry()
        words = mc.write_route_table(directory).read_text().split('\n')
        self.assertEqual(words[0], 'MACNLOPS_ROUTES_V1 %s P0_gg_gg 5 4' % fingerprint)
        self.assertEqual(words[1:3], ['BORN_PDGS 1', '21 21 21 21'])
        # The two final gluons are interchangeable, the beams are not.
        self.assertEqual(words[3], 'STATE_CLASSES 0 0 2 2')
        self.assertEqual(words[4], 'SECTORS 3')
        self.assertEqual(words[5:7], ['SECTOR 1 5 3 3 1 1', '21 21 21 21 21'])
        self.assertIn('SECTOR 3 5 4 4 0 1', words)
        start = words.index(next(word for word in words if word.startswith('TRANSPORT')))
        rows = [tuple(map(int, row.split())) for row in words[start + 1:-2]]
        basis = registry['outer']['component_basis']
        self.assertEqual(len(rows), len(basis['graphs']) * len(basis['flows']))
        self.assertEqual(words[-2:], ['END', ''])
        images = {row[:5]: row[5:] for row in rows}
        for key, image in images.items():
            self.assertEqual(key[:2], (2, 3))
            self.assertEqual(images[(2, 3) + image], key[2:])
        self.assertTrue(any(key[2:] != image for key, image in images.items()))

    def test_route_table_requires_a_matching_registry(self):
        directory, registry, fingerprint = self.gggg_registry()
        record = json.loads((directory / 'mc_histories.json').read_text())
        record['sha256'] = '0' * 64
        (directory / 'mc_histories.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(mc.MatchingError, 'branching routes'):
            mc.write_route_table(directory)

    def test_marker_gives_the_shower_hand_off(self):
        header = PREAMBLE % dict(rejected='kept')
        marker = mc.first_emission_header(header)
        self.assertEqual(marker['shower_cutoffs'], [0.50000000000000011, 0.53000000000000003])
        self.assertEqual(marker['rejected'], 'kept')
        self.assertIsNone(mc.first_emission_header('<header></header>'))
        self.assertFalse(mc.matched_lhe_header(header))
        for broken in (header.replace('weight="macnlops"', 'weight="other"'),
                       header.replace('cutoff_fsr="0.53000000000000003"', 'cutoff_fsr="-1"'),
                       header.replace('/>\n  <initrwgt>', '/>\n' + header.split('\n')[2] +
                                      '\n  <initrwgt>')):
            with self.assertRaises(mc.MatchingError):
                mc.first_emission_header(broken)
        collected = self.work / 'events.lhe'
        collected.write_text(header.replace('<header>\n', '<header>\n<MGRunCard>\n'
            '  12 = nevents ! events\n</MGRunCard>\n') + event(1, '0.1E+01', '0.1E+01') +
            '</LesHouchesEvents>\n')
        contract = mc.first_emission_contract(collected)
        self.assertEqual((contract['events'], contract['rejected'], contract['environment']),
                         (12, 'kept', None))
        self.assertEqual(contract['schema'], 'macnlops-first-emission-v1')

    def channels(self, rejected):
        jobs, summaries = [], []
        kept = dict(kept=[('1', 1.0, '0.10000000E+01'), ('2', 2.0, '0.00000000E+00'),
                          ('1', -0.5, '-.50000000E+00')],
                    dropped=[('1', 1.0, '0.10000000E+01'), ('1', -0.5, '-.50000000E+00')])
        for channel in (1, 2):
            output = self.work / ('channel_%d.lhe' % channel)
            rows = kept[rejected] if channel == 1 else [('2', 4.0, '0.40000000E+01')]
            output.write_text(PREAMBLE % dict(rejected=rejected) + ''.join(
                event(int(process), '%.8E' % weight, text) for process, weight, text in rows) +
                '</LesHouchesEvents>\n')
            processes = {}
            for process, weight, text in rows:
                item = processes.setdefault(process, dict(events=0, input=0., macnlops=0.,
                    macnlops_squares=0., macnlops_max=0.))
                value = mc.number(text)
                item['events'] += 1
                item['input'] += weight
                item['macnlops'] += value
                item['macnlops_squares'] += value * value
                item['macnlops_max'] = max(item['macnlops_max'], abs(value))
            accepted = sum(mc.number(text) != 0 for _, _, text in rows)
            if channel == 1 and rejected == 'dropped':
                # One process-2 event was vetoed and not written.
                processes['2'] = dict(events=1, input=2.0, macnlops=0., macnlops_squares=0.,
                                      macnlops_max=0.)
            jobs.append(dict(output=str(output)))
            summaries.append(dict(rejected_events=rejected, processes=processes,
                                  counts=dict(accepted_S=accepted, accepted_H=0)))
        return jobs, summaries

    def test_kept_rejections_collect_the_channel_files_unchanged(self):
        jobs, summaries = self.channels('kept')
        banner = self.work / 'banner.txt'
        banner.write_text('<MGRunCard>\n  6 = nevents\n</MGRunCard>\n')
        inputs, template = mc.first_emission_collection(jobs, summaries, str(banner), self.work)
        self.assertEqual((inputs, template), ([job['output'] for job in jobs], str(banner)))

    def test_dropped_rejections_restate_the_retained_cross_sections(self):
        jobs, summaries = self.channels('dropped')
        banner = self.work / 'banner.txt'
        banner.write_text('<header>\n<MGRunCard>\n  6 = nevents ! requested\n'
                          '</MGRunCard>\n</header>\n')
        inputs, template = mc.first_emission_collection(jobs, summaries, str(banner), self.work)
        self.assertEqual(inputs[1:], [job['output'] for job in jobs])
        with mc.LHEStream(inputs[0]) as stream:
            self.assertEqual(list(stream), [])
            init = re.search(r'<init>\n(.*?)</init>', stream.header, re.S)[1].splitlines()
        self.assertEqual(init[0].split()[-1], '2')
        rows = {int(row.split()[3]): [float(x) for x in row.split()[:3]] for row in init[1:]}
        self.assertAlmostEqual(rows[1][0], 0.5)
        self.assertAlmostEqual(rows[2][0], 4.0)
        self.assertEqual([rows[1][2], rows[2][2]], [1.0, 4.0])
        # Channel 1 has three trials (one dropped), channel 2 one.
        expected = mc.sample_sum_variance(3, 0.5, 1.25) + 0.
        self.assertAlmostEqual(rows[1][1], expected ** .5)
        self.assertIn('  3 = nevents ! requested', Path(template).read_text())

    def test_collector_carries_the_marker_and_weight_declaration(self):
        jobs, summaries = self.channels('kept')
        banner = self.work / 'banner.txt'
        banner.write_text('<LesHouchesEvents version="3.0">\n<header>\n<MGRunCard>\n'
                          '  4 = nevents\n</MGRunCard>\n</header>\n')
        output = self.work / 'collected.lhe'
        collect_events.collect_events(output=output, header_template=banner,
            template_header_keep_tags=['MGRunCard'], input_files=[job['output'] for job in jobs],
            seed=3, subset=None, mode='auto', prefer_pigz=False, gzip_level=6, verbose=False)
        with mc.LHEStream(output) as stream:
            header, blocks = stream.header, list(stream)
        inside = re.search(r'<header>(.*?)</header>', header, re.S)[1]
        self.assertEqual(mc.first_emission_header(inside)['rejected'], 'kept')
        self.assertIn("<weight id='macnlops'>", inside)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(Counter(re.search(r"<wgt id='macnlops'> (\S+) </wgt>", block)[1]
                                 for block in blocks)['0.00000000E+00'], 1)


# Stand-ins for macnlops_capi.f: fixed densities R=1, D=3 (u=2) with K=4 for
# a shower endpoint and K=1 for an H point, at alpha_s^K=0.118 and muF=100 GeV.
STUB = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char* registry = "%(registry)s";
void macnlops_c_setup(int* nlegs, char* reg) { *nlegs = 5; memcpy(reg, registry, 64); }
void macnlops_c_profile(double* cutoffs, double* xmax, double* masses, int* flavours) {
  FILE* f = fopen("stub_profile.txt", "r");
  if (!f || fscanf(f, "%%lf %%lf %%lf %%lf %%lf %%lf %%d %%d", cutoffs, cutoffs + 1, xmax,
                   xmax + 1, masses, masses + 1, flavours, flavours + 1) != 8) exit(3);
  fclose(f);
}
void macnlops_c_configure(double* xmax, double* isr, double* fsr) {}
void macnlops_c_density(int owner, const int* ids, const double* lab, int config,
  int graph, int flow, int damping, double symmetry, int has_born, const int* born_ids,
  const int* born_colours, const double* born_lab, const double* born_masses,
  double* r, double* d, double* k, double* alphas, double* q2fact, int* status) {
  if (owner < 1 || owner > 4 || damping != 1 || symmetry <= 0.) exit(4);
  *r = 1.; *d = 3.; *k = has_born ? 4. : 1.; *alphas = 0.118;
  q2fact[0] = q2fact[1] = 1.e4; *status = 0;
}
"""

ROUTES = """MACNLOPS_ROUTES_V1 %(registry)s P0_uux_emep 5 4
BORN_PDGS 2
2 -2 11 -11
4 -4 11 -11
STATE_CLASSES 0 1 2 3
SECTORS 4
SECTOR 1 5 1 1 1 2
2 -2 11 -11 21
4 -4 11 -11 21
SECTOR 2 5 2 2 1 2
2 -2 11 -11 21
4 -4 11 -11 21
SECTOR 3 5 1 1 1 2
21 -4 11 -11 -4
21 -2 11 -11 -2
SECTOR 4 5 2 2 1 2
2 21 11 -11 2
4 21 11 -11 4
TRANSPORT 0
END
"""


class TestFirstEmissionNative(unittest.TestCase):
    """MacNLOPSMatch.cc with stand-in densities on actual DY source events."""

    REGISTRY = '4c9dcf18f058e3159bcc1b88d941806220b8ccca6e97f1ca598d7eab1930fc1f'

    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get('PYTHIA8') or os.environ.get('PYTHIA8LOCATION')
        if not prefix:
            raise unittest.SkipTest('Set PYTHIA8 to compile the first-emission program')
        cls.prefix = Path(prefix).resolve()
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops_first_emission_native_')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        source = ROOT / 'Template/NLO/MCatNLO/srcPythia8'
        (cls.work / 'stub.c').write_text(STUB % dict(registry=cls.REGISTRY))
        common = ['-I' + str(cls.prefix / 'include'), '-L' + str(cls.prefix / 'lib'),
                  '-Wl,-rpath,' + str(cls.prefix / 'lib'), '-lpythia8', '-ldl']
        for command in (['gcc', '-c', 'stub.c', '-o', 'stub.o'],
                        [os.environ.get('CXX', 'g++'), '-std=c++11', '-O1',
                         str(source / 'MacNLOPSMatch.cc'), 'stub.o', '-o', 'match.exe'] + common,
                        [os.environ.get('CXX', 'g++'), '-std=c++11', '-O1',
                         str(source / 'MacNLOPS.cc'), '-o', 'probe.exe'] + common):
            built = subprocess.run(command, cwd=cls.work, capture_output=True, text=True)
            if built.returncode:
                raise AssertionError(built.stdout + built.stderr)
        cls.env = os.environ.copy()
        cls.pdf = cls.env.get('MACNLOPS_TEST_PDF', 'LHAPDF6:MSTW2008nlo90cl/0')
        cls.settings = cls.work / 'shower.cmnd'
        cls.settings.write_text(''.join('%d:m0 = %r\n' % item for item in (
            (1, 0.), (2, 0.), (3, 0.), (4, 0.), (5, 0.), (6, 173.), (11, 0.), (13, 0.),
            (15, 1.777), (21, 0.), (23, 91.188), (24, 80.419), (25, 125.))) +
            'SpaceShower:pTmin = 0.5\nTimeShower:pTmin = 0.5\n')
        # The generation profile is the stock probe of the same settings.
        probe = cls.work / 'probe.json'
        subprocess.run(['./probe.exe', '--probe-config', str(probe), '--beam-id-a', '2212',
            '--beam-id-b', '2212', '--beam-energy-a', '6500', '--beam-energy-b', '6500',
            '--pdf', cls.pdf, '--alpha-s-mz', repr(mc.TABLE_ALPHA_S), '--alpha-s-order',
            str(mc.TABLE_ALPHA_S_ORDER), '--settings', str(cls.settings), '--flavour-count',
            '5', '--mcatnlo-delta'], cwd=cls.work, env=cls.env, check=True,
            capture_output=True, text=True)
        support = json.loads(probe.read_text())
        (cls.work / 'stub_profile.txt').write_text('%r %r %r %r %r %r %d %d\n' % (
            *support['shower_cutoffs'], *support['beam_xmax'],
            support['heavy_mass_floors']['4'], support['heavy_mass_floors']['5'],
            support['n_quark_in'], support['n_gluon_to_quark']))
        (cls.work / 'routes.dat').write_text(ROUTES % dict(registry=cls.REGISTRY))
        (cls.work / 'res_1').write_text(' Final result [S]:   12.0  +/-  0.1\n'
                                        ' Final result [Born]:   10.0  +/-  0.1\n')
        shutil.copyfile(ROOT / 'tests/input_files/macnlops_first_emission_dy.lhe',
                        cls.work / 'events.lhe')
        with mc.LHEStream(cls.work / 'events.lhe') as stream:
            cls.source = list(stream)

    def run_match(self, name, *options):
        result = subprocess.run(['./match.exe', '--input', 'events.lhe', '--output',
            name + '.lhe', '--summary', name + '.json', '--routes', 'routes.dat',
            '--settings', str(self.settings), '--pdf', self.pdf, '--seed', '7349',
            *options], cwd=self.work, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with mc.LHEStream(self.work / (name + '.lhe')) as stream:
            header, blocks = stream.header, list(stream)
        return header, blocks, json.loads((self.work / (name + '.json')).read_text())

    def check_event(self, block, beta, pdf_ratio):
        decision = mc.xml_attributes(re.search(
            r'<macnlops_decision\b([^<>]*?)/>', block)[1])
        nominal = block.splitlines()[1].split()[2]
        weight = re.search(r"<wgt id='macnlops'> (\S+) </wgt>", block)[1]
        accepted, keep = decision['accepted'] == '1', mc.number(decision['keep'])
        self.assertEqual(weight, nominal if accepted else '0.00000000E+00')
        lines = block.splitlines()
        tag = next(i for i, line in enumerate(lines) if line.startswith('#aMCatNLO'))
        self.assertEqual([lines[tag + 1].strip(), lines[tag + 3].strip()], ['<rwgt>', '</rwgt>'])
        event = mc.Event(block)
        self.assertIsNotNone(event.continuation)
        if decision['rule'] == 'positive_h':
            self.assertEqual(keep, 1.)
        elif decision['rule'] == 'h_removal':
            self.assertTrue(event.weight < 0)
            self.assertEqual(keep, .5)
        elif decision['rule'] == 's_veto':
            alpha = mc.number(decision['alpha_s_shower'])
            ratio = mc.number(decision.get('pdf_ratio', '1'))
            if not pdf_ratio:
                self.assertEqual(ratio, 1.)
            scale = mc.number(decision['kernel_scale'])
            self.assertAlmostEqual(scale, beta * alpha / .118 * ratio, delta=1.e-12 * scale)
            self.assertAlmostEqual(keep, 1. - min(.5 / scale, 1.), delta=1.e-14)
        else:
            self.assertEqual((decision['rule'], keep), ('exhausted', 1.))
        return decision, accepted

    def test_every_event_is_written_with_its_macnlops_weight(self):
        header, blocks, summary = self.run_match('kept')
        self.assertEqual(len(blocks), len(self.source))
        marker = mc.first_emission_header(header)
        self.assertEqual((marker['prescription'], marker['rejected']),
                         (mc.FIRST_EMISSION_POLICY, 'kept'))
        self.assertIn("<weight id='macnlops'> macnlops </weight>", header)
        rules, accepted = Counter(), Counter()
        for block in blocks:
            decision, kept = self.check_event(block, 1.2, True)
            rules[decision['rule']] += 1
            accepted[kept] += 1
        self.assertEqual(rules['h_removal'], 3)
        self.assertEqual(rules['positive_h'], 3)
        self.assertEqual(rules['s_veto'] + rules['exhausted'] + rules['below_matching_scale'], 8)
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['channel']['beta'], 1.2)
        self.assertEqual(summary['counts']['input_S'] + summary['counts']['input_H'], 14)
        self.assertEqual(summary['counts'].get('rejected_H', 0), accepted[False] -
                         summary['counts'].get('rejected_S', 0))
        self.assertAlmostEqual(summary['weight_sums']['macnlops'], sum(
            mc.number(re.search(r"<wgt id='macnlops'> (\S+) </wgt>", block)[1])
            for block in blocks), delta=1.e-12)

    def test_v2_has_no_pdf_ratio_and_dropped_rejections_are_omitted(self):
        _, kept, _ = self.run_match('v2', '--s-veto', 'v2')
        for block in kept:
            self.check_event(block, 1.2, False)
        header, blocks, summary = self.run_match('dropped', '--drop-rejected')
        self.assertEqual(mc.first_emission_header(header)['rejected'], 'dropped')
        self.assertEqual(len(blocks), summary['counts'].get('accepted_S', 0) +
                         summary['counts'].get('accepted_H', 0))
        for block in blocks:
            self.assertTrue(self.check_event(block, 1.2, True)[1])


if __name__ == '__main__':
    unittest.main()
