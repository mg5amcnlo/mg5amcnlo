"""Read actual ISR/FSR limits after stock Pythia imports an endpoint LHE.

Unlike the formatting checks, these verify the global anchor, incoming
minimum-positive rule and directed final-dipole overrides inside the shower.
"""
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class TestNativeContinuationScales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix = os.environ.get('PYTHIA8') or os.environ.get('PYTHIA8LOCATION')
        if not prefix:
            raise unittest.SkipTest('Set PYTHIA8 for native continuation-scale checks')
        compiler = shutil.which(os.environ.get('CXX', 'g++'))
        if not compiler:
            raise unittest.SkipTest('Requires a C++ compiler')
        prefix = Path(prefix)
        cls.tmp = tempfile.TemporaryDirectory(prefix='macnlops-native-scales-')
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.work = Path(cls.tmp.name)
        cls.exe = cls.work/'probe'
        subprocess.run([compiler, '-std=c++11', '-O1',
            '-I'+str(ROOT/'Template/NLO/MCatNLO/srcPythia8'),
            '-I'+str(prefix/'include'), str(ROOT/'docs/audits/macnlops_continuation_scale_probe.cc'),
            '-L'+str(prefix/'lib'), '-Wl,-rpath,'+str(prefix/'lib'),
            '-lpythia8', '-ldl', '-o', str(cls.exe)],check=True,capture_output=True)

    def check(self, scale, pairs, kind):
        # Exactly on-shell, momentum-conserving ttbar+g fixture, with two
        # distinct colour connections on the final gluon.
        py = math.sqrt(200**2-173**2-50**2)
        anchors = ' '.join('scalup_%d="%s"' % (i,scale) for i in range(1,6))
        matrix = ' '.join('scalup_%d_%d="%s"' % (i,j,q) for (i,j),q in pairs.items())
        source = self.work/'input.lhe'
        source.write_text('''<LesHouchesEvents version="3.0">
<header></header>
<init>
2212 2212 6500 6500 0 0 0 0 -4 1
1 0 1 1
</init>
<event>
5 1 1 {scale} 0.007297 0.118
21 -1 0 0 501 502 0 0 250 250 0 0 9
21 -1 0 0 502 503 0 0 -250 250 0 0 9
6 1 1 2 501 0 -50 {py:.17g} 0 200 173 0 9
-6 1 1 2 0 504 -50 {minus_py:.17g} 0 200 173 0 9
21 1 1 2 504 503 100 0 0 100 0 0 9
#aMCatNLO {kind} 0 0 0 0 0 0 0 0 0 0 0 0 0 0
<scales {anchors} {matrix}></scales>
</event>
</LesHouchesEvents>
'''.format(scale=scale,py=py,minus_py=-py,kind=kind,anchors=anchors,matrix=matrix))
        settings = self.work/'settings.cmnd'
        settings.write_text('''Beams:frameType = 4
Beams:setProductionScalesFromLHEF = on
Beams:setDipoleShowerStartingScalesFromLHEF = on
SpaceShower:pTmaxMatch = 1
TimeShower:pTmaxMatch = 1
SpaceShower:pTmaxFudge = 1.
TimeShower:pTmaxFudge = 1.
TimeShower:globalRecoil = off
PartonLevel:MPI = off
HadronLevel:Hadronize = off
SpaceShower:QEDshowerByQ = off
SpaceShower:QEDshowerByL = off
TimeShower:QEDshowerByQ = off
TimeShower:QEDshowerByL = off
TimeShower:QEDshowerByGamma = off
TimeShower:QEDshowerByOther = off
6:mayDecay = off
LesHouches:setQuarkMass = 0
PDF:pSet = '''+os.environ.get('MACNLOPS_TEST_PDF','LHAPDF6:MSTW2008nlo90cl/0')+'\n')
        output = self.work/'scales.csv'
        result = subprocess.run([str(self.exe),str(source),str(settings),str(output),'inspect'],
                                capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        with output.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows),1)
        row = {k:float(v) for k,v in rows[0].items()}
        for name in ['global_error','production_error','dipole_error','missing_fsr_pairs']:
            self.assertEqual(row[name],0.)
        return row

    def test_s_common_winning_scale_is_the_native_global_and_dipole_ceiling(self):
        pairs = {pair:25 for pair in [(1,2),(1,3),(2,1),(2,5),(3,1),(4,5),(5,2),(5,4)]}
        row = self.check(25,pairs,1)
        self.assertEqual([row[n] for n in ['global','beam1','beam2','max_fsr']], [25.]*4)

    def test_h_directed_fsr_override_survives_the_production_minimum(self):
        pairs = {(1,2):8,(1,3):8,(2,1):12,(2,5):12,(3,1):30,(4,5):5,(5,2):12,(5,4):20}
        row = self.check(60,pairs,2)
        self.assertEqual([row[n] for n in ['global','beam1','beam2','max_fsr']], [60.,8.,12.,30.])

    def test_zero_h_pair_stops_that_fsr_end_without_zeroing_its_production_anchor(self):
        pairs = {(1,2):8,(1,3):8,(2,1):12,(2,5):12,(3,1):0,(4,5):5,(5,2):12,(5,4):20}
        row = self.check(60,pairs,2)
        self.assertEqual([row[n] for n in ['global','beam1','beam2','max_fsr']], [60.,8.,12.,20.])


if __name__ == '__main__':
    unittest.main()
