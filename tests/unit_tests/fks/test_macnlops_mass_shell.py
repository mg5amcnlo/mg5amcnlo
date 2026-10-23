"""Mass-shell preflight on a real boosted massive-FSR shower point."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMacNLOPSMassShell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from madgraph.various.macnlops import Event
        # This ordinary bbar -> bbar g point previously failed the guard:
        # the recoiling bottom has E=2711 GeV and a 1.20e-6 GeV^2
        # invariant cancellation error, despite native stored mass 4.7 GeV.
        fixture = ROOT / 'tests/input_files/macnlops_boosted_bottom_fsr.lhe'
        cls.point = Event(fixture.read_text()).external()
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-shell-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        source = (ROOT / 'Template/NLO/SubProcesses/driver_macnlops.f').read_text()
        guard = source[source.index('      hard_scale2=max('):
                       source.index('      if(.not.MC_HIST_COMPLETE(owner))')]
        driver = cls.work / 'guard.f'
        driver.write_text('''      program test
      implicit none
      integer ids(6),status,i,ios
      double precision lab(0:3,6)
      do
         do i=1,6
            read(*,*,iostat=ios)ids(i),lab(:,i)
            if(ios.ne.0)stop
         enddo
         call preflight(ids,lab,status)
         write(*,'(a,1x,i8)')'STATUS',status
      enddo
      end
      subroutine preflight(ids,lab,status)
      implicit none
      integer nexternal,nincoming
      parameter(nexternal=6,nincoming=2)
      integer ids(nexternal),status,leg
      double precision lab(0:3,nexternal),model_mass,mass2,
     $     tolerance,hard_scale2,get_mass_from_id
      external get_mass_from_id
      status=0
''' + guard + ''' 900  continue
      end
      double precision function get_mass_from_id(pid)
      implicit none
      integer pid
      get_mass_from_id=0d0
      if(abs(pid).eq.5)get_mass_from_id=4.7d0
      end
''')
        cls.executable = cls.work / 'guard'
        subprocess.run(['gfortran', '-O2', '-ffixed-line-length-132',
                        str(driver), '-o', str(cls.executable)],
                       capture_output=True, check=True, text=True)

    def evaluate(self, point):
        request = ''.join('%d %s\n' % (pdg, ' '.join('%.17e' % x for x in p))
                          for pdg, p in point)
        output = subprocess.run([str(self.executable)], input=request,
                                capture_output=True, text=True, check=True).stdout
        return int(next(line.split()[1] for line in output.splitlines()
                        if line.startswith('STATUS ')))

    def test_boosted_ordinary_bottom_fsr_preserves_native_point(self):
        self.assertEqual(self.evaluate(self.point), 0)

    def test_genuine_forced_charm_sister_is_rejected(self):
        point = list(self.point)
        _, p = point[-1]
        # A forced charm conversion puts m=1.2 GeV on a hard-model
        # massless sister; this is a physical mismatch, not roundoff.
        point[-1] = (4, (math.sqrt(sum(x*x for x in p[1:]) + 1.2**2),) + p[1:])
        self.assertEqual(self.evaluate(point), 24)

    def test_genuine_bottom_mass_change_is_rejected(self):
        point = list(self.point)
        pid, p = point[2]
        point[2] = (pid, (math.sqrt(sum(x*x for x in p[1:]) + 4.8**2),) + p[1:])
        self.assertEqual(self.evaluate(point), 24)
