"""Sampled source packets survive LHE collection without changing their law."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = ROOT / 'Template/NLO/SubProcesses'


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestSampledSourcePacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='macnlops-component-packet-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        source = (TEMPLATE / 'handling_lhe_events.f').read_text()
        (cls.work / 'packets.f').write_text(source[source.index(
            '      subroutine clear_macnlops_hard_event()'):])
        (cls.work / 'check.f90').write_text('''program check
 use weight_lines
 implicit none
 integer mode,meta(4)
 real(8) values(4),p(0:3,4),xbj(2)
 character(1000) line,arg
 character(64),parameter :: registry=repeat('a',64)
 call get_command_argument(1,arg)
 read(arg,*)mode
 call clear_macnlops_hard_event()
 if(mode==0)then
   meta=[3,2,7,4];values=[.25d0,.3d0,.4d0,2d0]
   call store_macnlops_component(meta,values,registry)
   call write_macnlops_hard_event(6)
   call write_macnlops_hard_event(6)
 elseif(mode==1.or.mode==2)then
   read(*,'(a)')line
   call read_macnlops_component(5,line)
   if(mode==2)then
     read(*,'(a)')line
     call read_macnlops_component(5,line)
   endif
   call write_macnlops_hard_event(6)
 else
   p=1d0
   xbj=[.1d0,.2d0]
   call check_component_source_born(p,xbj)
   call check_component_source_born(p,xbj)
   if(mode==3)then
     p(0,3)=2d0
     call check_component_source_born(p,xbj)
   endif
   if(mode==5)then
     xbj=[.2d0,.1d0]
     call check_component_source_born(p,xbj)
   endif
   call reset_component_contexts()
   p=2d0
   call check_component_source_born(p,xbj)
   if(any(component_source_born/=2d0))stop 1
   print *, 'PASS common sampled Born state'
 endif
end program
''')
        result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
            '-ffixed-line-length-none', str(TEMPLATE / 'weight_lines.f'),
            str(cls.work / 'packets.f'), str(cls.work / 'check.f90'),
            '-o', str(cls.work / 'check')], cwd=cls.work,
            capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def invoke(self, mode, text=''):
        return subprocess.run([str(self.work / 'check'), str(mode)], cwd=self.work,
            input=text, capture_output=True, text=True, timeout=10)

    def packet(self):
        result = self.invoke(0)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('<macnlops_component '), 1)
        return result.stdout

    def test_exact_roundtrip_and_single_use(self):
        packet = self.packet()
        self.assertEqual(list(map(int, packet.splitlines()[1].split())), [3, 2, 7, 4])
        self.assertEqual(list(map(float, packet.splitlines()[2].split())), [.25, .3, .4, 2.])
        result = self.invoke(1, packet)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, packet)
        self.assertNotEqual(self.invoke(2, packet + packet).returncode, 0)

    def test_rejects_invalid_identity_scalars_and_truncation(self):
        packet = self.packet()
        invalid = [packet.replace('version="1"', 'version="2"'),
                   packet.replace('registry="a', 'registry="g'),
                   packet.replace('</macnlops_component>', '')]
        for row, position, value in [(1, 0, '0'), (1, 2, '-1'),
                                     (2, 0, '0'), (2, 0, '1.1'),
                                     (2, 1, '-.1'), (2, 2, '1.1'),
                                     (2, 3, '0'), (2, 3, 'NaN')]:
            rows = packet.splitlines()
            fields = rows[row].split()
            fields[position] = value
            rows[row] = ' '.join(fields)
            invalid.append('\n'.join(rows) + '\n')
        for row in (1, 2):
            rows = packet.splitlines()
            rows[row] += ' 99'
            invalid.append('\n'.join(rows) + '\n')
        for text in invalid:
            with self.subTest(packet=text):
                self.assertNotEqual(self.invoke(1, text).returncode, 0)

    def test_combined_s_packet_requires_one_born_state(self):
        result = self.invoke(4)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS common sampled Born state', result.stdout)
        self.assertNotEqual(self.invoke(3).returncode, 0)
        self.assertNotEqual(self.invoke(5).returncode, 0)


if __name__ == '__main__':
    unittest.main()
