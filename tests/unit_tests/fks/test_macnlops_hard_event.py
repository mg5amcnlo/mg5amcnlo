"""Fortran hard-coordinate annotations are optional, precise and single-use."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestHardEventAnnotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='hard-event-annotation-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work=Path(cls.temp.name)
        source=(ROOT/'Template/NLO/SubProcesses/handling_lhe_events.f').read_text()
        (cls.work/'buffer.f').write_text(source[source.index(
            '      subroutine clear_macnlops_hard_event()'):])
        (cls.work/'check.f90').write_text('''program check
 implicit none
 integer mode,i,ic(7,4),meta(15)
 real(8) p(0:4,4)
 character(1000) line,arg
 call get_command_argument(1,arg)
 read(arg,*)mode
 call clear_macnlops_hard_event()
 if(mode==0.or.mode==1)then
  ic=0;p=0d0;meta=0
  ic(1,:)=[2,-2,11,-11];ic(6,:)=[-1,-1,1,1];ic(7,:)=9
  ic(2:3,3)=1;ic(2:3,4)=2
  ic(4,1)=501;ic(5,2)=501
  p(0,:)=10d0;p(3,1)=10d0;p(3,2)=-10d0
  p(1,3)=10d0;p(1,4)=-10d0
  meta(1:3)=[1,2,2];meta(7:13)=[1,1,0,0,1,1,4]
  meta(15)=1
  write(*,'(a)')'<event>'
  write(*,'(a)')'ordinary body, weights and SCALUP stay byte-identical'
  if(mode==1)call store_macnlops_hard_event(4,ic,p,meta)
  call write_macnlops_hard_event(6)
  write(*,'(a)')'</event>'
  ! Writing consumes the packet, so the next event cannot inherit it.
  write(*,'(a)')'<event>'
  call write_macnlops_hard_event(6)
  write(*,'(a)')'</event>'
 else
  read(*,'(a)')line
  call read_macnlops_hard_event(5,line)
  if(mode==3)then
   read(*,'(a)')line
   call read_macnlops_hard_event(5,line)
  endif
  call write_macnlops_hard_event(6)
 endif
end program
''')
        result=subprocess.run(['gfortran','-O2','-fcheck=all',
            '-ffixed-line-length-none',str(cls.work/'buffer.f'),
            str(cls.work/'check.f90'),'-o',str(cls.work/'check')],
            cwd=cls.work,capture_output=True,text=True)
        if result.returncode:raise AssertionError(result.stdout+result.stderr)

    def run_packet(self, mode, text=''):
        return subprocess.run([str(self.work/'check'),str(mode)],input=text,
            capture_output=True,text=True,timeout=10)

    def packet(self):
        output=self.run_packet(1).stdout
        first=output.index('  <macnlops_hard_event')
        last=output.index('  </macnlops_hard_event>')+len('  </macnlops_hard_event>\n')
        return output[first:last]

    def test_toggle_preserves_body_and_never_leaks_to_next_event(self):
        with_packet=self.run_packet(1)
        without=self.run_packet(0)
        self.assertEqual(with_packet.returncode,0,with_packet.stderr)
        self.assertEqual(with_packet.stdout.replace(self.packet(),''),without.stdout)
        self.assertEqual(with_packet.stdout.count('<macnlops_hard_event'),1)
        packet=self.packet()
        self.assertIn('source="ordinary" partition="global_sum"',packet)
        metadata=list(map(int,packet.splitlines()[1].split()))
        self.assertEqual(metadata,[4,1,2,2,0,0,0,1,1,0,0,1,1,4])
        for row in packet.splitlines()[2:-1]:
            self.assertEqual(len(row.split()),13)
            self.assertEqual(float(row.split()[10]),0.)

    def test_collect_reader_preserves_packet_exactly(self):
        packet=self.packet();result=self.run_packet(2,packet)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,packet)

    def test_generic_collector_preserves_one_incoming_decay(self):
        rows=self.packet().splitlines()
        fields=rows[3].split();fields[1]='1';rows[3]=' '.join(fields)
        packet='\n'.join(rows)+'\n'
        result=self.run_packet(2,packet)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,packet)

    def test_rejects_unsupported_version_and_false_global_partition(self):
        packet=self.packet()
        for invalid in [packet.replace('version="1"','version="2"'),
                        packet.replace('partition="global_sum"','partition="unknown"')]:
            self.assertNotEqual(self.run_packet(2,invalid).returncode,0)
        rows=packet.splitlines();meta=rows[1].split();meta[9]='1'
        rows[1]=' '.join(meta)
        self.assertNotEqual(self.run_packet(2,'\n'.join(rows)+'\n').returncode,0)

    def test_rejects_truncated_nonfinite_and_nonexternal_particles(self):
        packet=self.packet();self.assertNotEqual(self.run_packet(2,
            packet.replace('  </macnlops_hard_event>\n','')).returncode,0)
        for index,value in [(1,'2'),(9,'NaN')]:
            rows=packet.splitlines();fields=rows[2].split();fields[index]=value
            rows[2]=' '.join(fields)
            self.assertNotEqual(self.run_packet(2,'\n'.join(rows)+'\n').returncode,0)

    def test_rejects_duplicate_and_extra_fields(self):
        packet=self.packet()
        self.assertNotEqual(self.run_packet(3,packet+packet).returncode,0)
        for index in [1,2]:
            rows=packet.splitlines();rows[index]+=' 123'
            self.assertNotEqual(self.run_packet(2,'\n'.join(rows)+'\n').returncode,0)


if __name__=='__main__':unittest.main()
