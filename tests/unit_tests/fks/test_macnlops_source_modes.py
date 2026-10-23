"""Ordinary source records and physical proposal support stay independent."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMacNLOPSSourceModes(unittest.TestCase):
    def test_ordinary_real_transfer_and_flavour_resolved_proposal(self):
        source = (ROOT/'Template/NLO/SubProcesses/fks_singular.f').read_text()
        first = source.index('      subroutine macnlops_validate_profile')
        last = source.index('      subroutine recompute_xlum_for_wgt_mint', first)
        with tempfile.TemporaryDirectory(prefix='macnlops-source-modes-') as temp:
            work = Path(temp)
            includes = {
                'nexternal.inc': 'integer,parameter :: nexternal=5,nincoming=2',
                'nFKSconfigs.inc': 'integer,parameter :: fks_configs=1',
                'fks_info.inc': 'integer,parameter :: fks_i_d(1)=[5],fks_j_d(1)=[1]',
                'run.inc': '''logical macnlops,macnlops_resolved_source
common/mode/macnlops,macnlops_resolved_source
double precision macnlops_isr_cutoff,macnlops_fsr_cutoff,
     $ macnlops_charm_mass,macnlops_bottom_mass,macnlops_beam_xmax(2)
common/profile/macnlops_isr_cutoff,macnlops_fsr_cutoff,
     $ macnlops_charm_mass,macnlops_bottom_mass,macnlops_beam_xmax
integer macnlops_nquark_in,macnlops_ngluon_to_quark
common/flavours/macnlops_nquark_in,macnlops_ngluon_to_quark''',
            }
            for name, value in includes.items():
                (work/name).write_text('\n'.join(
                    row if row.startswith('     $') else '      '+row
                    for row in value.splitlines())+'\n')
            (work/'context.f90').write_text('module mc_native_context\n'
                'integer :: history_permutations(5,1)=0\nend module\n')
            (work/'factor.f').write_text(source[first:last])
            (work/'check.f90').write_text('''program check
 use weight_lines
 implicit none
 logical macnlops,macnlops_resolved_source
 common/mode/macnlops,macnlops_resolved_source
 real(8) macnlops_isr_cutoff,macnlops_fsr_cutoff, &
 macnlops_charm_mass,macnlops_bottom_mass,macnlops_beam_xmax(2)
 common/profile/macnlops_isr_cutoff,macnlops_fsr_cutoff, &
 macnlops_charm_mass,macnlops_bottom_mass,macnlops_beam_xmax
 integer macnlops_nquark_in,macnlops_ngluon_to_quark
 common/flavours/macnlops_nquark_in,macnlops_ngluon_to_quark
 real(8),external :: macnlops_flavour_factor
 integer row,mode_i
 macnlops=.true.
 macnlops_isr_cutoff=.5d0;macnlops_fsr_cutoff=.6d0
 macnlops_charm_mass=1.2d0;macnlops_bottom_mass=4d0
 macnlops_beam_xmax=.999d0
 macnlops_nquark_in=5;macnlops_ngluon_to_quark=5
 call weight_lines_allocated(5,1,1,2)
 nFKS(1)=1;native_ids(:,1)=0;H_event(1)=.true.
 parton_pdg(:,1,1)=[2,-2,11,-11,21]
 parton_pdg(:,2,1)=[5,-5,11,-11,21]
 parton_pdg_uborn(:,1,1)=[2,-2,11,-11]
 parton_pdg_uborn(:,2,1)=[5,-5,11,-11]
 ! Ordinary source type11 already contains (1-Pbogus)*R: preserve it.
 macnlops_resolved_source=.false.
 macnlops_meta(:,1)=0d0;itype(1)=11
 do row=1,2
  if(macnlops_flavour_factor(1,row)/=1d0) stop 1
 enddo
 ! Physical proposal support is independent of source construction.
 do mode_i=0,1
  macnlops_resolved_source=mode_i==1
  itype(1)=13
  macnlops_meta(:,1)=[9d0,10000d0,.05d0,.05d0,.7d0,2d0]
  if(abs(macnlops_flavour_factor(1,1)-1d0)>1d-14) stop 2
  if(macnlops_flavour_factor(1,2)/=0d0) stop 3
 enddo
 ! Only explicit legacy resolved source transfers its below-gate real.
 macnlops_resolved_source=.true.
 macnlops_meta(6,1)=1d0;itype(1)=11
 if(macnlops_flavour_factor(1,1)/=0d0) stop 4
 if(macnlops_flavour_factor(1,2)/=1d0) stop 5
 call reset_macnlops_line(1)
 if(macnlops_flavour_factor(1,1)/=0d0) stop 6
 print *, 'PASS independent source and physical-proposal modes'
end program
real(8) function get_mass_from_id(id)
 implicit none
 integer id
 get_mass_from_id=0d0
end function
''')
            result = subprocess.run(['gfortran','-O2','-fcheck=all',
                '-ffixed-line-length-none','-I.', str(work/'context.f90'),
                str(ROOT/'Template/NLO/SubProcesses/mc_pythia8_kernel.f90'),
                str(ROOT/'Template/NLO/SubProcesses/weight_lines.f'),
                str(work/'factor.f'),str(work/'check.f90'),'-o',str(work/'check')],
                cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(work/'check')],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS independent source',result.stdout)


if __name__ == '__main__':
    unittest.main()
