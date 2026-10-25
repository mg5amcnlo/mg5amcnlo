"""Generating-history metadata must survive weight-line storage changes."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMacNLOPSWeightLines(unittest.TestCase):
    def test_sampled_component_identity_survives_growth_copy_and_point_reset(self):
        with tempfile.TemporaryDirectory(prefix='macnlops-component-lines-') as temporary:
            work = Path(temporary)
            (work / 'check.f90').write_text('''program check
 use weight_lines
 implicit none
 integer first,i
 call begin_component_context(2,3,7,4,.25d0,.3d0,2d0)
 first=component_context_active
 call weight_lines_allocated(5,1,1,1)
 call reset_macnlops_line(1)
 call end_component_context(.4d0)
 call weight_lines_allocated(5,40,1,1)
 call copy_macnlops_line(1,2)
 if(.not.same_source_component(1,2))stop 1
 if(component_line_context(1)/=first)stop 2
 do i=1,20
   call begin_component_context(2,3,7,4,.25d0,.3d0,2d0)
   call reset_macnlops_line(3)
   call end_component_context(.4d0)
 enddo
 if(same_source_component(1,3))stop 3
 if(component_contexts(first)%flow_probability/=.25d0)stop 4
 if(component_contexts(first)%source_factor/=.4d0)stop 5
 if(component_contexts(first)%insertion/=.3d0)stop 6
 if(component_contexts(first)%config/=3.or.component_contexts(first)%graph/=7)stop 10
 if(component_contexts(first)%diagram_symmetry/=2d0)stop 11
 call reset_macnlops_line(3)
 if(component_line_context(3)/=0)stop 7
 call deallocate_weight_lines
 if(allocated(component_line_context).or.allocated(component_contexts))stop 8
 if(component_context_active/=0.or.component_context_count/=0)stop 9
 print *, 'PASS sampled component identity lifecycle'
end program
''')
            result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
                '-ffixed-line-length-none',
                str(ROOT / 'Template/NLO/SubProcesses/weight_lines.f'),
                str(work / 'check.f90'), '-o', str(work / 'check')],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS sampled component identity lifecycle', result.stdout)

    def test_partial_native_flavour_overlap_is_not_a_global_group(self):
        with tempfile.TemporaryDirectory(prefix='macnlops-group-coverage-') as temporary:
            work=Path(temporary)
            (work/'check.f90').write_text('''program check
 use weight_lines
 implicit none
 logical compatible(3)
 call weight_lines_allocated(5,3,1,2)
 icontr=3;H_event=.true.;niproc(1:3)=[2,2,1]
 group_owner(1:3)=[1,1,3]
 parton_pdg(:,1,1)=[2,-2,11,-11,21]
 parton_pdg(:,2,1)=[4,-4,11,-11,21]
 parton_pdg(:,1,2)=parton_pdg(:,2,1)
 parton_pdg(:,2,2)=parton_pdg(:,1,1)
 parton_pdg(:,1,3)=parton_pdg(:,1,1)
 compatible=.true.
 ! Third table partially overlaps: its u-row is outside selected group.
 if(group_flavour_complete(1,1,compatible)) stop 1
 if(.not.group_flavour_complete(1,2,compatible)) stop 2
 ! An unrelated colour/scale/momentum component is not this partition.
 compatible(3)=.false.
 if(.not.group_flavour_complete(1,1,compatible)) stop 3
 ! Reordered rows in the same group are complete without index equality.
 if(.not.group_flavour_complete(1,2,compatible)) stop 4
 print *, 'PASS physical-flavour group coverage'
end program
''')
            result=subprocess.run(['gfortran','-O2','-fcheck=all',
                '-ffixed-line-length-132',
                str(ROOT/'Template/NLO/SubProcesses/weight_lines.f'),
                str(work/'check.f90'),'-o',str(work/'check')],
                cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(work/'check')],cwd=work,
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_gate_history_survives_resize_copy_and_reuse(self):
        with tempfile.TemporaryDirectory(prefix='macnlops-weight-lines-') as temporary:
            work = Path(temporary)
            driver = work/'check.f90'
            driver.write_text('''program check
  use weight_lines
  implicit none
  integer :: old_capacity
  double precision :: expected(6)
  expected=[9d0,1000d0,.2d0,.3d0,.7d0,1d0]
  ! First add_wgt can allocate storage inside an active native gate.
  macnlops_gate_active=.true.
  macnlops_proposal_active=.true.
  macnlops_current_z=.7d0
  call weight_lines_allocated(5,1,1,1)
  if (.not.macnlops_gate_active.or.macnlops_current_z/=.7d0) stop 1
  if (.not.macnlops_proposal_active) stop 13
  if (any(macnlops_meta/=0d0)) stop 2
  macnlops_meta(:,1)=expected
  ! Grow subprocess, scale-weight and contribution capacities together.
  call weight_lines_allocated(5,2,3,4)
  if (any(macnlops_meta(:,1)/=expected)) stop 3
  if (any(macnlops_meta(:,2:)/=0d0)) stop 4
  call copy_macnlops_line(1,2)
  call reset_macnlops_line(1)
  if (any(macnlops_meta(:,1)/=0d0)) stop 5
  if (any(macnlops_meta(:,2)/=expected)) stop 6
  old_capacity=max_contr
  macnlops_meta(:,old_capacity)=2d0*expected
  call weight_lines_allocated(5,old_capacity+1,max_wgt,max_iproc)
  if (any(macnlops_meta(:,old_capacity)/=2d0*expected)) stop 7
  if (any(macnlops_meta(:,old_capacity+1:)/=0d0)) stop 8
  call deallocate_weight_lines
  if (allocated(macnlops_meta)) stop 9
  if (macnlops_gate_active.or.macnlops_current_z/=0d0) stop 10
  if (macnlops_proposal_active) stop 14
  call weight_lines_allocated(5,1,1,1)
  if (any(shape(macnlops_meta)/=[6,1])) stop 11
  if (any(macnlops_meta/=0d0)) stop 12
  call deallocate_weight_lines
  print *, 'PASS gate-history storage lifecycle'
end program check
''')
            executable = work/'check'
            compile_result = subprocess.run([
                'gfortran','-O2','-fcheck=all','-ffixed-line-length-132',
                str(ROOT/'Template/NLO/SubProcesses/weight_lines.f'),
                str(driver),'-o',str(executable)],cwd=work,
                capture_output=True,text=True)
            self.assertEqual(compile_result.returncode,0,
                             compile_result.stdout+compile_result.stderr)
            result = subprocess.run([str(executable)],cwd=work,
                                    capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS gate-history storage lifecycle',result.stdout)


if __name__ == '__main__':
    unittest.main()
