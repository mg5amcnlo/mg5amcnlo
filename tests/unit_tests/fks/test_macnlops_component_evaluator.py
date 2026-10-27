"""Small compiled fixtures for component measures and Born provenance.

No process export, PDF call, event generation or shower is run here.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(shutil.which('gfortran'), 'gfortran is required')
class TestMacNLOPSComponentEvaluator(unittest.TestCase):
    def test_native_born_provenance_and_individual_diagram_measure(self):
        driver = (ROOT / 'Template/NLO/SubProcesses/driver_macnlops.f').read_text()
        helpers = driver[driver.index('      subroutine macnlops_validate_component_born'):
                         driver.index('      double precision function macnlops_flavour_sum')]
        source = (ROOT / 'Template/NLO/SubProcesses/fks_singular.f').read_text()
        flow = source[source.index('      subroutine include_born_flow_weight'):
                      source.index('      subroutine include_native_recoil_weights')]
        with tempfile.TemporaryDirectory(prefix='macnlops-component-evaluator-') as temporary:
            work = Path(temporary)
            includes = {
                'nexternal.inc': 'integer,parameter :: nexternal=5,nincoming=2',
                'genps.inc': 'integer,parameter :: maxproc=2,maxflow=2,ngraphs=2,ncolor=2,lmaxconfigs=2',
                'born_nhel.inc': 'integer,parameter :: max_bcol=2',
                'nFKSconfigs.inc': 'integer,parameter :: fks_configs=7',
                'fks_symmetry.inc': '''integer FKS_FAC_I_D(7),FKS_FAC_J_D(7)
double precision FKS_IDEN_BORN_D(7),FKS_IDEN_REAL_D(7)
data FKS_FAC_I_D /2,3,2,2,1,1,1/
data FKS_FAC_J_D /1,2,1,2,1,1,1/
data FKS_IDEN_BORN_D /1d0,2d0,1d0,2d0,1d0,1d0,0d0/
data FKS_IDEN_REAL_D /2d0,6d0,2d0,4d0,1d0,2d0,1d0/''',
                'run.inc': 'double precision ebeam(2)\ncommon/beams/ebeam',
                'born_leshouche.inc': '''integer iproc_born
common/native_born_count/iproc_born
common/native_born_lh/idup,mothup,icolup''',
            }
            for name, text in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n' for line in text.splitlines()))
            (work / 'modules.f90').write_text('''module fks_phase_space_data
 implicit none
 real(8) p_born(0:3,4),xbjrk_cnt(2,0:2),p1_cnt(0:3,5,0:2)
end module
module weight_lines
 logical :: mc_H_only=.true.
end module
''')
            (work / 'helpers.f').write_text(helpers + flow)
            (work / 'check.f90').write_text('''program check
 use fks_phase_space_data
 use weight_lines
 implicit none
 integer idup(4,2),mothup(2,4,2),icolup(2,4,2),iproc_born
 common/native_born_count/iproc_born
 common/native_born_lh/idup,mothup,icolup
 real(8) ebeam(2)
 common/beams/ebeam
 real(8) factors(9),mc_factors(8),pdf_factors(3),dsym
 common/factor_n1body/factors
 common/factor_n1body_NLOPS/mc_factors
 common/factor_pdfsch/pdf_factors
 common/dsymfactor/dsym
 integer current_cms,calls,modes(3)
 real(8) current_alpha
 common/trace/current_cms,calls,modes,current_alpha
 integer ids(4),colours(2,4),status,i,j,emitters
 integer ring(2,3),opposite(2,3)
 real(8) lab(0:3,4),saved_lab(0:3,4),masses(4),p(0:3,5),boost
 real(8) normalization,expected(5),before,after,r,d,k,projected_x(2)
 integer expected_emitters(5)
 logical macnlops_same_colour_topology
 external macnlops_same_colour_topology
 p_born(:,1)=[5d0,0d0,0d0,5d0]
 p_born(:,2)=[5d0,0d0,0d0,-5d0]
 p_born(:,3)=[5d0,5d0,0d0,0d0]
 p_born(:,4)=[5d0,-5d0,0d0,0d0]
 idup(:,1)=[21,21,2,-2];idup(:,2)=[21,21,1,-1]
 iproc_born=2;mothup=0
 icolup(:,:,1)=reshape([501,502,502,503,501,0,0,503],[2,4])
 icolup(:,:,2)=reshape([503,502,501,503,501,0,0,502],[2,4])
 ids=idup(:,1);colours=icolup(:,:,1)
 where(colours/=0)colours=colours-300
 masses=0d0;ebeam=10d0;boost=.4d0
 do i=1,4
  lab(0,i)=cosh(boost)*p_born(0,i)+sinh(boost)*p_born(3,i)
  lab(1:2,i)=p_born(1:2,i)
  lab(3,i)=sinh(boost)*p_born(0,i)+cosh(boost)*p_born(3,i)
 enddo
 ! The massive-FSR second solution has no counterevents.
 xbjrk_cnt=0d0;projected_x=lab(0,1:2)/ebeam
 saved_lab=lab
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=0)stop 1
 ! Same undirected neighbours do not identify an oriented gluon flow.
 ring=reshape([1,2,2,3,3,1],[2,3])
 opposite=reshape([2,1,3,2,1,3],[2,3])
 if(macnlops_same_colour_topology(3,ring,opposite))stop 2
 colours=icolup(:,:,2)
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=36)stop 3
 colours=icolup(:,:,1);ids(3)=-2;ids(4)=2
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=36)stop 4
 ids=idup(:,1);masses(3)=1d0
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=37)stop 5
 masses=0d0
 ! A different polar Born geometry has the same external masses but
 ! different ordered Gram invariants with the incoming beams.
 lab(1,3)=sqrt(24d0);lab(3,3)=1d0;lab(0,3)=5d0
 lab(1,4)=-sqrt(24d0);lab(3,4)=-1d0;lab(0,4)=5d0
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=38)stop 6
 lab=saved_lab;projected_x(1)=projected_x(1)*1.01d0
 call macnlops_validate_component_born(1,ids,colours,lab,masses,projected_x,status)
 if(status/=38)stop 7
 ! Physical real and Born diagram partitions are distinct. Their
 ! source symmetry factor multiplies every contribution once.
 factors=1d0;mc_factors=1d0;pdf_factors=1d0
 p=101d0;p1_cnt=20d0;dsym=7d0;calls=0
 call macnlops_include_diagram_weights(p,2d0)
 if(.not.mc_H_only.or.dsym/=7d0)stop 8
 if(calls/=3.or.any(modes/=[4,3,2]))stop 9
 call include_born_flow_weight(.6d0,1d0)
 if(abs(factors(1)-.3d0)>1d-14)stop 10
 if(abs(factors(2)-.48d0)>1d-14)stop 11
 ! MC already has p_c in its Born; include_born_flow_weight must not
 ! multiply that fraction again. A scalar leading kernel will supply it.
 if(abs(mc_factors(8)-.8d0)>1d-14)stop 12
 if(abs(100d0*factors(1)-40d0*mc_factors(8)*.6d0 &
        -20d0*factors(2)-1.2d0)>1d-13)stop 13
 ! One/two Born gluons, and g->qq with other identical q or qbar,
 ! have different real/Born statistical ratios but integer emitter
 ! orbits. These factors convert all densities to the actual-emitter
 ! measure; a common positive conversion preserves both probabilities.
 expected=[2d0,3d0,2d0,2d0,1d0]
 expected_emitters=[1,2,1,2,1]
 do i=1,5
  call macnlops_component_normalization(i,normalization,emitters,status)
  if(status/=0.or.normalization/=expected(i).or. &
     emitters/=expected_emitters(i))stop 14
  r=3d0;d=5d0;k=4d0
  before=1d0-(d-r)/k
  r=r*normalization;d=d*normalization;k=k*normalization
  after=1d0-(d-r)/k
  if(abs(after-before)>1d-15)stop 15
 enddo
 call macnlops_component_normalization(6,normalization,emitters,status)
 if(status/=39)stop 16
 call macnlops_component_normalization(7,normalization,emitters,status)
 if(status/=39)stop 17
 print *, 'PASS component Born provenance and physical diagram measure'
end program

subroutine set_cms_stuff(cms)
 implicit none
 integer cms,current_cms,calls,modes(3)
 real(8) current_alpha
 common/trace/current_cms,calls,modes,current_alpha
 current_cms=cms
end subroutine

subroutine set_alphaS(p)
 implicit none
 real(8) p(0:3,5),current_alpha
 integer current_cms,calls,modes(3)
 common/trace/current_cms,calls,modes,current_alpha
 current_alpha=p(0,1)
end subroutine

subroutine include_multichannel_enhance(mode)
 use weight_lines
 implicit none
 integer mode,current_cms,calls,modes(3)
 real(8) current_alpha,factors(9),mc_factors(8),pdf_factors(3),dsym
 common/trace/current_cms,calls,modes,current_alpha
 common/factor_n1body/factors
 common/factor_n1body_NLOPS/mc_factors
 common/factor_pdfsch/pdf_factors
 common/dsymfactor/dsym
 if(mc_H_only)stop 21
 calls=calls+1;modes(calls)=mode
 if(mode==4)then
  if(current_cms/=-100.or.current_alpha/=101d0)stop 22
  mc_factors(7:8)=mc_factors(7:8)*.4d0*dsym
 elseif(mode==3)then
  if(current_cms/=0.or.current_alpha/=20d0)stop 23
  factors(2:9)=factors(2:9)*.4d0*dsym
 elseif(mode==2)then
  if(current_cms/=-100.or.current_alpha/=101d0)stop 24
  factors(1)=factors(1)*.25d0*dsym
 else
  stop 25
 endif
end subroutine

real(8) function get_mass_from_id(id)
 implicit none
 integer id
 get_mass_from_id=0d0
end function
''')
            result = subprocess.run(['gfortran', '-O0', '-fcheck=all',
                '-ffixed-line-length-none', '-I.', str(work / 'modules.f90'),
                str(work / 'helpers.f'), str(work / 'check.f90'),
                '-o', str(work / 'check')], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS component Born provenance', result.stdout)


if __name__ == '__main__':
    unittest.main()
