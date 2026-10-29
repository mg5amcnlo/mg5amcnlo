"""Exact amplitude keys differ from homogeneous helicity-mask keys."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from madgraph.iolibs import born_support as support


ROOT = Path(__file__).resolve().parents[3]
MEMBERS = [('g', 'real', '1'), ('mass', 'real', '2'),
           ('width', 'real', '3'), ('loop_scale', 'real', '4'),
           ('tree_a', 'complex', '1'), ('tree_b', 'complex', '2'),
           ('loop_coupling', 'complex', '3')]
SIGNATURE = dict(g=1, real=[2, 3], complex=[],
                 couplings=[('complex', 1, 1), ('complex', 2, 2)])


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestBornAmplitudeCache(unittest.TestCase):
    def test_custom_helper_dependencies_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='mg5_helper_cache_') as tmp:
            root=Path(tmp);helpers=root/'Source/DHELAS';helpers.mkdir(parents=True)
            path=helpers/'vertex.f'
            safe=['subroutine vertex(x,y)', 'double precision x,y,z(2)',
                  'z(1)=sqrt(abs(x))', 'call helper(z,y)', 'end',
                  'subroutine helper(z,y)', 'double precision z(2),y',
                  'y=z(1)', 'end']
            support.write_fortran(path,safe)
            self.assertTrue(support.explicit_helas_dependencies(root))
            cases=[['use hidden_model', 'double precision x,y', 'y=x'],
                   ['double precision x,y,hidden', 'external hidden', 'y=hidden(x)'],
                   # Fortran does not require EXTERNAL on a typed function.
                   ['double precision x,y,hidden', 'y=hidden(x)'],
                   ['double precision x,y', 'call model_helper(x,y)'],
                   ['double precision x,y,saved_value', 'save saved_value', 'y=x*saved_value'],
                   ['double precision x,y', 'if(x.gt.0d0)read *,y']]
            for body in cases:
                with self.subTest(body=body):
                    support.write_fortran(path,['subroutine vertex(x,y)']+body+['end'])
                    self.assertFalse(support.explicit_helas_dependencies(root))
            support.write_fortran(path,safe)
            (helpers/'custom.f90').write_text('module hidden\nreal(8)::scale\nend module\n')
            self.assertFalse(support.explicit_helas_dependencies(root))

    def test_hidden_module_counterexample_uses_full_state_key(self):
        with tempfile.TemporaryDirectory(prefix='mg5_hidden_amplitude_') as tmp:
            root=Path(tmp);helpers=root/'Source/DHELAS';helpers.mkdir(parents=True)
            born=root/'SubProcesses/P0';born.mkdir(parents=True)
            support.write_fortran(born/'born.f',['call vertex(tree_a,value)'])
            support.write_fortran(helpers/'vertex.f',[
                'subroutine vertex(coupling,value)', 'use hidden_model',
                'complex(8) coupling', 'real(8) value',
                'value=real(coupling,8)*loop_scale', 'end'])
            records=[dict(directory='P0',helicity_matrices=[
                dict(homogeneous=True,couplings={'tree_a':1,'tree_b':2})])]
            signature=support.helicity_state_signature(root,records,
                dict(parameters={('external',):[]}),MEMBERS)
            self.assertIsNone(signature)
            template=ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90'
            module=root/'mc_born_types.f90';shutil.copyfile(template,module)
            support.write_amplitude_state_comparator(module,MEMBERS,signature)
            # The former permissive certification would emit this narrow key.
            bad=root/'bad_types.f90';shutil.copyfile(template,bad)
            support.write_amplitude_state_comparator(bad,MEMBERS,SIGNATURE)
            bad.write_text(bad.read_text().replace('module mc_born_types','module bad_types'))
            (root/'hidden.f90').write_text('module hidden_model\nreal(8)::loop_scale\nend module\n')
            (root/'check.f90').write_text('''program check
use mc_born_types
use bad_types,only:BadState=>BornModelState,bad_equal=>born_amplitude_state_equal
use hidden_model
implicit none
type(BornModelState) a,b
type(BadState) bad_a,bad_b
real(8) old_value,new_value
a%real_values=[2d0,173d0,1.5d0,100d0]
a%complex_values=[(2d0,1d0),(-4d0,2d0),(7d0,-1d0)]
b=a
b%real_values(4)=200d0
bad_a%real_values=a%real_values
bad_a%complex_values=a%complex_values
bad_b%real_values=b%real_values
bad_b%complex_values=b%complex_values
loop_scale=a%real_values(4)
call vertex(a%complex_values(1),old_value)
loop_scale=b%real_values(4)
call vertex(b%complex_values(1),new_value)
if(old_value.eq.new_value)stop 1
if(.not.bad_equal(bad_a,bad_b))stop 2
if(born_amplitude_state_equal(a,b))stop 3
end program
''')
            command=['gfortran','-O2','-fno-automatic','-fcheck=all',str(module),str(bad),
                     str(root/'hidden.f90'),str(helpers/'vertex.f'),str(root/'check.f90'),'-o','check']
            result=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(root/'check')],cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_generated_adapter_cache_keys_and_full_state_installation(self):
        record = dict(provider=1, nexternal=2, ngraphs=1, ncolor=1,
                      nhelicity=2, namps=2, nsqamps=3, orders=['QCD', 'QED'],
                      sectors=[dict(fks=dict(splitting_type=['QCD'])),
                               dict(fks=dict(splitting_type=['QED']))], links=[],
                      helicity_matrices=[dict(amplitude_degrees=[1])])
        with mock.patch.object(support, 'expand', return_value=[]):
            adapter = support.provider_adapter(record, Path('/unused/SubProcesses/P0'), MEMBERS)
        start = adapter.index('model_changed=.not.born_model_state_equal(state,cached_state)')
        end = adapter.index('split_type_used(1)=.true.')
        # Use the generated cache declarations and point updates as well as
        # the lookup/epoch/state-transfer block. Only the matrix evaluation
        # is omitted; each probe retains its provider's independent cache.
        declarations = '\n'.join(adapter[1:adapter.index('status=0')]).replace(
            'implicit none', 'implicit none\ninteger,parameter::amp_split_size=3,nsplitorders=2')
        block = '\n'.join(adapter[start:end])
        update_start = adapter.index('if(.not.reuse_amplitudes.or.scaled)then')
        update_end = adapter.index('points(slot)%valid=.true.')+1
        update = '\n'.join(adapter[update_start:update_end])
        routines = []
        for provider in (1, 2):
            routines.append('subroutine probe%d(context,p,state,request,hit,epoch)\n' % provider+
                            declarations+'''
integer epoch
logical hit
real(8) g,mass,width,loop_scale
complex(8) tree_a,tree_b,loop_coupling
common/installed_real/g,mass,width,loop_scale
common/installed_complex/tree_a,tree_b,loop_coupling
'''+block+'''
hit=.not.rebuild
epoch=born_model_epoch
saveamp=(0d0,0d0)
if(rebuild)then
'''+update+'''
endif
active_slot=slot
end subroutine
''')
        with tempfile.TemporaryDirectory(prefix='mg5_amplitude_adapter_') as tmp:
            work=Path(tmp);module=work/'mc_born_types.f90'
            shutil.copyfile(ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90', module)
            support.write_helicity_state_comparator(module,MEMBERS,SIGNATURE)
            support.write_amplitude_state_comparator(module,MEMBERS,SIGNATURE)
            (work/'check.f90').write_text('''program check
use mc_born_types
implicit none
type(BornModelState) a,b
type(BornRequest) request
real(8) p(0:3,1),g,mass,width,loop_scale
complex(8) tree_a,tree_b,loop_coupling
common/installed_real/g,mass,width,loop_scale
common/installed_complex/tree_a,tree_b,loop_coupling
integer epoch
logical hit
a%real_values=[2d0,173d0,1.5d0,100d0]
a%complex_values=[(2d0,1d0),(-4d0,2d0),(7d0,-1d0)]
p=1d0
request%sector=1
call probe1(1,p,a,request,hit,epoch)
if(hit.or.epoch.ne.1)stop 1
call probe1(1,p,a,request,hit,epoch)
if(.not.hit.or.epoch.ne.1)stop 2
b=a
b%real_values(4)=200d0
b%complex_values(3)=(9d0,4d0)
call probe1(1,p,b,request,hit,epoch)
if(.not.hit.or.epoch.ne.1.or.loop_scale.ne.200d0.or.loop_coupling.ne.(9d0,4d0))stop 3
call probe1(1,p,a,request,hit,epoch)
if(.not.hit.or.epoch.ne.1.or.loop_scale.ne.100d0.or.loop_coupling.ne.(7d0,-1d0))stop 4
! A second provider has independent amplitudes and installs its full state.
call probe2(1,p,b,request,hit,epoch)
if(hit.or.loop_scale.ne.200d0)stop 5
call probe1(1,p,a,request,hit,epoch)
if(.not.hit.or.loop_scale.ne.100d0.or.loop_coupling.ne.(7d0,-1d0))stop 6
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 7
request%sector=2
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 8
p(1,1)=nearest(p(1,1),1d0)
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 9
call probe1(2,p,a,request,hit,epoch)
if(.not.hit)stop 10
a%real_values(2)=nearest(a%real_values(2),1d0)
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 11
a%real_values(3)=nearest(a%real_values(3),1d0)
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 12
a%complex_values(1)=a%complex_values(1)*(1d0,0.1d0)
call probe1(2,p,a,request,hit,epoch)
if(hit)stop 13
! Common G rescaling rebuilds results from scaled amplitudes, retaining masks.
epoch=-1
a%real_values(1)=2d0*a%real_values(1)
a%complex_values(1)=2d0*a%complex_values(1)
a%complex_values(2)=4d0*a%complex_values(2)
call probe1(2,p,a,request,hit,epoch)
if(hit.or.epoch.ne.4)stop 14
end program
'''+'\n'.join(routines))
            result=subprocess.run(['gfortran','-O2','-fno-automatic','-fcheck=all',
                                   str(module),str(work/'check.f90'),'-o','check'],
                                  cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(work/'check')],cwd=work,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def check(self, body, signature):
        with tempfile.TemporaryDirectory(prefix='mg5_amplitude_cache_') as tmp:
            work = Path(tmp)
            module = work/'mc_born_types.f90'
            shutil.copyfile(ROOT/'Template/NLO/Source/BornSupport/mc_born_types.f90', module)
            support.write_helicity_state_comparator(module, MEMBERS, signature)
            support.write_amplitude_state_comparator(module, MEMBERS, signature)
            (work/'check.f90').write_text('''program check
use mc_born_types
use, intrinsic :: ieee_arithmetic
implicit none
type(BornModelState) a,b
integer i
a%real_values=[2d0,173d0,1.5d0,100d0]
a%complex_values=[(2d0,1d0),(-4d0,2d0),(7d0,-1d0)]
b=a
if(.not.born_amplitude_state_equal(a,b))stop 1
''' + body + '\nend program\n')
            result = subprocess.run(['gfortran', '-O2', '-fno-automatic', '-fcheck=all',
                                     str(module), str(work/'check.f90'), '-o', 'check'],
                                    cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            result = subprocess.run([str(work/'check')], cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_loop_state_omitted_but_tree_inputs_exact(self):
        self.check('''
b%real_values(4)=200d0
b%complex_values(3)=(8d0,3d0)
if(born_model_state_equal(a,b))stop 2
if(.not.born_amplitude_state_equal(a,b))stop 3
do i=1,3
  b=a
  b%real_values(i)=nearest(a%real_values(i),1d0)
  if(born_amplitude_state_equal(a,b))stop 4
enddo
do i=1,2
  b=a
  b%complex_values(i)=cmplx(nearest(real(a%complex_values(i),8),1d0),aimag(a%complex_values(i)),8)
  if(born_amplitude_state_equal(a,b))stop 5
  b=a
  b%complex_values(i)=cmplx(real(a%complex_values(i),8),nearest(aimag(a%complex_values(i)),1d0),8)
  if(born_amplitude_state_equal(a,b))stop 6
enddo
! A homogeneous scaling preserves masks but cannot reuse amplitudes.
b=a
b%real_values(1)=2d0*a%real_values(1)
b%complex_values(1)=2d0*a%complex_values(1)
b%complex_values(2)=4d0*a%complex_values(2)
if(.not.born_helicity_state_equal(a,b))stop 7
if(born_amplitude_state_equal(a,b))stop 8
! Cancellation/interference phases must remain part of the exact key.
b=a
b%complex_values(2)=conjg(a%complex_values(2))
if(born_amplitude_state_equal(a,b))stop 9
b=a
b%real_values(2)=ieee_value(0d0,ieee_quiet_nan)
if(born_amplitude_state_equal(a,b))stop 10
''', SIGNATURE)

    def test_unsupported_signature_keeps_full_state(self):
        self.check('''
b%real_values(4)=200d0
if(born_amplitude_state_equal(a,b))stop 2
b=a
b%complex_values(3)=(8d0,3d0)
if(born_amplitude_state_equal(a,b))stop 3
''', None)

    def test_invalid_state_dimensions_fail_closed(self):
        self.check('''
deallocate(b%real_values)
if(born_amplitude_state_equal(a,b))stop 2
allocate(b%real_values(3))
if(born_amplitude_state_equal(a,b))stop 3
b=a
deallocate(b%complex_values)
if(born_amplitude_state_equal(a,b))stop 4
allocate(b%complex_values(4))
if(born_amplitude_state_equal(a,b))stop 5
''', SIGNATURE)

    def test_uncertified_or_hidden_dependencies_keep_strict_key(self):
        model = dict(parameters={('external',): []})
        with tempfile.TemporaryDirectory(prefix='mg5_amplitude_signature_') as tmp:
            root = Path(tmp)
            (root/'Source/DHELAS').mkdir(parents=True)
            sub = root/'SubProcesses/P0'
            sub.mkdir(parents=True)
            (sub/'born.f').write_text('      call vertex(tree_a,mass,width)\n')
            records = [dict(directory='P0', helicity_matrices=[
                dict(homogeneous=True, couplings={'tree_a': 1, 'tree_b': 2})])]
            self.assertIsNotNone(support.helicity_state_signature(root, records, model, MEMBERS))
            records[0]['helicity_matrices'].append(dict(homogeneous=False, couplings={'unknown': None}))
            self.assertIsNone(support.helicity_state_signature(root, records, model, MEMBERS))
            records[0]['helicity_matrices'].pop()
            (root/'Source/DHELAS/custom.f').write_text(
                '      subroutine custom\n      common/hidden/loop_scale\n      end\n')
            self.assertIsNone(support.helicity_state_signature(root, records, model, MEMBERS))


if __name__ == '__main__':
    unittest.main()
