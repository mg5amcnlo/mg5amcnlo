"""Native providers must evaluate each context in its own requested leg frame."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from madgraph.iolibs import born_support
from tests.unit_tests.fks.test_momentum_maps import ROOT, TEMPLATE, fortran_routine


@unittest.skipUnless(shutil.which('gfortran'), 'requires gfortran')
class TestNativeFrameMasks(unittest.TestCase):
    def test_frame_wrappers_scope_the_helicity_basis_even_at_rest(self):
        with tempfile.TemporaryDirectory(prefix='mg7_native_frame_scope_') as tmp:
            work = Path(tmp)
            bridge = work / 'boost_to_frame.f'
            shutil.copyfile(TEMPLATE / bridge.name, bridge)
            born_support.write_native_frame_bridge(bridge)
            wrappers = ('sborn_frame', 'sborn_onehel_frame', 'sborn_sf_frame',
                        'extra_cnt_frame', 'binothlha_frame')
            (work / 'wrappers.f').write_text('\n'.join(
                fortran_routine(bridge, name) for name in wrappers))
            for name, source in {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'orders.inc': 'integer nsplitorders\nparameter(nsplitorders=2)',
            }.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                                for line in source.splitlines()))
            (work / 'scope.f90').write_text(SCOPE_DRIVER)
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-ffree-line-length-none',
                str(TEMPLATE.parent / 'Source/BornSupport/mc_born_types.f90'),
                'scope.f90', 'wrappers.f', '-o', 'check'], cwd=work,
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS scoped native helicity basis', result.stdout)

    def test_context_changes_invalidate_born_and_real_frame_masks(self):
        with tempfile.TemporaryDirectory(prefix='mg7_native_frames_') as tmp:
            work = Path(tmp)
            bridge = work / 'boost_to_frame.f'
            shutil.copyfile(TEMPLATE / bridge.name, bridge)
            # Exercise the same native-only transformation as export finalization.
            born_support.write_native_frame_bridge(bridge)
            (work / 'masks.f').write_text('\n'.join(
                fortran_routine(bridge, name) for name in
                ('get_frame_mask_born', 'get_frame_mask_real')))
            includes = {
                'nexternal.inc': 'integer nexternal,nincoming\n'
                                 'parameter(nexternal=5,nincoming=2)',
                'nFKSconfigs.inc': 'integer fks_configs\nparameter(fks_configs=4)',
                'run.inc': 'integer frame_id\ncommon/to_frame_me/frame_id',
                'frame_info.inc': 'integer frame_map_born(4)\n'
                                  'data frame_map_born/1,2,3,4/',
                'fks_info.inc': 'integer fks_i_d(4)\ndata fks_i_d/5,5,4,4/',
            }
            for name, source in includes.items():
                (work / name).write_text(''.join('      ' + line + '\n'
                                                for line in source.splitlines()))
            (work / 'check.f90').write_text(DRIVER)
            result = subprocess.run([
                shutil.which('gfortran'), '-O2', '-fcheck=all',
                '-ffixed-line-length-none', '-ffree-line-length-none',
                str(TEMPLATE.parent / 'Source/BornSupport/mc_born_types.f90'),
                'check.f90', 'masks.f', '-o', 'check'],
                cwd=work, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(work / 'check')], cwd=work,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS native context frame masks', result.stdout)


DRIVER = '''module mc_native_context
use mc_born_types, only: BornMetadata
implicit none
type(BornMetadata), target :: contexts(3)
type(BornMetadata), pointer :: native_metadata => null()
integer :: native_epoch=0
end module

program check_native_frames
use mc_native_context
implicit none
integer frame_id,born(4),real(5)
common/to_frame_me/frame_id
frame_id=8
! Before activation the static owner map remains available.
call get_frame_mask_born(born)
if(any(born /= [0,0,1,0])) error stop 'static Born map'
call get_frame_mask_real(1,real)
if(any(real /= [0,0,1,0,0])) error stop 'static real map'

allocate(contexts(1)%frame_map(4),contexts(2)%frame_map(4))
contexts(1)%frame_map=[1,2,4,3]
contexts(2)%frame_map=[1,2,3,4]
native_metadata => contexts(1)
native_epoch=1
! Same frame_id and sector, but another context's user leg is selected.
! Calling the real mask first also tests its own cache invalidation.
call get_frame_mask_real(1,real)
if(any(real /= [0,0,0,1,0])) error stop 'first native real map'
call get_frame_mask_born(born)
if(any(born /= [0,0,0,1])) error stop 'first native Born map'

native_metadata => contexts(2)
native_epoch=2
call get_frame_mask_real(1,real)
if(any(real /= [0,0,1,0,0])) error stop 'second native real map'
call get_frame_mask_born(born)
if(any(born /= [0,0,1,0])) error stop 'second native Born map'
! An auxiliary native alias can insert the emitted leg in another slot.
frame_id=16
call get_frame_mask_real(3,real)
if(any(real /= [0,0,0,0,1])) error stop 'alias insertion map'

native_metadata => contexts(1)
native_epoch=3
call get_frame_mask_real(3,real)
if(any(real /= [0,0,1,0,0])) error stop 'alias context cache'
if(any(contexts(1)%frame_map /= [1,2,4,3])) error stop 'mutated context map'
! Returning to the static owner must invalidate the native masks too.
nullify(native_metadata)
native_epoch=4
call get_frame_mask_real(1,real)
if(any(real /= [0,0,0,1,0])) error stop 'static fallback after native'
print *, 'PASS native context frame masks'
end program
'''


SCOPE_DRIVER = '''module mc_native_context
implicit none
integer :: active_context=0,ensure_count=0
contains
subroutine ensure_native_context()
ensure_count=ensure_count+1
active_context=42
end subroutine
end module

module bridge_state
implicit none
integer :: selected_ids(4)=0,expected_key(2)=0,calls=0
logical :: move_momenta=.false.
end module

program check_scope
use mc_born_types,only:born_active_helicity_frame
use mc_native_context
use bridge_state
implicit none
real(8) :: p(0:3,4),ans,virt
complex(8) :: cnts(2,2)
integer phase
p=2d0
! Preserve an outer scope, including when the selected frame is already
! at rest and the boost returns the identical momenta.
do phase=1,3
  born_active_helicity_frame=[77,88]
  selected_ids=[0,0,1,0]
  expected_key=[42,8]
  move_momenta=phase.eq.1
  if(phase.eq.3)then
    selected_ids=0
    expected_key=[77,88]
  endif
  call sborn_frame(p,ans)
  call assert_restored()
  call sborn_onehel_frame(p,1,2,ans)
  call assert_restored()
  call sborn_sf_frame(p,1,2,ans)
  call assert_restored()
  call extra_cnt_frame(p,1,cnts)
  call assert_restored()
  call binothlha_frame(p,1d0,virt)
  call assert_restored()
enddo
if(calls.ne.15.or.ensure_count.ne.15)error stop 'missing scoped call'
if(any(p.ne.2d0))error stop 'mutated input momenta'
print *, 'PASS scoped native helicity basis'
contains
subroutine assert_restored()
if(any(born_active_helicity_frame.ne.[77,88]))error stop 'scope leaked'
end subroutine
end program

subroutine get_frame_mask_born(ids)
use bridge_state
integer ids(4)
ids=selected_ids
end subroutine

subroutine boost_to_me_frame(p,n,ids,pf)
use bridge_state
integer n,ids(n)
real(8) p(0:3,n),pf(0:3,n)
pf=p
if(move_momenta)pf(0,1)=p(0,1)+1d0
end subroutine

subroutine check_active_key()
use mc_born_types,only:born_active_helicity_frame
use bridge_state
if(any(born_active_helicity_frame.ne.expected_key))error stop 'wrong basis key'
calls=calls+1
end subroutine

subroutine sborn_native(p,ans)
real(8) p(0:3,4),ans
call check_active_key()
ans=1d0
end subroutine

subroutine sborn_onehel(p,hel1,chosen_hel,ans)
real(8) p(0:3,4),ans
integer hel1,chosen_hel
call check_active_key()
ans=1d0
end subroutine

subroutine sborn_sf_native(p,m,n,ans)
real(8) p(0:3,4),ans
integer m,n
call check_active_key()
ans=1d0
end subroutine

subroutine extra_cnt_native(p,icnt,cnts)
real(8) p(0:3,4)
integer icnt
complex(8) cnts(2,2)
call check_active_key()
cnts=1d0
end subroutine

subroutine binothlha(p,born,virt)
real(8) p(0:3,4),born,virt
call check_active_key()
virt=1d0
end subroutine
'''
