program check_mc_momentum_permutation
  use, intrinsic :: ieee_arithmetic
  use fks_phase_space_helpers, only: apply_momentum_permutation
  implicit none
  character(16) :: mode
  double precision :: p(0:3,6),out(0:3,6),saved(0:3,6),invalid(3)
  integer :: perm(6),i
  logical :: valid
  call get_command_argument(1,mode)
  p(:,1)=[200d0,0d0,0d0,200d0]
  p(:,2)=[200d0,0d0,0d0,-200d0]
  p(:,3)=[100d0,100d0,0d0,0d0]
  p(:,4)=[100d0,-100d0,0d0,0d0]
  p(:,5)=[100d0,0d0,100d0,0d0]
  p(:,6)=[100d0,0d0,-100d0,0d0]
  saved=p
  perm=[1,2,3,6,5,4]
  select case(trim(mode))
  case('valid')
    call apply_momentum_permutation(perm,p,out)
    if(any(out.ne.p(:,perm)))error stop 'changed a permuted component'
    p(0,[4,6])=sqrt(100d0**2+25d0**2)
    call apply_momentum_permutation(perm,p,out,valid)
    if(.not.valid.or.any(out.ne.p(:,perm)))error stop 'rejected equal massive legs'
    p(0,4)=p(0,4)+1d-12
    call apply_momentum_permutation(perm,p,out,valid)
    if(.not.valid)error stop 'rejected harmless roundoff'
  case('mass','strict')
    p(0,4)=p(0,4)+1d-4
    if(trim(mode).eq.'strict')then
      call apply_momentum_permutation(perm,p,out)
      error stop 'missing strict mass check'
    endif
    call apply_momentum_permutation(perm,p,out,valid)
    if(valid)error stop 'accepted inconsistent mass shells'
  case('w2j_sumkl')
    ! P0_gu_vedep/GF2.0_3, seed 33, offset 3: stopped after 4089 events.
    ! Identical massless quarks 4 and 6 are exchanged. Leg 6 has
    ! m^2=-1.00125e-5 GeV^2, above the 7.21611e-6 GeV^2 tolerance.
    p(:,1)=[2.68628172949970690d2,0d0,0d0,2.68628172949970690d2]
    p(:,2)=[6.73723013178982910d1,0d0,0d0,-6.73723013178982910d1]
    p(:,3)=[1.46427972774550952d0,1.95364886229746487d-1, &
         -2.34308025648977264d-1,-1.43214783853880556d0]
    p(:,4)=[5.46820131095570474d1,6.55180317420027603d0, &
         -8.24529508079451467d0,-5.36582849326261453d1]
    p(:,5)=[1.17014504522078600d1,1.37466156152688734d0, &
         -1.74804450197183203d0,-1.14881934478778529d1]
    p(:,6)=[2.68152730969034280d2,-8.12182962195691083d0, &
         1.02276476084153245d1,2.67834497860439512d2]
    call apply_momentum_permutation(perm,p,out,valid)
    if(valid)error stop 'accepted the failing w2j history permutation'
  case('nonfinite')
    invalid=[ieee_value(0d0,ieee_quiet_nan), &
         ieee_value(0d0,ieee_positive_inf),ieee_value(0d0,ieee_negative_inf)]
    do i=1,3
      p=saved
      p(1,4)=invalid(i)
      call apply_momentum_permutation(perm,p,out,valid)
      if(valid)error stop 'accepted nonfinite momenta'
    enddo
  case('range','duplicate','incoming')
    select case(trim(mode))
    case('range')
      perm(4)=7
    case('duplicate')
      perm(4)=4
    case('incoming')
      perm(1:2)=[2,1]
    end select
    call apply_momentum_permutation(perm,p,out,valid)
    error stop 'broken integer map was not fatal'
  case default
    error stop 'unknown permutation fixture'
  end select
  ! A rejected point cannot poison the following valid call.
  call apply_momentum_permutation(perm,saved,out,valid)
  if(.not.valid.or.any(out.ne.saved(:,perm)))error stop 'failed after recovery'
  write(*,*) 'PASS '//trim(mode)
end program
