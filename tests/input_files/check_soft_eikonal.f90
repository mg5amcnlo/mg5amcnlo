program check_soft_eikonal
  use, intrinsic :: ieee_arithmetic
  use fks_phase_space_data
  implicit none
  real(8) :: pp(0:3,7),small(0:3),direction(0:3),scale,a,b,c,dot
  integer :: k
  external dot
  ! P0_uux_uxvedep, outer sector 4, native history 17 (i,j)=(7,1).
  small=[1.9412798495029867d-4,4.2281074989622250d-5, &
         1.0792948936341049d-4,-1.5572158027819647d-4]
  direction=[175.38639907330929d0,38.668406976089592d0, &
             98.371914059055683d0,-139.95752857989231d0]
  if(dot(small,direction).ne.0d0)error stop 'fixture does not trigger legacy dot cutoff'
  resonance_recoil=.false.
  do k=-3,3,3
    scale=10d0**k
    pp=0d0
    pp(:,1)=scale*[direction(0),0d0,0d0,direction(0)]
    pp(:,5)=scale*small
    pp(:,6)=scale*[direction(0),0d0,0d0,-direction(0)]
    p_i_fks_cnt(:,0)=scale*direction
    sqrtshat=2d0*scale*direction(0)
    call eikonal_reduced(pp,5,1,7,1,0d0,-0.79799533669307987d0,a)
    call eikonal_reduced(pp,1,5,7,1,0d0,-0.79799533669307987d0,b)
    call eikonal_reduced(pp,5,6,7,1,0d0,-0.79799533669307987d0,c)
    if(.not.all(ieee_is_finite([a,b,c])))error stop 'nonfinite eikonal'
    write(*,'(a,4es26.17e3)') 'RESULT',scale,a,b,c
  enddo
end program
