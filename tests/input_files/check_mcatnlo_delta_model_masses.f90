! Link the actual model-generated mass getter to the Delta scale module.
program check_mcatnlo_delta_model_masses
  use mcatnlo_delta_scales
  implicit none
  real(dp), external :: get_mass_from_id
  real(dp) :: pr(0:3),pe(0:3),pk(0:3),mc,mb,tc,tb
  integer :: status

  mc=get_mass_from_id(4)
  mb=get_mass_from_id(5)
  pr=(/500._dp,0._dp,0._dp,500._dp/)
  pe=(/1._dp,0.6_dp,0._dp,0.8_dp/)
  pk=(/500._dp,0._dp,0._dp,-500._dp/)
  call pythia_pt_lund(pr,pe,pk,21,4,.false.,.false.,mc,mb,tc,status)
  if (status /= delta_ok) stop 1
  call pythia_pt_lund(pr,pe,pk,21,5,.false.,.false.,mc,mb,tb,status)
  if (status /= delta_ok) stop 2
  print '(6ES25.16)', mc,get_mass_from_id(-4),mb,get_mass_from_id(-5),tc,tb
end program check_mcatnlo_delta_model_masses
