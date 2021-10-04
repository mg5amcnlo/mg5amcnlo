double precision function get_rescale_alpha_factor(ntag,qed_pow)
C returns the power of ratios of alpha needed to compensate 
C for the presence of external tagged photons
C It is automatically written, and it detects whether the
C starting model is in the Gmu or alpha0 scheme.
C ntag is the number of tagged photons, qed_pow the power of gweak
C of the contribution considered
implicit none
integer ntag, qed_pow
INCLUDE '../../Source/MODEL/input.inc'

get_rescale_alpha_factor = 1d0
if (ntag.eq.0) return

get_rescale_alpha_factor = %(rescale_fact)s

return
end


double precision function get_virtual_a0Gmu_conv(qed_pow, ntagph, ivirt, born_wgt)
C returns the piece to compensate the a0<>Gmu change of scheme for the
C virtuals (single pole if ivirt = 1, finite if ivirt = 0
implicit none
integer qed_pow, ntagph, ivirt
double precision born_wgt
INCLUDE '../../Source/MODEL/input.inc'

if (ivirt.eq.1) then
! single 
get_virtual_a0Gmu_conv = %(virtual_a0Gmu_conv_pole)s
else if (ivirt.eq.0) then
! finite part
get_virtual_a0Gmu_conv = %(virtual_a0Gmu_conv_finite)s
else
write(*,*) 'Error get_virtual_a0Gmu_conv: Invalid ivirt', ivirt
endif

return
end
