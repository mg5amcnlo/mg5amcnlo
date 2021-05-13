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

