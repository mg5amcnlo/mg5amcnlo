subroutine sudakov_wrapper(p)
implicit none
include 'nexternal.inc'
include "born_nhel.inc"
double precision p(0:3, nexternal-1)

integer i,j
INTEGER IHEL
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )
INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
double precision ans_summed

include 'orders.inc'
DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD

DOUBLE COMPLEX AMP_SPLIT_EWSUD_LSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
DOUBLE COMPLEX AMP_SPLIT_EWSUD_SSC(AMP_SPLIT_SIZE)
common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
DOUBLE COMPLEX AMP_SPLIT_BORN_ONEHEL(AMP_SPLIT_SIZE)
common /to_amp_born_onehel/amp_split_born_onehel

integer iflist(nexternal-1), pdglist(nexternal-1), pdglist_oth(nexternal-1)

double precision invariants(nexternal-1, nexternal-1)

double complex get_lsc_diag, get_lsc_nondiag
external get_lsc_diag, get_lsc_nondiag
double complex get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c
external get_ssc_n_diag, get_ssc_n_nondiag_1, get_ssc_n_nondiag_2, get_ssc_c

LOGICAL CALCULATEDBORN
COMMON/CCALCULATEDBORN/CALCULATEDBORN

integer ewsud_helselect
common/to_ewsud_helselect/ewsud_helselect

double precision compensate_identical_factor
external compensate_identical_factor
double precision comp_idfac

CALCULATEDBORN=.FALSE.

AMP_SPLIT_EWSUD_LSC(:) = dcmplx(0d0,0d0)
AMP_SPLIT_EWSUD_SSC(:) = dcmplx(0d0,0d0)

%(iflist)s

call sdk_get_invariants(p, iflist, invariants)

C Main loop over the helicities
do ihel=1, ncomb
C if an helicity is selected, then filter it
if (ewsud_helselect.ne.0.and.ewsud_helselect.ne.ihel) cycle

C call the born amplitude and store the result
call sborn_onehel(p,nhel(1,ihel),ihel,ans_summed)
amp_split_born_onehel(:) = amp_split_ewsud(:) 

C call the amplitudes with the goldstones for the longitudinally-polarised legs and the born for all-transverse
%(calls_to_me)s
enddo

CALCULATEDBORN=.FALSE.

return
end





integer function sdk_get_ncomb()
implicit none
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )

sdk_get_ncomb = ncomb

return
end


subroutine sdk_get_hels(ihel, hels)
implicit none
include 'nexternal.inc'
INTEGER IHEL, i
INTEGER HELS(nexternal-1)
INTEGER     NCOMB
PARAMETER ( NCOMB=  %(ncomb)d )
INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s

hels(:) = NHEL(:, ihel) 

return
end


double precision function compensate_identical_factor(igold_me)
C     provides the multiplicative factor for the result of the
C      sudakov MEs so that they have the same identical-particle
C      factor as the Born in the specific FKS configuration
implicit none
integer igold_me
C 
integer sdk_ident_goldstone(%(ngoldstone_me)d)
data sdk_ident_goldstone / %(sdk_ident_goldstone)s / 
C
integer nfksprocess
common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
integer bornspincol
data bornspincol / %(bornspincol)d /

  compensate_identical_factor = dble(sdk_ident_goldstone(igold_me))/dble(iden_values(nfksprocess)/bornspincol)

return
end

