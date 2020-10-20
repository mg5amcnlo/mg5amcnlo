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

integer iflist(nexternal-1), pdglist(nexternal-1)

double precision invariants(nexternal-1, nexternal-1)

double precision get_lsc_diag, get_lsc_nondiag

LOGICAL CALCULATEDBORN
COMMON/CCALCULATEDBORN/CALCULATEDBORN

CALCULATEDBORN=.FALSE.

AMP_SPLIT_EWSUD_LSC(:) = 0d0

%(iflist)s

call sdk_get_invariants(p, iflist, invariants)

C Main loop over the helicities
do ihel=1, ncomb
C call the amplitudes with the goldstones for the longitudinally-polarised legs
%(calls_to_me)s
enddo

CALCULATEDBORN=.FALSE.

return
end





