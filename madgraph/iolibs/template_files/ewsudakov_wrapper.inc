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

LOGICAL CALCULATEDBORN
COMMON/CCALCULATEDBORN/CALCULATEDBORN

CALCULATEDBORN=.FALSE.

C Main loop over the helicities
do ihel=1, ncomb
C call the amplitudes with the goldstones for the longitudinally-polarised legs
%(calls_to_goldstones)s
else
   call sborn_onehel(p,nhel(1,ihel),ihel,ans_summed)
endif

enddo

CALCULATEDBORN=.FALSE.

return
end
