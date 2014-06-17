subroutine extra_cnt(p, icnt, cnts)
C call the extra cnt corresponding to icnt
C may be a dummy function, depending on the process
implicit none
integer icnt, i
include 'nexternal.inc'
double precision p(0:3, nexternal)
include 'orders.inc'
double complex cnts(2,nsplitorders)

do i = 1, nsplitorders
cnts(1,i) = dcmplx(0d0,0d0)
cnts(2,i) = dcmplx(0d0,0d0)
enddo

%(iflines)s
return
end

