subroutine macnlops_diag_born(out) bind(C)
use iso_c_binding
use fks_phase_space_data, only: p_born
implicit none
include 'nexternal.inc'
real(c_double), intent(out) :: out(4*(nexternal-1))
integer i,j
do i=1,nexternal-1
 do j=0,3
  out(4*(i-1)+j+1)=p_born(j,i)
 enddo
enddo
end subroutine
