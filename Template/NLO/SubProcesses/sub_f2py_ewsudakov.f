      subroutine ewsudakov_py(p_born_in, nexternal, gstr_in, results) 
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      implicit none
CF2PY double precision, intent(in), dimension(0:3,nexternal) :: p_born_in
CF2PY integer, intent(in) :: nexternal
CF2PY double precision, intent(in) :: gstr_in
CF2PY double precision, intent(out) :: results(6)

C arguments

C      include 'nexternal.inc'
      integer nexternal
      double precision p_born_in(0:3,nexternal)
      double precision gstr_in, results(6)
      ! results contain (born, sud0, sud1)
      call ewsudakov_f77(p_born_in, gstr_in, results)
      return 
      end
