      PROGRAM INITIALIZE
      IMPLICIT NONE

      open(unit=23,file="parameters.inc")
      call setpara('param_card.dat')  !first call to set up the parameters
      close(23)

      end

