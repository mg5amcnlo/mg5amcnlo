      PROGRAM INITIALIZE
      IMPLICIT NONE

      open(unit=23,file="parameters.inc")
      call setpara('param_card.dat')  !first call to setup the paramaters
      close(23)

      end

