      subroutine qlratreal(si,ta,rat,ieps)
*****comment:************************************************************
*									*
*	Calculate the function                                          *
*									*
*	sigma-i*ep                                                      *
* rat = ----------                                                      *
*       tau-i*ep                                                        *
*									*
*	where sigma and tau are real and ieps gives the sign if i*pi    *
*									*
***comment:*************************************************************
      implicit none
      include 'qlconstants.f' 
      double precision si,ta,ieps,rat
      rat=si/ta
      if (rat .gt. zip) then
      ieps=0d0
      return
      elseif (si .lt. zip) then
      ieps=-1d0 
      return
      elseif (ta .lt. zip) then
      ieps=+1d0 
      return
      elseif (ta .eq. zip) then
      write(6,*) 'error in qlratreal, ta=',ta
      stop
      endif
      end
