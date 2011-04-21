c
c ----------------------------------------------------------------------
c
      subroutine ficxxx(fi , fic)
c
c this subroutine charge conjugates a flowing-in fermion wavefunction.  
c                                                                       
c input:                                                                
c       complex fi(6)          : flowing-in fermion                 |fi>
c                                                                       
c output:                                                               
c       complex fic(6)         : charge conjugated fermion         <fic|
c
      implicit none
      double complex fi(6), fic(6)
c
      fic(1) = -fi(2)
      fic(2) =  fi(1)
      fic(3) =  fi(4)
      fic(4) = -fi(3)
      fic(5) = -fi(5)
      fic(6) = -fi(6)
c
      return
      end
