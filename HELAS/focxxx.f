c
c ----------------------------------------------------------------------
c
      subroutine focxxx(fo , foc)
c
c this subroutine charge conjugates a flowing-out fermion wavefunction. 
c                                                                       
c input:                                                                
c       complex fo(6)          : flowing-out fermion                <fo|
c                                                                       
c output:                                                               
c       complex foc(6)         : charge conjugated fermion         |foc>
c
      implicit none
      double complex fo(6), foc(6)
c
      foc(1) =  fo(2)
      foc(2) = -fo(1)
      foc(3) = -fo(4)
      foc(4) =  fo(3)
      foc(5) = -fo(5)
      foc(6) = -fo(6)
c
      return
      end
