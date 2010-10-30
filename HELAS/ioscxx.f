c
c ----------------------------------------------------------------------
c
      subroutine ioscxx(fic,foc,sc,gc , vertex)
c
c This subroutine computes an amplitude of the antifermion-antifermion- 
c scalar coupling.                                                      
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex sc(3)          : input    scalar                      s 
c       complex gc(2)          : coupling constants                 gchf
c                                                                       
c output:                                                               
c       complex vertex         : amplitude                   <foc|s|fic>
c
      implicit none
      double complex fic(6),foc(6),sc(3),gc(2),vertex
c
      vertex = sc(1)*( gc(1)*(fic(1)*foc(1)+fic(2)*foc(2))
     &                +gc(2)*(fic(3)*foc(3)+fic(4)*foc(4)) )
c
      return
      end
