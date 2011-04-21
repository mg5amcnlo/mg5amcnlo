c
c ----------------------------------------------------------------------
c
      subroutine iovcxx(fic,foc,vc,gc , vertex)
c
c this subroutine computes an amplitude of the antifermion-antifermion- 
c vector coupling.                                                      
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c                                                                       
c output:                                                               
c       complex vertex         : amplitude                   <foc|v|fic>
c
      implicit none
      double complex fic(6),foc(6),vc(6),gc(2),vertex

      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      vertex = - gc(1)*( (foc(1)*fic(3)+foc(2)*fic(4))*vc(1)
     &                  -(foc(1)*fic(4)+foc(2)*fic(3))*vc(2)
     &                  +(foc(1)*fic(4)-foc(2)*fic(3))*vc(3)*cImag
     &                  -(foc(1)*fic(3)-foc(2)*fic(4))*vc(4)       )

      if ( gc(2).ne.cZero ) then
         vertex = vertex
     &            - gc(2)*( (foc(3)*fic(1)+foc(4)*fic(2))*vc(1)
     &                     +(foc(3)*fic(2)+foc(4)*fic(1))*vc(2)
     &                     -(foc(3)*fic(2)-foc(4)*fic(1))*vc(3)*cImag
     &                     +(foc(3)*fic(1)-foc(4)*fic(2))*vc(4)       )
      end if
c
      return
      end
