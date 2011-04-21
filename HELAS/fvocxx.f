c
c ----------------------------------------------------------------------
c
      subroutine fvocxx(foc,vc,gc,fmass,fwidth , fvoc)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-out external antifermion and a vector boson.                  
c                                                                       
c input:                                                                
c       complex foc(6)         : flow-out antifermion              <foc|
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output antifermion f'         
c       real    fwidth         : width of output antifermion f'         
c                                                                       
c output:                                                               
c       complex fvoc(6)        : off-shell antifermion       <foc,v,fc'|
c
      implicit none
      double complex foc(6),vc(6),gc(2),fvoc(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvoc(5) = foc(5)+vc(5)
      fvoc(6) = foc(6)+vc(6)

      pf(0) = dble( fvoc(5))
      pf(1) = dble( fvoc(6))
      pf(2) = dimag(fvoc(6))
      pf(3) = dimag(fvoc(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      d = rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sr1 =   (vc(1)-      vc(4))*foc(1)
     &      - (vc(2)+cImag*vc(3))*foc(2)
      sr2 = - (vc(2)-cImag*vc(3))*foc(1)
     &      + (vc(1)+      vc(4))*foc(2)

      if ( gc(2).ne.cZero ) then
         sl1 =   (vc(1)+      vc(4))*foc(3)
     &         + (vc(2)+cImag*vc(3))*foc(4)
         sl2 =   (vc(2)-cImag*vc(3))*foc(3)
     &         + (vc(1)-      vc(4))*foc(4)

         fvoc(1) = ( gc(1)*( (pf(0)+pf(3))*sr1   +       fvoc(6)*sr2)
     &              +gc(2)*fmass*sl1 )*d
         fvoc(2) = ( gc(1)*( dconjg(fvoc(6))*sr1 + (pf(0)-pf(3))*sr2)
     &              +gc(2)*fmass*sl2 )*d
         fvoc(3) = ( gc(2)*( (pf(0)-pf(3))*sl1   -       fvoc(6)*sl2)
     &              +gc(1)*fmass*sr1 )*d
         fvoc(4) = ( gc(2)*(-dconjg(fvoc(6))*sl1 + (pf(0)+pf(3))*sl2)
     &              +gc(1)*fmass*sr2 )*d
      else
         d = d * gc(1)
         fvoc(1) = ((pf(0)+pf(3))*sr1          +fvoc(6)*sr2)*d
         fvoc(2) = (dconjg(fvoc(6))*sr1 + (pf(0)-pf(3))*sr2)*d
         fvoc(3) = fmass*sr1*d
         fvoc(4) = fmass*sr2*d
      end if
c
      return
      end
