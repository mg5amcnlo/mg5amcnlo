c
c ----------------------------------------------------------------------
c
      subroutine fvicxx(fic,vc,gc,fmass,fwidth , fvic)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-in external antifermion and a vector boson.                   
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion              |fic>
c       complex vc(6)          : input    vector                      v 
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output antifermion f'         
c       real    fwidth         : width of output antifermion f'         
c                                                                       
c output:                                                               
c       complex fvic(6)        : off-shell antifermion       |fc',v,fic>
c
      implicit none
      double complex fic(6),vc(6),gc(2),fvic(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvic(5) = fic(5)-vc(5)
      fvic(6) = fic(6)-vc(6)

      pf(0) = dble( fvic(5))
      pf(1) = dble( fvic(6))
      pf(2) = dimag(fvic(6))
      pf(3) = dimag(fvic(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      d = rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sr1 =   (vc(1)-      vc(4))*fic(3)
     &      - (vc(2)-cImag*vc(3))*fic(4)
      sr2 = - (vc(2)+cImag*vc(3))*fic(3)
     &      + (vc(1)+      vc(4))*fic(4)

      if ( gc(2).ne.cZero ) then
         sl1 =   (vc(1)+      vc(4))*fic(1)
     &         + (vc(2)-cImag*vc(3))*fic(2)
         sl2 =   (vc(2)+cImag*vc(3))*fic(1)
     &         + (vc(1)-      vc(4))*fic(2)

         fvic(1) = ( gc(2)*((pf(0)-pf(3))*sl1 - dconjg(fvic(6))*sl2)
     &              +gc(1)*fmass*sr1 )*d
         fvic(2) = ( gc(2)*(     -fvic(6)*sl1 +   (pf(0)+pf(3))*sl2)
     &              +gc(1)*fmass*sr2 )*d
         fvic(3) = ( gc(1)*((pf(0)+pf(3))*sr1 + dconjg(fvic(6))*sr2)
     &              +gc(2)*fmass*sl1 )*d
         fvic(4) = ( gc(1)*(      fvic(6)*sr1 +   (pf(0)-pf(3))*sr2)
     &              +gc(2)*fmass*sl2 )*d
      else
         d = d * gc(1)
         fvic(1) = fmass*sr1*d
         fvic(2) = fmass*sr2*d
         fvic(3) = ((pf(0)+pf(3))*sr1 + dconjg(fvic(6))*sr2)*d
         fvic(4) = (      fvic(6)*sr1 +   (pf(0)-pf(3))*sr2)*d
      end if
c
      return
      end
