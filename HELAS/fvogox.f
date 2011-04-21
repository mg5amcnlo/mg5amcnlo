      subroutine fvogox(fo,vc,gc,fmass,fwidth , fvo)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-OUT external fermion and a vector boson in the case when 
c they are all color octets.
c
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of OUTPUT fermion f'
c       real    fwidth         : width of OUTPUT fermion f'
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,v,f'|
c
      implicit none
      double complex fo(6),vc(6),gc(2),fvo(6),sl1,sl2,sr1,sr2,d
      double precision pf(0:3),fmass,fwidth,pf2

      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

      fvo(5) = fo(5)+vc(5)
      fvo(6) = fo(6)+vc(6)

      pf(0) = dble( fvo(5))
      pf(1) = dble( fvo(6))
      pf(2) = dimag(fvo(6))
      pf(3) = dimag(fvo(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      sl1 =   (vc(1)+      vc(4))*fo(3)
     &      + (vc(2)+cImag*vc(3))*fo(4)
      sl2 =   (vc(2)-cImag*vc(3))*fo(3)
     &      + (vc(1)-      vc(4))*fo(4)

      if ( gc(2).ne.cZero ) then
         sr1 =   (vc(1)-      vc(4))*fo(1)
     &         - (vc(2)+cImag*vc(3))*fo(2)
         sr2 = - (vc(2)-cImag*vc(3))*fo(1)
     &         + (vc(1)+      vc(4))*fo(2)

         fvo(1) = ( gc(2)*( (pf(0)+pf(3))*sr1  +        fvo(6)*sr2)
     &             +gc(1)*fmass*sl1 )*d
         fvo(2) = ( gc(2)*( dconjg(fvo(6))*sr1 + (pf(0)-pf(3))*sr2)
     &             +gc(1)*fmass*sl2 )*d
         fvo(3) = ( gc(1)*( (pf(0)-pf(3))*sl1  -        fvo(6)*sl2)
     &             +gc(2)*fmass*sr1 )*d
         fvo(4) = ( gc(1)*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)
     &             +gc(2)*fmass*sr2 )*d

      else
         d = d * gc(1)
         fvo(1) = fmass*sl1*d
         fvo(2) = fmass*sl2*d
         fvo(3) = (  (pf(0)-pf(3))*sl1 -        fvo(6)*sl2)*d
         fvo(4) = (-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)*d
      end if
c
      return
      end
