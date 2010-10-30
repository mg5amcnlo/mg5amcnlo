c
c ----------------------------------------------------------------------
c
      subroutine fsocxx(foc,sc,gc,fmass,fwidth , fsoc)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-out external antifermion and a vector boson.                  
c                                                                       
c input:                                                                
c       complex foc(6)         : flow-out fermion               <foc|
c       complex sc(6)          : input    scalar                   s 
c       complex gc(2)          : coupling constants              gchf
c       real     fmass         : mass  of output antifermion fc'     
c       real     fwidth        : width of output antifermion fc'     
c                                                                       
c output:                                                               
c       complex fsoc(6)        : off-shell fermion         <fo,s,fc'|
c
      implicit none
      double complex foc(6),sc(6),fsoc(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
c
      fsoc(5) = foc(5)+sc(2)
      fsoc(6) = foc(6)+sc(3)

      pf(0) = dble( fsoc(5))
      pf(1) = dble( fsoc(6))
      pf(2) = dimag(fsoc(6))
      pf(3) = dimag(fsoc(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(2)*(p0p3*foc(3)       +fsoc(6) *foc(4))
      sl2 = gc(2)*(p0m3*foc(4)+dconjg(fsoc(6))*foc(3))
      sr1 = gc(1)*(p0m3*foc(1)       -fsoc(6) *foc(2))
      sr2 = gc(1)*(p0p3*foc(2)-dconjg(fsoc(6))*foc(1))

      fsoc(1) = ( gc(1)*fmass*foc(1) + sl1 )*ds
      fsoc(2) = ( gc(1)*fmass*foc(2) + sl2 )*ds
      fsoc(3) = ( gc(2)*fmass*foc(3) + sr1 )*ds
      fsoc(4) = ( gc(2)*fmass*foc(4) + sr2 )*ds
c
      return
      end
