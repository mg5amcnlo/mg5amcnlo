c
c ----------------------------------------------------------------------
c
      subroutine fsicxx(fic,sc,gc,fmass,fwidth , fsic)
c
c this subroutine computes an off-shell antifermion wavefunction from a 
c flowing-in external antifermion and a vector boson.                   
c                                                                       
c input:                                                                
c       complex fic(6)         : flow-in  antifermion           |fic>
c       complex sc(3)          : input    scalar                   s 
c       complex gc(2)          : coupling constants              gchf
c       real    fmass          : mass  of output antifermion fc'     
c       real    fwidth         : width of output antifermion fc'     
c                                                                       
c output:                                                               
c       complex fsic(6)        : off-shell fermion        |fc',s,fic>
c
      implicit none
      double complex fic(6),sc(3),fsic(6),gc(2),sl1,sl2,sr1,sr2,ds
      double precision pf(0:3),fmass,fwidth,pf2,p0p3,p0m3
c
      fsic(5) = fic(5)-sc(2)
      fsic(6) = fic(6)-sc(3)

      pf(0) = dble( fsic(5))
      pf(1) = dble( fsic(6))
      pf(2) = dimag(fsic(6))
      pf(3) = dimag(fsic(5))
      pf2 = pf(0)**2-(pf(1)**2+pf(2)**2+pf(3)**2)

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )
      p0p3 = pf(0)+pf(3)
      p0m3 = pf(0)-pf(3)
      sl1 = gc(1)*(p0p3*fic(1)+dconjg(fsic(6))*fic(2))
      sl2 = gc(1)*(p0m3*fic(2)       +fsic(6) *fic(1))
      sr1 = gc(2)*(p0m3*fic(3)-dconjg(fsic(6))*fic(4))
      sr2 = gc(2)*(p0p3*fic(4)       -fsic(6) *fic(3))

      fsic(1) = ( gc(1)*fmass*fic(1) + sr1 )*ds
      fsic(2) = ( gc(1)*fmass*fic(2) + sr2 )*ds
      fsic(3) = ( gc(2)*fmass*fic(3) + sl1 )*ds
      fsic(4) = ( gc(2)*fmass*fic(4) + sl2 )*ds
c
      return
      end
