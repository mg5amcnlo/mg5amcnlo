c
c ----------------------------------------------------------------------
c
      subroutine fsigld(fi,sc,gc,fmass,fwidth,smass,mNLSP,idecay , fsi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-in external fermion and a scalar boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fsi(6)         : off-shell fermion             |f',s,fi>
c
      implicit none
      double complex  fi(6), sc(3), gc(2), fsi(6), s1, s2, s3, s4, ds
      double complex  p14p, p14m, p23p, p23m
      double precision  pf(0:3), fmass, fwidth, mNLSP, smass, pf2
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fsi(5) = fi(5) - sc(2)
      fsi(6) = fi(6) - sc(3)

      pf(0) = dble( fsi(5))
      pf(1) = dble( fsi(6))
      pf(2) = dimag(fsi(6))
      pf(3) = dimag(fsi(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of FSIGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )

      s1 = -idecay*gc(1)*fi(1)*smass**2
      s2 = -idecay*gc(1)*fi(2)*smass**2
      s3 = gc(1)*mNLSP*( fi(1)*p14p + fi(2)*p23m ) 
      s4 = gc(1)*mNLSP*( fi(1)*p23p + fi(2)*p14m )

      if ( gc(2).ne.cZero ) then
         s1 = s1 + gc(2)*mNLSP*( fi(3)*p14m - fi(4)*p23m )
         s2 = s2 + gc(2)*mNLSP*(-fi(3)*p23p + fi(4)*p14p )
         s3 = s3 - gc(2)*idecay*fi(3)*smass**2
         s4 = s4 - gc(2)*idecay*fi(4)*smass**2
      end if

      fsi(1) = ( (pf(0)-pf(3))*s3 - dconjg(fsi(6))*s4 + fmass*s1 )*ds
      fsi(2) = (       -fsi(6)*s3 +  (pf(0)+pf(3))*s4 + fmass*s2 )*ds
      fsi(3) = ( (pf(0)+pf(3))*s1 + dconjg(fsi(6))*s2 + fmass*s3 )*ds
      fsi(4) = (        fsi(6)*s1 +  (pf(0)-pf(3))*s2 + fmass*s4 )*ds
c
      return          
      end
