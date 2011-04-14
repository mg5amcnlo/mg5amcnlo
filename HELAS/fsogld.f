c
c ----------------------------------------------------------------------
c
      subroutine fsogld(fo,sc,gc,fmass,fwidth,smass,mNLSP,idecay , fso)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-out external fermion and a scalar boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fso(6)         : off-shell fermion             <fo,s,f'|
c
      implicit none
      double complex  fo(6), sc(3), gc(2), fso(6), s1, s2, s3, s4, ds
      double precision  pf(0:3), fmass, fwidth, mNLSP, smass, pf2
      double precision  p14p, p14m, p23p, p23m
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fso(5) = fo(5) + sc(2)
      fso(6) = fo(6) + sc(3)

      pf(0) = dble( fso(5))
      pf(1) = dble( fso(6))
      pf(2) = dimag(fso(6))
      pf(3) = dimag(fso(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of FSOGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      ds = -sc(1)/dcmplx( pf2-fmass**2, fmass*fwidth )

      s1 = -idecay*gc(1)*fo(1)*smass**2
      s2 = -idecay*gc(1)*fo(2)*smass**2
      s3 = gc(1)*mNLSP*( fo(1)*p14m - fo(2)*p23p )
      s4 = gc(1)*mNLSP*(-fo(1)*p23m + fo(2)*p14p )

      if ( gc(2).ne.cZero ) then
         s1 = s1 + gc(2)*mNLSP*( fo(3)*p14p + fo(4)*p23p )
         s2 = s2 + gc(2)*mNLSP*( fo(3)*p23m + fo(4)*p14m )
         s3 = s3 - gc(2)*idecay*fo(3)*smass**2
         s4 = s4 - gc(2)*idecay*fo(4)*smass**2
      end if

      fso(1) = (  (pf(0)+pf(3))*s3 +         fso(6)*s4 + fmass*s1 )*ds
      fso(2) = ( dconjg(fso(6))*s3 +  (pf(0)-pf(3))*s4 + fmass*s2 )*ds
      fso(3) = (  (pf(0)-pf(3))*s1 -         fso(6)*s2 + fmass*s3 )*ds
      fso(4) = (-dconjg(fso(6))*s1 +  (pf(0)+pf(3))*s2 + fmass*s4 )*ds
c
      return          
      end
