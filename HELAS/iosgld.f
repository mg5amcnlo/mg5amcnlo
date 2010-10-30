c
c ----------------------------------------------------------------------
c
      subroutine iosgld(fi,fo,sc,gc,smass,mNLSP,idecay , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-scalar
c SUSY Goldstino coupling. In this routine, the NLSP is decaying to a
c boson and a Goldstino. The h.c. of the NLSP decay is handled via the
c input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex sc(3)          : input    scalar                      s
c       complex gc(2)          : coupling constants                  gsf
c       real    mNLSP          : mass of the NLSP
c       real    smass          : mass of the scalar boson
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex vertex         : amplitude                     <fo|s|fi>
c
      implicit none
      double complex  fi(6), fo(6), gc(2), sc(3), vertex
      double complex  p14p, p14m, p23p, p23m
      double precision  mNLSP, smass
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of IOSGLD'
         stop
      end if

      p14p = dble(sc(2)) + dimag(sc(2))
      p14m = dble(sc(2)) - dimag(sc(2))
      p23p = dble(sc(3)) + dimag(sc(3))*ci
      p23m = dble(sc(3)) - dimag(sc(3))*ci

      vertex = gc(1) *
     &         ( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &            +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*mNLSP
     &          -( fo(1)*fi(1) + fo(2)*fi(2) )*idecay*smass**2 )

      if ( gc(2).ne.cZero ) then
         vertex = vertex + gc(2) *
     &            ( ( ( fo(1)*p14m - fo(2)*p23p )*fi(3)
     &               -( fo(1)*p23m - fo(2)*p14p )*fi(4) )*mNLSP
     &             -( fo(3)*fi(3) + fo(4)*fi(4) )*idecay*smass**2 )
      end if

      vertex = vertex * sc(1)
c
      return
      end
