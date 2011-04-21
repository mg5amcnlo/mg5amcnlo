c
c ----------------------------------------------------------------------
c
      subroutine iovgld(fi,fo,vc,gc,idecay , vertex)
c
c This subroutine computes an amplitude of the fermion-fermion-vector
c SUSY Goldstino coupling. In this routine, the NLSP is decaying to a
c boson and a Goldstino. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex vertex         : amplitude                     <fo|v|fi>
c
      implicit none
      double complex  fi(6), fo(6), gc(2), vc(6), vertex
      double complex  AdotpG, A14p, A14m, A23p, A23m
      double complex  p14p, p14m, p23p, p23m
      double precision  pdotpG
      integer idecay

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      if ( idecay.eq.1 ) then
         pdotpG =  dble( vc(5))*dble( fo(5))
     &           - dble( vc(6))*dble( fo(6))
     &           - dimag(vc(6))*dimag(fo(6))
     &           - dimag(vc(5))*dimag(fo(5))
         AdotpG =  vc(1)*dble( fo(5)) - vc(2)*dble( fo(6))
     &           - vc(3)*dimag(fo(6)) - vc(4)*dimag(fo(5))
      else if ( idecay.eq.-1 ) then
         pdotpG =  dble( vc(5))*dble( fi(5))
     &           - dble( vc(6))*dble( fi(6))
     &           - dimag(vc(6))*dimag(fi(6))
     &           - dimag(vc(5))*dimag(fi(5))
         AdotpG =  vc(1)*dble( fi(5)) - vc(2)*dble( fi(6))
     &           - vc(3)*dimag(fi(6)) - vc(4)*dimag(fi(5))
      else
         write(6,*) 'error in idecay of IOVGLD'
         stop
      end if

      p14p = dble(vc(5)) + dimag(vc(5))
      p14m = dble(vc(5)) - dimag(vc(5))
      p23p = vc(6)
      p23m = dconjg(vc(6))
      A14p = vc(1) + vc(4)
      A14m = vc(1) - vc(4)
      A23p = vc(2) + vc(3)*ci
      A23m = vc(2) - vc(3)*ci

      vertex = gc(1)*( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &                  +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*AdotpG
     &                -( ( fo(3)*A14p + fo(4)*A23p )*fi(1)
     &                  +( fo(3)*A23m + fo(4)*A14m )*fi(2) )*pdotpG )

      if ( gc(2).ne.cZero ) then
         vertex = vertex
     &          + gc(2)*( ( (fo(1)*p14m - fo(2)*p23p )*fi(3)
     &                     -(fo(1)*p23m - fo(2)*p14p )*fi(4) )*AdotpG
     &                   -( (fo(1)*A14m - fo(2)*A23p )*fi(3)
     &                     -(fo(1)*A23m - fo(2)*A14p )*fi(4) )*pdotpG )
      end if

      vertex = vertex * idecay
c
      return
      end
