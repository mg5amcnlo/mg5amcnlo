c
c ----------------------------------------------------------------------
c
      subroutine hiogld(fi,fo,gc,smass,swidth,mNLSP,idecay , hio)
c
c This subroutine computes an off-shell scalar current for the NLSP-
c Goldstino vertex from the external fermion pair. The h.c. of the NLSP
c decay is handled via the input parameter idecay.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gsf
c       real    smass          : mass  of output scalar s
c       real    swidth         : width of output scalar s
c       real    mNLSP          : mass of NLSP
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex hio(3)         : scalar current          j^mu(<fo|s|fi>)
c
      implicit none
      double complex fi(6), fo(6), gc(2), hio(3)
      double complex dn, p14p, p14m, p23p, p23m
      double precision q(0:3), smass, swidth, mNLSP, q2
      double precision pG(1:4)
      integer idecay

      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      hio(2) = -fi(5) + fo(5)
      hio(3) = -fi(6) + fo(6)

      if ( idecay.ne.1 .or. idecay.ne.-1 ) then
         write(6,*) 'error in idecay of HIOGLD'
         stop
      end if

      q(0) = dble( hio(2))
      q(1) = dble( hio(3))
      q(2) = dimag(hio(3))
      q(3) = dimag(hio(2))
      q2 = q(0)**2 - q(1)**2 - q(2)**2 - q(3)**2

      p14p = q(0) + q(3)
      p14m = q(0) - q(3)
      p23p = q(1) + q(2)*ci
      p23m = q(1) - q(2)*ci

      dn = -dcmplx( q2-smass**2, smass*swidth )

      hio(1) = gc(1)*( ( ( fo(3)*p14p + fo(4)*p23p )*fi(1)
     &                  +( fo(3)*p23m + fo(4)*p14m )*fi(2) )*mNLSP
     &                -( fo(1)*fi(1) + fo(2)*fi(2) )*idecay*smass**2 )

      if ( gc(2).ne.cZero ) then
         hio(1) = hio(1) + gc(2) *
     &            ( ( ( fo(1)*p14m - fo(2)*p23p )*fi(3)
     &               -( fo(1)*p23m - fo(2)*p14p )*fi(4) )*mNLSP
     &             -( fo(3)*fi(3) + fo(4)*fi(4) )*idecay*smass**2 )
      end if

      hio(1) = hio(1)/dn
c
      return
      end
