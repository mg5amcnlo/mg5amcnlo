c
c ----------------------------------------------------------------------
c
      subroutine fvogld(fo,vc,gc,fmass,fwidth,idecay , fvo)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-out external fermion and a vector boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fvo(6)         : off-shell fermion             <fo,v,f'|
c
      implicit none
      double complex  fo(6), vc(6), gc(2), fvo(6), sl1, sl2, sr1, sr2, d
      double complex  p14p, p14m, p23p, p23m, A14p, A14m, A23p, A23m
      double complex  AdotpG
      double precision  fmass, fwidth
      double precision  pf(0:3), pv(4), pf2, pdotpG
      integer idecay

      double precision rOne
      parameter( rOne = 1.0d0 )
      double complex ci, cZero
      parameter( ci = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )
c
      fvo(5) = fo(5) + vc(5)
      fvo(6) = fo(6) + vc(6)

      pv(1) = dble( vc(5))
      pv(2) = dble( vc(6))
      pv(3) = dimag(vc(6))
      pv(4) = dimag(vc(5))

      pf(0) = dble( fvo(5))
      pf(1) = dble( fvo(6))
      pf(2) = dimag(fvo(6))
      pf(3) = dimag(fvo(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.eq.1 ) then
         pdotpG = pv(1)*pf(0) - pv(2)*pf(1) - pv(3)*pf(2) - pv(4)*pf(3)
         AdotpG = vc(1)*pf(0) - vc(2)*pf(1) - vc(3)*pf(2) - vc(4)*pf(3)
      else if ( idecay.eq.-1 ) then
         pdotpG =  pv(1)*dble( fo(5)) - pv(2)*dble( fo(6))
     &           - pv(3)*dimag(fo(6)) - pv(4)*dimag(fo(5))
         AdotpG =  vc(1)*dble( fo(5)) - vc(2)*dble( fo(6))
     &           - vc(3)*dimag(fo(6)) - vc(4)*dimag(fo(5))
      else
         write(6,*) 'error in idecay of FVOGLD'
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

      d = -rOne/dcmplx( pf2-fmass**2, fmass*fwidth )
      d = d*idecay

      sl1 =  (p14p*AdotpG - A14p*pdotpG)*fo(3)
     &      +(p23p*AdotpG - A23p*pdotpG)*fo(4)
      sl2 =  (p23m*AdotpG - A23m*pdotpG)*fo(3)
     &      +(p14m*AdotpG - A14m*pdotpG)*fo(4)

      if ( gc(2).ne.cZero ) then
         sr1 =  (p14m*AdotpG - A14m*pdotpG)*fo(1)
     &         -(p23p*AdotpG - A23p*pdotpG)*fo(2)
         sr2 = -(p23m*AdotpG - A23m*pdotpG)*fo(1)
     &         +(p14p*AdotpG - A14p*pdotpG)*fo(2)

         fvo(1) = ( gc(2)*(  (pf(0)+pf(3))*sr1         +fvo(6)*sr2 )
     &             +gc(1)*fmass*sl1 )*d
         fvo(2) = ( gc(2)*( dconjg(fvo(6))*sr1 + (pf(0)-pf(3))*sr2 )
     &             +gc(1)*fmass*sl2 )*d
         fvo(3) = ( gc(1)*(  (pf(0)-pf(3))*sl1         -fvo(6)*sl2 )
     &             +gc(2)*fmass*sr1 )*d
         fvo(4) = ( gc(1)*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2 )
     &             +gc(2)*fmass*sr2 )*d

      else
         d = d*gc(1)
         fvo(1) = d*fmass*sl1
         fvo(2) = d*fmass*sl2
         fvo(3) = d*(  (pf(0)-pf(3))*sl1         -fvo(6)*sl2)
         fvo(4) = d*(-dconjg(fvo(6))*sl1 + (pf(0)+pf(3))*sl2)
      end if
c
      return          
      end
