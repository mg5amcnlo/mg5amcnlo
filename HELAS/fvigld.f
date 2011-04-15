c
c ----------------------------------------------------------------------
c
      subroutine fvigld(fi,vc,gc,fmass,fwidth,idecay , fvi)
c
c This subroutine computes an off-shell fermion wavefunction from a
c flowing-in external fermion and a vector boson, for the NLSP-boson-
c Goldstino vertex. The h.c. of the NLSP decay is handled via the
c input parameter idecay (picks out correct Goldstino momentum).
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex vc(6)          : input    vector                      v
c       complex gc(2)          : coupling constants                  gvf
c       real    fmass          : mass  of output fermion f'
c       real    fwidth         : width of output fermion f'
c       integer idecay         :  1 for NLSP decay to Goldstino
c                              : -1 for Goldstino to NLSP (h.c. of above)
c
c output:
c       complex fvi(6)         : off-shell fermion             |f',v,fi>
c
      implicit none
      double complex  fi(6), vc(6), gc(2), fvi(6), sl1, sl2, sr1, sr2, d
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
      fvi(5) = fi(5) - vc(5)
      fvi(6) = fi(6) - vc(6)

      pv(1) = dble( vc(5))
      pv(2) = dble( vc(6))
      pv(3) = dimag(vc(6))
      pv(4) = dimag(vc(5))

      pf(0) = dble( fvi(5))
      pf(1) = dble( fvi(6))
      pf(2) = dimag(fvi(6))
      pf(3) = dimag(fvi(5))
      pf2 = pf(0)**2 - pf(1)**2 - pf(2)**2 - pf(3)**2

      if ( idecay.eq.1 ) then
         pdotpG = pv(1)*pf(0) - pv(2)*pf(1) - pv(3)*pf(2) - pv(4)*pf(3)
         AdotpG = vc(1)*pf(0) - vc(2)*pf(1) - vc(3)*pf(2) - vc(4)*pf(3)
      else if ( idecay.eq.-1 ) then
         pdotpG =  pv(1)*dble( fi(5)) - pv(2)*dble( fi(6))
     &           - pv(3)*dimag(fi(6)) - pv(4)*dimag(fi(5))
         AdotpG =  vc(1)*dble( fi(5)) - vc(2)*dble( fi(6))
     &           - vc(3)*dimag(fi(6)) - vc(4)*dimag(fi(5))
      else
         write(6,*) 'error in idecay of FVIGLD'
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

      sl1 =  (p14p*AdotpG - A14p*pdotpG)*fi(1)
     &      +(p23m*AdotpG - A23m*pdotpG)*fi(2)
      sl2 =  (p23p*AdotpG - A23p*pdotpG)*fi(1)
     &      +(p14m*AdotpG - A14m*pdotpG)*fi(2)

      if ( gc(2).ne.cZero ) then
         sr1 =  (p14m*AdotpG - A14m*pdotpG)*fi(3)
     &         -(p23m*AdotpG - A23m*pdotpG)*fi(4)
         sr2 = -(p23p*AdotpG - A23p*pdotpG)*fi(3)
     &         +(p14p*AdotpG - A14p*pdotpG)*fi(4)

         fvi(1) = ( gc(1)*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2 )
     &             +gc(2)*fmass*sr1 )*d
         fvi(2) = ( gc(1)*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2 )
     &             +gc(2)*fmass*sr2 )*d
         fvi(3) = ( gc(2)*((pf(0)+pf(3))*sr1 + dconjg(fvi(6))*sr2 )
     &             +gc(1)*fmass*sl1 )*d
         fvi(4) = ( gc(2)*(       fvi(6)*sr1 +  (pf(0)-pf(3))*sr2 )
     &             +gc(1)*fmass*sl2 )*d

      else
         d = d*gc(1)
         fvi(1) = d*((pf(0)-pf(3))*sl1 - dconjg(fvi(6))*sl2)
         fvi(2) = d*(      -fvi(6)*sl1 +  (pf(0)+pf(3))*sl2)
         fvi(3) = d*fmass*sl1
         fvi(4) = d*fmass*sl2
      end if
c
      return          
      end
