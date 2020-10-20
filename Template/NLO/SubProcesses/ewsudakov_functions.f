      double precision function get_lsc_diag(pdgs, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double precision bigL, smallL, sdk_cew_diag, sdk_iz2_diag

      integer i

      get_lsc_diag = 0d0

      lzow = dlog(mdl_mz**2/mdl_mw**2)

      do i = 1, nexternal-1
        get_lsc_diag = get_lsc_diag - 0.5d0 * 
     %   (sdk_cew_diag(pdgs(i),hels(i),iflist(i)) * bigL(invariants(1,2))
     %    - 2d0*lzow*sdk_iz2_diag*smallL(invariants(1,2)))
      enddo
      return
      end


      double precision function get_lsc_nondiag(invariants)
      implicit none
      include 'nexternal.inc'
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double precision bigL, smallL, sdk_cew_nondiag

      lzow = dlog(mdl_mz**2/mdl_mw**2)

      get_lsc_nondiag = -0.5d0 * sdk_cew_nondiag() * bigL(invariants(1,2))

      return
      end
      
      
      double precision function bigL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      bigL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)**2

      return
      end
      
      
      double precision function smallL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      smallL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)

      return
      end
      
      
      double precision function sdk_chargesq(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      double precision sdk_charge

      sdk_chargesq = sdk_charge(pdg, hel, ifsign)**2

      return
      end


      double precision function sdk_charge(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      s_pdg = pdg*ifsign

      sdk_charge = 0d0

C lepton 
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_charge = -1d0
C antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_charge = 1d0

C up quark 
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_charge = 2d0/3d0
C anti up quark 
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_charge = -2d0/3d0

C down quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_charge = -1d0/3d0
C antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_charge = 1d0/3d0

C charged goldstones / W boson
      if (s_pdg.eq.251.or.s_pdg.eq.24) sdk_charge = 1d0
      if (s_pdg.eq.-251.or.s_pdg.eq.-24) sdk_charge = -1d0

      return
      end


      double precision function sdk_iz2_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity
      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_iz2_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_iz2_diag =1d0 / (4*sw2*cw2) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_iz2_diag = sw2/cw2
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_iz2_diag = (cw2-sw2)**2 / (4*sw2*cw2) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_iz2_diag = 4*sw2/(9*cw2)
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_iz2_diag =(3*cw2-sw2)**2 / (36*sw2*cw2) 

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_iz2_diag = sw2/(9*cw2)
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_iz2_diag = (3*cw2+sw2)**2 / (36*sw2*cw2) 

C goldstones, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251) write(*,*) 'IZ2DIAG not implemented' 
      if (abs(s_pdg).eq.250) write(*,*) 'IZ2DIAG not implemented'
      if (abs(s_pdg).eq.25) write(*,*) 'IZ2DIAG not implemented'

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_iz2_diag = cw2 / sw2

      return
      end




      double precision function sdk_cew_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity
      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_cew_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_cew_diag = 1d0/cw2
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_cew_diag = 4d0/(9*cw2)
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_cew_diag = (sw2+27*cw2) / (36*sw2*cw2) 

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_cew_diag = 1d0/(9*cw2)
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_cew_diag = (sw2+27*cw2) / (36*sw2*cw2) 

C goldstones, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)
      if (abs(s_pdg).eq.250) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_cew_diag = 2 / sw2

C transverse Z boson
      if (abs(s_pdg).eq.23) sdk_cew_diag = 2 * cw2 / sw2

C (transverse) photon
      if (abs(s_pdg).eq.22) sdk_cew_diag = 2d0 

      return
      end


      double precision function sdk_cew_nondiag()
      implicit none
C returns the gamma/z mixing of sdk_cew
      include "coupl.inc"
      double precision sw2, cw2

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_cew_nondiag = - 2 / sw2 * dsqrt(cw2*sw2) 

      return
      end


      subroutine sdk_get_invariants(p, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      double precision p(0:3, nexternal-1)
      integer iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer i,j
      double precision sumdot

      do i = 1, nexternal-1
        do j = i+1, nexternal-1
          invariants(i,j) = sumdot(p(0,i),p(0,j),dble(ifsign(i)*ifsign(j)))
          invariants(j,i) = invariants(i,j)
        enddo
      enddo

      return 
      end
