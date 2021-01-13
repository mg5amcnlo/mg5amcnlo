      !! MZ declare all functions as double complex, since some (few)
      !  terms can be imaginary
      
      
      double complex function get_lsc_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_diag, sdk_iz2_diag
      external sdk_iz2_diag
      integer i

      get_lsc_diag = 0d0

c exit and do nothing
      return

      lzow = dlog(mdl_mz**2/mdl_mw**2)

      do i = 1, nexternal-1
        get_lsc_diag = get_lsc_diag - 0.5d0 * 
     %   (sdk_cew_diag(pdglist(i),hels(i),iflist(i)) * bigL(invariants(1,2))
     %    - 2d0*lzow*sdk_iz2_diag(pdglist(i),hels(i),iflist(i))*smallL(invariants(1,2)))
      enddo
      return
      end


      double complex function get_lsc_nondiag(invariants, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      double precision invariants(nexternal-1, nexternal-1)
      integer pdg_old, pdg_new
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_nondiag

      ! this function is non zero only for Z/gamma mixing)
      get_lsc_nondiag = 0d0

c exit and do nothing
      return

      if ((pdg_old.eq.23.and.pdg_new.eq.22).or.
     $    (pdg_old.eq.22.and.pdg_new.eq.23)) then
        lzow = dlog(mdl_mz**2/mdl_mw**2)
        get_lsc_nondiag = -0.5d0 * sdk_cew_nondiag() * bigL(invariants(1,2))
      endif

      return
      end


      double complex function get_ssc_c(ileg1, ileg2, pdglist, pdgp1, pdgp2, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer ileg1, ileg2, pdgp1, pdgp2
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_tpm

      double precision s, rij

      logical   printinewsdkf
      common /to_printinewsdkf/printinewsdkf

      get_ssc_c = 0d0
c exit and do nothing
c      return

      lzow = dlog(mdl_mz**2/mdl_mw**2)
      s = invariants(1,2)

      rij = invariants(ileg1,ileg2)
      get_ssc_c = get_ssc_c + 2d0*smallL(s) * dlog(dabs(rij)/s) 
     $    * sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)
     $    * sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)


      if (printinewsdkf) WRITE (72,*) , hels, ileg1, ileg2, pdgp1, pdgp2,
     $ dble(sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)*CMPLX(1d0,-1000d0)),
     $ dble(sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)*CMPLX(1d0,-1000d0)) 



      if (printinewsdkf) print*,"get_ssc_c=",get_ssc_c,"    rij=",rij
      if (printinewsdkf) print*,"sdk_tpm(",pdglist(ileg1),",", hels(ileg1),
     . ",", iflist(ileg1),",", pdgp1,")=",sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1)
      if (printinewsdkf) print*,"sdk_tpm(",pdglist(ileg2),",", hels(ileg2),
     . ",", iflist(ileg2),",", pdgp2,")=",sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2)
      if (printinewsdkf) print*,"rij=","r(",ileg1,",",ileg2,")=",rij

c      if (printinewsdkf) then
c      if (sdk_tpm(pdglist(ileg1), hels(ileg1), iflist(ileg1), pdgp1) *
c     $   sdk_tpm(pdglist(ileg2), hels(ileg2), iflist(ileg2), pdgp2).ne.0d0.and. dabs(dlog(dabs(rij)/s)).ge.1d-6) then
c        print*,"DIVERSO DA ZERO", dlog(dabs(rij)/s)
c      else 
c         print*,"UGUALE A ZERO"
c      endif
c      endif

      return
      end


      double complex function get_ssc_n_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_ia_diag, sdk_iz_diag

      integer i,j
      double precision s, rij

      get_ssc_n_diag = 0d0
      return
c exit and do nothing


c      print*,"CI ENTRO"

      lzow = dlog(mdl_mz**2/mdl_mw**2)
      s = invariants(1,2)

      do i = 1, nexternal-1
        do j = 1, i-1
          rij = invariants(i,j)
          ! photon, Lambda = MW
          get_ssc_n_diag = get_ssc_n_diag + 2d0*smallL(s) * dlog(dabs(rij/s)) 
     %      * sdk_ia_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_ia_diag(pdglist(j),hels(j),iflist(j))

c      print*,"sdk_ia_diag(pdglist(,",i,"),hels(",i,"),iflist(",i,"))=", sdk_ia_diag(pdglist(i),hels(i),iflist(i))
c      print*,"sdk_ia_diag(pdglist(,",j,"),hels(",j,"),iflist(",j,"))=", sdk_ia_diag(pdglist(j),hels(j),iflist(j))
c      print*,"get_ssc_n_diag=",get_ssc_n_diag
          ! Z
          get_ssc_n_diag = get_ssc_n_diag + 2d0*smallL(s) * dlog(dabs(rij/s)) 
     %      * sdk_iz_diag(pdglist(i),hels(i),iflist(i))
     %      * sdk_iz_diag(pdglist(j),hels(j),iflist(j))

c      print*,"sdk_iz_diag(pdglist(,",i,"),hels(",i,"),iflist(",i,"))=", sdk_iz_diag(pdglist(i),hels(i),iflist(i))
c      print*,"sdk_iz_diag(pdglist(,",j,"),hels(",j,"),iflist(",j,"))=", sdk_iz_diag(pdglist(j),hels(j),iflist(j))
c      print*,"get_ssc_n_diag=",get_ssc_n_diag



        enddo
      enddo
      return
      end


      double complex function get_ssc_n_nondiag_1(pdglist, hels, iflist,
     $                              invariants, ileg, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg, pdg_old, pdg_new
      double complex bigL, smallL, sdk_iz_nondiag, sdk_iz_diag
      integer i
      double precision s

      ! this function corresponds to the case when *one* out of the two particles
      ! that enters the SSC contributions mixes as Chi <--> H (mediated
      ! by the Z).
      get_ssc_n_nondiag_1 = 0d0
      s = invariants(1,2)

      if ((pdg_old.eq.25.and.pdg_new.eq.250).or.
     $    (pdg_old.eq.250.and.pdg_new.eq.25)) then
        do i = 1, nexternal-1
          if (i.eq.ileg) cycle
          get_ssc_n_nondiag_1 = get_ssc_n_nondiag_1 +
     $              sdk_iz_diag(pdglist(i),hels(i),iflist(i)) *
     $              sdk_iz_nondiag(pdg_new,hels(ileg),iflist(ileg))
     $              * 2d0 * smallL(s) * dlog(abs(invariants(i,ileg))/s)
        enddo
      endif

      return
      end


      double complex function get_ssc_n_nondiag_2(pdglist, hels, iflist,
     $                          invariants, ileg1, pdg_old1,pdg_new1,
     $                                      ileg2, pdg_old2,pdg_new2)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      integer ileg1, pdg_old1, pdg_new1, ileg2, pdg_old2, pdg_new2
      double complex bigL, smallL, sdk_iz_nondiag, sdk_iz_diag
      double precision s

      ! this function corresponds to the case when both the two particles
      ! that enters the SSC contributions mixes as Chi <--> H (mediated
      ! by the Z).
      get_ssc_n_nondiag_2 = 0d0
      s = invariants(1,2)

      if (((pdg_old1.eq.25.and.pdg_new1.eq.250).or.
     $     (pdg_old1.eq.250.and.pdg_new1.eq.25)).and.
     $    ((pdg_old2.eq.25.and.pdg_new2.eq.250).or.
     $     (pdg_old2.eq.250.and.pdg_new2.eq.25))) then
        get_ssc_n_nondiag_2 = get_ssc_n_nondiag_2 +
     $              sdk_iz_nondiag(pdg_new1,hels(ileg1),iflist(ileg1)) * 
     $              sdk_iz_nondiag(pdg_new2,hels(ileg2),iflist(ileg2))
     $              * 2d0 * smallL(s) * dlog(abs(invariants(ileg1,ileg2))/s)
      endif

      return
      end



      
      double complex function get_xxc_diag(pdglist, hels, iflist, invariants)
      implicit none
      include 'nexternal.inc'
      integer pdglist(nexternal-1), hels(nexternal-1), iflist(nexternal-1)
      double precision invariants(nexternal-1, nexternal-1)
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_diag, sdk_iz2_diag
      external sdk_iz2_diag
      integer i

      get_xxc_diag = 0d0

c exit and do nothing
      return

      return
      end


      double complex function get_xxc_nondiag(invariants, pdg_old, pdg_new)
      implicit none
      include 'nexternal.inc'
      double precision invariants(nexternal-1, nexternal-1)
      integer pdg_old, pdg_new
      include 'coupl.inc'
      double precision lzow
      double complex bigL, smallL, sdk_cew_nondiag

      ! this function is non zero only for Z/gamma mixing)
      get_xxc_nondiag = 0d0

c exit and do nothing
      return

      if ((pdg_old.eq.23.and.pdg_new.eq.22).or.
     $    (pdg_old.eq.22.and.pdg_new.eq.23)) then
        continue
      endif

      return
      end

 


 
      double complex function bigL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      bigL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)**2

      return
      end
      
      
      double complex function smallL(s)
      implicit none
      double precision s
      include 'coupl.inc'
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      smallL = dble(gal(1))**2 / (4d0*pi)**2 * dlog(s/mdl_mw**2)

      return
      end
      
      
      double complex function sdk_chargesq(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      double complex sdk_charge

      sdk_chargesq = sdk_charge(pdg, hel, ifsign)**2

      return
      end


      double complex function sdk_charge(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign

      integer s_pdg

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


      double complex function sdk_tpm(pdg, hel, ifsign, pdgp)
      implicit none
      integer pdg, hel, ifsign, pdgp
      ! PDGP is the pdg of the born leg after its charge has been 
      !  changed by ±1. It is only necessary for vector bosons, where
      !  one can have either w+ > w+ gamma or w+ > w+ z
      integer s_pdg 

      include "coupl.inc"
      double precision sw2, cw2

C the product of pdg code * helicity  
C Hel=+1/-1 -> R/L. Note that for transverse polarisations it does not depend 
C on "ifsign", since switching from final to initial changes both the pdg and the helicity

C!!!!						ATTENTION						      !!!!
C!!!! following the notation of Denner and Pozzorini, the prime index pdgp is the first and pdg is the second !!!!
C!!!! It is not working for logitudinally polarised Z in the initial state (ifsign=-1), both as pdg or pdgp   !!!!                        

      if (ifsign.eq.-1.and.(pdg.eq.250.or.pdgp.eq.250)) then
        print*,"Error: Tpm invovling longitudinally polarised 
     .          Z not implemented for the initial state"
      endif

      if (hel.ne.0) then
        s_pdg = pdg*hel
      else
        s_pdg = pdg*ifsign
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_tpm = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_tpm = sign(1d0,dble(ifsign*pdg)) 

C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_tpm = sign(1d0,dble(ifsign*pdg)) 

C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_tpm = sign(1d0,dble(ifsign*pdg))

C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_tpm = sign(1d0,dble(ifsign*pdg))

      ! if it has already been set, then add the correct normalisation
      ! and return
      if (sdk_tpm.ne.0d0) then
        sdk_tpm = sdk_tpm / dsqrt(2d0*sw2)
        return
      endif

C goldstones, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251.and.pdgp.eq.25) sdk_tpm = sign(1d0,dble(s_pdg)) 
      if (abs(s_pdg).eq.251.and.pdgp.eq.250) sdk_tpm = CMPLX(0d0,-1d0)


c following last .and. conditions are not strictly necessary
      if (abs(s_pdg).eq.250.and.abs(pdgp*ifsign).eq.251) sdk_tpm =  CMPLX(0d0,1d0)
      if (abs(s_pdg).eq.25.and.abs(pdgp*ifsign).eq.251) sdk_tpm = sign(1d0,dble((pdgp*ifsign))) 

      if (sdk_tpm.ne.0d0) then
         sdk_tpm = sdk_tpm / (2d0 * dsqrt(sw2))
         return
      endif
      


C vector bosons
      if (abs(pdg*ifsign).eq.24.and.pdgp.eq.22) sdk_tpm = sign(1d0,dble(pdg*ifsign))
      if (pdg.eq.22.and.abs(pdgp*ifsign).eq.24) sdk_tpm = sign(1d0,dble(pdgp*ifsign))
      if (abs(pdg*ifsign).eq.24.and.pdgp.eq.23) sdk_tpm = -sign(1d0,dble(pdg*ifsign)) * dsqrt(cw2/sw2)
      if (pdg.eq.23.and.abs(pdgp*ifsign).eq.24) sdk_tpm = -sign(1d0,dble(pdgp*ifsign)) * dsqrt(cw2/sw2)


      return
      end



      double complex function sdk_t3_diag(pdg, hel, ifsign)
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

      sdk_t3_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_t3_diag = sign(0.5d0,dble(ifsign*pdg)) 

C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_t3_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_t3_diag = sign(0.5d0,dble(ifsign*pdg))

C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_t3_diag = -sign(0.5d0,dble(ifsign*pdg))

C goldstones, they behave like left handed leptons (charged); neutrals
C mix
      if (abs(s_pdg).eq.251) sdk_t3_diag = sign(0.5d0,dble(s_pdg)) 

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_t3_diag = sign(1d0,dble(ifsign*pdg))

      return
      end



      double complex function sdk_yo2_diag(pdg, hel, ifsign)
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
        s_pdg = pdg
      endif

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_yo2_diag = 0d0

C left handed neutrino / right handed antineutrino
      if (s_pdg.eq.-12.or.s_pdg.eq.-14.or.s_pdg.eq.-16) sdk_yo2_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C right handed lepton / left handed antilepton
      if (s_pdg.eq.11.or.s_pdg.eq.13.or.s_pdg.eq.15) sdk_yo2_diag = -sign(1d0,dble(ifsign*pdg))
C left handed lepton / right handed antilepton
      if (s_pdg.eq.-11.or.s_pdg.eq.-13.or.s_pdg.eq.-15) sdk_yo2_diag = -sign(0.5d0,dble(ifsign*pdg)) 

C right handed up quark / left handed antiup quark
      if (s_pdg.eq.2.or.s_pdg.eq.4.or.s_pdg.eq.6) sdk_yo2_diag = sign(2d0/3d0,dble(ifsign*pdg))
C left handed up quark / right handed antiup quark
      if (s_pdg.eq.-2.or.s_pdg.eq.-4.or.s_pdg.eq.-6) sdk_yo2_diag = sign(1d0/6d0,dble(ifsign*pdg))

C right handed down quark / left handed antidown quark
      if (s_pdg.eq.1.or.s_pdg.eq.3.or.s_pdg.eq.5) sdk_yo2_diag = -sign(1d0/3d0,dble(ifsign*pdg))
C left handed down quark / right handed antidown quark
      if (s_pdg.eq.-1.or.s_pdg.eq.-3.or.s_pdg.eq.-5) sdk_yo2_diag = sign(1d0/6d0,dble(ifsign*pdg))

C goldstones, they behave like left handed leptons (charged); neutrals
C mix
      if (abs(s_pdg).eq.251) sdk_yo2_diag = sign(0.5d0,dble(s_pdg))

      return
      end


      double complex function sdk_iz_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      include "coupl.inc"
      double precision sw2, cw2
      double complex sdk_t3_diag, sdk_charge

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      sdk_iz_diag = sdk_t3_diag(pdg,hel,ifsign) - sw2*sdk_charge(pdg,hel,ifsign)
      sdk_iz_diag = sdk_iz_diag / sqrt(sw2*cw2)

      return
      end


      double complex function sdk_iz_nondiag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      include "coupl.inc"
      double precision sw2, cw2

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      ! only works for the mixing chi/H
      sdk_iz_nondiag = 0d0
      if (pdg.eq.250.or.pdg.eq.25) then
          ! pdg (new) = 25
        if (pdg.eq.25) sdk_iz_nondiag = ifsign * dcmplx(0d0,-1d0) / 2d0
        if (pdg.eq.250) sdk_iz_nondiag = ifsign * dcmplx(0d0,1d0) / 2d0
        sdk_iz_nondiag = sdk_iz_nondiag / sqrt(sw2*cw2)
      endif

      return
      end


      double complex function sdk_ia_diag(pdg, hel, ifsign)
      implicit none
      integer pdg, hel, ifsign
      double complex sdk_charge

      sdk_ia_diag = -sdk_charge(pdg,hel,ifsign)

      return
      end



      double complex function sdk_iz2_diag(pdg, hel, ifsign)
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
      if (abs(s_pdg).eq.251) sdk_iz2_diag = (cw2-sw2)**2 / (4*sw2*cw2) 
      if (abs(s_pdg).eq.250) sdk_iz2_diag =1d0 / (4*sw2*cw2)
      if (abs(s_pdg).eq.25)  sdk_iz2_diag =1d0 / (4*sw2*cw2)  

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_iz2_diag = cw2 / sw2

      return
      end




      double complex function sdk_cew_diag(pdg, hel, ifsign)
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

C goldstones and Higgs, they behave like left handed leptons (charged) or neutrinos (neutrals)
      if (abs(s_pdg).eq.251) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)
      if (abs(s_pdg).eq.250) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)
      if (abs(s_pdg).eq.25) sdk_cew_diag = (1+2*cw2) / (4*sw2*cw2)

C transverse W boson
      if (abs(s_pdg).eq.24) sdk_cew_diag = 2 / sw2

C transverse Z boson
      if (abs(s_pdg).eq.23) sdk_cew_diag = 2 * cw2 / sw2

C (transverse) photon
      if (abs(s_pdg).eq.22) sdk_cew_diag = 2d0 

      return
      end


      double complex function sdk_cew_nondiag()
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
          invariants(i,j) = sumdot(p(0,i),p(0,j),dble(iflist(i)*iflist(j)))
          invariants(j,i) = invariants(i,j)
        enddo
      enddo

      return 
      end



      subroutine sdk_test_functions()
      implicit none
      ! performs consistency checks between the various functions
      integer npdgs
      parameter(npdgs=33)
      integer pdg_list(npdgs)
      data pdg_list /-251,-24,-16,-15,-14,-13,-12,-11,
     %               -6,-5,-4,-3,-2,-1,1,2,3,4,5,6,
     %               11,12,13,14,15,16,21,22,23,24,25,
     %               250,251/
      integer ihel, ifsign
      integer i

      include "coupl.inc"
      double precision sw2, cw2

      double complex sdk_charge, sdk_t3_diag, sdk_yo2_diag, sdk_iz2_diag
      double complex q, t3, yo2, iz2 

      cw2 = mdl_mw**2 / mdl_mz**2
      sw2 = 1d0 - cw2

      do ifsign = -1, 1, 2 
        do ihel = -1, 1, 2 
          do i = 1,npdgs
            ! t3-y-q relation
            write(*,*) 'Q=t3+yo2_diag'
            q = sdk_charge(pdg_list(i), ihel, ifsign)
            t3 = sdk_t3_diag(pdg_list(i), ihel, ifsign) 
            yo2 = sdk_yo2_diag(pdg_list(i),ihel, ifsign)
            if (abs(q - (t3+yo2)).gt.1d-4) then
              write(*,*) 'WRONG', pdg_list(i), ihel, ifsign,
     %                 q, t3, yo2
            endif
            
            ! t3-q-iz2 relation
            write(*,*) 'IZ=t3-sw Q / sw cw'
            iz2 = sdk_iz2_diag(pdg_list(i),ihel, ifsign)

            if (abs(iz2 - (t3-sw2*q)**2/sw2/cw2).gt.1d-4) then
              write(*,*) 'WRONG', pdg_list(i), ihel, ifsign,
     %                 q, t3, (t3-sw2*q)**2/sw2/cw2, iz2
            endif
          enddo
        enddo
      enddo

      return
      end
