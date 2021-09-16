      double precision function pdg2pdf_timed(ih,ipdg,ibeam,x,xmu)
c        function
         double precision pdg2pdf
         external pdg2pdf

c        argument

         integer ih, ipdg, ibeam
         DOUBLE  PRECISION x,xmu

c timing statistics
         include "timing_variables.inc"

         call cpu_time(tbefore)
         pdg2pdf_timed = pdg2pdf(ih,ipdg,ibeam,x,xmu)
         call cpu_time(tAfter)
         tPDF = tPDF + (tAfter-tBefore)
         return

      end
      
      double precision function pdg2pdf(ih,ipdg,ibeam,x,xmu)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c     ih is now signed <0 for antiparticles
c     if ih<0 does not have a dedicated pdf, then the one for ih>0 will be called
c     and the sign of ipdg flipped accordingly.
c
c     ibeam is the beam identity 1/2
c      if set to -1/-2 it meand that ipdg should not be flipped even if ih<0
c      usefull for re-weighting
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg, ibeam
C
C     Include
C
      include 'pdf.inc'
C      
      double precision Ctq3df,Ctq4Fn,Ctq5Pdf,Ctq6Pdf,Ctq5L
      integer mode,Irt,i,j,i_replace,ii
      double precision xlast(20),xmulast(20),pdflast(-7:7,20),q2max
      character*7 pdlabellast(20)
      double precision epa_electron,epa_proton
      integer ipart,ireuse,iporg,ihlast(20)
      save xlast,xmulast,pdflast,pdlabellast,ihlast
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/300*-99d9/
      data pdlabellast/20*'abcdefg'/
      data ihlast/20*-99/
      data i_replace/20/
C dressed lepton stuff
      include '../eepdf.inc'
      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee


      integer i_ee, ih_local
      double precision compute_eepdf
      double precision tolerance
      parameter (tolerance=1.d-2)
      if (ih.eq.0) then
c     Lepton collisions (no PDF). 
         pdg2pdf=1d0
         return
      endif
c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return pdg2pdf=0
c     instead of stopping the code, as this might accidentally happen.
      if (x.eq.0d0) then
         pdg2pdf=0d0
         return
      elseif (x.lt.0d0 .or. x.gt.1d0) then
       if (x-1d0.lt.tolerance) then
         x=1d0
       else
         write (*,*) 'PDF not supported for Bjorken x ', x
         open(unit=26,file='../../../error',status='unknown')
         write(26,*) 'Error: PDF not supported for Bjorken x ',x
         stop 1
       endif
      endif


C     dressed leptons so force lpp to be 3/4 (electron/muon beam)
C      and check that it is not a photon initial state --elastic photon is handle below --
      if ((abs(ih).eq.3.or.abs(ih).eq.4).and.ipdg.ne.22) then
c         if (ibeam.lt.0) then
c            ipart=sign(1,ih)*ipdg
c         else
            ipart = ipdg
c         endif
c          ! change e/mu/tau = 8/9/10 to 11/13/15 


         if (abs(ipart).eq.8) then
          ipart = sign(1,ipart) * 11
        else if (abs(ipart).eq.9) then
          ipart = sign(1,ipart) * 13
        else if (abs(ipart).eq.10) then
          ipart = sign(1,ipart) * 15
        endif
        if (ibeam.lt.0) then
           ih_local = ipart
        elseif (abs(ih) .eq.3) then
           ih_local = sign(1,ih) * 11
        else if (abs(ih) .eq.4) then
           ih_local = sign(1,ih) * 13
        else
           write(*,*) "not supported beam type"
           stop 1
        endif

        pdg2pdf = 0d0
        do i_ee = 1, n_ee 
           ee_components(i_ee) = compute_eepdf(x, omx_ee(ibeam), xmu,i_ee,ipart,ih_local)
        enddo
c           write(*,*) x, omx_ee(ibeam), xmu,i_ee,ipart,ih_local,  ee_components(1)
        return
      endif

      if (ibeam.gt.0) then
         ipart=sign(1,ih)*ipdg
      else
         ipart = ipdg
      endif
      
      if(iabs(ipart).eq.21) then
         ipart=0
      else if(iabs(ipart).eq.22) then
         ipart=7
      else if(iabs(ipart).eq.7) then
         ipart=7
      else if(iabs(ipart).gt.7)then
c     This will be called for any PDG code, but we only support up to 7
C         write(*,*) 'PDF not supported for pdg ',ipdg
C         write(*,*) 'For lepton colliders, please set the lpp* '//
C     $    'variables to 0 in the run_card'  
C         open(unit=26,file='../../../error',status='unknown')
C         write(26,*) 'Error: PDF not supported for pdg ',ipdg
C         stop 1
         pdg2pdf=0d0
         return
      endif

      iporg=ipart
      ireuse = 0
      ii=i_replace
      do i=1,20
c     Check if result can be reused since any of last twenty
c     calls. Start checking with the last call and move back in time
         if (ih.eq.ihlast(ii)) then
            if (x.eq.xlast(ii)) then
               if (xmu.eq.xmulast(ii)) then
                  if (pdlabel.eq.pdlabellast(ii)) then
                     ireuse = ii
                     exit
                  endif
               endif
            endif
         endif
         ii=ii-1
         if (ii.eq.0) ii=ii+20
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0) then
         if (pdflast(iporg,ireuse).ne.-99d9) then
            pdg2pdf=pdflast(iporg,ireuse)
            return 
         endif
      endif

c Calculated a new value: replace the value computed longest ago.
      i_replace=mod(i_replace,20)+1
      
c     Give the current values to the arrays that should be
c     saved. 'pdflast' is filled below.
      xlast(i_replace)=x
      xmulast(i_replace)=xmu
      pdlabellast(i_replace)=pdlabel
      ihlast(i_replace)=ih

      if(iabs(ipart).eq.7.and.abs(ih).gt.1) then
         q2max=xmu*xmu
         if(abs(ih).eq.3) then       !from the electron
            pdg2pdf=epa_electron(x,q2max)
         elseif(abs(ih) .eq. 2) then !from a proton without breaking
            pdg2pdf=epa_proton(x,q2max)
         endif 
         pdflast(iporg,i_replace)=pdg2pdf
         return
      endif

c The actual call to the PDFs (in Source/PDF/pdf.f)
      call pftopdg(abs(ih),x,xmu,pdflast(-7,i_replace))
      pdg2pdf=pdflast(iporg,i_replace)
      return
      end


      double precision function get_ee_expo()
      ! return the exponent used in the
      ! importance-sampling transformation to sample
      ! the Bjorken x's
      implicit none
      double precision expo
      parameter (expo=0.96d0)
      get_ee_expo = expo
      return
      end




      double precision function compute_eepdf(x,omx_ee, xmu, n_ee, id, idbeam)
      implicit none
      double precision x, xmu
      integer n_ee, id, idbeam

      double precision xmu2
      double precision k_exp

      double precision eps
      parameter (eps=1e-20)

      double precision eepdf_tilde, eepdf_tilde_power
      double precision get_ee_expo
      double precision ps_expo

      double precision omx_ee

      if (id.eq.7) then
        compute_eepdf = 0d0
        return
      endif

      xmu2=xmu**2

      compute_eepdf = eepdf_tilde(x,xmu2,n_ee,id,idbeam) 
      ! this does not include a factor (1-x)^(-kappa)
      ! where k is given by
      k_exp = eepdf_tilde_power(xmu2,n_ee,id,idbeam)
      ps_expo = get_ee_expo()

      if (k_exp.gt.ps_expo) then
          write(*,*) 'WARNING, e+e- exponent exceeding limit', k_exp, ps_expo
      endif

      compute_eepdf = compute_eepdf * (omx_ee)**(-k_exp+ps_expo)

      return
      end



      double precision function ee_comp_prod(comp1, comp2)
      ! compute the scalar product for the two array
      ! of eepdf components
      implicit none
      include '../eepdf.inc'
      double precision comp1(n_ee), comp2(n_ee)
      integer i

      ee_comp_prod = 0d0
      do i = 1, n_ee
        ee_comp_prod = ee_comp_prod + comp1(i) * comp2(i)
      enddo
      return
      end




