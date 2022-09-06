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
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg,ibeam
C
C     Include
C
      include 'pdf.inc'
C      
      double precision Ctq3df,Ctq4Fn,Ctq5Pdf,Ctq6Pdf,Ctq5L
      integer mode,Irt,i,j,i_replace,ii
      double precision xlast(20),xmulast(20),pdflast(-7:7,20),q2max
      character*7 pdlabellast(20)
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
      integer i_ee
      double precision call_epdf_bs, call_epdf_nobs
      double precision tolerance
      parameter (tolerance=1.d-2)

      logical photons_from_lepton
      common /to_afromee/ photons_from_lepton
C  PDFs with beamstrahlung use specific initialisation/evaluation
      logical has_bstrahl
      common /to_has_bs/ has_bstrahl

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee

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

C     dressed leptons
      if (abs(ih).eq.3) then
        ! change e/mu/tau = 8/9/10 to 11/13/15 
        if (abs(ipdg).eq.8) then
          ipart = sign(1,ipdg) * 11
        else if (abs(ipdg).eq.9) then
          ipart = sign(1,ipdg) * 13
        else if (abs(ipdg).eq.10) then
          ipart = sign(1,ipdg) * 15
        else 
          ipart = ipdg
        endif

        !Kill the photon if asked for
        if ((ipart.eq.22.or.ipart.eq.7).and..not.photons_from_lepton) then
            pdg2pdf=0d0
            ee_components(:)=0d0
            return
        endif

        pdg2pdf = 0d0
        if (.not.has_bstrahl) then
          ! pure ISR case
          ! we pass ih/abs(ih)*ipart as PDG id because
          ! the eMELA/ePDF convention always refers as an electron beam
          ! Note that the photon from the positron will have -7!!
          ee_components(1) = call_epdf_nobs(x,omx_ee(ibeam),xmu,ih/abs(ih)*ipart)
          ee_components(2:n_ee) = 0d0
        else
          ! case with beamstrahlung. This case may not be symmetric
          ! because of different beam parameters for e+ and e-; need to
          ! pass both ih (transformed to +-1) and ipart
          do i_ee = 1, n_ee 
            ! we pass ih/abs(ih)*ipart as PDG id because
            ! the eMELA/ePDF convention always refers as an electron beam
            ! Note that the photon from the positron will have -7!!
            ee_components(i_ee) = call_epdf_bs(x,omx_ee(ibeam),xmu,i_ee,abs(ih)/ih,ipart)
          enddo
        endif
      endif

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




      double precision function call_epdf_nobs(x, omx, xmu, id)
      implicit none
      double precision x, omx, xmu
      integer id

      double precision xmu2
      double precision k_exp

      double precision eps
      parameter (eps=1e-20)

      double precision eepdf_tilde, eepdf_tilde_power
      double precision get_ee_expo
      double precision ps_expo

C ePDF/eMELA specific parameters
      integer id_epdf

C  PDFs with beamstrahlung use specific initialisation/evaluation
      logical has_bstrahl
      common /to_has_bs/ has_bstrahl

      xmu2=xmu**2

      if (id.eq.11) then
          ! e+ in e+ / e- in e-
          id_epdf = 11
      else if (iabs(id).eq.7.or.abs(id).eq.22) then
          ! photon in e+/e-; 
          !  the abs comes because photon from positron has -7
          id_epdf = 22
      else
          ! this is tipically the case of e+ in e-, quarks in e-, etc
          ! for which we will return zero
          id_epdf = -1
      endif

      if (id_epdf.eq.-1) then
          call_epdf_nobs = 0d0
          return
      endif

      ps_expo = get_ee_expo()

      ! the elpdfq2 call will return the pdf without the singular factor
      !  1/(1-x)^ps_expo, which is taken into account by the
      !  phase-space parameterization
      ! The first argument is 0 to use grids, 1 to evaluate the PDF
      ! The second argument is 11 electron, 22 photon, -11 positron
      ! assuming an electron as incoming hadron
      call elpdfq2(0,id_epdf,x,omx,xmu2,1d0-ps_expo,call_epdf_nobs)

      return
      end


      double precision function call_epdf_bs(x, omx, xmu, i_ee, ih, id)
      implicit none
      double precision x, omx, xmu
      integer id, i_ee, ih
      ! i_ee is the BS component
      ! ih is +-1 for electron/positron
      ! ipart is the pdg id of the parton

      double precision xmu2
      double precision k_exp

      double precision eps
      parameter (eps=1e-20)

      double precision eepdf_tilde, eepdf_tilde_power
      double precision get_ee_expo
      double precision ps_expo

C ePDF/eMELA specific parameters
      integer id_epdf

C  PDFs with beamstrahlung use specific initialisation/evaluation
      logical has_bstrahl
      common /to_has_bs/ has_bstrahl

      xmu2=xmu**2

      if (id*ih.eq.11) then
          ! e+ in e+ / e- in e-
          id_epdf = id
      else if (id.eq.7.or.id.eq.22) then
          ! photon in e+/e-; 
          !  the abs comes because photon from positron has -7
          id_epdf = 22
      else
          ! this is tipically the case of e+ in e-, quarks in e-, etc
          ! for which we will return zero
          id_epdf = -1
      endif

      if (id_epdf.eq.-1) then
          call_epdf_bs = 0d0
          return
      endif

      ps_expo = get_ee_expo()

      ! the elpdfq2 call will return the pdf without the singular factor
      !  1/(1-x)^ps_expo, which is taken into account by the
      !  phase-space parameterization
      ! In this case the frist argument is the BS component
      ! the second is the parton id 
      ! the third is the beam_id (+-1 for electron/positron)
      call bs_elpdfq2(i_ee,id_epdf,ih,x,omx,xmu2,1d0-ps_expo,call_epdf_bs) 

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
