      double precision function pdg2pdf_timed(ih,ipdg,x,xmu)
c        function
         double precision pdg2pdf
         external pdg2pdf

c        argument

         integer ih, ipdg
         DOUBLE  PRECISION x,xmu

c timing statistics
         include "timing_variables.inc"

         call cpu_time(tbefore)
         pdg2pdf_timed = pdg2pdf(ih,ipdg,x,xmu)
         call cpu_time(tAfter)
         tPDF = tPDF + (tAfter-tBefore)
         return

      end
      
      double precision function pdg2pdf(ih,ipdg,x,xmu)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg
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
      integer ee_ibeam
      common /to_ee_ibeam/ee_ibeam
      double precision call_epdf
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


C     dressed leptons
      if (abs(ih).eq.4) then
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

        ! MZ Kill the photon
        if (abs(ipart).ne.11) then
            pdg2pdf=0d0
            ee_components(:)=0d0
            return
        endif

        pdg2pdf = 0d0
        do i_ee = 1, n_ee 
          ee_components(i_ee) = call_epdf(x,xmu,i_ee,ipart,ee_ibeam)
        enddo
        return
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




      double precision function call_epdf(x, xmu, n_ee, id, idbeam)
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
      common /to_ee_omx/omx_ee

C ePDF specific parameters
      integer base, sol
      logical use_grid
      integer id_epdf, idbeam_epdf

      base = 0 ! physical base: e-, gamma, e+
      sol = 0 ! 0: numerical+matching, 1: only numerical                   
      use_grid = .true.

      if (id.eq.7) then
        call_epdf = 0d0
        return
      endif

      xmu2=xmu**2

      ! at the moment we support only the first component (no beamstrahlung)
      if (n_ee.ne.1) then
          call_epdf = 0d0
          return
      endif

      if (id.eq.idbeam.and.abs(id).eq.11) then
          id_epdf = 0
          idbeam_epdf = 0
      else
          write(*,*) 'CALL EPDF not implemented', id, idbeam
      endif

      ps_expo = get_ee_expo()
      ! the pdfq will return the pdf without the singular factor
      !  1/(1-x)^ps_expo, which is taken into account by the
      !  phase-space parameterization
      call pdfq(id_epdf,x,omx_ee,xmu,1d0-ps_expo,idbeam_epdf,base,sol,use_grid,call_epdf) 


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



      subroutine store_ibeam_ee(ibeam)
      implicit none
      ! just store the identity of beam ibeam 
      ! in the common to_ee_ibeam using the information
      ! from initial_states_map
      integer ibeam

      integer beams(2), idum
      logical firsttime
      data firsttime /.true./

      integer ee_ibeam
      common /to_ee_ibeam/ee_ibeam

      double precision omx1_ee, omx2_ee
      common /to_ee_omx1/ omx1_ee, omx2_ee

      double precision omx_ee
      common /to_ee_omx/omx_ee

      save beams

      if (firsttime) then
        open (unit=71,status='old',file='initial_states_map.dat')
        read (71,*)idum,idum,beams(1),beams(2)
        close (71)
        firsttime = .false.
      endif

      ee_ibeam = beams(ibeam)

      if (ibeam.eq.1) omx_ee = omx1_ee
      if (ibeam.eq.2) omx_ee = omx2_ee

      return
      end


