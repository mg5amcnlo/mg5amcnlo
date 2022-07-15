      double precision function pdg2pdf(ih,ipdg,beamid,x,xmu)
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
      INTEGER IH,ipdg
      integer beamid            ! 1 or 2 (for left or right beam)
C                                -1/-2  same as 1/2 but no change on ipdg needed
C
C     Include
C
      include 'pdf.inc'
C dressed lepton stuff
      include '../eepdf.inc'
      integer i_ee, ih_local

      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee

      double precision compute_eepdf
      double precision tolerance
      parameter (tolerance=1.d-2)
c
      
      double precision tmp1, tmp2
      integer nb_proton(2), nb_neutron(2) 
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      integer nb_hadron
C      

      double precision Ctq6Pdf, get_ion_pdf
      integer mode,Irt,i,j
      double precision xlast(2),xmulast(2),pdflast(-7:7,2),q2max
      character*7 pdlabellast(2)
      double precision epa_lepton,epa_proton
      integer ipart,ireuse,iporg,ihlast(2)
      save xlast,xmulast,pdflast,pdlabellast,ihlast
      data xlast/2*-99d9/
      data xmulast/2*-99d9/
      data pdflast/30*-99d9/
      data pdlabellast/2*'abcdefg'/
      data ihlast/2*-99/

c     effective w/z/a approximation (leading log fixed order, not resummed)
      double precision eva_get_pdf_by_PID
      external eva_get_pdf_by_PID
      integer ppid
      integer ievo,ievo_eva
      common/to_eva/ievo_eva
      integer hel,helMulti,hel_picked
      double precision hel_jacobian
      common/hel_picked/hel_picked,hel_jacobian
      integer get_nhel
      external get_nhel
      real*8 pol(2),fLPol
      common/to_polarization/pol


c     collider configuration
      integer lpp(2)
      double precision ebeam(2),xbk(2),q2fact(2)
      common/to_collider/ebeam,xbk,q2fact,lpp

      if (iabs(ih).eq.9) then
         pdg2pdf = 1d0
         return
      endif

      nb_hadron = (nb_proton(iabs(beamid))+nb_neutron(iabs(beamid)))
c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return pdg2pdf=0
c     instead of stopping the code, as this might accidentally happen.
      if (x.eq.0d0) then
         pdg2pdf=0d0
         return
      elseif (x.lt.0d0 .or. (x*nb_hadron).gt.1d0) then
         if (nb_hadron.eq.1.or.x.lt.0d0) then
            write (*,*) 'PDF not supported for Bjorken x ', x*nb_hadron
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF not supported for Bjorken x ',x*nb_hadron
            stop 1
         else
            pdg2pdf = 0d0
            return
         endif
      endif

C     dressed leptons so force lpp to be 3/4 (electron/muon beam)
C      and check that it is not a photon initial state --elastic photon is handle below --
      if (pdlabel.eq.'dressed')then
c       change e/mu/tau = 8/9/10 to 11/13/15
        ipart = ipdg
        if (abs(ipart).eq.8) then
          ipart = sign(1,ipart) * 11
        else if (abs(ipart).eq.9) then
          ipart = sign(1,ipart) * 13
        else if (abs(ipart).eq.10) then
          ipart = sign(1,ipart) * 15
        endif
	pdg2pdf = 0d0

        if (beamid.lt.0) then
           ih_local = ipart
        elseif (abs(ih) .eq.3) then
           ih_local = sign(1,ih) * 11
        else if (abs(ih) .eq.4) then
           ih_local = sign(1,ih) * 13
        else
           write(*,*) "not supported beam type"
           stop 1
        endif
        do i_ee = 1, n_ee
          ee_components(i_ee) = compute_eepdf(x,omx_ee(iabs(beamid)),xmu,i_ee,ipart,ih_local)
	enddo
        pdg2pdf =  ee_components(1) ! temporary to test pdf load
c        write(*,*), x, beamid ,omx_ee(iabs(beamid)),xmu,1,ipart,ih_local,pdg2pdf
        return
      endif
      

c     If group_subprocesses is true, then IH=abs(lpp) and ipdg=ipdg*sgn(lpp) in export_v4.
c     For EVA,  group_subprocesses is false and IH=LPP and ipdg are passed, instead.
c     If group_subprocesses is false, the following sets ipdg=ipdg*sgn(IH) if not in EVA
      if(pdlabel.eq.'eva'.or.pdsublabel(iabs(beamid)).eq.'eva') then
         ipart=ipdg
      else
         ipart=sign(1,ih)*ipdg
      endif      

      if(iabs(ipart).eq.21) then ! g      
         ipart=0
c      else if(ipart.eq.12) then ! ve
c         ipart=12
c      else if(ipart.eq.-12) then ! ve~
c         ipart=-12
c      else if(ipart.eq.14) then ! vm
c         ipart=14
c      else if(ipart.eq.-14) then ! vm~
c         ipart=-14
      else if(ipart.eq.24) then  ! w+
         ipart=24
      else if(ipart.eq.-24) then ! w-
         ipart=-24
      else if(iabs(ipart).eq.23) then ! z
         ipart=23
      else if(iabs(ipart).eq.22) then ! a
         ipart=7
      else if(iabs(ipart).eq.7) then  ! a
         ipart=7
c     This will be called for any PDG code. We only support (for now) 0-7, and 22-24
      else if(iabs(ipart).gt.7)then
         write(*,*) 'PDF not supported for pdg ',ipdg
         write(*,*) 'For lepton colliders, please set the lpp* '//
     $    'variables to 0 in the run_card current is' , ih  
         open(unit=26,file='../../../error',status='unknown')
         write(26,*) 'Error: PDF not supported for pdg ',ipdg
         stop 1
      endif

      iporg=ipart
      ireuse = 0
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (x*nb_hadron.eq.xlast(i) .and. xmu.eq.xmulast(i) .and.
     $        pdlabel.eq.pdlabellast(i) .and. ih.eq.ihlast(i)) then
            ireuse = i
         endif
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0.and.iabs(iporg).lt.8)then
         if (pdflast(iporg,ireuse).ne.-99d9) then
            pdg2pdf = get_ion_pdf(pdflast(-7, ireuse), iporg, nb_proton(iabs(beamid)),
     $                         nb_neutron(iabs(beamid)))
            return 
         endif
      endif

c     Bjorken x and/or facrorization scale and/or PDF set are not
c     identical to the saved values: this means a new event and we
c     should reset everything to compute new PDF values. Also, determine
c     if we should fill ireuse=1 or ireuse=2.
      if (ireuse.eq.0.and.xlast(1).ne.-99d9.and.xlast(2).ne.-99d9)then
         do i=1,2
            xlast(i)=-99d9
            xmulast(i)=-99d9
            do j=-7,7
               pdflast(j,i)=-99d9
            enddo
            pdlabellast(i)='abcdefg'
            ihlast(i)=-99
         enddo
c     everything has been reset. Now set ireuse=1 to fill the first
c     arrays of saved values below
         ireuse=1
      else if(ireuse.eq.0.and.xlast(1).ne.-99d9)then
c     This is first call after everything has been reset, so the first
c     arrays are already filled with the saved values (hence
c     xlast(1).ne.-99d9). Fill the second arrays of saved values (done
c     below) by setting ireuse=2
         ireuse=2
      else if(ireuse.eq.0)then
c     Special: only used for the very first call to this function:
c     xlast(i) are initialized as data statements to be equal to -99d9
         ireuse=1
      endif

c     Give the current values to the arrays that should be
c     saved. 'pdflast' is filled below.
      xlast(ireuse)=x*nb_hadron
      xmulast(ireuse)=xmu
      pdlabellast(ireuse)=pdlabel
      ihlast(ireuse)=ih

      if(pdlabel.eq.'eva'.or.pdsublabel(iabs(beamid)).eq.'eva') then
         if(iabs(ipart).ne.7.and.
c     &      iabs(ipart).ne.12.and.
c     &      iabs(ipart).ne.14.and.     
     &      iabs(ipart).ne.23.and.
     &      iabs(ipart).ne.24 ) then
            write(*,*) 'ERROR: EVA PDF only supported for A/Z/W, not for pdg = ',ipart
            stop 1
         else
c         write(*,*) 'running eva'
            select case (iabs(ih))
            case (0:2)
               write(*,*) 'ERROR: EVA PDF only supported for e+/- and mu+/- beams, not for lpp/ih=',ih
               stop 24
            case (3) ! e+/-
               ppid = 11
            case (4) ! mu+/-
               ppid = 13
            case default
               write(*,*) 'ERROR: EVA PDF only supported for e+/- and mu+/- beams, not for lpp/ih=',ih
               stop 24
            end select
            ppid  = ppid * ih/iabs(ih) ! get sign of parent
            fLPol = pol(iabs(beamid))        ! see setrun.f for treatment of polbeam*
            q2max = xmu*xmu
            ievo = ievo_eva
            hel      = GET_NHEL(HEL_PICKED, beamid) ! helicity of v
            helMulti = GET_NHEL(0, beamid)          ! helicity multiplicity of v to undo spin averaging
            pdg2pdf  = helMulti*eva_get_pdf_by_PID(ipart,ppid,hel,fLpol,x,q2max,ievo)
            return
         endif
      else ! this ensure backwards compatibility
         if(iabs(ipart).eq.7.and.iabs(ih).gt.1) then
            q2max=xmu*xmu
            if(iabs(ih).eq.3.or.iabs(ih).eq.4) then       !from the electron or muonn
               pdg2pdf=epa_lepton(x,q2max, iabs(ih))
            elseif(iabs(ih) .eq. 2) then !from a proton without breaking
               pdg2pdf=epa_proton(x,q2max,beamid)
            endif 
            pdflast(iporg,ireuse)=pdg2pdf
            return
         endif         
      endif



      
      if (pdlabel(1:5) .eq. 'cteq6') then
C        Be carefull u and d are flipped inside cteq6
         if (nb_proton(iabs(beamid)).gt.1.or.nb_neutron(iabs(beamid)).ne.0)then
            if (ipart.eq.1.or.ipart.eq.2)then
               pdflast(1,ireuse)=Ctq6Pdf(2,x*nb_hadron,xmu) ! remember u/d flipping in cteq
               pdflast(2,ireuse)=Ctq6Pdf(1,x*nb_hadron,xmu)
               pdg2pdf = get_ion_pdf(pdflast(-7,ireuse), ipart, nb_proton(iabs(beamid)), nb_neutron(iabs(beamid)))
            else if (ipart.eq.-1.or.ipart.eq.-2)then
               pdflast(-1,ireuse)=Ctq6Pdf(-2,x*nb_hadron,xmu) ! remember u/d flipping in cteq
               pdflast(-2,ireuse)=Ctq6Pdf(-1,x*nb_hadron,xmu)
               pdg2pdf = get_ion_pdf(pdflast(-7,ireuse), ipart, nb_proton(iabs(beamid)), nb_neutron(iabs(beamid)))
            else
               pdflast(ipart,ireuse)=Ctq6Pdf(ipart,x*nb_hadron,xmu)
               pdg2pdf = get_ion_pdf(pdflast(-7,ireuse), ipart, nb_proton(iabs(beamid)), nb_neutron(iabs(beamid)))
            endif 
         else
            if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $           ipart=sign(3-iabs(ipart),ipart)
            pdg2pdf=Ctq6Pdf(ipart,x,xmu)
            pdflast(iporg,ireuse)=pdg2pdf
         endif
      else
         call pftopdg(iabs(ih),x*nb_hadron,xmu,pdflast(-7,ireuse))
         pdg2pdf = get_ion_pdf(pdflast(-7, ireuse), iporg, nb_proton(iabs(beamid)),
     $                         nb_neutron(iabs(beamid)))
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
      include 'eepdf.inc'
      double precision comp1(n_ee), comp2(n_ee)
      integer i

      ee_comp_prod = 0d0
      do i = 1, n_ee
        ee_comp_prod = ee_comp_prod + comp1(i) * comp2(i)
      enddo
      return
      end



     
