      double precision function pdg2pdf(ih,ipdg,beamid,x,xmu)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg
      integer beamid
C
C     Include
C
      include 'pdf.inc'
C
      integer nb_proton(2)
      integer nb_neutron(2)
      common/to_heavyion_pdg/ nb_proton, nb_neutron
      integer nb_hadron

C      
      double precision get_ion_pdf
      integer i,j,ihlast(20),ipart,iporg,ireuse,imemlast(20),iset,imem
     &     ,i_replace,ii,ipartlast(20)
      double precision xlast(20),xmulast(20),pdflast(-7:7,20)
      double precision epa_proton, epa_lepton
      save ihlast,xlast,xmulast,pdflast,imemlast,ipartlast
      data ihlast/20*-99/
      data ipartlast/20*-99/
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/300*-99d9/
      data imemlast/20*-99/
      data i_replace/20/

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

      nb_hadron = (nb_proton(iabs(beamid))+nb_neutron(iabs(beamid)))
c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return pdg2pdf=0
c     instead of stopping the code, as this might accidentally happen.
      if (x.eq.0d0) then
         pdg2pdf=0d0
         return
      elseif (x.lt.0d0 .or. (x*nb_hadron).gt.1d0) then
         if(nb_hadron.eq.1.or.x.lt.0d0)then
            write (*,*) 'PDF not supported for Bjorken x ', x*nb_hadron
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF not supported for Bjorken x ',x*nb_hadron
            stop 1
         else
            pdg2pdf=0d0
            return
         endif
      endif

c     If group_subprocesses is true, then IH=abs(lpp) and ipdg=ipdg*sgn(lpp) in export_v4.
c     For EVA,  group_subprocesses is false and IH=LPP and ipdg are passed, instead.
c     If group_subprocesses is false, the following sets ipdg=ipdg*sgn(IH) if not in EVA
      if(pdsublabel(iabs(beamid)).eq.'eva') then
         ipart=ipdg
      else 
         ipart=ipdg*ih/iabs(ih)
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
c      else if(iabs(ipart).gt.7)then
c         write(*,*) 'PDF not supported for pdg ',ipdg
c         write(*,*) 'For lepton colliders, please set the lpp* '//
c     $    'variables to 0 in the run_card'  
c         open(unit=26,file='../../../error',status='unknown')
c         write(26,*) 'Error: PDF not supported for pdg ',ipdg
c         stop 1
      endif
      
      if(pdsublabel(iabs(beamid)).eq.'eva') then
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
c              q2max = xmu*xmu
            ievo = ievo_eva
            hel      = GET_NHEL(HEL_PICKED, beamid) ! helicity of v
            helMulti = GET_NHEL(0, beamid)          ! helicity multiplicity of v to undo spin averaging
            pdg2pdf  = helMulti*eva_get_pdf_by_PID(ipart,ppid,hel,fLpol,x,xmu*xmu,ievo)
            return
         endif
      else
         if(iabs(ipart).eq.24.or.iabs(ipart).eq.23) then  ! w/z
            write(*,*) 'LHAPDF not supported for pdg ',ipdg
            write(*,*) 'For EVA, check if pdlabel and pdsublabel* '//
     $    'are set correctly in the run_card'  
            open(unit=26,file='../../../error',status='unknown')
            write(26,*) 'Error: PDF not supported for pdg ',ipdg
            stop 1
         endif
      endif

      iporg=ipart
c     Determine the iset used in lhapdf
      call getnset(iset)
      if (iset.ne.1) then
         write (*,*) 'PDF not supported for Bjorken x ', x
         open(unit=26,file='../../../error',status='unknown')
         write(26,*) 'Error: PDF not supported for Bjorken x ',x
         stop 1
      endif

c     Determine the member of the set (function of lhapdf)
      call getnmem(iset,imem)

      ireuse = 0
      ii=i_replace
      do i=1,20
         if (abs(ipart).gt.7)then
            exit
         endif
c     Check if result can be reused since any of last twenty
c     calls. Start checking with the last call and move back in time
         if (ih.eq.ihlast(ii)) then
            if (ipart.eq.ipartlast(ii)) then
               if (x*nb_hadron.eq.xlast(ii)) then
                  if (xmu.eq.xmulast(ii)) then
                     if (imem.eq.imemlast(ii)) then
                        ireuse = ii
                        exit
                     endif
                  endif
               endif
            endif
         endif
         ii=ii-1
         if (ii.eq.0) ii=ii+20
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0.and.abs(ipart).le.7) then
         if (pdflast(ipart,ireuse).ne.-99d9) then
            pdg2pdf = get_ion_pdf(pdflast(-7,ireuse), ipart, nb_proton(iabs(beamid)), nb_neutron(iabs(beamid)))/x
            return 
         endif
      endif

c Calculated a new value: replace the value computed longest ago
      i_replace=mod(i_replace,20)+1

c     Call lhapdf and give the current values to the arrays that should
c     be saved
      if(iabs(ih).eq.1) then
         if (nb_proton(iabs(beamid)).eq.1.and.nb_neutron(iabs(beamid)).eq.0) then
            call evolvepart(ipart,x,xmu,pdg2pdf)
            if (abs(ipart).le.7)   pdflast(ipart, i_replace)=pdg2pdf
         else
            if (ipart.eq.1.or.ipart.eq.2) then
               call evolvepart(1,x*nb_hadron
     &                         ,xmu,pdflast(1, i_replace))
               call evolvepart(2,x*nb_hadron
     &                         ,xmu,pdflast(2, i_replace))
            else if (ipart.eq.-1.or.ipart.eq.-2)then
               call evolvepart(-1,x*nb_hadron
     &                         ,xmu,pdflast(-1, i_replace))
               call evolvepart(-2,x*nb_hadron
     &                         ,xmu,pdflast(-2, i_replace))
            else
               call evolvepart(ipart,x*nb_hadron
     &                         ,xmu,pdflast(ipart, i_replace))
            endif 
            pdg2pdf = get_ion_pdf(pdflast(-7, i_replace), ipart, nb_proton(iabs(beamid)), nb_neutron(iabs(beamid)))
         endif
         pdg2pdf=pdg2pdf/x
      else if(iabs(ih).eq.3.or.iabs(ih).eq.4) then       !from the electron
            pdg2pdf=epa_lepton(x,xmu*xmu, iabs(ih))
      else if(iabs(ih).eq.2) then ! photon from a proton without breaking
          pdg2pdf = epa_proton(x,xmu*xmu, beamid)

      else
         write (*,*) 'beam type not supported in lhadpf'
         stop 1
      endif
      xlast(i_replace)=x*nb_hadron
      xmulast(i_replace)=xmu
      ihlast(i_replace)=ih
      imemlast(i_replace)=imem
c
      return
      end

      double precision function get_ee_expo()
      ! return the exponent used in the
      ! importance-sampling transformation to sample
      ! the Bjorken x's
      implicit none
      stop 21
      return
      end

      double precision function compute_eepdf(x,omx_ee, xmu, n_ee, id, idbeam)
      implicit none
      double precision x, xmu, omx_ee(*)
      integer n_ee, id, idbeam
      stop 21
      return
      end

      double precision function ee_comp_prod(comp1, comp2)
      ! compute the scalar product for the two array
      ! of eepdf components
      implicit none
      double precision comp1(*), comp2(*)
      stop 21
      return
      end
