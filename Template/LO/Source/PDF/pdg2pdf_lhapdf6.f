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
      double precision tmp1, tmp2
      integer nb_proton1, nb_proton2
      integer nb_neutron1, nb_neutron2
      common/to_heavyion_pdg/ nb_proton1, nb_proton2, nb_neutron1, 
     &                        nb_neutron2
      integer nb_proton, nb_neutron
C      
      integer i,j,ihlast(20),ipart,iporg,ireuse,imemlast(20),iset,imem
     &     ,i_replace,ii,ipartlast(20)
      double precision xlast(20),xmulast(20),pdflast(20)
      double precision epa_proton
      save ihlast,xlast,xmulast,pdflast,imemlast,ipartlast
      data ihlast/20*-99/
      data ipartlast/20*-99/
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/20*-99d9/
      data imemlast/20*-99/
      data i_replace/20/


c     Make sure we have a reasonable Bjorken x. Note that even though
c     x=0 is not reasonable, we prefer to simply return pdg2pdf=0
c     instead of stopping the code, as this might accidentally happen.
      if (x.eq.0d0) then
         pdg2pdf=0d0
         return
      elseif (x.lt.0d0 .or. x.gt.1d0) then
         write (*,*) 'PDF not supported for Bjorken x ', x
         open(unit=26,file='../../../error',status='unknown')
         write(26,*) 'Error: PDF not supported for Bjorken x ',x
         stop 1
      endif

      ipart=ipdg
      if(iabs(ipart).eq.21) ipart=0
      if(iabs(ipart).eq.22) ipart=7
      iporg=ipart

c     This will be called for any PDG code, but we only support up to 7
      if(iabs(ipart).gt.7)then
         write(*,*) 'PDF not supported for pdg ',ipdg
         write(*,*) 'For lepton colliders, please set the lpp* '//
     $    'variables to 0 in the run_card'  
         open(unit=26,file='../../../error',status='unknown')
         write(26,*) 'Error: PDF not supported for pdg ',ipdg
         stop 1
      endif

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
c     Check if result can be reused since any of last twenty
c     calls. Start checking with the last call and move back in time
         if (ih.eq.ihlast(ii)) then
            if (ipart.eq.ipartlast(ii)) then
               if (x.eq.xlast(ii)) then
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
      if (ireuse.gt.0) then
         if (pdflast(ireuse).ne.-99d9) then
            pdg2pdf=pdflast(ireuse)
            return 
         endif
      endif

c Calculated a new value: replace the value computed longest ago
      i_replace=mod(i_replace,20)+1

c     Call lhapdf and give the current values to the arrays that should
c     be saved
      if(ih.eq.1) then
         if (ibeam.eq.1)then
            nb_proton = nb_proton1
            nb_neutron = nb_neutron1
         else
            nb_proton = nb_proton2
            nb_neutron = nb_neutron2
         endif
         if (nb_proton.gt.1.or.nb_neutron.ne.0)then
            if (ipart.eq.1)then
               call evolvepart(1,x,xmu,tmp1)
               call evolvepart(2,x,xmu,tmp2)
               pdg2pdf = nb_protron * tmp1 + nb_neutron * tmp2
            else if (ipart.eq.-1)then
               call evolvepart(-1,x,xmu,tmp1)
               call evolvepart(-2,x,xmu,tmp2)
               pdg2pdf = nb_protron * tmp1 + nb_neutron * tmp2
            else if (ipart.eq.2)then
               call evolvepart(1,x,xmu,tmp1)
               call evolvepart(2,x,xmu,tmp2)
               pdg2pdf =  nb_proton * tmp2 + nb_neutron * tmp1
            else if (ipart.eq.2)then
               call evolvepart(-1,x,xmu,tmp1)
               call evolvepart(-2,x,xmu,tmp2)
               pdg2pdf = nb_proton * tmp2 + nb_neutron * tmp1
            else
               call evolvepart(ipart,x,xmu,pdg2pdf)
               pdg2pdf = pdf2pdf*(nb_proton+nb_neutron)
            endif 
         else
               call evolvepart(ipart,x,xmu,pdg2pdf)
         endif
      endif
         pdg2pdf=pdg2pdf/x
      else if(ih.eq.2) then ! photon from a proton without breaking
          pdg2pdf = epa_proton(x,xmu*xmu)
      else
         write (*,*) 'beam type not supported in lhadpf'
         stop 1
      endif
      pdflast(i_replace)=pdg2pdf
      xlast(i_replace)=x
      xmulast(i_replace)=xmu
      ihlast(i_replace)=ih
      imemlast(i_replace)=imem
c
      return
      end

