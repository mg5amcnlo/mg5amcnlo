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
      integer i,j,ihlast(20),ipart,iporg,ireuse,imemlast(20),iset,imem
     &     ,i_replace,ii
      double precision xlast(20),xmulast(20),pdflast(-7:7,20)
      save ihlast,xlast,xmulast,pdflast,imemlast
      data ihlast/20*-99/
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/300*-99d9/
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

c      ireuse = 0
c      ii=i_replace
c      do i=1,20
cc     Check if result can be reused since any of last twenty
cc     calls. Start checking with the last call and move back in time
c         if (ih.eq.ihlast(ii)) then
c            if (x.eq.xlast(ii)) then
c               if (xmu.eq.xmulast(ii)) then
c                  if (imem.eq.imemlast(ii)) then
c                     ireuse = ii
c                     exit
c                  endif
c               endif
c            endif
c         endif
c         ii=ii-1
c         if (ii.eq.0) ii=ii+20
c      enddo

c     Reuse previous result, if possible
c      if (ireuse.gt.0) then
c         if (pdflast(iporg,ireuse).ne.-99d9) then
c            pdg2pdf=pdflast(iporg,ireuse)
c            return 
c         endif
c      endif

c Calculated a new value: replace the value computed longest ago
c      i_replace=mod(i_replace,20)+1

c     Call lhapdf and give the current values to the arrays that should
c     be saved
c      call pftopdglha(ih,x,xmu,pdflast(-7,i_replace))
c      xlast(i_replace)=x
c      xmulast(i_replace)=xmu
c      ihlast(i_replace)=ih
c      imemlast(i_replace)=imem
c
c      pdg2pdf=pdflast(ipart,i_replace)
      call evolvepart(ipart,x,xmu,pdg2pdf)
      pdg2pdf=pdg2pdf/x
      return
      end

