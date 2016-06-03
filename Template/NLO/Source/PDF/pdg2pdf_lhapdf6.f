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
      integer i,j,ihlast(20),ipart,iporg,ireuse,imemlast(20),iset,imem
     &     ,i_replace,ii,ipartlast(20),isetlast(20)
      double precision xlast(20),xmulast(20),pdflast(20)
      save ihlast,xlast,xmulast,pdflast,imemlast,ipartlast
      data ihlast/20*-99/
      data ipartlast/20*-99/
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/20*-99d9/
      data imemlast/20*-99/
      data isetlast/20*-99/
      data i_replace/20/

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
         write (*,*) 'PDF not supported for Bjorken x ', x
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
         stop 1
      endif

c     Determine the iset used in lhapdf
      call getnset(iset)
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
                        if (iset.eq.isetlast(ii)) then
                           ireuse = ii
                           exit
                        endif
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
      call evolvepartm(iset,ipart,x,xmu,pdg2pdf)
      pdg2pdf=pdg2pdf/x
      pdflast(i_replace)=pdg2pdf
      xlast(i_replace)=x
      xmulast(i_replace)=xmu
      ihlast(i_replace)=ih
      imemlast(i_replace)=imem
      isetlast(i_replace)=iset
c
      return
      end

