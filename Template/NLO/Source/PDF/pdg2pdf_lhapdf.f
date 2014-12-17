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
     &     ,i_replace
      double precision xlast(20),xmulast(20),pdflast(-7:7,20)
      save ihlast,xlast,xmulast,pdflast,imemlast
      data ihlast/20*-99/
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/300*-99d9/
      data imemlast/20*-99/
      data i_replace/0/

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
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (x.eq.xlast(i) .and. xmu.eq.xmulast(i) .and.
     $        imem.eq.imemlast(i) .and. ih.eq.ihlast(i)) then
            ireuse = i
         endif
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0) then
         if (pdflast(iporg,ireuse).ne.-99d9) then
            pdg2pdf=pdflast(iporg,ireuse)
            return 
         endif
      endif

      i_replace=mod(i_replace,20)+1
      
c$$$c     Bjorken x and/or facrorization scale and/or PDF set are not
c$$$c     identical to the saved values: this means a new event and we
c$$$c     should reset everything to compute new PDF values. Also, determine
c$$$c     if we should fill ireuse=1 or ireuse=2.
c$$$      if (ireuse.eq.0.and.xlast(1).ne.-99d9.and.xlast(2).ne.-99d9)then
c$$$         do i=1,2
c$$$            xlast(i)=-99d9
c$$$            xmulast(i)=-99d9
c$$$            do j=-7,7
c$$$               pdflast(j,i)=-99d9
c$$$            enddo
c$$$            imemlast(i)=-99
c$$$            ihlast(i)=-99
c$$$         enddo
c$$$c     everything has been reset. Now set ireuse=1 to fill the first
c$$$c     arrays of saved values below
c$$$         ireuse=1
c$$$      else if(ireuse.eq.0.and.xlast(1).ne.-99d9)then
c$$$c     This is first call after everything has been reset, so the first
c$$$c     arrays are already filled with the saved values (hence
c$$$c     xlast(1).ne.-99d9). Fill the second arrays of saved values (done
c$$$c     below) by setting ireuse=2
c$$$         ireuse=2
c$$$      else if(ireuse.eq.0)then
c$$$c     Special: only used for the very first call to this function:
c$$$c     xlast(i) are initialized as data statements to be equal to -99d9
c$$$         ireuse=1
c$$$      endif

c     Call lhapdf and give the current values to the arrays that should
c     be saved
      call pftopdglha(ih,x,xmu,pdflast(-7,i_replace))
      xlast(i_replace)=x
      xmulast(i_replace)=xmu
      ihlast(i_replace)=ih
      imemlast(i_replace)=imem
c
      pdg2pdf=pdflast(ipart,i_replace)
      return
      end

