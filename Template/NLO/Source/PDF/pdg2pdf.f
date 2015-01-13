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
      double precision epa_electron,epa_proton
      integer ipart,ireuse,iporg,ihlast(20)
      save xlast,xmulast,pdflast,pdlabellast,ihlast
      data xlast/20*-99d9/
      data xmulast/20*-99d9/
      data pdflast/300*-99d9/
      data pdlabellast/20*'abcdefg'/
      data ihlast/20*-99/
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

      if(iabs(ipart).eq.7.and.ih.gt.1) then
         q2max=xmu*xmu
         if(ih.eq.3) then       !from the electron
            pdg2pdf=epa_electron(x,q2max)
         elseif(ih .eq. 2) then !from a proton without breaking
            pdg2pdf=epa_proton(x,q2max)
         endif 
         pdflast(iporg,i_replace)=pdg2pdf
         return
      endif
      
      if (pdlabel(1:5) .eq. 'cteq3') then
C     
         if (pdlabel .eq. 'cteq3_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq3_l') then
            mode=2
         elseif (pdlabel .eq. 'cteq3_d') then
            mode=3
         endif

         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq3df(mode,ipart,x,xmu,Irt)/x

         if(ipdg.ge.1.and.ipdg.le.2)
     $      pdg2pdf=pdg2pdf+Ctq3df(mode,-ipart,x,xmu,Irt)/x

C     
      elseif (pdlabel(1:5) .eq. 'cteq4') then
C     
         if (pdlabel .eq. 'cteq4_m') then
            mode=1
         elseif (pdlabel .eq. 'cteq4_d') then
            mode=2
         elseif (pdlabel .eq. 'cteq4_l') then
            mode=3
         elseif (pdlabel .eq. 'cteq4a1') then
            mode=4
         elseif (pdlabel .eq. 'cteq4a2') then
            mode=5
         elseif (pdlabel .eq. 'cteq4a3') then
            mode=6
         elseif (pdlabel .eq. 'cteq4a4') then
            mode=7
         elseif (pdlabel .eq. 'cteq4a5') then
            mode=8
         elseif (pdlabel .eq. 'cteq4hj') then
            mode=9
         elseif (pdlabel .eq. 'cteq4lq') then
            mode=10
         endif
         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq4Fn(mode,ipart,x,xmu)
C
      elseif (pdlabel .eq. 'cteq5l1') then
C
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq5L(ipart,x,xmu)
C         
      elseif ((pdlabel(1:5) .eq. 'cteq5') .or. 
     .        (pdlabel(1:4) .eq. 'ctq5')) then
C         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq5Pdf(ipart,x,xmu)
C                  
      elseif (pdlabel(1:5) .eq. 'cteq6') then
C         
         if(iabs(ipart).ge.1.and.iabs(ipart).le.2)
     $      ipart=sign(3-iabs(ipart),ipart)

         pdg2pdf=Ctq6Pdf(ipart,x,xmu)
      else
         call pftopdg(ih,x,xmu,pdflast(-7,i_replace))
         pdg2pdf=pdflast(iporg,i_replace)
      endif      

      pdflast(iporg,i_replace)=pdg2pdf
      return
      end

