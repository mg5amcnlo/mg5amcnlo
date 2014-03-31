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
      integer mode,Irt,i,j
      double precision xlast(2),xmulast(2),pdflast(-7:7,2),q2max
      character*7 pdlabellast(2)
      double precision epa_electron,epa_proton
      integer ipart,ireuse,iporg,ihlast(2)
      save xlast,xmulast,pdflast,pdlabellast,ihlast
      data xlast/2*-99d9/
      data xmulast/2*-99d9/
      data pdflast/30*-99d9/
      data pdlabellast/2*'abcdefg'/
      data ihlast/2*-99/

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
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (x.eq.xlast(i) .and. xmu.eq.xmulast(i) .and.
     $        pdlabel.eq.pdlabellast(i) .and. ih.eq.ihlast(i)) then
            ireuse = i
         endif
      enddo

c     Reuse previous result, if possible
      if (ireuse.gt.0.)then
         if (pdflast(iporg,ireuse).ne.-99d9) then
            pdg2pdf=pdflast(iporg,ireuse)
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
      xlast(ireuse)=x
      xmulast(ireuse)=xmu
      pdlabellast(ireuse)=pdlabel
      ihlast(ireuse)=ih

      if(iabs(ipart).eq.7.and.ih.gt.1) then
         q2max=xmu*xmu
         if(ih.eq.3) then       !from the electron
            pdg2pdf=epa_electron(x,q2max)
         elseif(ih .eq. 2) then !from a proton without breaking
            pdg2pdf=epa_proton(x,q2max)
         endif 
         pdflast(iporg,ireuse)=pdg2pdf
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
         call pftopdg(ih,x,xmu,pdflast(-7,ireuse))
         pdg2pdf=pdflast(iporg,ireuse)
      endif      

      pdflast(iporg,ireuse)=pdg2pdf
      return
      end

