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
      double precision epa_electron,epa_proton
      integer ipart,ireuse,iporg
      save xlast,xmulast,pdflast
      data xlast/2*0d0/
      data pdflast/30*0d0/

      ipart=ipdg
      if(iabs(ipart).eq.21) ipart=0
      if(iabs(ipart).eq.22) ipart=7
      iporg=ipart

c     This will be called for any PDG code, but we only support up to 7
      if(iabs(ipart).gt.7)then
         pdg2pdf=0d0
         return 
      endif

      ireuse = 0
      do i=1,2
c     Check if result can be reused since any of last two calls
         if (x.eq.xlast(i).and.xmu.eq.xmulast(i)) then
            ireuse = i
         endif
      enddo

c     If both x non-zero and not ireuse, then zero x
      if (ireuse.eq.0.and.xlast(1).ne.0d0.and.xlast(2).ne.0d0)then
         do i=1,2
            xlast(i)=0d0
            xmulast(i)=0d0
            do j=-7,7
               pdflast(j,i)=0d0
            enddo
         enddo
         ireuse=1
      else if(ireuse.eq.0.and.xlast(1).ne.0d0)then
         ireuse=2
      else if(ireuse.eq.0)then
         ireuse=1
      endif

c     Reuse previous result, if possible
      if (ireuse.gt.0.and.pdflast(iporg,ireuse).ne.0d0) then
         pdg2pdf=pdflast(iporg,ireuse)
         return 
      endif

      xlast(ireuse)=x
      xmulast(ireuse)=xmu

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

