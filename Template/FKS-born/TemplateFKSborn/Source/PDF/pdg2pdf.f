      double precision function pdg2pdf(ih,ipdg,x,xmu)
c***************************************************************************
c     Based on pdf.f, wrapper for calling the pdf of MCFM
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,xmu
      INTEGER IH,ipdg,ipart
C
C     Include
C
      include 'pdf.inc'
C      
      double precision Ctq3df,Ctq4Fn,Ctq5Pdf,Ctq6Pdf,Ctq5L
      integer mode,Irt,ihlast
      double precision xlast,xmulast,pdflast(-7:7),q2max
      double precision epa_electron,epa_proton
      save ihlast,xlast,xmulast,pdflast

      ipart=ipdg
      if(iabs(ipart).eq.21) ipart=0
      if(iabs(ipart).eq.22) ipart=7

      if(iabs(ipart).eq.7.and.ih.gt.1) then
         q2max=xmu*xmu
         if(ih.eq.3) then       !from the electron
            pdg2pdf=epa_electron(x,q2max)
         elseif(ih .eq. 2) then !from a proton without breaking
            pdg2pdf=epa_proton(x,q2max)
         endif 
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
        if(ih.eq.ihlast.and.x.eq.xlast.and.xmu.eq.xmulast)then
          pdg2pdf=pdflast(ipart);
        else
          call pftopdg(ih,x,xmu,pdflast)
          ihlast=ih
          xlast=x
          xmulast=xmu
          pdg2pdf=pdflast(ipart);
        endif
      endif      

      end

