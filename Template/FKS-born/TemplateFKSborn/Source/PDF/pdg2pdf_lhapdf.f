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
      double precision xlast,xmulast,pdflast(-7:7)
      save ihlast,xlast,xmulast,pdflast

      ipart=ipdg
      if(ipart.eq.21) ipart=0

      if(ih.eq.ihlast.and.x.eq.xlast.and.xmu.eq.xmulast)then
         pdg2pdf=pdflast(ipart);
      else
         call pftopdglha(ih,x,xmu,pdflast)
         ihlast=ih
         xlast=x
         xmulast=xmu
         pdg2pdf=pdflast(ipart);
      endif

      end

