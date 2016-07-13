      subroutine pftopdglha(ih,x,q,pdf)
c***************************************************************************
c     Wrapper for calling the pdf from lhapdf
c***************************************************************************
      implicit none
c
c     Arguments
c
      DOUBLE  PRECISION x,q,pdf(-7:7)
      DOUBLE  PRECISION f(-6:6)
      INTEGER IH,I
      double precision  photon
      LOGICAL has_photon
      double precision epa_electron,epa_proton
C
C     Include
C
      include 'pdf.inc'
C      
      if(abs(ih).eq.1) then
         pdf(-7)=0d0
         if(has_photon())then
             call evolvePDFphoton(x, q, f, photon)
             pdf(7)= photon/x
         else
             pdf(7) = 0d0
             call evolvePDF(x, q, f)
         endif
         do i=-6,6
            pdf(i)=f(i)/x
         enddo
      elseif(ih .eq. 2) then  !from a proton without breaking
          pdf(7) = epa_proton(x, q * q)
      else
         write (*,*) 'beam type not supported in lhadpf'
         stop 1
      endif

      return	
      end
  

