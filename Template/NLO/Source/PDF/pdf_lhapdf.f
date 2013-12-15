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
C
C     Include
C
      include 'pdf.inc'
C      
      if(abs(ih).eq.1) then
         pdf(-7)=0d0
         pdf(7)=0d0
         call evolvePDF(x, q, f)
         do i=-6,6
            pdf(i)=f(i)/x
         enddo
      else
         write (*,*) 'beam type not supported in lhadpf'
         do i=-7,7
            pdf(i)=0d0
         enddo
      endif

      return	
      end
  

