      subroutine HWABEG
      return
      end

      subroutine HWAEND
      return
      end


      subroutine HWANAL
      INCLUDE 'HEPMC.INC'
      INTEGER IP,I
c print event to screen
     
      PRINT *,' EVENT ',NEVHEP
      DO IP=1,NHEP
         PRINT '(I4,I8,I4,4I4,1P,5D11.3)',IP,IDHEP(IP),ISTHEP(IP),
     &        JMOHEP(1,IP),JMOHEP(2,IP),JDAHEP(1,IP),JDAHEP(2,IP),
     &        (PHEP(I,IP),I=1,5)
      ENDDO

      return
      end
