      subroutine HWABEG
      open(unit=99,file='mcatnlo_hepmcout.txt',status='unknown')
      return
      end

      subroutine HWAEND
      close(99)
      return
      end


      subroutine HWANAL
      INCLUDE 'HEPMC.INC'
      INTEGER IP,I
c print event to screen
     
      WRITE(99,*)' EVENT ',NEVHEP
      DO IP=1,NHEP
         WRITE(99,100)IP,IDHEP(IP),ISTHEP(IP),
     &        JMOHEP(1,IP),JMOHEP(2,IP),JDAHEP(1,IP),JDAHEP(2,IP),
     &        (PHEP(I,IP),I=1,5)
      ENDDO
 100  FORMAT(I4,I8,I4,4I4,1P,5D11.3)
      return
      end
