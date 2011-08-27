      subroutine qltrisort(psq,msq)
C-----sort arguments of triangle so that the are ordered in mass
C-----m1sq<m2sq<m3sq
      implicit none
      double precision msq(3),psq(3),msqtmp(3),psqtmp(3)
      INTEGER j,x1(3),x2(3)
      data x1/3,1,2/
      data x2/2,3,1/
      save x1,x2
      do j=1,3
      msqtmp(j)=msq(j)
      psqtmp(j)=psq(j)
      enddo

      if (max(msqtmp(1),msqtmp(2),msqtmp(3)) .eq. msqtmp(1)) then
      do j=1,3
      msq(x1(j))=msqtmp(j)
      psq(x1(j))=psqtmp(j)
      enddo
      endif

      if (max(msqtmp(1),msqtmp(2),msqtmp(3)) .eq. msqtmp(2)) then
      do j=1,3
      msq(x2(j))=msqtmp(j)
      psq(x2(j))=psqtmp(j)
      enddo
      endif

      if (msq(1) .gt. msq(2)) then
      do j=1,2
      msqtmp(j)=msq(j)
      psqtmp(j+1)=psq(j+1)
      enddo

      msq(1)=msqtmp(2) 
      msq(2)=msqtmp(1) 
      psq(2)=psqtmp(3) 
      psq(3)=psqtmp(2) 
      endif

      return
      END
