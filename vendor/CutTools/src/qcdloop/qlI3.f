      double complex function qlI3(p1,p2,p3,m1,m2,m3,mu2,ep)
      implicit none
      double precision p1,p2,p3,m1,m2,m3,mu2
      integer ep
C     pi=p(i)^2, i=1,2,3 are the four-momentum squared of the external lines
C     mi=m(i)^2, i=1,2,3,are the squares of the masses of the internal lines
C     mu2 is the square of the scale mu
C     ep=-2,-1,0 chooses the coefficient in the Laurent series.

      double precision psq(3),msq(3)
      double precision p1o,p2o,p3o,m1o,m2o,m3o,mu2o,scalefac,newmu2
      double complex Ival(-2:0)
      integer epdum
      logical qlzero 
      data p1o,p2o,p3o,m1o,m2o,m3o/6*0d0/
      save Ival,p1o,p2o,p3o,m1o,m2o,m3o,mu2o
      

C--If we have already calculated, use the saved value
C--else setup the arrays
      if    ((p1 .eq. p1o)
     . .and. (p2 .eq. p2o)
     . .and. (p3 .eq. p3o)
     . .and. (m1 .eq. m1o)
     . .and. (m2 .eq. m2o)
     . .and. (m3 .eq. m3o)
     . .and. (mu2 .eq. mu2o)) then
      qlI3=Ival(ep)
      return
      else
C---recalculate
      p1o=p1
      p2o=p2
      p3o=p3
      m1o=m1
      m2o=m2
      m3o=m3
      mu2o=mu2
      endif

      scalefac=max(abs(m1),abs(m2),abs(m3),abs(p1),abs(p2),abs(p3))

      msq(1)=m1/scalefac
      msq(2)=m2/scalefac
      msq(3)=m3/scalefac
      psq(1)=p1/scalefac
      psq(2)=p2/scalefac
      psq(3)=p3/scalefac
      newmu2=mu2/scalefac
C----sort msq in ascending order (and reorder psq correspondingly)
      call qltrisort(psq,msq)

C----If internal masses all qlzero, reorder abs(psq) in ascending order
      if    (qlzero(abs(msq(1)))
     . .and. qlzero(abs(msq(2)))
     . .and. qlzero(abs(msq(3)))) then
      call qlsnglsort(3,psq)
      endif

C-----calculate value of integral
      call qlI3sub(msq,psq,newmu2,Ival)

C---apply the rescaling to the integral
      do epdum=-2,0
      Ival(epdum)=Ival(epdum)/scalefac
      enddo

      qlI3=Ival(ep)

      return
      end
