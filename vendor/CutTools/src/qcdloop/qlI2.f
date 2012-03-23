      double complex function qlI2(p1,m1,m2,mu2,ep) 
      implicit none
      include 'qlconstants.f'
      double precision p1,m1,m2,mu2
      integer ep
C     p1=p1(1) is the squared four-momentum of the external particle i 
C     mi=m(i)^2, i=1,2 are the squares of the mass of the propagator i 
C     mu2 is the square of the scale mu
C     ep=-2,-1,0 chooses the coefficient in the Laurent series.
      double precision p1o,m1o,m2o,mu2o,pp1,mm1,mm2,newmu2,scalefac
      logical qlzero 
      double complex Ival(-2:0),qlI2fin
      data p1o,m1o,m2o,mu2o/3*0d0,-1d0/
      save Ival,p1o,m1o,m2o,mu2o

C--If we have already calculated, use the saved value
C--else setup the arrays
      if    ((p1 .eq. p1o)
     . .and. (m1 .eq. m1o)
     . .and. (m2 .eq. m2o)
     . .and. (mu2 .eq. mu2o)) then
      qlI2=Ival(ep)
      return
      else
      p1o=p1
      m1o=m1
      m2o=m2
      mu2o=mu2
      endif
      if (mu2 .le. 0d0) then
      write(6,*) 'stopping because mu2 .le. 0d0 in qlI2, mu2=',mu2
      write(6,*) 'Rerun with positive mu2'
      stop
      endif

      scalefac=max(abs(p1),abs(m1),abs(m2),abs(mu2))
      pp1=p1/scalefac
      mm1=m1/scalefac
      mm2=m2/scalefac
      newmu2=mu2/scalefac

      Ival(-2)=czip
      Ival(-1)=czip
      Ival(0)=czip
      if ((qlzero(pp1)).and.(qlzero(mm1)).and.(qlzero(mm2))) then
      qlI2=Ival(ep)
      else
      Ival(-1)=cone
      Ival(0)=qlI2fin(pp1,mm1,mm2,newmu2)
      qlI2=Ival(ep)
      endif

      return
      end
