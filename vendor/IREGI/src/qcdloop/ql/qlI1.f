      double complex function qlI1(m1,mu2,ep)
      implicit none
      include 'qlconstants.f'
      double precision m1,mu2
      integer ep
C     m1=m(i)^2 is the square of the mass of the propagator 1 
C     mu2 is the square of the scale mu
C     ep=-2,-1,0 chooses the appropriate coefficienrt in the Laurent series.
      double precision m1o,mu2o
      logical qlzero 
      double complex Ival(-2:0)
      data m1o/0d0/
      save Ival,m1o,mu2o

      if (mu2 .le. 0d0) then
      write(6,*) 'stopping because mu2 .le. 0d0 in qlI1, mu2=',mu2
      write(6,*) 'Rerun with positive mu2'
      stop
      endif

C--If we have already calculated, use the saved value
C--else setup the arrays
      if ((m1 .eq. m1o) .and. (mu2 .eq. mu2o)) then
         qlI1=Ival(ep)
         return
      else
         Ival(-2)=czip
         Ival(-1)=czip
         Ival(0)=czip
         if (qlzero(m1)) then
              qlI1=Ival(ep)
         else
              Ival(-1)=dcmplx(m1)
              Ival( 0)=Ival(-1)*dcmplx(log(mu2/m1)+1d0)
              qlI1=Ival(ep)
         endif
      m1o=m1
      mu2o=mu2
      endif
      return
      end
