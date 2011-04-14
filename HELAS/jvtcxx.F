      subroutine jvtcxx(vc,tc,gt,vmass,vwidth , jvt)
c-------------------CP3  2009.10-----------------
c
c This subroutine computes an off-shell vector current from 
c the coupling of two gauge bosons and a non-propagating tensor boson.
c
c input:
c       complex vc(6)          : input vector                                                     v
c       complex tc(18)         : input non-propagating tensor                          T
c       complex gt            : coupling constant         gt=gs
c       real    vmass         : mass  of output vector  v'
c       real    vwidth         :  width of output vector    v'
c
c output:
c       complex jvt(6)         : vector current             j^mu(v':v,T)
c
      implicit none
      double complex vc(6), tc(18), jvt(6)
      double precision vmass, vwidth,gt

      double complex ft(6,4)
      double precision pv2(4), pp2

      integer i, j

      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )


      ft(1,1) = tc(1)
      ft(1,2) = tc(2)
      ft(1,3) = tc(3)
      ft(1,4) = tc(4)
      ft(2,1) = tc(5)
      ft(2,2) = tc(6)
      ft(2,3) = tc(7)
      ft(2,4) = tc(8)
      ft(3,1) = tc(9)
      ft(3,2) = tc(10)
      ft(3,3) = tc(11)
      ft(3,4) = tc(12)
      ft(4,1) = tc(13)
      ft(4,2) = tc(14)
      ft(4,3) = tc(15)
      ft(4,4) = tc(16)
      ft(5,1) = tc(17)
      ft(6,1) = tc(18)
 
      jvt(5) = vc(5)+ft(5,1)
      jvt(6) = vc(6)+ft(6,1)
      pv2(1) = dreal(jvt(5))
      pv2(2) = dreal(jvt(6))
      pv2(3) = dimag(jvt(6))
      pv2(4) = dimag(jvt(5))
      pp2 = pv2(1)**2 - pv2(2)**2 - pv2(3)**2 - pv2(4)**2


        do i=1,4
            jvt(i)=-(ft(1,i)*vc(1)) + ft(2,i)*vc(2) + ft(3,i)*vc(3) 
     & + ft(4,i)*vc(4) 
            jvt(i)=jvt(i)*gt/dcmplx(pp2, 0.0d0)
         enddo

         

      return
      end
