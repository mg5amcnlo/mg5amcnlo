      subroutine uvvcxx(v1,v2,gt,vmass,tmass,twidth , uvvh)
c
c This subroutine computes an off-shell tensor current from 
c the two gauge bosons-pseudo tensor boson coupling.
c
c input:
c       complex v1(3)          : first  vector                        v1
c       complex v2(3)          : second vector                        v2
c       real    gt             : coupling constant                gtv= gs
c       real    vmass          : vector boson mass                   m_v
c       real    tmass          : mass  of output tensor T
c       real    twidth         : width of output tensor T
c
c output:
c       complex uvv(18)        : tensor current         j^mu^nu(T:v1,v2)
c
      implicit none
      double complex v1(6), v2(6), uvvh(18)
      double precision vmass, tmass, twidth,gt
      integer i,j
      double complex yvv(6,4)
      double complex cZero
      double precision rZero, rTwo
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )

      
      yvv(5,1) = v1(5)+v2(5)
      yvv(6,1) = v1(6)+v2(6)

      do i=1,4
      do j=1,4
      yvv(i,j)=gt*v1(i)*v2(j)
      enddo
      enddo

      uvvh(1) = yvv(1,1)
      uvvh(2) = yvv(1,2)
      uvvh(3) = yvv(1,3)
      uvvh(4) = yvv(1,4)
      uvvh(5) = yvv(2,1)
      uvvh(6) = yvv(2,2)
      uvvh(7) = yvv(2,3)
      uvvh(8) = yvv(2,4)
      uvvh(9) = yvv(3,1)
      uvvh(10) = yvv(3,2)
      uvvh(11) = yvv(3,3)
      uvvh(12) = yvv(3,4)
      uvvh(13) = yvv(4,1)
      uvvh(14) = yvv(4,2)
      uvvh(15) = yvv(4,3)
      uvvh(16) = yvv(4,4)
      uvvh(17) = yvv(5,1)
      uvvh(18) = yvv(6,1)

      return
      end
