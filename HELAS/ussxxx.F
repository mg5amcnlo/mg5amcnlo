      subroutine ussxxx(s1,s2,gt,smass,tmass,twidth , uss)
c
c This subroutine computes an off-shell tensor current 
c from the scalar-scalar-tensor boson coupling.
c
c input:
c       complex s1(3)          : first  scalar                        s1
c       complex s2(3)          : second scalar                        s2
c       complex gt             : coupling constant         gts=-1/Lambda
c       real    smass          : scalar mass                         m_s
c       real    tmass          : mass  of output tensor T 
c       real    twidth         : width of output tensor T
c
c output:
c       complex uss(18)        : tensor current         j^mu^nu(T:s1,s2)
c     
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex s1(3), s2(3), gt, uss(18)
      double precision smass, tmass, twidth

      integer i,j
      double complex yss(6,4)
      double precision ps1(4), ps2(4), pT(4)
      double precision MET(4,4)
      double complex cZero, d
      double precision rZero, rTwo
      double precision p1p2,pT2,p1pT,p2pT
      parameter( rZero = 0.0d0, rTwo = 2.0d0 )
      parameter( cZero = ( 0.0d0, 0.0d0 ) )


      yss(5,1) = s1(2)+s2(2)
      yss(6,1) = s1(3)+s2(3)
      
      ps1(1) = dreal(s1(2))
      ps1(2) = dreal(s1(3))
      ps1(3) = dimag(s1(3))
      ps1(4) = dimag(s1(2))
      
      ps2(1) = -dreal(s2(2))
      ps2(2) = -dreal(s2(3))
      ps2(3) = -dimag(s2(3))
      ps2(4) = -dimag(s2(2))
      
      pT(1) = dreal(yss(5,1))
      pT(2) = dreal(yss(6,1))
      pT(3) = dimag(yss(6,1))
      pT(4) = dimag(yss(5,1))
      
      do i=1,4
         do j=1,4
            MET(i,j) = 0.0d0
         enddo 
      enddo
      MET(1,1) =  1.0d0
      MET(2,2) = -1.0d0
      MET(3,3) = -1.0d0
      MET(4,4) = -1.0d0
      
      p1p2 = ps1(1)*ps2(1)-ps1(2)*ps2(2)-ps1(3)*ps2(3)-ps1(4)*ps2(4)
      p1pT = ps1(1)*pT(1)-ps1(2)*pT(2)-ps1(3)*pT(3)-ps1(4)*pT(4)
      p2pT = pT(1)*ps2(1)-pT(2)*ps2(2)-pT(3)*ps2(3)-pT(4)*ps2(4)
      pT2  = pT(1)**2-pT(2)**2-pT(3)**2-pT(4)**2
      
      if ( tmass.gt.rZero ) then
         d = - gt/dcmplx( pT2-tmass**2, tmass*twidth )
      else
         d = - gt/dcmplx( pT2, rZero )
      end if
    
      do i = 1,4
         do j=1,4
            yss(i,j) = -2.0d0/3.0d0*MET(i,j)*smass**2
     &	-4.0d0/3.0d0*smass**2/tmass**2*pT(i)*pT(j)
     &    +2.0d0/3.0d0*smass**2/tmass**2*PT2*MET(i,j)
     &    +4.0d0/3.0d0*smass**2/tmass**4*PT2*pT(i)*pT(j)
     &    +2.0d0*(ps1(i)*ps2(j)+ps1(j)*ps2(i))
     &    -2.0d0/tmass**2*p1pT*(ps2(i)*pT(j)+ps2(j)*pT(i)) 
     &    -2.0d0/tmass**2*p2pT*(ps1(i)*pT(j)+ps1(j)*pT(i)) 
     &    +4.0d0/3.0d0/tmass**2*MET(i,j)*p1pT*p2pT
     &    +8.0d0/3.0d0/tmass**4*p1pT*p2pT*pT(i)*pT(j)
     &    -2.0d0/3.0d0*p1p2*MET(i,j)
     &    +8.0d0/3.0d0/tmass**2*p1p2*pT(i)*pT(j)
     &    -2.0d0/3.0d0/tmass**2*p1p2*pT2*MET(i,j)
     &    -4.0d0/3.0d0/tmass**4*PT2*p1p2*pT(i)*pT(j)

            yss(i,j) = yss(i,j)*d*s1(1)*s2(1)/2.0d0

         end do
      enddo

      uss(1) = yss(1,1)
      uss(2) = yss(1,2)
      uss(3) = yss(1,3)
      uss(4) = yss(1,4)
      uss(5) = yss(2,1)
      uss(6) = yss(2,2)
      uss(7) = yss(2,3)
      uss(8) = yss(2,4)
      uss(9) = yss(3,1)
      uss(10) = yss(3,2)
      uss(11) = yss(3,3)
      uss(12) = yss(3,4)
      uss(13) = yss(4,1)
      uss(14) = yss(4,2)
      uss(15) = yss(4,3)
      uss(16) = yss(4,4)
      uss(17) = yss(5,1)
      uss(18) = yss(6,1)

      return
      end
