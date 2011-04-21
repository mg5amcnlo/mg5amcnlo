      subroutine hstxxx(tc,sc,gt,smass,swidth , hst)
c
c This subroutine computes an off-shell scalar current from
c the scalar-scalar-tensor boson coupling.
c
c input:
c       complex tc(18)         : input tensor                          T
c       complex sc(3)          : input scalar                          s
c       complex gt             : coupling constant         gts=-1/Lambda
c       real    smass          : mass  of output scalar s'
c       real    swidth         : width of output scalar s'
c
c output:
c       complex hst(3)         : scalar current                j(s':T,s)     
c     
c- by Q.Li - OCT. 2006
c
      implicit none
      double complex tc(18), sc(3), hst(3)
      double precision smass, swidth
      double complex gt

      double complex ft(6,4)
      double complex T12, T13, T14, T23, T24, T34
      double complex TKK
      double precision ps1(4), ps2(4)
      integer i
      double complex cZero, d
      double precision rZero, rTwo, pf2
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

      hst(2) = sc(2)+ft(5,1)
      hst(3) = sc(3)+ft(6,1)

      ps1(1) = dreal(sc(2))
      ps1(2) = dreal(sc(3))
      ps1(3) = dimag(sc(3))
      ps1(4) = dimag(sc(2))

      ps2(1) = dreal(hst(2))
      ps2(2) = dreal(hst(3))
      ps2(3) = dimag(hst(3))
      ps2(4) = dimag(hst(2))

      pf2 = ps2(1)**2 - ps2(2)**2 - ps2(3)**2 - ps2(4)**2
      
      if ( smass.gt.rZero ) then
         d = - gt/dcmplx( pf2-smass**2, smass*swidth )
      else
         d = - gt/dcmplx( pf2, rZero )
      end if

      T12 = ft(1,2) + ft(2,1)
      T13 = ft(1,3) + ft(3,1)
      T14 = ft(1,4) + ft(4,1)
      T23 = ft(2,3) + ft(3,2)
      T24 = ft(2,4) + ft(4,2)
      T34 = ft(3,4) + ft(4,3)

      TKK   = cZero
    
      do i = 1,4
         TKK=TKK+ft(i,i)*ps1(i)*ps2(i)
      end do

      TKK   = rTwo*TKK
    
      TKK = TKK - T12*(ps1(1)*ps2(2) + ps1(2)*ps2(1))
     &          - T13*(ps1(1)*ps2(3) + ps1(3)*ps2(1))
     &          - T14*(ps1(1)*ps2(4) + ps1(4)*ps2(1))
     &          + T23*(ps1(2)*ps2(3) + ps1(3)*ps2(2))
     &          + T24*(ps1(2)*ps2(4) + ps1(4)*ps2(2))
     &          + T34*(ps1(3)*ps2(4) + ps1(4)*ps2(3))

      hst(1) = TKK+(ft(1,1)-ft(2,2)-ft(3,3)-ft(4,4))
     &	*(smass**2-ps1(1)*ps2(1)+ps1(2)*ps2(2)
     &      +ps1(3)*ps2(3)+ps1(4)*ps2(4))

      hst(1) = hst(1) * d*sc(1)

      return
      end
