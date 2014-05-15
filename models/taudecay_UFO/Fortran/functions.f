      
      subroutine taudecay_param()
      implicit none

      DOUBLE PRECISION PIM,PI0
      COMMON/PARAM_PI/PIM,PI0

      DOUBLE PRECISION ROM,ROG,ROM1,ROG1
      COMMON/PARAM_RHO/ROM,ROG,ROM1,ROG1

      DOUBLE PRECISION a1M,a1G,fpi
      COMMON/PARAM_A1/a1M,a1G,fpi

      PIM=0.13957018d0
      PI0=0.1349766d0

      ROM=0.77549d0
      ROG=0.1491d0
      ROM1=1.465d0
      ROG1=0.40d0

      a1M=1.23d0
      a1G=0.42d0
      fpi=0.13041d0

      return
      end

      double complex function FFCT2(S)
      implicit none
      double complex S,Frho
      external Frho
      S=real(S)
      FFCT2=Frho(S,1)
      return
      end

      double complex function FFCT3(S)
      implicit none
      double complex S,Fa1
      external Fa1
      S=real(S)
      FFCT3=Fa1(S,1)
      return
      end

      double complex function FFCT3F1(S)
      implicit none
      double complex S,Brho
      external Brho
      S=real(S)
      FFCT3F1=Brho(S,1)
      return
      end

      double complex function FFCT3F0(S)
      implicit none
      double complex S,Brho
      external Brho
      S=real(S)
      FFCT3F0=Brho(S,0)
      return
      end

      DOUBLE COMPLEX FUNCTION Frho(S,mode) ! Frho=f2
      IMPLICIT NONE
      DOUBLE COMPLEX Brho
      DOUBLE PRECISION S
      integer mode
      EXTERNAL Brho
      Frho=dsqrt(2d0)*Brho(S,mode)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION Brho(S,mode)
      IMPLICIT NONE
      DOUBLE COMPLEX BWrho
      DOUBLE PRECISION S
      integer mode
      EXTERNAL BWrho
      DOUBLE PRECISION BETA1
      PARAMETER (BETA1=-0.145d0) !TAUOLA parameters
      DOUBLE PRECISION ROM,ROG,ROM1,ROG1
      COMMON/PARAM_RHO/ROM,ROG,ROM1,ROG1
      !!!PARAMETER (ROM=0.773d0,ROG=0.145d0)   !TAUOLA parameters
      !!!PARAMETER (ROM1=1.370d0,ROG1=0.510d0) !TAUOLA parameters
      call taudecay_param()
      Brho=(BWrho(S,ROM,ROG,mode)+BETA1*BWrho(S,ROM1,ROG1,mode))/(1d0+BETA1)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION BWrho(S,M,G,mode) !Breit-Wigner for the rho mode
      IMPLICIT NONE
      DOUBLE PRECISION S,M,G
      integer mode
      DOUBLE PRECISION QS,QM,W,GS,pi1,pi2
      double precision klambda
      external klambda
      DOUBLE PRECISION PIM,PI0
      COMMON/PARAM_PI/PIM,PI0
      !!!PARAMETER (PIM=0.139d0,PI0=0.139d0)  !TAUOLA parameters
      call taudecay_param()
      pi1=pim
      if (mode.eq.1) then
         pi2=pi0                ! rho- decay
      else
         pi2=pim                ! rho0 decay
      endif
      IF (S.GT.(PI1+PI2)**2) THEN
         QS=dsqrt(klambda(PI1**2/S,PI2**2/S))
         QM=dsqrt(klambda(PI1**2/M**2,PI2**2/M**2))
         W=DSQRT(S)
         GS=G*(W/M)*(QS/QM)**3
      ELSE
         GS=0d0
      ENDIF
      BWrho=-M**2/DCMPLX(S-M**2,W*GS)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION Fa1(S,mode) ! Fa1=f3
      IMPLICIT NONE
      DOUBLE COMPLEX BWa1
      DOUBLE PRECISION S
      integer mode
      EXTERNAL BWa1
      DOUBLE PRECISION a1M,a1G,fpi
      COMMON/PARAM_A1/a1M,a1G,fpi
      call taudecay_param()
      Fa1=4d0/3d0/fpi *BWa1(S,a1M,a1G,mode)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION BWa1(S,M,G,mode) !Breit-Wigner for the a1 mode
      IMPLICIT NONE
      DOUBLE PRECISION S,M,G
      integer mode
      DOUBLE PRECISION GS,GFUN,W,pi3
      DOUBLE PRECISION PIM,PI0
      COMMON/PARAM_PI/PIM,PI0
      !!!PARAMETER (PIM=0.139d0,PI0=0.139d0) !TAUOLA parameters
      call taudecay_param()
      W=DSQRT(S)
      if (mode.eq.1) then
         pi3=2d0*pi0+pim        ! rho- decay
      else
         pi3=3d0*pim            ! rho0 decay
      endif      
      IF (S.GT.pi3**2) THEN
        GS=G*(W/M)*GFUN(S,mode)/GFUN(M**2,mode)
      ELSE
        GS=0d0
      ENDIF
      BWa1=-M**2/DCMPLX(S-M**2,W*GS)
      RETURN
      END

      DOUBLE PRECISION FUNCTION GFUN(QKWA,mode) ! running width for a1
      IMPLICIT NONE
      DOUBLE PRECISION QKWA
      integer mode
      double precision pi1,pi3
      DOUBLE PRECISION PIM,PI0
      COMMON/PARAM_PI/PIM,PI0
      DOUBLE PRECISION ROM,ROG,ROM1,ROG1
      COMMON/PARAM_RHO/ROM,ROG,ROM1,ROG1
      call taudecay_param()
      if (mode.eq.1) then
         pi3=2d0*pi0+pim        ! rho- decay
         pi1=pi0
      else
         pi3=3d0*pim            ! rho0 decay
         pi1=pim
      endif        
      IF (QKWA.LT.(ROM+pi1)**2) THEN
        GFUN=4.1d0/QKWA*(QKWA-pi3**2)**3*(1d0-3.3d0*(QKWA-pi3**2)+5.8d0*(QKWA-pi3**2)**2)
      ELSE
        GFUN=1.623d0+10.38d0/QKWA-9.32d0/QKWA**2+0.65d0/QKWA**3
      ENDIF
      END

      double precision function klambda(a,b)
      implicit none 
      double precision a,b,c
      klambda=1d0+a**2+b**2-2d0*(a*b+b+a)
      return
      end

      DOUBLE PRECISION FUNCTION pSumDot(P1,P2,dsign) ! invariant mass
      IMPLICIT NONE
      double precision p1(0:3),p2(0:3),dsign
      integer i
      double precision ptot(0:3)
      double precision pdot
      external pdot
      do i=0,3
        ptot(i)=p1(i)+dsign*p2(i)
      enddo
      pSumDot = pdot(ptot,ptot)
      RETURN
      END

      double precision function pdot(p1,p2) !4-vector dot product
      implicit none
      double precision p1(0:3),p2(0:3)
      pdot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      if(dabs(pdot).lt.1d-6)then ! solve numerical problem 
         pdot=0d0
      endif
      return
      end

      subroutine psum(p1,p2, q) !4-vector sum
      implicit none
      double precision p1(0:3),p2(0:3), q(0:3)
      q(0)=p1(0)+p2(0)
      q(1)=p1(1)+p2(1)
      q(2)=p1(2)+p2(2)
      q(3)=p1(3)+p2(3)
      return
      end

      subroutine psub(p1,p2, q) !4-vector subtract
      implicit none
      double precision p1(0:3),p2(0:3), q(0:3)
      q(0)=p1(0)-p2(0)
      q(1)=p1(1)-p2(1)
      q(2)=p1(2)-p2(2)
      q(3)=p1(3)-p2(3)
      return
      end

      subroutine psum3(p1,p2,p3, q) !4-vector sum
      implicit none
      double precision p1(0:3),p2(0:3),p3(0:3), q(0:3)
      q(0)=p1(0)+p2(0)+p3(0)
      q(1)=p1(1)+p2(1)+p3(1)
      q(2)=p1(2)+p2(2)+p3(2)
      q(3)=p1(3)+p2(3)+p3(3)
      return
      end
