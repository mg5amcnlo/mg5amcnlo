      DOUBLE COMPLEX FUNCTION FF2(S) ! Frho=f2
      IMPLICIT NONE
      DOUBLE COMPLEX Brho,S
      !DOUBLE PRECISION S
      integer mode
      EXTERNAL Brho
      S=real(S)
      mode=1
      FF2=dsqrt(2d0)*Brho(S,mode)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION Brho(S,mode)
      IMPLICIT NONE
      DOUBLE COMPLEX BWrho
      DOUBLE PRECISION S
      integer mode
      EXTERNAL BWrho
      DOUBLE PRECISION BETA1
      !include 'ffparam.inc' 
      DOUBLE PRECISION PIM,PI0
      PARAMETER (PIM=0.13957018d0,PI0=0.1349766d0)
      DOUBLE PRECISION ROM,ROG,ROM1,ROG1
      PARAMETER (ROM=0.77549d0,ROG=0.1491d0)
      PARAMETER (ROM1=1.465d0,ROG1=0.40d0)
      DOUBLE PRECISION a1M,a1G,fpi
      PARAMETER (a1M=1.23d0,a1G=0.42d0)
      PARAMETER (fpi=0.13041d0)
      Brho=(BWrho(S,ROM,ROG,mode)+BETA1*BWrho(S,ROM1,ROG1,mode))/(1d0+BETA1)
      RETURN
      END

      DOUBLE COMPLEX FUNCTION BWrho(S,M,G,mode) !Breit-Wigner for the rho mode
      IMPLICIT NONE
      DOUBLE PRECISION S,M,G
      integer mode
      DOUBLE PRECISION QS,QM,W,GS,pi1,pi2
      !include 'ffparam.inc' 
      DOUBLE PRECISION PIM,PI0
      PARAMETER (PIM=0.13957018d0,PI0=0.1349766d0)
      DOUBLE PRECISION ROM,ROG,ROM1,ROG1
      PARAMETER (ROM=0.77549d0,ROG=0.1491d0)
      PARAMETER (ROM1=1.465d0,ROG1=0.40d0)
      DOUBLE PRECISION a1M,a1G,fpi
      PARAMETER (a1M=1.23d0,a1G=0.42d0)
      PARAMETER (fpi=0.13041d0)
      double precision klambda
      external klambda
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
