
MODULE kinematics
USE global
CONTAINS
  FUNCTION cscalarprod(p1,p2)
    IMPLICIT NONE
    COMPLEX(KIND(1d0))::cscalarprod
    COMPLEX(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p1,p2
!    REAL(KIND(1d0)),PARAMETER::EPS2=1d-6
!    REAL(KIND(1d0))::max
    cscalarprod=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
!    max=ABS(p1(0)*p2(0))+ABS(p1(1)*p2(1))+ABS(p1(2)*p2(2))+ABS(p1(3)*p2(3))
!    IF(ABS(cscalarprod/max).LT.EPS2)cscalarprod=DCMPLX(0d0)
  END FUNCTION cscalarprod

  FUNCTION scalarprod(p1,p2)
    IMPLICIT NONE
    REAL(KIND(1d0))::scalarprod
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p1,p2
!    REAL(KIND(1d0)),PARAMETER::EPS2=1d-6
!    REAL(KIND(1d0))::max
    scalarprod=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
    IF(check.OR.ONSHELL_IREGI)THEN
       CALL makeonshell(nmass,massref(1:nmass),scalarprod)
!    ELSE
!       IF(ABS(scalarprod/mu_R_IREGI**2).LT.1d-6)scalarprod=0d0
    ENDIF
!    max=ABS(p1(0)*p2(0))+ABS(p1(1)*p2(1))+ABS(p1(2)*p2(2))+ABS(p1(3)*p2(3))
!    IF(ABS(scalarprod/max).LT.EPS2)scalarprod=0d0
  END FUNCTION scalarprod

 ! this is the ad-hoc function only for testing 
  SUBROUTINE makeonshell(n,mt2,ma)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n
    REAL(KIND(1d0)),INTENT(INOUT)::ma
    REAL(KIND(1d0)),DIMENSION(n),INTENT(IN)::mt2
    INTEGER::i
    IF(n.GT.0)THEN
       IF(ABS(ma/mt2(1)).LT.ZEROTHR_IREGI)THEN
          ma=0d0
          RETURN
       ENDIF
       DO i=1,n
          IF(ABS(ma-mt2(i))/mt2(i).LT.ZEROTHR_IREGI)THEN
             ma=mt2(i)
             RETURN
          ENDIF
       ENDDO
    ELSE
       IF(ABS(ma).LT.1d-3)THEN
          ma=0d0
          RETURN
       ENDIF
    ENDIF
    RETURN
  END SUBROUTINE makeonshell
END MODULE kinematics
