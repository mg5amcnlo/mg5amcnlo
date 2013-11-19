MODULE mis_warp
USE global
USE kinematics
USE avh_olo
IMPLICIT NONE
LOGICAL::first=.TRUE.
SAVE first
CONTAINS

  FUNCTION I0C2(NLOOPLINE,PijMatrix,M2L)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix ! PijMatrix(i,j)=pi.pj
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::I0C2
    REAL(KIND(1d0))::p12,p22,p32,p42,s12,s23
    SELECT CASE(NLOOPLINE)
       CASE(1)
          I0C2(1:4)=A0C1(M2L(1))
          RETURN
       CASE(2)
          p12=PijMatrix(2,2)+PijMatrix(1,1)-2d0*PijMatrix(1,2)
          I0C2(1:4)=B0C1(p12,M2L(1),M2L(2))
          RETURN
       CASE(3)
          p12=PijMatrix(2,2)+PijMatrix(1,1)-2d0*PijMatrix(1,2)
          p22=PijMatrix(3,3)+PijMatrix(2,2)-2d0*PijMatrix(2,3)
          p32=PijMatrix(3,3)+PijMatrix(1,1)-2d0*PijMatrix(1,3)
          I0C2(1:4)=C0C1(p12,p22,p32,M2L(1),M2L(2),M2L(3))
          RETURN
       CASE(4)          
          p12=PijMatrix(2,2)+PijMatrix(1,1)-2d0*PijMatrix(1,2)
          p22=PijMatrix(3,3)+PijMatrix(2,2)-2d0*PijMatrix(3,2)
          p32=PijMatrix(4,4)+PijMatrix(3,3)-2d0*PijMatrix(4,3)
          p42=PijMatrix(4,4)+PijMatrix(1,1)-2d0*PijMatrix(4,1)
          s12=PijMatrix(3,3)+PijMatrix(1,1)-2d0*PijMatrix(3,1)
          s23=PijMatrix(4,4)+PijMatrix(2,2)-2d0*PijMatrix(4,2)
          !PRINT *, p12,p22,p32,p42,s12,s23,M2L(1),M2L(2),M2L(3),M2L(4)
          I0C2(1:4)=D0C1(p12,p22,p32,p42,s12,s23,&
               M2L(1),M2L(2),M2L(3),M2L(4))
          RETURN
    END SELECT
  END FUNCTION I0C2

  FUNCTION I0C1(NLOOPLINE,PCL,M2L)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::I0C1
    REAL(KIND(1d0))::p12,p22,p32,p42,s12,s23
    REAL(KIND(1d0)),DIMENSION(0:3)::p1,p2,p3,p4,p1p2,p2p3
    SELECT CASE(NLOOPLINE)
       CASE(1)
          I0C1(1:4)=A0C1(M2L(1))
          RETURN
       CASE(2)
          p1(0:3)=PCL(2,0:3)-PCL(1,0:3)
          p12=scalarprod(p1,p1)
          I0C1(1:4)=B0C1(p12,M2L(1),M2L(2))
          RETURN
       CASE(3)
          p1(0:3)=PCL(2,0:3)-PCL(1,0:3)
          p2(0:3)=PCL(3,0:3)-PCL(2,0:3)
          p3(0:3)=PCL(1,0:3)-PCL(3,0:3)
          p12=scalarprod(p1,p1)
          p22=scalarprod(p2,p2)
          p32=scalarprod(p3,p3)
          I0C1(1:4)=C0C1(p12,p22,p32,M2L(1),M2L(2),M2L(3))
          RETURN
       CASE(4)
          p1(0:3)=PCL(2,0:3)-PCL(1,0:3)
          p2(0:3)=PCL(3,0:3)-PCL(2,0:3)
          p3(0:3)=PCL(4,0:3)-PCL(3,0:3)
          p4(0:3)=PCL(1,0:3)-PCL(4,0:3)
          p1p2(0:3)=PCL(3,0:3)-PCL(1,0:3)
          p2p3(0:3)=PCL(4,0:3)-PCL(2,0:3)
          p12=scalarprod(p1,p1)
          p22=scalarprod(p2,p2)
          p32=scalarprod(p3,p3)
          p42=scalarprod(p4,p4)
          s12=scalarprod(p1p2,p1p2)
          s23=scalarprod(p2p3,p2p3)
          !PRINT *,p12,p22,p32,p42,s12,s23,M2L(1),M2L(2),M2L(3),M2L(4) 
          I0C1(1:4)=D0C1(p12,p22,p32,p42,s12,s23,&
               M2L(1),M2L(2),M2L(3),M2L(4))
          RETURN
    END SELECT
  END FUNCTION I0C1
  ! dummy functions
  ! replace the correct mis package,e.g.QCDLoop,OneLOop here

  FUNCTION A0C1(m12)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::m12
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::A0C1
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-1:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
    COMPLEX(KIND(1d0)),EXTERNAL::qlI1
    IF(first)THEN
!       OPEN(UNIT=202,FILE="scalarlib.inp")
!       READ(202,*)scalarlib
!       CLOSE(202)
       IF(scalarlib.EQ.1)THEN
          CALL qlinit
!       ELSE
!          CALL ltini
       ENDIF
       first=.FALSE.
    ENDIF
    musq=MU_R_IREGI**2
    A0C1(2)=DCMPLX(0d0)
    A0C1(3)=DCMPLX(0d0)
    IF(scalarlib.EQ.1)THEN
       DO ep=0,-1,-1
          temp(ep)=qlI1(m12,musq,ep)
       ENDDO
    ELSE
       CALL olo_scale(MU_R_IREGI)
       CALL olo(zolo, DCMPLX(m12))
       temp(-1)=zolo(1)
       temp(0)=zolo(0)
    ENDIF
    A0C1(1)=temp(-1)*ipi2
    A0C1(4)=temp(0)*ipi2+factor1*temp(-1)
    RETURN
  END FUNCTION A0C1

  FUNCTION B0C1(p12,m12,m22)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,m12,m22
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::B0C1
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-1:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
    COMPLEX(KIND(1d0)),EXTERNAL::qlI2
    IF(first)THEN
!       OPEN(UNIT=202,FILE="scalarlib.inp")
!       READ(202,*)scalarlib
!       CLOSE(202)
       IF(scalarlib.EQ.1)THEN
          CALL qlinit
       ENDIF
       first=.FALSE.
    ENDIF
    IF((ABS(p12)+ABS(m12)+ABS(m22))/3d0.LT.EPS)THEN
       B0C1(1)=ipi2
       B0C1(2)=-ipi2
       B0C1(3:4)=DCMPLX(0d0)
       RETURN
    ENDIF
    musq=MU_R_IREGI**2
    B0C1(2)=DCMPLX(0d0)
    B0C1(3)=DCMPLX(0d0)
    IF(scalarlib.EQ.1)THEN
       DO ep=0,-1,-1
          temp(ep)=qlI2(p12,m12,m22,musq,ep)
       ENDDO
    ELSE
       CALL olo_scale(MU_R_IREGI)
       CALL olo(zolo, DCMPLX(p12),DCMPLX(m12),DCMPLX(m22))
       temp(-1)=zolo(1)
       temp(0)=zolo(0)
    ENDIF
    B0C1(1)=temp(-1)*ipi2
    B0C1(4)=temp(0)*ipi2+factor1*temp(-1)
    RETURN
  END FUNCTION B0C1


  FUNCTION C0C1(p12,p22,p32,m12,m22,m32)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,p22,p32,m12,m22,m32
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::C0C1
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-2:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2 
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
    COMPLEX(KIND(1d0)),EXTERNAL::qlI3
    IF(first)THEN
!       OPEN(UNIT=202,FILE="scalarlib.inp")
!       READ(202,*)scalarlib
!       CLOSE(202)
       IF(scalarlib.EQ.1)THEN
          CALL qlinit
       ENDIF
       first=.FALSE.
    ENDIF
    musq=MU_R_IREGI**2
    C0C1(1)=DCMPLX(0d0)
    IF(scalarlib.EQ.1)THEN
       DO ep=0,-2,-1
          temp(ep)=qlI3(p12,p22,p32,m12,m22,m32,musq,ep)
       ENDDO
    ELSE
       CALL olo_scale(MU_R_IREGI)
       CALL olo(zolo, DCMPLX(p12),DCMPLX(p22),DCMPLX(p32),&
            DCMPLX(m12),DCMPLX(m22),DCMPLX(m32))
       temp(-2)=zolo(2)
       temp(-1)=zolo(1)
       temp(0)=zolo(0)
    ENDIF
    C0C1(3)=temp(-2)*ipi2
    C0C1(2)=temp(-1)*ipi2+temp(-2)*factor1
    C0C1(4)=temp(0)*ipi2+temp(-1)*factor1+temp(-2)*factor2
    RETURN
  END FUNCTION C0C1


  FUNCTION D0C1(p12,p22,p32,p42,s12,s23,m12,m22,m32,m42)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,p22,p32,p42,s12,s23,m12,m22,m32,m42
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::D0C1
    REAL(KIND(1d0))::musq
    INTEGER::ep,flag
    COMPLEX(KIND(1d0)),DIMENSION(-2:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2 
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
    COMPLEX(KIND(1d0)),EXTERNAL::qlI4
    IF(first)THEN
!       OPEN(UNIT=202,FILE="scalarlib.inp")
!       READ(202,*)scalarlib
!       CLOSE(202)
       IF(scalarlib.EQ.1)THEN
          CALL qlinit
       ENDIF
       first=.FALSE.
    ENDIF
    musq=MU_R_IREGI**2
    D0C1(1)=DCMPLX(0d0)
    flag=0
    IF(scalarlib.EQ.1)THEN
       IF((ABS(s12-m32).LT.EPS.OR.ABS(s23-m42).LT.EPS.OR.&
            ABS(s12-m12).LT.EPS.OR.ABS(s23-m22).LT.EPS))THEN
            IF(ABS(p42-m42).GE.EPS.AND.ABS(p22-m32).GE.EPS&
            .AND.ABS(p42-m12).GE.EPS.AND.ABS(p22-m22).GE.EPS)THEN
               ! {p12, s23, p32, s12, p42, p22, m12, m22, m42, m32}
               DO ep=0,-2,-1
                  temp(ep)=qlI4(p12,s23,p32,s12,p42,p22,m12,m22,m42,m32,musq,ep)
               ENDDO
               flag=1
            ELSEIF(ABS(p12-m12).GE.EPS.AND.ABS(p12-m22).GE.EPS&
                 .AND.ABS(p32-m32).GE.EPS.AND.ABS(p32-m42).GE.EPS)THEN
               ! {s12, p22, s23, p42, p12, p32, m12, m32, m22, m42}
               DO ep=0,-2,-1
                  temp(ep)=qlI4(s12,p22,s23,p42,p12,p32,m12,m32,m22,m42)
               ENDDO
               flag=1
            ENDIF
       ENDIF
       IF(flag.EQ.0)THEN
          ! {p12, p22, p32, p42, s12, s23, m12, m22, m32, m42}
          DO ep=0,-2,-1
             temp(ep)=qlI4(p12,p22,p32,p42,s12,s23,m12,m22,m32,m42,musq,ep)
          ENDDO
       ENDIF
    ELSE
       CALL olo_scale(MU_R_IREGI)
       IF((ABS(s12-m32).LT.EPS.OR.ABS(s23-m42).LT.EPS&
            .OR.ABS(s12-m12).LT.EPS.OR.ABS(s23-m22).LT.EPS))THEN
          IF(ABS(p42-m12).GE.EPS.AND.ABS(p42-m42).GE.EPS&
               .AND.ABS(p22-m32).GE.EPS.AND.ABS(p22-m22).GE.EPS)THEN
             ! {p12, s23, p32, s12, p42, p22, m12, m22, m42, m32}
             CALL olo(zolo, DCMPLX(p12),DCMPLX(s23),DCMPLX(p32),DCMPLX(s12),&
                  DCMPLX(p42),DCMPLX(p22),DCMPLX(m12),DCMPLX(m22),&
                  DCMPLX(m42),DCMPLX(m32))
             flag=1
          ELSEIF(ABS(p12-m12).GE.EPS.AND.ABS(p12-m22).GE.EPS&
               .AND.ABS(p32-m32).GE.EPS.AND.ABS(p32-m42).GE.EPS)THEN
             ! {s12, p22, s23, p42, p12, p32, m12, m32, m22, m42} 
             CALL olo(zolo, DCMPLX(s12),DCMPLX(p22),DCMPLX(s23),DCMPLX(p42),&
                  DCMPLX(p12),DCMPLX(p32),DCMPLX(m12),DCMPLX(m32),&
                  DCMPLX(m22),DCMPLX(m42))
             flag=1
          ENDIF
       ENDIF
       IF(flag.EQ.0)THEN
          ! {p12, p22, p32, p42, s12, s23, m12, m22, m32, m42}
          CALL olo(zolo, DCMPLX(p12),DCMPLX(p22),DCMPLX(p32),DCMPLX(p42),&
               DCMPLX(s12),DCMPLX(s23),DCMPLX(m12),DCMPLX(m22),&
               DCMPLX(m32),DCMPLX(m42))
       ENDIF
       temp(-2)=zolo(2)
       temp(-1)=zolo(1)
       temp(0)=zolo(0)
    ENDIF
    D0C1(3)=temp(-2)*ipi2
    D0C1(2)=temp(-1)*ipi2+temp(-2)*factor1
    D0C1(4)=temp(0)*ipi2+temp(-1)*factor1+temp(-2)*factor2
    RETURN
  END FUNCTION D0C1
END MODULE mis_warp
