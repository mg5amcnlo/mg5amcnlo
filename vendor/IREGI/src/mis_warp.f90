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
    INTEGER::ii
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

  FUNCTION I0CC1(NLOOPLINE,PCL,M2L)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::I0CC1
    REAL(KIND(1d0))::p12,p22,p32,p42,s12,s23
    REAL(KIND(1d0)),DIMENSION(0:3)::p1,p2,p3,p4,p1p2,p2p3
    SELECT CASE(NLOOPLINE)
       CASE(1)
          I0CC1(1:4)=A0CC1(M2L(1))
          RETURN
       CASE(2)
          p1(0:3)=PCL(2,0:3)-PCL(1,0:3)
          p12=scalarprod(p1,p1)
          I0CC1(1:4)=B0CC1(p12,M2L(1),M2L(2))
          RETURN
       CASE(3)
          p1(0:3)=PCL(2,0:3)-PCL(1,0:3)
          p2(0:3)=PCL(3,0:3)-PCL(2,0:3)
          p3(0:3)=PCL(1,0:3)-PCL(3,0:3)
          p12=scalarprod(p1,p1)
          p22=scalarprod(p2,p2)
          p32=scalarprod(p3,p3)
          I0CC1(1:4)=C0CC1(p12,p22,p32,M2L(1),M2L(2),M2L(3))
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
          I0CC1(1:4)=D0CC1(p12,p22,p32,p42,s12,s23,&
               M2L(1),M2L(2),M2L(3),M2L(4))
          RETURN
    END SELECT
  END FUNCTION I0CC1
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

  FUNCTION A0CC1(m12)
    IMPLICIT NONE
    COMPLEX(KIND(1d0)),INTENT(IN)::m12
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::A0CC1
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-1:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
!    COMPLEX(KIND(1d0)),EXTERNAL::qlI1
!    IF(first)THEN
!       IF(scalarlib.EQ.1)THEN
!          CALL qlinit
!       ENDIF
!       first=.FALSE.
!    ENDIF
    musq=MU_R_IREGI**2
    A0CC1(2)=DCMPLX(0d0)
    A0CC1(3)=DCMPLX(0d0)
! complex mass can only be used in OneLOop
    CALL olo_scale(MU_R_IREGI)
    CALL olo(zolo, m12)
    temp(-1)=zolo(1)
    temp(0)=zolo(0)
    A0CC1(1)=temp(-1)*ipi2
    A0CC1(4)=temp(0)*ipi2+factor1*temp(-1)
    RETURN
  END FUNCTION A0CC1

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
!       CALL olo_unit(6,'printall')
       CALL olo_scale(MU_R_IREGI)
       CALL olo(zolo, DCMPLX(p12),DCMPLX(m12),DCMPLX(m22))
       temp(-1)=zolo(1)
       temp(0)=zolo(0)
    ENDIF
!    IF(ABS(p12*m12*m22).GT.1d0)THEN
!       temp(-1)=DCMPLX(1d0)
!       temp(0)=DCMPLX(0.745754405349275d0,-4.747187659478749d0)
!    ENDIF
    B0C1(1)=temp(-1)*ipi2
    B0C1(4)=temp(0)*ipi2+factor1*temp(-1)
!    IF(ABS(p12*m12*m22).GT.1d0)THEN
!       B0C1(4)=DCMPLX(46.852864216788554d0,-9.634620424960666d0)
!    ENDIF
!    PRINT *,"======================================"
!    PRINT *,"mass=",p12,m12,m22
!    WRITE(*,'(i2,1x,4e25.17)')1,MU_R_IREGI,p12,m12,m22
!    PRINT *,"mass=",p12,m12,m22
!    PRINT *,zolo(0)
!    PRINT *,zolo(1)
!    CALL olo(zolo,DCMPLX(p12),DCMPLX(m12),DCMPLX(0d0))
!    PRINT *,zolo(0)
    RETURN
  END FUNCTION B0C1

  FUNCTION B0CC1(p12,m12,m22)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12
    COMPLEX(KIND(1d0)),INTENT(IN)::m12,m22
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::B0CC1
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-1:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
!    COMPLEX(KIND(1d0)),EXTERNAL::qlI2
!    IF(first)THEN
!       IF(scalarlib.EQ.1)THEN
!          CALL qlinit
!       ENDIF
!       first=.FALSE.
!    ENDIF
!    STOP
    IF((ABS(p12)+ABS(m12)+ABS(m22))/3d0.LT.EPS)THEN
       B0CC1(1)=ipi2
       B0CC1(2)=-ipi2
       B0CC1(3:4)=DCMPLX(0d0)
       RETURN
    ENDIF
    musq=MU_R_IREGI**2
    B0CC1(2)=DCMPLX(0d0)
    B0CC1(3)=DCMPLX(0d0)
    CALL olo_scale(MU_R_IREGI)
    CALL olo(zolo, DCMPLX(p12),m12,m22)
    temp(-1)=zolo(1)
    temp(0)=zolo(0)
    B0CC1(1)=temp(-1)*ipi2
    B0CC1(4)=temp(0)*ipi2+factor1*temp(-1)
    RETURN
  END FUNCTION B0CC1

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
    INTEGER::ijij
    REAL(KIND(1d0))::p12_col,p22_col,p32_col,m12_col,m22_col,m32_col
    REAL(KIND(1d0)),PARAMETER::eulogpi=1.7219455507509330347d0 ! EulerGamma+Log[Pi]
    REAL(KIND(1d0))::sqrtm22m32
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
!   to determine whether it is a columb singularity or not
    IF(ABS(m12*m22*m32).EQ.0d0.AND.&
         ABS(m12*m22)+ABS(m22*m32)+ABS(m12*m32).NE.0d0.AND.&
         p12*p22*p32.NE.0d0)THEN
      DO ijij=1,6
         ! propagator symmetry
         SELECT CASE(ijij)
         CASE(1)
            p12_col=p12
            p22_col=p22
            p32_col=p32
            m12_col=m12
            m22_col=m22
            m32_col=m32
         CASE(2)
            p12_col=p12
            p22_col=p32
            p32_col=p22
            m12_col=m22
            m22_col=m12
            m32_col=m32
         CASE(3)
            p12_col=p22
            p22_col=p12
            p32_col=p32
            m12_col=m32
            m22_col=m22
            m32_col=m12
         CASE(4)
            p12_col=p32
            p22_col=p22
            p32_col=p12
            m12_col=m12
            m22_col=m32
            m32_col=m22
         CASE(5)
            p12_col=p22
            p22_col=p32
            p32_col=p12
            m12_col=m22
            m22_col=m32
            m32_col=m12
         CASE(6)
            p12_col=p32
            p22_col=p12
            p32_col=p22
            m12_col=m32
            m22_col=m12
            m32_col=m22
         END SELECT
         IF(ColumbCase(p12_col,p22_col,p32_col,m12_col,m22_col,m32_col))THEN
            C0C1(3)=DCMPLX(0d0)
            sqrtm22m32=DSQRT(m22_col*m32_col)
            C0C1(2)=-ipi2/2d0/sqrtm22m32
            C0C1(4)=ipi2*(0.5d0/sqrtm22m32*eulogpi&
                 -DLOG(musq/m22_col)/2d0/sqrtm22m32&
                 -DLOG(m22_col/m32_col)/2d0/(sqrtm22m32+m32)&
                 +1d0/sqrtm22m32)
            RETURN
         ENDIF
      END DO
    ENDIF
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

  FUNCTION ColumbCase(p12,p22,p32,m12,m22,m32)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,p22,p32,m12,m22,m32
    LOGICAL::ColumbCase
    REAL(KIND(1d0))::m2m3
    REAL(KIND(1d0)),PARAMETER::osthr=1d-4
    ColumbCase=.FALSE.
    IF(ABS(m12).NE.0)RETURN
    IF(ABS(m22).EQ.0.OR.ABS(p12-m22)/m22.GT.osthr)RETURN
    IF(ABS(m32).EQ.0.OR.ABS(p32-m32)/m22.GT.osthr)RETURN
    m2m3=(SQRT(m22)+SQRT(m32))**2
    IF(ABS(p22-m2m3)/m2m3.GT.osthr)RETURN
    ColumbCase=.TRUE.
    RETURN
  END FUNCTION ColumbCase

  FUNCTION C0CC1(p12,p22,p32,m12,m22,m32)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,p22,p32
    COMPLEX(KIND(1d0)),INTENT(IN)::m12,m22,m32
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::C0CC1
    REAL(KIND(1d0))::musq
    INTEGER::ep
    COMPLEX(KIND(1d0)),DIMENSION(-2:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
!    COMPLEX(KIND(1d0)),EXTERNAL::qlI3
!    IF(first)THEN
!       IF(scalarlib.EQ.1)THEN
!          CALL qlinit
!       ENDIF
!       first=.FALSE.
!    ENDIF
    musq=MU_R_IREGI**2
    C0CC1(1)=DCMPLX(0d0)
    CALL olo_scale(MU_R_IREGI)
    CALL olo(zolo, DCMPLX(p12),DCMPLX(p22),DCMPLX(p32),&
         m12,m22,m32)
    temp(-2)=zolo(2)
    temp(-1)=zolo(1)
    temp(0)=zolo(0)
    C0CC1(3)=temp(-2)*ipi2
    C0CC1(2)=temp(-1)*ipi2+temp(-2)*factor1
    C0CC1(4)=temp(0)*ipi2+temp(-1)*factor1+temp(-2)*factor2
    RETURN
  END FUNCTION C0CC1

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

  FUNCTION D0CC1(p12,p22,p32,p42,s12,s23,m12,m22,m32,m42)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::p12,p22,p32,p42,s12,s23
    COMPLEX(KIND(1d0)),INTENT(IN)::m12,m22,m32,m42
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::D0CC1
    REAL(KIND(1d0))::musq
    INTEGER::ep,flag
    COMPLEX(KIND(1d0)),DIMENSION(-2:0)::temp
    COMPLEX(KIND(1d0)),DIMENSION(0:2)::zolo
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2
    COMPLEX(KIND(1d0)),PARAMETER::factor1=DCMPLX(0d0,-16.994921386127647d0)
    COMPLEX(KIND(1d0)),PARAMETER::factor2=DCMPLX(0d0,6.514740380268655d0)
!    COMPLEX(KIND(1d0)),EXTERNAL::qlI4
!    IF(first)THEN
!       IF(scalarlib.EQ.1)THEN
!          CALL qlinit
!       ENDIF
!       first=.FALSE.
!    ENDIF
    musq=MU_R_IREGI**2
    D0CC1(1)=DCMPLX(0d0)
    flag=0
    CALL olo_scale(MU_R_IREGI)
    IF((ABS(s12-m32).LT.EPS.OR.ABS(s23-m42).LT.EPS&
         .OR.ABS(s12-m12).LT.EPS.OR.ABS(s23-m22).LT.EPS))THEN
       IF(ABS(p42-m12).GE.EPS.AND.ABS(p42-m42).GE.EPS&
            .AND.ABS(p22-m32).GE.EPS.AND.ABS(p22-m22).GE.EPS)THEN
          ! {p12, s23, p32, s12, p42, p22, m12, m22, m42, m32}
          CALL olo(zolo, DCMPLX(p12),DCMPLX(s23),DCMPLX(p32),DCMPLX(s12),&
                  DCMPLX(p42),DCMPLX(p22),m12,m22,m42,m32)
          flag=1
       ELSEIF(ABS(p12-m12).GE.EPS.AND.ABS(p12-m22).GE.EPS&
            .AND.ABS(p32-m32).GE.EPS.AND.ABS(p32-m42).GE.EPS)THEN
          ! {s12, p22, s23, p42, p12, p32, m12, m32, m22, m42}
          CALL olo(zolo, DCMPLX(s12),DCMPLX(p22),DCMPLX(s23),DCMPLX(p42),&
               DCMPLX(p12),DCMPLX(p32),m12,m32,m22,m42)
          flag=1
       ENDIF
    ENDIF
    IF(flag.EQ.0)THEN
       ! {p12, p22, p32, p42, s12, s23, m12, m22, m32, m42}
       CALL olo(zolo, DCMPLX(p12),DCMPLX(p22),DCMPLX(p32),DCMPLX(p42),&
            DCMPLX(s12),DCMPLX(s23),m12,m22,m32,m42)
    ENDIF
    temp(-2)=zolo(2)
    temp(-1)=zolo(1)
    temp(0)=zolo(0)
    D0CC1(3)=temp(-2)*ipi2
    D0CC1(2)=temp(-1)*ipi2+temp(-2)*factor1
    D0CC1(4)=temp(0)*ipi2+temp(-1)*factor1+temp(-2)*factor2
    RETURN
  END FUNCTION D0CC1
END MODULE mis_warp
