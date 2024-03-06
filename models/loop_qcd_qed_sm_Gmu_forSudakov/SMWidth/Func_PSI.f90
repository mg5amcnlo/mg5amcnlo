MODULE func_psi
USE use_func
USE ParamModule
IMPLICIT NONE
!INTEGER,PARAMETER::SGL=SELECTED_REAL_KIND(p=6)
!INTEGER,PARAMETER::DBL=SELECTED_REAL_KIND(p=13)
!REAL(KIND=DBL),PARAMETER::pi=3.1415926535897932385d0
!REAL(KIND=DBL),PARAMETER::EulerGamma=0.57721566490153286061d0
INTERFACE unit_step
   MODULE PROCEDURE unit_step_d
   MODULE PROCEDURE unit_step_s
   MODULE PROCEDURE unit_step_i
END INTERFACE unit_step
INTERFACE Arg
   MODULE PROCEDURE ArgI
   MODULE PROCEDURE ArgR
   MODULE PROCEDURE ArgC
END INTERFACE Arg
INTERFACE Log2
   MODULE PROCEDURE LogR
   MODULE PROCEDURE LogI
   MODULE PROCEDURE LogC
END INTERFACE Log2
INTERFACE CSQRT2
   MODULE PROCEDURE CSQRTC
   MODULE PROCEDURE CSQRTR
   MODULE PROCEDURE CSQRTI
END INTERFACE CSQRT2
INTERFACE li2
   MODULE PROCEDURE li2I
   MODULE procedure li2R
   MODULE procedure li2C
END INTERFACE li2
CONTAINS
FUNCTION lamda_func(a,b,c)
REAL(KIND=DBL)::lamda_func
REAL(KIND=DBL),INTENT(IN)::a,b,c
lamda_func=a**2+b**2+c**2-2*a*b-2*a*c-2*b*c
END FUNCTION lamda_func

FUNCTION unit_step_d(x)
REAL(KIND=DBL)::unit_step_d
REAL(KIND=DBL)::x
IF(x.LT.0d0) THEN
   unit_step_d=0d0
ELSE
   unit_step_d=1d0
END IF
END FUNCTION unit_step_d

FUNCTION unit_step_s(x)
REAL(KIND=DBL)::unit_step_s
REAL(KIND=SGL)::x
IF(x.LT.0.0) THEN
   unit_step_s=0d0
ELSE
   unit_step_s=1d0
END IF
END FUNCTION unit_step_s

FUNCTION unit_step_i(x)
REAL(KIND=DBL)::unit_step_i
INTEGER::x
IF(x.LT.0) THEN
   unit_step_i=0d0
ELSE
   unit_step_i=1d0
END IF
END FUNCTION unit_step_i

FUNCTION ACosh_p(x)
REAL(KIND=DBL),INTENT(IN)::x
REAL(KIND=DBL)::ACosh_p
IF(DABS(x).LT.1d0) THEN
  WRITE(*,*)"Error in ACosh_p!"
  ACosh_p=0
ELSE
  ACosh_p=DLOG(x+DSQRT(x**2-1d0))
END IF
END FUNCTION ACosh_p
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Li2 function
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION li2R(x)
  COMPLEX(KIND=DBL)::li2R
  REAL(KIND=DBL),INTENT(IN)::x
  REAL(KIND=DBL)::y
  y=1d0-x
  IF(y.GT.0d0) THEN
     !   IF(y.GT.1d0)WRITE(*,*)"ATTENTION!"
     li2R=func_li2a(y,0)
  ELSEIF(y.LT.0d0)THEN
     li2R=func_li2a(-y,1)
  ELSE
     li2R=DCMPLX(0d0)
  END IF
  RETURN
END FUNCTION li2R

FUNCTION li2I(x)
  COMPLEX(KIND=DBL)::li2I
  INTEGER,INTENT(IN)::x
  REAL(KIND(1d0))::xx
  REAL(KIND=DBL)::y
  xx=x
  y=1d0-xx
  IF(y.GT.0d0) THEN
     li2I=func_li2a(y,0)
  ELSEIF(y.LT.0d0)THEN
     li2I=func_li2a(-y,1)
  ELSE
     li2I=DCMPLX(0d0)
  END IF
  RETURN
END FUNCTION li2I

FUNCTION li2C(x)
  COMPLEX(KIND=DBL)::li2C
  COMPLEX(KIND=DBL),INTENT(IN)::x
  COMPLEX(KIND=DBL)::y,newy
  REAL(KIND=DBL)::yr,yi
  INTEGER::iphi
  y=DCMPLX(1d0)-x
  yr=DBLE(y)
  yi=DIMAG(y)
  IF(yr.EQ.0d0.AND.yi.EQ.0d0)THEN
     li2C=DCMPLX(0d0)
     RETURN
  ENDIF
  IF(yr.GE.0d0.AND.yi.GE.0d0)THEN
     iphi=0
     newy=y
  ELSEIF(yr.GE.0d0.AND.yi.LT.0d0)THEN
     iphi=0
     newy=y
  ELSEIF(yr.LT.0d0.AND.yi.GE.0d0)THEN
     iphi=1
     newy=DCMPLX(-yr,-yi)
  ELSEIF(yr.LT.0d0.AND.yi.LT.0d0)THEN
     iphi=-1
     newy=DCMPLX(-yr,-yi)
  END IF
  li2C=func_li2c(newy,iphi)
  RETURN
END FUNCTION li2C

FUNCTION PolyLog(n,x)
  COMPLEX(KIND=DBL)::PolyLog
  REAL(KIND=DBL),INTENT(IN)::x
  INTEGER,INTENT(IN)::n
  IF(n.NE.2)THEN
     WRITE(*,*)"ERROR:Don't know n = ",n
     STOP
  ENDIF
  PolyLog=li2R(x)
  RETURN
END FUNCTION PolyLog
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! DLOG2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION DLOG2(x)
REAL(KIND=DBL)::DLOG2
REAL(KIND=DBL),INTENT(IN)::x
IF(x.LT.0d0)THEN
   DLOG2=DLOG(-x)
ELSE
   DLOG2=DLOG(x)
ENDIF
END FUNCTION DLOG2

FUNCTION LogI(x)
  COMPLEX(KIND(1d0))::LogI
  INTEGER,INTENT(IN)::x
  REAL(KIND(1d0))::xx
  xx=x
  LogI=DLOG(xx)
  RETURN
END FUNCTION LogI

FUNCTION LogR(x)
  COMPLEX(KIND(1d0))::LogR
  REAL(KIND(1d0)),INTENT(IN)::x
  LogR=DLOG(x)
  RETURN
END FUNCTION LogR

FUNCTION LogC(x)
  COMPLEX(KIND(1d0))::LogC
  COMPLEX(KIND(1d0)),INTENT(IN)::x
  LogC=CDLOG(x)
  RETURN
END FUNCTION LogC
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Arg
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION ArgR(x)
  IMPLICIT NONE
  REAL(KIND=DBL),INTENT(IN)::x
  REAL(KIND=DBL)::ArgR
  IF(x.GE.0d0)THEN
     ArgR=0d0
  ELSE
     ArgR=pi
  ENDIF
  RETURN
END FUNCTION ArgR

FUNCTION ArgI(x)
  IMPLICIT NONE
  INTEGER,INTENT(IN)::x
  REAL(KIND=DBL)::ArgI
  IF(x.GE.0)THEN
     ArgI=0d0
  ELSE
     ArgI=pi
  ENDIF
  RETURN
END FUNCTION ArgI

FUNCTION ArgC(x)
  IMPLICIT NONE
  COMPLEX(KIND=DBL),INTENT(IN)::x
  REAL(KIND=DBL)::ArgC
  REAL(KIND=DBL)::xr,xi
  REAL(KIND=DBL)::add,time
  xr=DBLE(x)
  xi=DIMAG(x)
  IF(xr.EQ.0d0)THEN
     IF(xi.GT.0d0)THEN
        ArgC=pi/2d0
        RETURN
     ENDIF
     IF(xi.LT.0d0)THEN
        ArgC=-pi/2d0
        RETURN
     ENDIF
     IF(xi.EQ.0d0)THEN
        ArgC=0d0
        RETURN
     ENDIF
  ENDIF
  IF(xr.GE.0d0.AND.xi.GE.0d0)THEN
     add=0d0
     time=1d0
  ELSEIF(xr.GE.0d0.AND.xi.LT.0d0)THEN
     add=0d0
     time=-1d0
  ELSEIF(xr.LT.0d0.AND.xi.GE.0d0)THEN
     add=pi
     time=-1d0
  ELSEIF(xr.LT.0d0.AND.xi.LE.0d0)THEN
     add=-pi
     time=1d0
  ENDIF
  ArgC=time*atan(abs(xi/xr))+add
  RETURN
END FUNCTION ArgC
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! CSQRT2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FUNCTION CSQRTI(x)
  IMPLICIT NONE
  COMPLEX(KIND(1d0))::CSQRTI
  INTEGER,INTENT(IN)::x
  REAL(KIND(1d0))::xx
  COMPLEX(KIND(1d0))::xxx
  IF(x.GE.0)THEN
     xx=x
     CSQRTI=DCMPLX(DSQRT(xx))
  ELSE
     xxx=x
     CSQRTI=CDSQRT(xxx)
  ENDIF
  RETURN
END FUNCTION CSQRTI

FUNCTION CSQRTR(x)
  IMPLICIT NONE
  COMPLEX(KIND(1d0))::CSQRTR
  REAL(KIND(1d0)),INTENT(IN)::x
  COMPLEX(KIND(1d0))::xx
  IF(x.GE.0d0)THEN
     CSQRTR=DCMPLX(DSQRT(x))
  ELSE
     xx=x
     CSQRTR=CDSQRT(xx)
  ENDIF
  RETURN
END FUNCTION CSQRTR

FUNCTION CSQRTC(x)
  IMPLICIT NONE
  COMPLEX(KIND(1d0))::CSQRTC
  COMPLEX(KIND(1d0)),INTENT(IN)::x
  CSQRTC=CDSQRT(x)
  RETURN
END FUNCTION CSQRTC
END MODULE func_psi
