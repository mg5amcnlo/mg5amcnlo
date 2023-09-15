MODULE ReadParam
  USE ParamModule
  IMPLICIT NONE
  CHARACTER*20::prefix='mdl_'
CONTAINS
  SUBROUTINE ReadParamCard(paramcard)
    IMPLICIT NONE
    CHARACTER(len=*),INTENT(IN)::paramcard
    INTEGER::npara
    INTEGER,PARAMETER::maxpara=1000
    CHARACTER*20,DIMENSION(maxpara)::param,value
    INTEGER::i
    LOGICAL::lexist
!    REAL(KIND(1d0)),PARAMETER::pi=3.1415926535897932384626433832795d0
    IF(.NOT.print_banner)THEN
       INCLUDE "banner.inc"
       print_banner=.TRUE.
    ENDIF
    INQUIRE(FILE=identcard,EXIST=lexist)
    IF(.NOT.lexist)THEN
       CALL WriteIdenCard
       identcard="ident_card.dat"
    ENDIF
    CALL LHA_loadcard(paramcard,npara,param,value)
    CALL LHA_get_real(npara,param,value,'mu_r',Decay_MU_R,9.118800D+01)
    CALL LHA_get_real(npara,param,value,TRIM(prefix)//'mt',Decay_MT,173.3d0)
    CALL LHA_get_real(npara,param,value,TRIM(prefix)//'mh',Decay_MH,125.0d0)
    CALL LHA_get_real(npara,param,value,TRIM(prefix)//'mz',Decay_MZ,91.188d0)
    CALL LHA_get_real(npara,param,value,TRIM(prefix)//'mw',Decay_MW,80.409d0)
    IF(Decay_scheme.EQ.1)THEN
       CALL LHA_get_real(npara,param,value,'aewm1',Decay_aEWM1,1.325070D+02)
    ELSE
       CALL LHA_get_real(npara,param,value,'gf',Decay_Gf,1.166390D-05)
    ENDIF
    CALL LHA_get_real(npara,param,value,'as',Decay_As,1.180000D-01)
    IF(Decay_scheme.EQ.1)THEN
       ! alpha(MZ) scheme
       Decay_e=DSQRT(4d0*pi*Decay_aEWM1**(-1))
       alphaQCD2=Decay_As
       Decay_gs=DSQRT(4d0*pi*Decay_aS)
       Decay_Gf=Decay_aEWM1**(-1)/DSQRT(2d0)*pi/Decay_MW**2&
            /(1d0-Decay_MW**2/Decay_MZ**2)
    ELSE
       ! Gmu scheme
       Decay_aEWM1=DSQRT(2d0)*Decay_Gf*Decay_MW**2&
            *(1d0-Decay_MW**2/Decay_MZ**2)/pi
       Decay_aEWM1=Decay_aEWM1**(-1)
       Decay_e=DSQRT(4d0*pi*Decay_aEWM1**(-1))
       alphaQCD2=Decay_As
       Decay_gs=DSQRT(4d0*pi*Decay_aS)
    ENDIF
    Decay_R_CW2=Decay_MW**2/Decay_MZ**2
    Decay_R_SW2=1d0-Decay_R_CW2
    Decay_SW=DSQRT(Decay_R_SW2)
    Decay_CW=DSQRT(Decay_R_CW2)
    WRITE(*,*)' External Params in SMWidth'
    WRITE(*,*)' -------------------------------------'
    WRITE(*,*)' '
    WRITE(*,*)'MU_R = ',Decay_MU_R
    WRITE(*,*)'aEWM1 = ',Decay_aEWM1
    WRITE(*,*)'Gf = ',Decay_Gf
    WRITE(*,*)'aS(MZ) = ',Decay_aS
    WRITE(*,*)'MT = ',Decay_MT
    WRITE(*,*)'MZ = ',Decay_MZ
    WRITE(*,*)'MW = ',Decay_MW
    WRITE(*,*)'MH = ',Decay_MH
    RETURN
  END SUBROUTINE ReadParamCard

  SUBROUTINE WriteIdenCard
    IMPLICIT NONE
    INTEGER::iunit=432
    LOGICAL::lexist
    !INTEGER::decay_scheme
    IF(.NOT.Skip_scheme)THEN
       INQUIRE(FILE="scheme.dat",EXIST=lexist)
       IF(.NOT.lexist)THEN
          WRITE(*,*)"ERROR:Cannot open scheme.dat ! Don't know which scheme !"
          STOP
       ENDIF
       OPEN(UNIT=iunit,FILE="scheme.dat",ACTION="READ",STATUS="OLD")
       READ(iunit,*)Decay_scheme
    ENDIF
    IF(Decay_scheme.NE.1.AND.Decay_scheme.NE.2)THEN
       WRITE(*,*)"ERROR:Ilegal scheme in scheme.dat:",Decay_scheme
       WRITE(*,*)"Specify 1(alpha(MZ)-scheme) or 2(Gmu-scheme)"
       IF(.NOT.Skip_scheme)THEN
          CLOSE(iunit)
       ENDIF
       STOP
    ENDIF
    IF(.NOT.Skip_scheme)THEN
       CLOSE(iunit)
    ENDIF
    OPEN(UNIT=iunit,FILE="ident_card.dat")
    WRITE(iunit,*)" "
    WRITE(iunit,*)" "
    WRITE(iunit,*)"loop 666 MU_R"
    WRITE(iunit,*)" "
    IF(Decay_scheme.EQ.1)THEN
       WRITE(iunit,*)"sminputs 1 aEWM1"
    ELSE
       WRITE(iunit,*)"sminputs 2 Gf"
    ENDIF
    WRITE(iunit,*)" "
    WRITE(iunit,*)"sminputs 3 aS"
    WRITE(iunit,*)" "
    WRITE(iunit,*)"mass 6 "//TRIM(prefix)//"MT"
    WRITE(iunit,*)" "
    WRITE(iunit,*)"mass 23 "//TRIM(prefix)//"MZ"
    WRITE(iunit,*)" "
    WRITE(iunit,*)"mass 24 "//TRIM(prefix)//"MW"
    WRITE(iunit,*)" "
    WRITE(iunit,*)"mass 25 "//TRIM(prefix)//"MH"
    WRITE(iunit,*)" "
    CLOSE(iunit)
  END SUBROUTINE WriteIdenCard
END MODULE ReadParam
