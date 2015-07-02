!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                !
! THIS FILE IS THE INTERFACE OF IREGI FOR MADLOOP5 CONVENTION    !
!                                                                !
! H.-S.SHAO, 29/10/2013                                          !
!                                                                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
SUBROUTINE INITIREGI(RECY,LOOPLIB,THR)
  USE global
  IMPLICIT NONE
  INTEGER,INTENT(IN)::LOOPLIB
  REAL(KIND(1d0)),INTENT(IN)::THR
  LOGICAL,INTENT(IN)::RECY
  !RECYCLING=.TRUE.! USE RECYCLING, THEN FOR EACH PS, CLEAR CACHING PLEASE
  NULLIFY(ibp_save)
  NULLIFY(pave_save)
  NULLIFY(shiftpaveden_save)
  NULLIFY(ibp_save2)
  NULLIFY(pave_save2)
  NULLIFY(shiftpaveden_save2)
  NULLIFY(xyzmatrices_save)
  NULLIFY(rsmatrices_save)
  RECYCLING=RECY
  CHECK=.FALSE.
  SELECT CASE(LOOPLIB)
  CASE(1)
     ! LOOPTOOLS, WE WILL USE QCDLOOP INSTEAD
     scalarlib=1
  CASE(2)
     ! AVH,i.e. ONELOOP
     scalarlib=2
  CASE(3)
     ! QCDLOOP
     scalarlib=1
  CASE DEFAULT
     ! USE QCDLOOP FOR OTHER CASES
     scalarlib=1
  END SELECT
  ZEROTHR_IREGI=THR
  ONSHELL_IREGI=.TRUE. ! always set on shell to cure PS
  ML5_CONVENTION=.TRUE. ! set ML5 CONVENTION TO MATCH THAT OF PJFRY++ OR CUTTOOLS
END SUBROUTINE INITIREGI
SUBROUTINE IMLOOP(CTMODE,IMODE,NLOOPLINE,NLOOPCOEFS,RANK,PDEN,M2L,MUR,TIRCOEFS,STABLE)
! CTMODE=1 ---> DO NOTHING
! CTMODE=2 ---> ROTATE THE LOOP PROPAGATOR DIRECTION,i.e D0...DN-1 to DN-1...D0
! CTMODE=4 ---> DO NOTHING BUT IN QUADRUPLE PRECISION (NOT DONE)
! CTMODE=5 ---> ROTATE THE LOOP PROPAGATOR DIRECTION IN QUADRUPLE PRECISION (NOT DONE)
! IMODE=0, IBP reduction
! IMODE=1, PaVe reduction
! IMODE=2, PaVe reduction with stablility improved by IBP reduction 
  USE funlib
  USE ti_reduce
  USE cti_reduce
  USE global
  IMPLICIT NONE
  INTEGER,INTENT(IN)::NLOOPLINE,RANK,NLOOPCOEFS,CTMODE
  REAL(KIND(1d0)),DIMENSION(0:3,NLOOPLINE-1),INTENT(IN)::PDEN
  COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
  REAL(KIND(1d0)),INTENT(IN)::MUR
  COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPCOEFS-1,3),INTENT(OUT)::TIRCOEFS
  LOGICAL,INTENT(OUT)::STABLE
  COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPCOEFS-1,4)::TCOEFS
  !COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPCOEFS-1,3)::TIRCOEFS2
  LOGICAL::TSTABLE
  REAL(KIND(1d0)),DIMENSION(NLOOPLINE)::M2LR
  INTEGER,INTENT(IN)::IMODE
  INTEGER::nlenrank,i,j
  REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PL
  !    REAL(KIND(1d0)),PARAMETER::pi=3.141592653589793d0
  REAL(KIND(1d0)),PARAMETER::eulergamma=0.5772156649015329d0
  REAL(KIND(1d0)),PARAMETER::eumlog4pi=-1.9538085820677579324d0 ! eulergamma-log(4*pi)
  REAL(KIND(1d0)),PARAMETER::euplogpi=1.7219455507509331d0 ! eulergamma+log(pi)
  ! (6 EulerGamma^2 + \[Pi]^2 - 12 EulerGamma Log[4 \[Pi]] + 6 Log[4 \[Pi]]^2)/12
  REAL(KIND(1d0)),PARAMETER::exp4eulergamma=2.7311510211049246100d0
  ! (6 EulerGamma^2 + \[Pi]^2 + 12 EulerGamma Log[\[Pi]] + 6 Log[\[Pi]]^2)/12 
  REAL(KIND(1d0)),PARAMETER::exp4eulergamma2=2.3050152732995803d0
  COMPLEX(KIND(1d0)),PARAMETER::cnorm=DCMPLX(0d0,-0.1013211836423377714438794632d0) ! 1/pi**2/i
  INTEGER::nnonzeromass,ml5_sign,ipostemp
  LOGICAL::exist
  LOGICAL::comp_mass_scheme
  INTEGER::scal_save
  !    LOGICAL::INIT=.TRUE.
  !    SAVE INIT
  !    IF(INIT)THEN
  !       RECYCLING=.TRUE. ! USE RECYCLING, THEN FOR EACH PS, CLEAR CACHING PLEASE
  !       INIT=.FALSE.
  !    ENDIF
  nlenrank=0
  DO I=0,RANK
     nlenrank=nlenrank+(3+I)*(2+I)*(1+I)/6
  ENDDO
  IF(nlenrank.GT.NLOOPCOEFS)THEN
     STOP "ERROR IN IMLOOP FOR INCORRECT RANK OR NLOOPCOEFS"
  ENDIF
  nnonzeromass=0
  massref(1:MAXNLOOP_IREGI)=0d0
  comp_mass_scheme=.FALSE.
  DO i=1,NLOOPLINE
     M2LR(i)=DREAL(M2L(i))
     IF(ABS(DIMAG(M2L(i)))/MAX(ABS(M2L(i)),1D-2).GT.1D-15)THEN
        comp_mass_scheme=.TRUE.
     ENDIF
     IF(ABS(M2LR(i)).GT.ZEROTHR_IREGI)THEN
        exist=.FALSE.
        DO J=1,nnonzeromass
           IF(ABS(M2LR(I)-massref(J))/M2LR(I).LT.ZEROTHR_IREGI)THEN
              exist=.TRUE.
              EXIT
           ENDIF
        ENDDO
        IF(.NOT.exist)THEN
           nnonzeromass=nnonzeromass+1
           IF(nnonzeromass.GT.20)STOP 'ERROR:TOO MANY INDEPENDENT MASSES !!!!'
           massref(nnonzeromass)=M2LR(I)
        ENDIF
     ENDIF
     IF(i.EQ.1)THEN
        PL(i,0:3)=0d0
     ELSE
        PL(i,0:3)=PDEN(0:3,i-1)
     ENDIF
  ENDDO
  NMASS=nnonzeromass
  STABLE_IREGI=.TRUE.
  ! IMODE,NLOOPLINE,MAXRANK,NCOEFS,PDEN,M2L,MU,TICOEFS,TSTABLE
  ! IMODE=0, IBP reduction
  ! IMODE=1, PaVe reduction
  ! IMODE=2, PaVe reduction with stablility improved by IBP reduction
  ! I HARD-CODED IMODE=2 HERE
  !IMODE=2
  IF(.NOT.comp_mass_scheme)THEN
     CALL tensor_integral_reduce(IMODE,NLOOPLINE,RANK,nlenrank,&
          PL,M2LR,MUR,TCOEFS(0:nlenrank-1,1:4),TSTABLE,CTMODE)
  ELSE
     scal_save=scalarlib
     scalarlib=2
     ! complex mass scheme
     CALL comp_tensor_integral_reduce(IMODE,NLOOPLINE,RANK,nlenrank,&
          PL,M2L,MUR,TCOEFS(0:nlenrank-1,1:4),TSTABLE,CTMODE)
     scalarlib=scal_save
  ENDIF
  STABLE=TSTABLE
  ! TO MATCH THE CONVENTION OF PJFRY++
  ! MULTIPLY GAMMA(1-eps)*(Pi)^eps instead of GAMMA(1-eps)*(4*Pi)^eps
  DO i=0,nlenrank-1
     ipostemp=POS2RANK(i)
     IF(MOD(ipostemp,2).EQ.0)THEN
        ml5_sign=1
     ELSE
        ml5_sign=-1
     ENDIF
     TIRCOEFS(i,3)=TCOEFS(i,3)*cnorm*ml5_sign
     ! GAMMA(1-eps)*(4*Pi)^eps
     !TIRCOEFS2(i,2)=(TCOEFS(i,1)+TCOEFS(i,2)+TCOEFS(i,3)*eumlog4pi)*cnorm*ml5_sign
     !TIRCOEFS2(i,1)=(TCOEFS(i,4)+(TCOEFS(i,1)+TCOEFS(i,2))*eumlog4pi&
     !     +TCOEFS(i,3)*exp4eulergamma)*cnorm*ml5_sign
     ! GAMMA(1-eps)*(Pi)^eps
     TIRCOEFS(i,2)=(TCOEFS(i,1)+TCOEFS(i,2)+TCOEFS(i,3)*euplogpi)*cnorm*ml5_sign
     TIRCOEFS(i,1)=(TCOEFS(i,4)+(TCOEFS(i,1)+TCOEFS(i,2))*euplogpi&
          +TCOEFS(i,3)*exp4eulergamma2)*cnorm*ml5_sign
  ENDDO
  ! SORT THE ORDER OF COEFFICIENTS (OLD ML5 CONV) WITH THE NEW ML5 CONV
  !CALL SORT_IREGICOEFS(RANK,nlenrank,TIRCOEFS2(0:nlenrank-1,1:3),&
  !     TIRCOEFS(0:nlenrank-1,1:3))
  RETURN
END SUBROUTINE IMLOOP
SUBROUTINE IREGI_FREE_PS
  USE global
  USE binary_tree
  IMPLICIT NONE
  CALL free_ibppave_save
  CALL free_xyzmatrices_bt(xyzmatrices_save)
  CALL free_cxyzmatrices_bt(cxyzmatrices_save)
  CALL free_rsmatrices_bt(rsmatrices_save)
  CALL free_crsmatrices_bt(crsmatrices_save)
  RETURN
END SUBROUTINE IREGI_FREE_PS
