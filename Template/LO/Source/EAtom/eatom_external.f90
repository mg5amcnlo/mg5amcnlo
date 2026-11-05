! External (non-module) entry points of the EAtom library, callable from
! fixed-form Fortran without a USE statement (e.g. the madevent matrix
! elements and genps.f).

MODULE eatom_frame
  ! Electron mass used to build the atom-rest-frame kinematics (set by the
  ! phase-space initialisation). When set, EATOM_WF checks p(0) = m_e.
  IMPLICIT NONE
  REAL(KIND(1d0)),SAVE::eatom_me_frame=0d0
  LOGICAL,SAVE::eatom_me_set=.FALSE.
END MODULE eatom_frame

SUBROUTINE EATOM_WF(nuclearA,nuclearZ,HEL,p,wf)
  ! Same as the module procedure eatom_WF (HEL = +-(10000+ishell), shell
  ! averaged spinor), with a guard: the bound-electron wavefunction is only
  ! defined in the rest frame of the atom, where p(0) = m_e.
  USE atomic_electron_wavefunctions, ONLY: Atomic_electron_WF
  USE eatom_frame
  IMPLICIT NONE
  INTEGER,INTENT(IN)::nuclearA,nuclearZ,HEL
  REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p
  COMPLEX(KIND(1d0)),DIMENSION(6),INTENT(OUT)::wf
  INTEGER,PARAMETER::offset=10000
  INTEGER::ishell
  REAL(KIND(1d0))::m
  ! run_card settings of madevent (run.inc); zero, i.e. all shells, otherwise
  REAL(KIND(1d0))::run_pmax
  INTEGER::run_mode,run_shell
  COMMON/TO_EATOM_RUN/run_pmax,run_mode,run_shell
  IF(eatom_me_set)THEN
     IF(ABS(p(0)-eatom_me_frame).GT.1d-9*eatom_me_frame)THEN
        WRITE(*,*)"ERROR: EATOM_WF called outside the atom rest frame:", &
             " E_B=",p(0)," m_e=",eatom_me_frame
        STOP 1
     ENDIF
  ENDIF
  ishell=ABS(HEL)-offset
  IF(run_shell.NE.0.AND.ishell.NE.run_shell)THEN
     ! shell excluded by eatom_shell in the run_card
     wf(1)=dcmplx(p(0),p(3))*(-1)
     wf(2)=dcmplx(p(1),p(2))*(-1)
     wf(3:6)=(0d0,0d0)
     RETURN
  ENDIF
  m=SIGN(1000d0,DBLE(HEL))
  CALL Atomic_electron_WF(nuclearA,nuclearZ,ishell,m,p,wf)
  RETURN
END SUBROUTINE EATOM_WF
