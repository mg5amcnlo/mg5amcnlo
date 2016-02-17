      PROGRAM DRIVER
!**************************************************************************
!     THIS IS THE DRIVER FOR CHECKING THE STABILITY OF THE LOOP MATRIX
!     ELEMENTS
!**************************************************************************
      IMPLICIT NONE
!     
!     CONSTANTS  
!     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

!     
!     INCLUDE FILES
!     
!---  the include file with the values of the parameters and masses      
      INCLUDE "coupl.inc"
!---  integer nexternal ! number particles (incoming+outgoing) in the me 
      INCLUDE "nexternal.inc" 
      INCLUDE "MadLoopParams.inc"

!     
!     LOCAL
!     
      INTEGER I,J,K
      REAL*8 P(0:3,NEXTERNAL)   ! four momenta. Energy is the zeroth component.
      
      INTEGER MATELEM_ARRAY_DIM
      REAL*8 , allocatable :: MATELEM(:,:)

      REAL*8 SQRTS,BORNELEM,AO2PI           ! sqrt(s)= center of mass energy 
      REAL*8 PIN(0:3), POUT(0:3)
      CHARACTER*120 BUFF(NEXTERNAL)
      CHARACTER*1 EX 
      INTEGER HELCHOICE, SOCHOICE

!
!     SAVED VARIABLES
!
      LOGICAL INIT
      DATA INIT/.TRUE./
      COMMON/INITCHECKSA/INIT

!     
!     EXTERNAL
!     
      REAL*8 DOT
      EXTERNAL DOT

!-----
!     BEGIN CODE
!-----
!     
!---  INITIALIZATION CALLS
!    

      IF (INIT) THEN
         INIT=.FALSE.
         CALL %(proc_prefix)sGET_ANSWER_DIMENSION(MATELEM_ARRAY_DIM)
         ALLOCATE(MATELEM(0:3,0:MATELEM_ARRAY_DIM))

!---  Call to initialize the values of the couplings, masses and widths 
!     used in the evaluation of the matrix element. The primary parameters of the
!     models are read from Cards/param_card.dat. The secondary parameters are calculated
!     in Source/MODEL/couplings.f. The values are stored in common blocks that are listed
!     in coupl.inc .

         call setpara('param_card.dat')  !first call to setup the paramaters

      ENDIF

      AO2PI=G**2/(8.D0*(3.14159265358979323846d0**2))


      call printout()

      do while (.TRUE.)
        write(*,*) "Exit? ['y','n']"
        read(*,*) EX
        if (EX.eq.'y'.or.EX.eq.'Y') THEN
          exit
        endif
        write(*,*) "Enter CTModeRun"
        read(*,*) CTMODERUN
        write(*,*) "PS pt, parts. as in proc and format (E,p_x,p_y,p_z)"
        do i=1,NExternal
          read(*,*) P(0,i),P(1,i),P(2,i),P(3,i)
        enddo
        do i=0,3
          PIN(i)=0.0d0
          do j=1,nincoming
            PIN(i)=PIN(i)+p(i,j)
          enddo
        enddo
        SQRTS=dsqrt(dabs(DOT(PIN(0),PIN(0))))
        write(*,*) "Enter MU_R, -1.0d0 = default"
        read(*,*) MU_R
        if (MU_R.lt.0.0d0) then
          MU_R=SQRTS            
        endif
        write(*,*) "Enter Helicity tag, -1 = summed. For loops only."
        read(*,*) HELCHOICE
        write(*,*) "Enter split_orders choice, -1 = all."
        read(*,*) SOCHOICE
        IF (SOCHOICE.NE.-1) THEN
          CALL %(proc_prefix)sSET_COUPLINGORDERS_TARGET(SOCHOICE)
        ENDIF
!---  Update the couplings with the new MU_R
        CALL UPDATE_AS_PARAM()
!     
!     Now we can call the matrix element!
!
        IF (HELCHOICE.EQ.-1) THEN
          CALL %(proc_prefix)sSLOOPMATRIX(P,MATELEM)
        ELSE
          CALL %(proc_prefix)sSLOOPMATRIXHEL(P,HELCHOICE,MATELEM)
        ENDIF
        write(*,*) '##TAG#RESULT_START#TAG##'
        do i=1,nexternal      
          write (*,'(a2,1x,5ES30.15E4)') 'PS',P(0,i),P(1,i),P(2,i),P(3,i)
        enddo
        write (*,'(a3,1x,i3)') 'EXP',-(2*nexternal-8)
        write (*,'(a4,1x,1ES30.15E4)') 'BORN',0.0d0
        write (*,'(a3,1x,1ES30.15E4)') 'FIN',MATELEM(1,0)
        write (*,'(a4,1x,1ES30.15E4)') '1EPS',0.0d0
        write (*,'(a4,1x,1ES30.15E4)') '2EPS',0.0d0
        write (*,*) 'Export_Format LoopInduced'
        write(*,*) '##TAG#RESULT_STOP#TAG##'      
      enddo

      end
      
        
        
        
         double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end
