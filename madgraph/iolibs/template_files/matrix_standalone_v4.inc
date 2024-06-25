      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) ::HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSMATRIX(P,ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX(P,ANS)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO StandAlone Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINITIAL 
      PARAMETER (NINITIAL=%(nincoming)d)
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
C  put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL 
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./

C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      DATA USERHEL/-1/    
      LOGICAL HELRESET
      common/%(proc_prefix)shelreset/HELRESET
      data HELRESET/.true./
      
%(helicity_lines)s
%(den_factor_line)s

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C
C     FUNCTIONS
C
      LOGICAL %(proc_prefix)sIS_BORN_HEL_SELECTED

c----------
c   Check if helreset mode is on
c---------
if (HELRESET) then
   NTRY = 0
   do i=1,NCOMB
      GOODHEL(I) = .false.
   enddo
   HELRESET = .false.
endif

C ----------
C BEGIN CODE
C ----------
      IF(USERHEL.EQ.-1) NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          GOODHEL(IHEL)=.TRUE.
        ENDDO
      ENDIF
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN       
                IF (GOODHEL(IHEL) .OR. NTRY .LT. 20.OR.USERHEL.NE.-1) THEN
                    IF(NTRY.GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                        CYCLE
                    ENDIF
c           write (*,*) 'HEL = ',NHEL(:,IHEL)
                    T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1))
                    IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                        ANS=ANS+T
                    ENDIF
                    IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                        GOODHEL(IHEL)=.TRUE.
                    ENDIF
                ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)
c     write (*,*) "Spyros: IDEN         = ", IDEN
c     write (*,*) "Spyros: USERHEL      = ", USERHEL
c     write (*,*) "Spyros: HELAVGFACTOR = ", HELAVGFACTOR
      IF(USERHEL.NE.-1) THEN
          ANS=ANS*HELAVGFACTOR    
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
c       write (*,*) "Spyros: InPart ", J, " BEAMS_HELAVGFACTOR = ", BEAMS_HELAVGFACTOR(J), ", POLARIZATION = ", POLARIZATIONS(J,0)  
          ENDIF
        ENDDO
      ENDIF
      END
       
       
      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,IC)
C  
%(info_lines)s
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 %(proc_prefix)sCF(NCOLOR,NCOLOR)
      common /color_matrix/ %(proc_prefix)sCF
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
%(global_variable)s
C COLOR DATA
%(color_data_lines)s
C  
C  
C ----------
C BEGIN CODE
C ----------
      call %(proc_prefix)sGET_AMP(P,NHEL,IC,AMP)
c      WRITE (*,*) '  -> AMP = ', AMP
      call %(proc_prefix)sGET_JAMP(AMP,JAMP)
c      WRITE (*,*) '  -> JAMP = ', JAMP
      call %(proc_prefix)sGET_MATRIX(JAMP,%(proc_prefix)sMATRIX)
c      write (*,*) "  -> col.ave. |M|^2 for HEL=[", NHEL ,"] = ", %(proc_prefix)sMATRIX

%(amp2_lines)s

      END

      SUBROUTINE %(proc_prefix)sGET_NHEL(IDEN_STAR,NHEL_STAR)
C CONSTANTS
C
CF2PY INTENT(OUT) :: NHEL_STAR
CF2PY INTENT(OUT) :: IDEN_STAR
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB         
      PARAMETER (             NCOMB=%(ncomb)d)
  
      INTEGER NHEL(NEXTERNAL,NCOMB),NHEL_STAR(NEXTERNAL,NCOMB)  
      INTEGER IDEN,IDEN_STAR

%(helicity_lines)s
%(den_factor_line)s   
      IDEN_STAR = IDEN
      NHEL_STAR = NHEL
      END
      
      SUBROUTINE %(proc_prefix)sGET_AMP(P,NHEL,IC,AMP)
C  
%(process_lines)s
C 
CF2PY INTENT(OUT) :: AMP  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: IC

      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      COMPLEX*16 AMP(NGRAPHS)
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
%(global_variable)s
C  
C
%(helas_calls)s
%(amp2_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_JAMP(AMP,JAMP)
C  
%(process_lines)s
C 
CF2PY INTENT(OUT) :: JAMP 
CF2PY INTENT(IN) :: AMP

      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NCOLOR
      PARAMETER ( NCOLOR=%(ncolor)d) 
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)

%(jamp_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP,MATRIX) 
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
CF2PY INTENT(OUT) :: MATRIX 
CF2PY INTENT(IN) :: JAMP


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d) 
      REAL*8     ZERO,MATRIX
      PARAMETER (ZERO=0D0)
C  
  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 %(proc_prefix)sCF(NCOLOR,NCOLOR)
      COMMON /color_matrix/ %(proc_prefix)sCF
      COMPLEX*16 JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C

C COLOR DATA
C  

      MATRIX = 0.D0 
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + %(proc_prefix)sCF(J,I)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      END



      SUBROUTINE %(proc_prefix)sGET_INTER(JAMP_1,JAMP_2, INTER)

CF2PY INTENT(OUT) :: INTER   
CF2PY INTENT(IN) :: JAMP_1
CF2PY INTENT(IN) :: JAMP_2
     
      INTEGER I,J
      INTEGER NCOLOR 
      PARAMETER (NCOLOR=%(ncolor)d)
      REAL*8 %(proc_prefix)sCF(NCOLOR,NCOLOR)
      COMMON /color_matrix/ %(proc_prefix)sCF 
      COMPLEX*16 JAMP_1(NCOLOR),JAMP_2(NCOLOR),ZTEMP,INTER

C COLOR DATA
C  
      
      INTER = (0.D0,0.D0)  
      DO I = 1, NCOLOR 
        ZTEMP = DCONJG(JAMP_2(I))
    DO J=1, NCOLOR
          INTER = INTER + %(proc_prefix)sCF(J,I) * JAMP_1(J) * ZTEMP 
        ENDDO 
      ENDDO

      END       



      SUBROUTINE  %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER((NCOMB*NCOMB+1)/2: all interference term (not the symmetric one)
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      DOUBLE COMPLEX INTER(99)
      DOUBLE COMPLEX TMP_INTER(99)
      INTEGER NB_NHEL
      PARAMETER (NB_NHEL=%(ncomb)d)
c LOCAL
      INTEGER I,IHEL,IPART
C
      INTEGER NHEL(NEXTERNAL,NB_NHEL),NTRY
C     put in common block to expose this variable to python interface
      COMMON/PROCESS_NHEL/NHEL
      
      DO I=1, N_COMB*(N_COMB+1)/2
        INTER(I) = 0
      enddo

      DO IHEL =1, NB_NHEL
        THISNHEL(:) = NHEL(:, IHEL)
	DO IPART=1,N_CHANGING
            if(THISNHEL(POS(IPART)).NE.ALLOW_HEL(1+N_COMB*(IPART-1))) GOTO 10 !BYPASS COMPUTATION FOR HELICITY
        ENDDO
        TMP_INTER(:) = 0
        call  %(proc_prefix)sGET_ALL_INTER(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, TMP_INTER)
        do I = 1, N_COMB*(N_COMB+1)/2
           INTER(I) = INTER(I) + TMP_INTER(I)
        enddo 
 10   enddo   
      return 
      end


      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER((NCOMB*NCOMB+1)/2: all interference term (not the symmetric one)
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(OUT) :: INTER(NCOMB*(NCOMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      DOUBLE COMPLEX INTER(99)
c
c     Intermediate array
c
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER IC(NEXTERNAL)

      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX JAMP(3*NEXTERNAL, NCOLOR)
c
c     LOCAL
c
      INTEGER I,J,SOL,N

C ----------
C BEGIN CODE
C ----------
      IC(:)=1
      write(*,*) ''
      do I = 1, N_COMB
         do N = 1, N_CHANGING
            NHEL(POS(N)) = ALLOW_HEL((N-1)*N_COMB+I)
            call %(proc_prefix)sGET_AMP(P,NHEL,IC,AMP)
            call %(proc_prefix)sGET_JAMP(AMP,JAMP(1,I))
          enddo
      enddo

    SOL = 0 
    DO I = 1, N_COMB
       DO J= I, N_COMB
         SOL = SOL +1
         call %(proc_prefix)sGET_INTER(JAMP(1,I), JAMP(1,J), INTER(SOL))
       ENDDO
    ENDDO


    RETURN
    END





      SUBROUTINE %(proc_prefix)sGET_value(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE   
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS 
      real*8 pi     
CF2PY INTENT(OUT) :: ANS  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: ALPHAS
C     ROUTINE FOR F2PY to read the benchmark point.    
C     the include file with the values of the parameters and masses 
      include "coupl.inc"
      
      pi = 3.141592653589793d0
      G = 2* DSQRT(ALPHAS*pi)
      call UPDATE_AS_PARAM()
      if (NHEL.ne.0) then
        CALL %(proc_prefix)sSMATRIXHEL(P, NHEL, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, ANS)
      endif
      return 
      end

      SUBROUTINE %(proc_prefix)sINITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE   
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH 
      call setpara(PATH)  !first call to setup the paramaters    
      return 
      end      

      LOGICAL FUNCTION %(proc_prefix)sIS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C
C     ARGUMENTS
C
      INTEGER HELID
C
C     LOCALS
C
      INTEGER I,J
      LOGICAL FOUNDIT
C
C     GLOBALS
C
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/HELC

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
C     ----------
C     BEGIN CODE
C     ----------
      
      %(proc_prefix)sIS_BORN_HEL_SELECTED = .True.
      if (POLARIZATIONS(0,0).eq.-1) THEN
        RETURN
      ENDIF

      DO I=1,NEXTERNAL
        IF (POLARIZATIONS(I,0).eq.-1) THEN
          CYCLE
        ENDIF
        FOUNDIT = .FALSE.
        DO J=1,POLARIZATIONS(I,0)
          IF (HELC(I,HELID).eq.POLARIZATIONS(I,J)) THEN
            FOUNDIT = .True.
            EXIT
          ENDIF
        ENDDO
        IF(.not.FOUNDIT) THEN
          %(proc_prefix)sIS_BORN_HEL_SELECTED = .False.
          RETURN
        ENDIF
      ENDDO

      RETURN
      END
