      SUBROUTINE PDFWRAP
      IMPLICIT NONE
C     
C     INCLUDE
C     
      INCLUDE 'pdf.inc'
      INCLUDE '../alfas.inc'
      INCLUDE '../vector.inc'
      INCLUDE '../coupl.inc'
      REAL*8 ZMASS
      DATA ZMASS/91.188D0/
      CHARACTER*150 LHAPATH
      CHARACTER*20 PARM(20)
      DOUBLE PRECISION VALUE(20)
      REAL*8 ALPHASPDF
      EXTERNAL ALPHASPDF


C     -------------------
C     START THE CODE
C     -------------------      

C     initialize the pdf set
      CALL FINDPDFPATH(LHAPATH)
      CALL SETPDFPATH(LHAPATH)
      VALUE(1)=LHAID
      PARM(1)='DEFAULT'
      IF (PDLABEL.EQ.'lhapdf') THEN
        CALL PDFSET(PARM,VALUE)
        CALL GETORDERAS(NLOOP)
        NLOOP=NLOOP+1
        ASMZ=ALPHASPDF(ZMASS)
      ELSE
        WRITE(*,*) 'Unknown PDLABEL', PDLABEL
        STOP 1
      ENDIF

      RETURN
      END


      SUBROUTINE FINDPDFPATH(LHAPATH)
C     *****************************************************************
C     ***
C     generic subroutine to open the table files in the right
C      directories
C     *****************************************************************
C     ***
      IMPLICIT NONE
C     
      CHARACTER LHAPATH*150,UP*3
      DATA UP/'../'/
      LOGICAL EXISTS
      INTEGER I, POS
      CHARACTER*300  TEMPNAME2
      CHARACTER*300 PATH  ! path of the executable
      INTEGER FINE2
      CHARACTER*30  UPNAME  ! sequence of ../

C     first try in the current directory
      LHAPATH='./PDFsets'
      INQUIRE(FILE=LHAPATH, EXIST=EXISTS)
      IF(EXISTS)RETURN

      DO I=1,6
        LHAPATH=UP//LHAPATH
        INQUIRE(FILE=LHAPATH, EXIST=EXISTS)
        IF(EXISTS)RETURN
      ENDDO

C     
C     getting the path of the executable
C     
      CALL GETARG(0,PATH)  !path is the PATH to the madevent executable (either global or from launching directory)
      POS = INDEX(PATH,'/',.TRUE.)
      PATH = PATH(:POS)
      FINE2=INDEX(PATH,' ')-1


C     
C     check path from the executable
C     
      LHAPATH='lib/PDFsets'
      INQUIRE(FILE=LHAPATH, EXIST=EXISTS)
      IF(EXISTS)RETURN
      UPNAME='../../../../../../../'
      DO I=1,6
        TEMPNAME2=PATH(:FINE2)//UPNAME(:3*I)//LHAPATH
C       LHAPath=up//LHAPath
        INQUIRE(FILE=TEMPNAME2, EXIST=EXISTS)
        IF(EXISTS)THEN
          LHAPATH = TEMPNAME2
          RETURN
        ENDIF
      ENDDO
      PRINT*,'Could not find PDFsets directory, quitting'
      STOP

      RETURN
      END


