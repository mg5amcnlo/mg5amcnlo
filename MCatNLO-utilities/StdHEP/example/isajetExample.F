      PROGRAM TSTISA
C   
C     JTDKY = +/- unit number for decay table file.    
C                 If it is negative, decay table is not printed.   
C     JTEVT = +/- unit number for output event file.   
C                 If it is negative, only stable particles are written on it. 
C     JTCOM =     unit number for command file.    
C     JTLIS =     unit number for listing. 
C   
      IMPLICIT NONE 
#include "itapes.inc"
#include "idrun.inc"
#include "primar.inc"
C
#include "stdhep.inc"
#include "stdlun.inc"
C   
      integer istr,nevt,itotal
      INTEGER JTDKY,JTEVT,JTCOM,JTLIS,IFL,ILOOP,IPRT,LOK,ILOOP2
      INTEGER INDEC,INDEC2
      CHARACTER*132 ISADEC
      LOGICAL OK,DONE
      SAVE ILOOP,JTDKY,JTEVT,JTCOM,JTLIS

      EXTERNAL ALDATA
C   
C          Initialize ISAJET    
C   
      JTDKY=-1
      JTEVT=23
      JTCOM=21
      JTLIS=22
      nevt=1000
      itotal=0
C initialize HEP logical units
      lnhwrt=JTEVT
      lnhrd=0
      lnhdcy=0
      lnhout=JTLIS
      lnhpdf=0
      lnhdmp=0
C  point to standard decay tables
      CALL GETENV( 'ISAJET_DIR', ISADEC )
      IF ( ISADEC .EQ. ' ' ) THEN
         ISADEC = 'isadecay.dat'
      ELSE
         INDEC   = INDEX ( ISADEC , ' ' )
         ISADEC(INDEC:INDEC+13)  = '/isadecay.dat'
         INDEC2  = INDEX ( ISADEC , ' ' ) - 1
         print *,'looking for ',ISADEC(:INDEC2)
      ENDIF
      OPEN(UNIT=1,FILE=ISADEC,STATUS='OLD')
      OPEN(UNIT=JTCOM,FILE='isajetExample.cmd',STATUS='OLD')
      OPEN(UNIT=JTLIS,FILE='isajetExample.lpt',STATUS='NEW')
      CALL ISAINI(JTDKY,JTEVT,JTCOM,JTLIS)  
C initialize MCFio
      call stdxwinit('isajetExample.io','StdHep/Isajet example',
     1               nevt,istr,LOK)
      CALL TSTBKEM(40)
      call cntcode(1)
      IPRT=1
C   
C print list of defined particles and their translations
C    (you have to call ISAINI first)
C use bogus masses for 4th generation quarks so we can see their definitions
C
      CALL PRTLST(JTLIS,200.,220.)
C
C    compare particle ID for isajet and standard
C
      CALL HPTRLSTI
C   
C          Read instructions; terminate for STOP command or error.  
C   
      ILOOP2 = 0
 201  IFL=0 
      CALL ISABEG(IFL)  
      IF(IFL.NE.0)THEN
        write(lnhout,1001) itotal
c...close event file
        call stdxend(istr)
        call cntcode(3)
C...print histograms
        CALL HOUTPU(LNHOUT)
        CALL HISTDO
      ENDIF
      IF(IFL.NE.0) STOP
C          Fill Stdhep common block 1
        CALL STDFLISXSEC
C          Write Stdhep begin-run record   
        call stdxwrt(100,istr,LOK)
        IF(LOK.NE.0) WRITE(LNHOUT,*)
     1        ' Problem writing StdHep begin run record'
C   
C          Event loop   
C   
      ILOOP=0   
 301  CONTINUE  
      ILOOP=ILOOP+1   
C        Generate one event - discard if .NOT.OK  
      CALL ISAEVT(ILOOP,OK,DONE)  
C        Write event record   
      IF(OK)THEN
        CALL ISAHEP(1)
	itotal = itotal + 1
        IF(NEVHEP.LE.IPRT)THEN
           CALL PRTEVT(0)
           CALL HEPLST(1)
           CALL HEPTREE
        ENDIF
        CALL TSTFLEM(40)
	call cntcode(2)
        call stdxwrt(1,istr,LOK)
C          Fill Stdhep common block 1
        CALL STDFLISXSEC
        CALL TSTFLXS(40,ILOOP2)
        IF(LOK.NE.0) WRITE(LNHOUT,*) ' Problem writing event record'
      ENDIF
      IF(.NOT.DONE) GO TO 301   
C   
C          Calculate cross section and luminosity   
C   
      CALL ISAEND   
C          Fill Stdhep common block 1
      CALL STDFLISXSEC
C          Write Stdhep end-run record   
      call stdxwrt(200,istr,LOK)
      IF(LOK.NE.0) WRITE(LNHOUT,*)
     1        ' Problem writing stdhep end run record'
      ILOOP2 = ILOOP2 + ILOOP
      GO TO 201  
1001  format(' grand total: ',i10,' events generated')
      END 

      SUBROUTINE TSTFLXS(IOFF,ILP)
C... fill histograms
#include "stdhep.inc"
#include "stdcm1.inc"

      INTEGER IOFF,ILP
      REAL BN

      BN = FLOAT(NEVHEP+ILP)+0.5
      CALL HFILL(IOFF+20,BN,0.,STDXSEC)
      RETURN
      END

      subroutine cntcode(ii)
c
c  make a table of isthep values
c
#include "stdhep.inc"
#include "stdlun.inc"
      integer ii,i,icnt(200)
      save icnt

      if(ii.eq.1)then
         do i=1,200
	   icnt(i)=0
	 enddo
      elseif(ii.eq.2)then
         do i=1,nhep
	   ist = isthep(i)
	   if(ist.le.200 .and. ist.gt.0) icnt(ist) = icnt(ist) + 1
	 enddo
      elseif(ii.eq.3)then
         write(lnhout,1001) 
	 do i=1,200
            write(lnhout,1002) i,icnt(i)
	 enddo
      else
      endif

      return
 1001 format(10x,'List of used isthep values')
 1002 format(' isthep ',i3,2x,i10)
      end
