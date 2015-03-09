C************************************************************************
C
C                           HISTOGRAMMING PACKAGE 
C                                 M. MANGANO
C
C************************************************************************
C
C**********************************************************************
C    SIMPLE HISTOGRAMMING PACKAGE --  SIMPLIFIED VERSION OF HBOOK
C    BY Michelangelo Mangano    NOVEMBER 1988
C    LAST REVISED NOVEMBER 9, 1988
C    LAST REVISED JUNE 12, 1989  (ADD SCATTER PLOTS)
C    LAST REVISED oct 1990 (Add multi-plots on one page, routines MULTITOP,
C         			MTFILL,...)
C    LAST REVISED Jun 2003 by FM: passed to double precision AND 
C    included the date for linux      
C**********************************************************************
C
C Fills up to 500 histograms with up to 100 bins. 
C Gives a data file (to be specified in the calling program by assigning 
C a file name to unit 98) and a topdrawer file (to be specified in the 
C calling program by assigning a file name to unit 99).
C
C INITIALIZATION:
C Call once INIHIST; this just resets a few counters and logicals
C Call MBOOK(N,'TITLE',DEL,XMIN,XMAX) for each histogram to be booked.
C N (an integer) is the label of the histogram;
C 'TITLE' is the name of the histogram (no more then 100 characters);
C DEL (real*8) is the bin size;
C XMIN (real*8) is the lower limit of the first bin;
C XMAX (real*8)is the upper limit of the last  bin
C Example:
C      call mbook(2,'pt distribution',1.,10,70)
C This call initializes histogram number 2, called 'pt distribution';
C The bin size will be 1. (possibly GeV, if that's what you want), the
C first bin being  10<x<11. and the last one being 69.<x<70
C
C FILLING:
C When it's time, call MFILL(N,X,Y); this will add Y (real*8) to the bin 
C in which X (real*8) happens to be, within histogram N. 
C
C PLAYING AROUND:
C At the end of the day you may want to sum, divide, cancel, etc.etc.
C various histograms (bin by bin). Then you call MOPERA(I,'O',J,K,X,Y). 
C The 1-character string O can take the following values:
C +  : sums       X*(hist I) with Y*(hist J) and puts the result in hist K;
C -  : subtracts  X*(hist I) with Y*(hist J) and puts the result in hist K;
C *  : multiplies X*(hist I) with Y*(hist J) and puts the result in hist K;
C /  : divides    X*(hist I) with Y*(hist J) and puts the result in hist K;
C F  : multiplies hist I by the factor X, and puts the result in hist K;
C R  : takes the square root of  hist  I, and puts the result in hist K;if
C      the value at a given bin is less than or equal to 0, puts 0 in K
C S  : takes the square      of  hist  I, and puts the result in hist K;
C L  : takes the log_10 of  hist  I, and puts the result in hist K; if the
C      value at a given bin is less than or equal to 0, puts 0 in K
C M  : statistical analysis; if I contains the weights (let's say WGT),
C      J contains variable times weight (F*WGT) and K contains the
C      variable squared times the weight (F**2*WGT), then, after using 'M',
C      J will contain the average value of the variable <F> and K will 
C      contain the sigma of the average: sigma=sqrt(<F**2>-<F>**2).
C      If WGT=1. for all the entries, then it is enough to put I=J, and
C      it is not necessary to book a hist with the weights.
C V  : estimates errors for vegas evaluation of differential distributions.
C      Fill I with the values of
C      the functions do integrate times the Vegas weight (fun*wgt); fill
C      J with fun**2*wgt; then K will contain an estimate of the error
C      of the integration. Putting X=1/(#of iterations) performs the 
C      avegare over the iterations, and gives the right normalization to 
C      the differential distribution, I, and to the errors, K. J stays the same.
C
C FINAL ACCOUNTING:
C Now we can finalize our histograms; MFINAL(N) will calculate the integral
C of the histogram N, the mean value of the X variable and its RMS.
C If we now want to renormalize the hist's, we can call MNORM(N,X), which
C will normalize the integral to X  -- CAUTION: do not call MNORM before
C MFINAL, it will blow up.
C
C OUTPUT:
C To get a .dat file containing the values of the histograms, together with
C some information (like integral, mean values, etc.etc.) call MPRINT(N),
C for each hist N that you want in the .dat file. Before the call to MPRINT
C you want to open unit 98 and give it a name:                       
C     OPEN(UNIT=98,NAME='NAME.DAT',STATUS='NEW')
C If you want a topdrawer file with a plot of the hist values, call 
C MTOP(N,M,'X','Y','SCALE'). The points of the plot will be taken from histogram
C N, the error bars from histogram M. 'SCALE', character*(*), determines
C the scale for y, logarithmic or linear (SCALE=LOG,LIN). 
C If you do not want error bars, keep
C a histogram of zeros, or just call a hist that had not been booked.
C X will appear as a 'bottom title', and Y will appear as a 'left title'.
C The top title is by default the name of the histogram itself.
C A little box below the plot will contain some information on the plot
C itself. Before calling MTOP,
C     OPEN(UNIT=99,NAME='NAME.TOP',STATUS='NEW')
C--------------------------------------------------------------------------
C
C  COMMON/HISTO/  Histogram N
C                           
C   BOOK(N),      Three-letter character-string: 'NO' if histogram was not 
C		  Booked, 'YES' otherwise
C   TITLE(N),     Title of the histogram
C
C   HMIN(N),      Min value of x range
C   HMAX(N),      Max value of x range
C   HDEL(N),      Bin width
C   NBIN(N),      Total number of bins
C   USCORE(N),    Total integral of underscores with x < HMIN(N)
C   OSCORE(N),    Total integral of onderscores with x > HMAX(N)
C   IUSCORE(N),   Number of entries with x < HMIN(N)
C   IOSCORE(N),   Number of entries with x > HMAX(N)
C   IENT(N),      Total number of entries within x range HMIN(N)<x<HMAX(N)
C   HINT(N),      Integral of the histogram within HMIN(N)<x<HMAX(N)
C   HAVG(N),      Average value of x, weighted over the x range of the histo
C   HSIG(N),      Quadratic dispersion of x around the average
C   HIST(N,L),    Value of bin L-th
C   XHIS(N,L),    Central x value of bin L-th
C   IHIS(N,L),    Number of entries within bin L-th
C   NHIST         Total number of booked histograms
C


      SUBROUTINE INIHIST
C*************************************************************
c     initialization
      IMPLICIT NONE
      include 'dbook.inc'
c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
c
c     LOCAL
c
      INTEGER I
C
C     START
C
      NHIST=0
      DO 1, I=1,NPLOTS             
   1  BOOK(I)=' NO'
C     Initialize the number of bins of the aMCfast grids
      if(iappl.ne.0) appl_obs_nbins = 0
      END  
 

    
      SUBROUTINE MBOOK(N,TIT,DEL,XMIN,XMAX)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
      CHARACTER*(*) TIT
      REAL*8 DEL,XMIN,XMAX
C
C     GLOBAL
C
      include 'dbook.inc'
C
C     LOCAL
C
      INTEGER I
Cq
C     START
C
      NHIST = MAX(N,NHIST)
      IF(BOOK(N)(1:1).EQ.'Y') THEN
	 CALL MWARN('MBOOK')
         WRITE(*,*) 'Histogram',N,TITLE(N),' already in use. '
         WRITE(*,*) 'superseded by ',TIT
      ENDIF
      BOOK(N) = 'YES'
      TITLE(N) = ' '//TIT
1     HDEL(N) = DEL
      NBIN(N) = INT((XMAX-XMIN)/(DEL*0.999999d0))
      IF(NBIN(N).GT.100) THEN
	WRITE(*,*) 'TOO MANY BINS (',NBIN(N),') REQUIRED IN HIST ',N
	WRITE(*,*) 'RE-ENTER BIN SIZE DELTA (OLD BIN = ',DEL,' ):'
	READ(*,*) DEL
	GO TO 1
      ENDIF
      HMIN(N) = XMIN
      HMAX(N) = NBIN(N)*DEL+XMIN
      IF(abs(HMAX(N)-XMAX).gt.0.001d0*DEL) THEN
	 CALL MWARN('MBOOK')
         WRITE(*,*)
     #'Histogram ', TIT, ' Change of upper limit:',xmax,'-->',HMAX(N)
      ENDIF
      IENT(N) = 0
      IUSCORE(N) = 0
      IOSCORE(N) = 0
      USCORE(N) = 0
      OSCORE(N) = 0
      HAVG(N) = 0
      HINT(N) = 0
      HSIG(N) = 0
      DO I=1,NBIN(N)
         XHIS(N,I)=HMIN(N)+HDEL(N)*(DFLOAT(I)-0.5d0)
         IHIS(N,I)=0
         HIST(N,I)=0
      ENDDO
      END

      SUBROUTINE MFILL4(N,X,Y)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
      REAL*8 X,Y
C
C     LOCAL
C
      REAL*8 XI
      INTEGER I
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
c
c     START
c

      IF(X.LT.HMIN(N)) THEN
         USCORE(N) = USCORE(N) + Y
         IUSCORE(N) = IUSCORE(N) + 1
      ELSEIF(X.GT.HMAX(N)) THEN
         OSCORE(N) = OSCORE(N) + Y
         IOSCORE(N) = IOSCORE(N) + 1
      ELSE
         XI=((X-HMIN(N))/HDEL(N))+1
         I=INT(XI)
         IENT(N)=IENT(N)+1
         IHIS(N,I)=IHIS(N,I)+1
         HIST(N,I)=HIST(N,I)+Y
      ENDIF
      END


      SUBROUTINE MINTEG(NIN,NOUT,IDIR,IPOW)
C If IPOW=1 performs the integral of the distribution contained in histogram
C NIN up to the value specified by the abscissa (if IDIR=1) or from this
C value on (if IDIR=-1). The resulting integral distribution is put into 
C NOUT, which is automatically booked if NOUT.ne.NIN .  Choosing IPOW=2
C the routine will return the square root of the integral of the squares,
C as is required, for example, for the propagation of the mean quadratic error
C of a given distribution. Overscores and underscores are included.
      IMPLICIT NONE
C
C     ARGUMENTS
C     
      INTEGER NIN,NOUT,IDIR,IPOW
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
c
c     LOCAL
c
      INTEGER M,I,L
      CHARACTER*14  C
      DIMENSION C(2) 
      DATA C/' INTEG BELOW X',' INTEG ABOVE X'/
C
C     EXTERNAL
C
      REAL*8 SUMPOW
C
C     START
C
      IF(BOOK(NIN)(1:1).NE.'Y') RETURN
      M = NBIN(NIN)                                           
      I = (IDIR + 3)/2
      IF(NOUT.NE.NIN.AND.BOOK(NOUT)(1:1).NE.'Y') THEN
      	CALL MBOOK(NOUT,TITLE(NIN)//C(I), 
     &                HDEL(NIN),HMIN(NIN),HMAX(NIN))
      ENDIF
      IF(IDIR.EQ.1) THEN
         HIST(NOUT,1) = SUMPOW(HIST(NIN,1),USCORE(NIN),IPOW)
         IHIS(NOUT,1) = IHIS(NIN,1) + IUSCORE(NIN)
         XHIS(NOUT,1) = XHIS(NIN,1) + HDEL(NIN)/2
         DO L=2,M                      
            HIST(NOUT,L) = SUMPOW(HIST(NIN,L),HIST(NOUT,L-1),IPOW)
            IHIS(NOUT,L) = IHIS(NIN,L) + IHIS(NOUT,L-1) 
            XHIS(NOUT,L) = XHIS(NIN,L) + HDEL(NIN)/2
         ENDDO
         OSCORE(NOUT) = SUMPOW(OSCORE(NIN),HIST(NIN,M),IPOW)
         IOSCORE(NOUT) = IOSCORE(NIN) + IHIS(NIN,M)
      ELSEIF(IDIR.EQ.-1) THEN
         HIST(NOUT,M) = SUMPOW(HIST(NIN,M),OSCORE(NIN),IPOW)
         IHIS(NOUT,M) = IHIS(NIN,M) + IOSCORE(NIN)
         XHIS(NOUT,M) = XHIS(NIN,M) - HDEL(NIN)/2
         DO L=M-1,1,-1                        
            HIST(NOUT,L) = SUMPOW(HIST(NIN,L),HIST(NOUT,L+1),IPOW)
            IHIS(NOUT,L) = IHIS(NIN,L) + IHIS(NOUT,L+1)
            XHIS(NOUT,L) = XHIS(NIN,L) - HDEL(NIN)/2
         ENDDO
         USCORE(NOUT) = SUMPOW(USCORE(NIN),HIST(NIN,1),IPOW)
         IUSCORE(NOUT) = IUSCORE(NIN)+IHIS(NIN,1)
      ELSE                                 
         CALL MWARN('MINTEG')
         WRITE(*,*) 'Wrong idir in minteg: OPERATION NOT PERFORMED'
         RETURN
      ENDIF
      END

      REAL*8 FUNCTION SUMPOW(X,Y,IPOW)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      REAL*8 X,Y
      INTEGER IPOW
C
C     START
C
      IF(IPOW.EQ.1) THEN
         SUMPOW = X + Y
      ELSEIF(IPOW.EQ.2) THEN
         SUMPOW = DSQRT(X**2+Y**2)
      ELSEIF(IPOW.EQ.0) THEN
         CALL MWARN('SUMPOW')
         WRITE(*,*)'Error: IPOW=0 not allowed in SUMPOW'
      ELSE
         SUMPOW = (X**IPOW+Y**IPOW)**(1./IPOW)
      ENDIF
      END

      SUBROUTINE MOPERA(I,OPER,J,K,X,Y)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER I,J,K
      CHARACTER OPER*1
      REAL*8 X,Y
C
C     LOCAL
C
      REAL*8 XXX,XSUM,XSUMSQ,XNORM,XAVG,XSQAVG
      INTEGER L
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      IF( (BOOK(I)(1:1).NE.'Y'.AND.BOOK(I).NE.'NST') .OR.
     &    (BOOK(J)(1:1).NE.'Y'.AND.BOOK(J).NE.'NST') ) RETURN
      IF(NBIN(I).NE.NBIN(J).AND.(OPER.EQ.'+'.OR.OPER.EQ.'-'.OR.OPER.EQ.
     &    '*'.OR.OPER.EQ.'/'.OR.OPER.EQ.'M'.OR.OPER.EQ.'A')) THEN
	  CALL MWARN('MOPERA')
          WRITE(*,*) I,J                               
  20      FORMAT(' ****** INCOMPATIBLE OPERATION HIST ',I2,' &',I2,
     &    '*******'/)
          RETURN
      ENDIF
      IF(BOOK(K)(1:1).NE.'Y') 
     &  CALL MBOOK(K,TITLE(I),HDEL(I),HMIN(I),HMAX(I))
      IF(OPER.EQ.'E') THEN
c If I contains the accumulated weights, J the accumulated squares of the
c weights and IHIS(J,1) the number of accumulated entries, 'E' will add
c the average value of I to K and will put in J the quadratic dispersion.
         IF(IHIS(J,1).NE.0) THEN
            XXX = 1./IHIS(J,1)
         ELSE
            XXX = 0
         ENDIF
         DO L=1,NBIN(I)
            XSUM   = HIST(I,L)
            XSUMSQ = HIST(J,L)
            HIST(K,L)=HIST(K,L) + XXX*HIST(I,L)
            HIST(J,L)=XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
         ENDDO
         IENT(K)=IENT(K)+IENT(I)
         XSUM = USCORE(I)
         XSUMSQ = USCORE(J)
         USCORE(K) = USCORE(K)+XXX*XSUM
         USCORE(J) = XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
         XSUM = OSCORE(I)
         XSUMSQ = OSCORE(J)
         OSCORE(K) = OSCORE(K)+XXX*XSUM
         OSCORE(J) = XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
      ELSEIF(OPER.EQ.'Q') THEN
         DO L=1,NBIN(I)
            HIST(K,L) = SQRT(HIST(J,L)**2+HIST(I,L)**2)
         ENDDO
         USCORE(K) = SQRT(USCORE(J)**2+USCORE(I)**2)
         OSCORE(K) = SQRT(OSCORE(J)**2+OSCORE(I)**2)
      ELSEIF(OPER.EQ.'A') THEN
         DO L=1,NBIN(I)
            HIST(J,L) = HIST(J,L) + HIST(I,L)
            IHIS(J,L) = IHIS(J,L) + IHIS(I,L)
            HIST(K,L) = HIST(K,L) + HIST(I,L)**2
            IHIS(K,L) = IHIS(K,L) + 1
            HIST(I,L) = 0
            IHIS(I,L) = 0
         ENDDO
         IENT(J) = IENT(J)+IENT(I)
         IUSCORE(J) = IUSCORE(J) + IUSCORE(I)
         USCORE(J) = USCORE(J) + USCORE(I)
         IOSCORE(J) = IOSCORE(J) + IOSCORE(I)
         OSCORE(J) = OSCORE(J) + OSCORE(I)
         IENT(K) = IENT(K)+1
         IUSCORE(K) = IUSCORE(K) + 1
         USCORE(K) = USCORE(K) + USCORE(I)**2
         IOSCORE(K) = IOSCORE(K) + 1
         OSCORE(K) = OSCORE(K) + OSCORE(I)**2
         IENT(I) = 0
         IUSCORE(I) = 0
         IOSCORE(I) = 0
         USCORE(I) = 0
         OSCORE(I) = 0
      ELSEIF(OPER.EQ.'X') THEN
         DO L=1,NBIN(I)
            HIST(I,L) = 0
            IHIS(I,L) = 0
         ENDDO
         IENT(I) = 0
         IUSCORE(I) = 0
         IOSCORE(I) = 0
         USCORE(I) = 0
         OSCORE(I) = 0
      ELSE
        DO L=1,NBIN(I)
      	IF(OPER.EQ.'+') THEN
       	  HIST(K,L)=X*HIST(I,L) + Y*HIST(J,L)
      	ELSEIF(OPER.EQ.'-') THEN
      	  HIST(K,L)=X*HIST(I,L) - Y*HIST(J,L)
      	ELSEIF(OPER.EQ.'*') THEN
      	  HIST(K,L)=X*HIST(I,L) * Y*HIST(J,L)
      	ELSEIF(OPER.EQ.'/') THEN
          IF(Y.EQ.0..OR.HIST(J,L).EQ.0.) THEN
            HIST(K,L)=0.
          ELSE
            HIST(K,L)=X*HIST(I,L) / (Y*HIST(J,L))
          ENDIF
       	ELSEIF(OPER.EQ.'F') THEN
      	  HIST(K,L)=X*HIST(I,L)
      	ELSEIF(OPER.EQ.'R') THEN
          IF(HIST(I,L).GT.0.) THEN
            HIST(K,L)=X*DSQRT(HIST(I,L))
          ELSE                           
            HIST(K,L)=0.
          ENDIF
      	ELSEIF(OPER.EQ.'S') THEN
          HIST(K,L)=X*HIST(I,L)**2
      	ELSEIF(OPER.EQ.'L') THEN  
          IF(HIST(I,L).EQ.0..OR.J.EQ.0.) THEN
             HIST(K,L)=0.
           ELSE
             HIST(K,L)=X*LOG10(Y*HIST(I,L))
           ENDIF
      	ELSEIF(OPER.EQ.'M') THEN
           IF(I.NE.J) XNORM=HIST(I,L)
           IF(I.EQ.J) XNORM=DFLOAT(IHIS(J,L))
           IF(XNORM.NE.0.) THEN
             XAVG=HIST(J,L)/XNORM
             HIST(K,L)=
     &       DSQRT(ABS(-XAVG**2+HIST(K,L)/XNORM)/DFLOAT(IHIS(I,L)))
             HIST(J,L)=XAVG 
           ELSE 
             HIST(K,L)=0.
             HIST(J,L)=0.
           ENDIF
      	ELSEIF(OPER.EQ.'V') THEN                 
           XAVG=HIST(I,L)*X
           XSQAVG=HIST(J,L)*X
           XNORM=DFLOAT(IHIS(I,L))*X
           IF(XNORM.NE.0.) THEN
              HIST(K,L)=DSQRT(ABS(XSQAVG-XAVG**2)/XNORM)
              HIST(I,L)=XAVG
           ELSE  
              HIST(K,L)=0.
           ENDIF 
      	ELSE 
	 CALL MWARN('MOPERA')
         WRITE(*,*) OPER
   5     FORMAT(' ****** OPERATION ="',A1,'" UNKNOWN ********'/)
         RETURN
        ENDIF
        ENDDO
      	IF(OPER.EQ.'+') THEN
       	  USCORE(K)=X*USCORE(I) + Y*USCORE(J)  
       	  OSCORE(K)=X*OSCORE(I) + Y*OSCORE(J)  
      	ELSEIF(OPER.EQ.'-') THEN     
      	  USCORE(K)=X*USCORE(I) - Y*USCORE(J)
      	  OSCORE(K)=X*OSCORE(I) - Y*OSCORE(J)
      	ELSEIF(OPER.EQ.'*') THEN     
      	  USCORE(K)=X*USCORE(I) * Y*USCORE(J)
      	  OSCORE(K)=X*OSCORE(I) * Y*OSCORE(J)
      	ELSEIF(OPER.EQ.'/') THEN     
          IF(Y.EQ.0..OR.USCORE(J).EQ.0.) THEN
            USCORE(K)=0.
          ELSE
            USCORE(K)=X*USCORE(I) / (Y*USCORE(J))
          ENDIF
          IF(Y.EQ.0..OR.OSCORE(J).EQ.0.) THEN
            OSCORE(K)=0.
          ELSE
            OSCORE(K)=X*OSCORE(I) / (Y*OSCORE(J))
          ENDIF
       	ELSEIF(OPER.EQ.'F') THEN
      	  USCORE(K)=X*USCORE(I)
      	  OSCORE(K)=X*OSCORE(I)
      	ELSEIF(OPER.EQ.'R') THEN
          IF(USCORE(I).GT.0.) THEN
            USCORE(K)=X*DSQRT(USCORE(I))
          ELSE                           
            USCORE(K)=0.
          ENDIF     
          IF(OSCORE(I).GT.0.) THEN
            OSCORE(K)=X*DSQRT(OSCORE(I))
          ELSE                           
            OSCORE(K)=0.
          ENDIF     
      	ELSEIF(OPER.EQ.'S') THEN
          USCORE(K)=X*USCORE(I)**2
          OSCORE(K)=X*OSCORE(I)**2
      	ELSEIF(OPER.EQ.'L') THEN  
          IF(USCORE(I).EQ.0..OR.J.EQ.0.) THEN
             USCORE(K)=0.
           ELSE
             USCORE(K)=X*LOG10(Y*USCORE(I))
           ENDIF                         
          IF(OSCORE(I).EQ.0..OR.J.EQ.0.) THEN
             OSCORE(K)=0.
           ELSE
             OSCORE(K)=X*LOG10(Y*OSCORE(I))
           ENDIF                         
        ENDIF
      ENDIF
      RETURN
      END

      SUBROUTINE MZERO(N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N 
C
C     LOCAL
C
      INTEGER I
C     
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      BOOK(N)='RES'
      IENT(N)=0
      IUSCORE(N)=0
      IOSCORE(N)=0
      HAVG(N)=0.
      HINT(N)=0.
      DO 1 I=1,NBIN(N)
   1  HIST(N,I)=0.
      END

      SUBROUTINE MRESET(N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      BOOK(N)='RES'
      END

      SUBROUTINE PUTTAG(J,NAME)
C*************************************************************
c Per marcare un istogramma
      IMPLICIT NONE
C
C     ARGUMENTS
C
      CHARACTER*(*) NAME
      INTEGER J
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     LOCAL
C
      CHARACTER*(*) TAG
C
C     START
C
      BOOK(J) = NAME
      RETURN
      ENTRY GETTAG(J,TAG)
      TAG = BOOK(J)
      END

      SUBROUTINE MFINAL(N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
C
C     LOCAL
C
      INTEGER J,IF
      REAL*8 AVG,XIN,SIG,X
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      IF(BOOK(N)(1:1).NE.'Y') RETURN
      AVG=0
      XIN=0                                  
      SIG=0
      IF=0
      DO J=1,NBIN(N)
         X=HIST(N,J)
 	 AVG=AVG+X*XHIS(N,J)
         XIN=XIN+X
	 IF(X.NE.0) IF=1
      ENDDO             
      IF(XIN.EQ.0) GO TO 10
      AVG = AVG/XIN
      DO J=1,NBIN(N)
         SIG=HIST(N,J)*(XHIS(N,J)-AVG)**2+SIG
      ENDDO
      SIG=DSQRT(ABS(SIG/XIN))
 10   CONTINUE
      HINT(N) = XIN
      HAVG(N) = AVG
      HSIG(N) = SIG
      END               

      SUBROUTINE MNORM(N,X)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
      REAL*8  X
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     LOCAL
C
      INTEGER I
      REAL*8 Y
C
C     START
C
      IF(BOOK(N)(:1).NE.'Y')RETURN
      IF(HINT(N).EQ.0.) THEN
	CALL MWARN('MNORM')
	WRITE(*,*)' INTEGRAL HIST ',N,' IS ZERO: CANNOT RENORMALIZE'
	RETURN               
      ELSE
	Y=X/HINT(N)
      ENDIF
      DO 1, I=1,NBIN(N)
    1 HIST(N,I)=HIST(N,I)*Y
      HINT(N)=X            
      OSCORE(N)=OSCORE(N)*Y
      USCORE(N)=USCORE(N)*Y
      END                  

      SUBROUTINE MPRINT(N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N
C
C     LOCAL
C
      INTEGER INI,J
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      DATA INI/0/
      IF(INI.EQ.0) THEN
c     CALL IDATE(IMON,IDAY,IYEAR)
c     CALL TIME(CTIME)
      INI=1
      ENDIF
      IF(BOOK(N)(:1).NE.'Y') RETURN
C      WRITE(98,7) N,IYEAR,IMON,IDAY,CTIME(1:5)
      WRITE(98,*) TITLE(N)
      DO 1 J=1,NBIN(N)
      IF(HIST(N,J).EQ.0.) GO TO 1
      WRITE(98,'(3X,F10.4,2X,E15.4)')  
     &                            XHIS(N,J),HIST(N,J)
    1 CONTINUE
      WRITE(98,15) HAVG(N),HSIG(N),HINT(N)
      WRITE(98,20) IENT(N),IUSCORE(N),IOSCORE(N)
C    7 FORMAT(4X,'HIST = ',I3,'   19',I2,'-',I2,'-',I2,1X,A5/)
   10 FORMAT(4X,2E10.3)
   15 FORMAT(/' AVG =',E10.3,4X,' RMS =',E10.3,' INTEGRAL =',E10.3,/)
   20 FORMAT(' ENTRIES=',I5,4X,'UNDERSCORE=',I5,4x,'OVERSCORE=',I5,//)
      END


      SUBROUTINE MTOP4(N,M,BTIT,LTIT,SCALE)
      
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N,M
      CHARACTER*(*) LTIT,BTIT
      CHARACTER*(*) SCALE
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     LOCAL
C
      INTEGER INI,J
      real*8 maxbin
C
C     START
C
      DATA INI/0/
      IF(INI.EQ.0) THEN
c      CALL IDATE(IMON,IDAY,IYEAR)
c      CALL TIME(CTIME)
      INI=1
      ENDIF
      IF(BOOK(N)(:1).NE.'Y') RETURN
cRF
      IF (N.eq.1)WRITE(99,'(A)')' SET DEVICE POSTSCRIPT ORIENT=3'

      WRITE(99,100) TITLE(N),BTIT,LTIT,SCALE,HMIN(N),HMAX(N)
  100 FORMAT( /1x,                               
     &' SET INTENSITY 4'/,1X,
     &' SET WINDOW Y 2.5 TO 9.'/,1X,
     &' SET WINDOW X 2.5 TO 12.'/,1X,
     &' SET FONT DUPLEX '/1X, 
     &' SET SYMBOL 5O SIZE 1.8'/,1X,
     &' TITLE TOP ','"',A20,'"',/1X,
     &' TITLE BOTTOM ','"',A20,'"',/1X,
     &' TITLE LEFT ','"',A20,'"',/1X,
     &' SET SCALE Y ',A,/1X,
     &' (SET TICKS TOP OFF)   '/1x,     
     &' SET LIMITS X ',F10.5,' ',F10.5,/1X,
     &' SET ORDER X Y DY ')
      maxbin=0d0
      DO J=1,NBIN(N)
         maxbin=max(maxbin,abs(HIST(N,J)))
      ENDDO
      DO 1 J=1,NBIN(N)
         WRITE(99,'(3X,F10.4,2(2X,E15.4))')  
     &        XHIS(N,J),HIST(N,J),HIST(M,J)
    1 CONTINUE
      WRITE(99,200)
  200 FORMAT('   HISTO')
  300 FORMAT( /1x,
     &' (INFO-BOX'/,1X,                               
     &' BOX 7.3 1.3 SIZE 9.5 0.7'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 3.8 1.4 "X-sect =',1PE10.3,'(pb)   AVG =',1PE10.3,
     &             '   RMS =',1PE10.3,'"',/1X,
     &' TITLE 3.8 1.1 "Entries =',I8,2x,'Entries =',I8,2X
     &                ,'Undersc =',I6,2X,'Oversc =',I6,'"',/1X,
     &' SET TITLE SIZE -2')                            
      WRITE(99,400)
  400 FORMAT('   NEW PLOT')
      END


      SUBROUTINE MULTITOP4(NH,NE,N,M,BTIT,LTIT,SCA)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER NH,NE,N,M
      CHARACTER*(*) LTIT,BTIT
      CHARACTER*(*) SCA
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     LOCAL
C
      REAL*8 YTIT,XTIT,FEXP,FMAX,FMIN,X,XMX,FMX,FMN
      REAL*8 YU,XU,YL,XL,YTIT0,YD,XTIT0,TITX,TITS,TICS,SRED,XD
      INTEGER IBIN,IPNS,J,I,NOLD,IFRAME,IFRMAX,IP,NS,MOLD
      INTEGER INI
C
C     START
C
      CHARACTER SCALE*3
      CHARACTER*7  PLOT(4)
      DATA PLOT/'SOLID','DASHES','DOTS','DOTDASH'/
C  PLOT SIZE, CORNERS
      REAL*8 WIDTH,HEIGHT,XCORN,YCORN
      DATA WIDTH,HEIGHT/11.5,8.5/,XCORN,YCORN/1.5,1./
C  PLOT VERSUS TEXT FRACTION                  
      REAL*8 XPFRAC,YPFRAC,XTFRAC,YTFRAC
      DATA XPFRAC,YPFRAC/0.75,0.75/,XTFRAC,YTFRAC/0.25,0.25/
C  DEFAULT SIZES                                           
      REAL*8 TIC0,LAB0,TIT0,LABS
      DATA TIT0,LAB0,TIC0/3d0,3d0,0.06d0/
      DATA INI/0/                                          
      IF(INI.EQ.0) THEN
c      CALL IDATE(IMON,IDAY,IYEAR)
c      CALL TIME(CTIME)
      IFRAME=0        
C      WRITE(99,71) IYEAR,IMON,IDAY,CTIME(1:5)
C   71 FORMAT(4X,' (   19',I2,' -',I2,' -',I2,1X,A5/)
      INI=1         
      ENDIF
      IF(SCA.EQ.'REF') THEN
	IFRAME=0
	RETURN
      ENDIF
      IF(BOOK(NH)(:1).NE.'Y') RETURN
      IFRMAX=N*M         
      IFRAME=IFRAME+1
      IF(IFRAME.GT.IFRMAX.OR.N.NE.NOLD.OR.M.NE.MOLD) THEN
      	IFRAME=1
        WRITE(99,202)   
C        WRITE(99,1) IMON,IDAY,CTIME(1:5)
C  1     FORMAT(' SET FONT DUPLEX',/,'  SET TITLE SIZE 2',/,
C     +      ' TITLE 12.8 9 ANGLE -90 ','" MLM   ',I2,'-',I2,1X,A5,'"')
      ENDIF                                
      IF(IFRAME.EQ.1) THEN
    	I=1
	J=1
      ELSEIF(IFRAME.LE.IFRMAX) THEN
	IF(I.LE.N) I=I+1
        IF(I.GT.N) THEN
		I=1
		J=J+1
	ENDIF
      ENDIF
      IF(N.EQ.NOLD) GO TO 10
      NS=N-1
      XD=WIDTH/DFLOAT(N)
      SRED=DSQRT(DFLOAT(N*M))
      TITS=TIT0/SRED          
      LABS=LAB0/SRED
      TICS=TIC0/SRED
      XTIT0=0.55*XPFRAC*XD
      NOLD=N            
10    IF(M.EQ.MOLD) GO TO 20
      YD=HEIGHT/DFLOAT(M)
      YTIT0=0.06*YD
      MOLD=M        
20    CONTINUE
      XL=(I-1)*XD + XCORN
      YL=(M-J)*YD + YCORN
      XU=XL+XD*XPFRAC
      YU=YL+YD*YPFRAC        
      IP=0
      XMX=0.
      DO IBIN=1,NBIN(NH)
        X=HIST(NH,IBIN)
        IF(X.NE.0.) THEN
           IF(XMX.EQ.0.) THEN
              FMX = X + HIST(NE,IBIN)
              FMN = X - HIST(NE,IBIN)
           ELSE
              FMX=MAX(FMX,X + HIST(NE,IBIN))
              FMN=MIN(FMN,X - HIST(NE,IBIN))
           ENDIF
        ENDIF
        XMX=MAX(XMX,ABS(X)+ HIST(NE,IBIN))
      ENDDO
      SCALE=SCA
50    IF(SCALE.EQ.'LIN') THEN
        IF(FMN.GE.0.)   FMIN=0.
        IF(FMN.LT.0.)   FMIN=FMN*1.3
        IF(FMX.GT.0.)   FMAX=FMX*1.3
        IF(FMX.LT.0.)   FMAX=0.
      ELSEIF(SCALE.EQ.'LOG') THEN
cRF
c$$$        IF(FMN.LE.0.) THEN
c$$$                SCALE='LIN'
c$$$                GO TO 50
c$$$        ENDIF
        FMAX=10.**( AINT(LOG10(ABS(FMX))+1000001) - 1000000 )
        FMIN=10.**( AINT(LOG10(ABS(FMN))+1000000) - 1000000 )
      ENDIF                         
      WRITE(99,100) TITS,LABS,TICS,XL,XU,YL,YU
100   FORMAT(2X,'( SET FONT DUPLEX',/,
     *       2X,'SET TITLE SIZE ',F8.4,/,
     *       2X,'SET LABEL SIZE ',F8.4,/,
     *       2X,'SET TICKS TOP OFF SIZE ',F8.4,/,
     *       2X,'SET WINDOW X ',F8.4,' TO ',F8.4,/,
     *       2X,'SET WINDOW Y ',F8.4,' TO ',F8.4)
      XTIT=XL+XTIT0
      YTIT=YU+YTIT0
      WRITE(99,101) XL,YTIT,TITLE(NH)(1:40)
101   FORMAT('  TITLE ',2(F8.4,1X),'"',A,'"')                  
      YTIT=YTIT-2.*YTIT0
      WRITE(99,102) XTIT,YTIT,HINT(NH)
102   FORMAT('  TITLE ',2(F8.4,1X),'" INT=',1PE10.3,'"')                  
      YTIT=YTIT-YTIT0
      WRITE(99,103) XTIT,YTIT,IENT(NH)
103   FORMAT('  TITLE ',2(F8.4,1X),'" ENT=',I9,'"')                  
      YTIT=YTIT-YTIT0                         
      IF(USCORE(NH).NE.0.) THEN
        WRITE(99,104) XTIT,YTIT,USCORE(NH)
104     FORMAT('  TITLE ',2(F8.4,1X),'" UFL=',1PE10.3,'"')                  
        YTIT=YTIT-YTIT0                      
      ENDIF
      IF(OSCORE(NH).NE.0.) THEN
        WRITE(99,105) XTIT,YTIT,OSCORE(NH)
105     FORMAT('  TITLE ',2(F8.4,1X),'" OFL=',1PE10.3,'"')                  
        YTIT=YTIT-YTIT0                      
      ENDIF      
      WRITE(99,106) XTIT,YTIT,XU,YTIT,XTIT,YTIT,XTIT,YU
106   FORMAT(2X,'SET ORD X Y ',/,2(F8.4,1X),/,2(F8.4,1X),/,
     *       2X,'JOIN TEXT',/,
     *       2X,2(F8.4,1X),/,2(F8.4,1X),/,
     *       2X,'JOIN TEXT')                                    
      WRITE(99,108) TITS*1.5
108   FORMAT(2X,'SET TITLE SIZE ',F8.4)
      WRITE(99,107) BTIT,XL-0.75*XD*XTFRAC,YL+(YU-YL)/3.,LTIT,SCALE,
     * HMIN(NH),HMAX(NH),FMIN,FMAX
107   FORMAT(                                           
     &' TITLE BOTTOM ','"',A,'"',/1X,
     &' TITLE ',f10.5,f10.5,' ANGLE 90 ','"',A,'"',/1X,
     &' SET SCALE Y ',A,/1X,
     &' SET TICKS TOP OFF   '/1x,     
     &' SET LIMITS X ',F10.5,' ',F10.5,/1X,
     &' SET LIMITS Y ',1PE10.3,' ',1PE10.3,/1X,
     &' SET ORDER X Y DY')               
C                       
C  END HEADER , FILL TOPDRAWER WITH DATA
C
      ENTRY MTFILL(NH,NE,N,M,BTIT,LTIT,SCA)
      IP=IP+1                             
      IF(IP.GT.4) IP=1
      WRITE(99,110) TITLE(NH),HINT(NH),IENT(NH)
110   FORMAT(' ( ',A,/,' ( INT=',1PE10.3,'  ENTRIES=',I12)
      DO 200 IBIN=1,NBIN(NH)           
      WRITE(99,'(3X,F10.4,2(2X,E15.4))')  
     &          XHIS(NH,IBIN),HIST(NH,IBIN),HIST(NE,IBIN)
200   CONTINUE                                           
      WRITE(99,201)  PLOT(IP)
      IF(BOOK(NE).NE.'NO')   WRITE(99,*)  '  PLOT'
201   FORMAT(2X,'HIST ',A)           
202   FORMAT('   NEW PLOT',/,/)
203   RETURN     
      END                     


      SUBROUTINE MWARN(ROUT)
C*************************************************************
      CHARACTER*(*) ROUT
      WRITE(*,*) '***********************************************'
      WRITE(*,*) '***** WARNING CALLED FROM ROUTINE ',ROUT,':'
      END


C*******************************************************************
C     END OF THE HISTOGRAMMING PACKAGE
C*******************************************************************





      subroutine mfill(n,var,www)
      implicit none
      integer n,n_by4
      double precision var,www
c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
      integer j
      if(iappl.ne.0)then
         do j=1,nh_obs
            if(n.eq.ih_obs(j))then
               appl_obs_num   = j
               appl_obs_histo = var 
c     Fill the reference APPLgrid histograms
               call APPL_fill_ref
c     Fill the APPLgrid files
               call APPL_fill
            endif
         enddo
      endif
      n_by4=4*(n-1)+1
      call mfill4(n_by4,var,www)
      return
      end

      subroutine bookup(n,string,del,xl,xu)
      implicit none
      integer i,n,n_by4
      character*(*) string
      double precision del,xl,xu
c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
c     Initialize the grids only if the switch "iappl" is different from zero
c     and if the title string containes the word "central" and does not contain
c     the word "Born". 
      if(iappl.ne.0.and.index(string,"central").ne.0.and.
     1                  index(string,"Born").eq.0)then
c     Observable parameters
c     Compute number of bins and edges only if they have not been given by the user.
         if(appl_obs_nbins.eq.0)then
            appl_obs_nbins = int( ( xu - xl ) / del / 0.999999d0 )
c     compute bin edges
            do i=0,appl_obs_nbins
               appl_obs_bins(i) = xl + i * del
            enddo
         endif
         appl_obs_min = appl_obs_bins(0)
         appl_obs_max = appl_obs_bins(appl_obs_nbins)
         if(abs(appl_obs_max-xu).gt.0.00000001d0)then
            write(*,*) 'APPLgrid Histogram: ', 
     1                 'Change of the upper limit:',xu,'-->',
     2                  appl_obs_max
         endif
c     Initialize APPLgrid routines
         call APPL_init
c     Keep track of the position of this histogram
         nh_obs = nh_obs + 1
         ih_obs(nh_obs) = n
c     Reset number of bins to zero
         appl_obs_nbins = 0
      endif
      n_by4=4*(n-1)+1
      call bookup4(n_by4,string,del,xl,xu)
      return
      end

      subroutine MTOP(n,string1,string2,string3)
      implicit none
      integer n,n_by4,m_by4
      character*(*) string1,string2,string3
c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
      integer j
      if(iappl.ne.0)then
         do j=1,nh_obs
            if(n.eq.ih_obs(j))then
               appl_obs_num = j
               call APPL_fill_ref_out
               call APPL_term
            endif
         enddo
      endif
      n_by4=4*(n-1)+1
      m_by4=n_by4+3
c write the 'n' plots with the 'n+3' error bars
      call mtop4(n_by4,m_by4,string1,string2,string3)
      return
      end

      subroutine multitop(n,lr,lh,string1,string2,string3)
      implicit none
      integer n,n_by4,lr,lh,m_by4
      character*(*) string1,string2,string3
c     APPLgrid commons
      include "reweight_appl.inc"
      include "appl_common.inc"
      integer iappl
      common /for_applgrid/ iappl
      integer j
      if(iappl.ne.0)then
         do j=1,nh_obs
            if(n.eq.ih_obs(j))then
               appl_obs_num = j
               call APPL_fill_ref_out
               call APPL_term
            endif
         enddo
      endif
      n_by4=4*(n-1)+1
      m_by4=n_by4+3
c write the 'n' plots with the 'n+3' error bars
      call multitop4(n_by4,m_by4,lr,lh,string1,string2,string3)
      return
      end



      subroutine bookup4(n,string,del,xl,xu)
      implicit none
      integer n
      character*(*) string
      double precision del,xl,xu
c
c Per ogni istogramma da fare, ne sono richiesti quattro
c In n si accumulano i valori in outfun.
c A ogni iterazione di vegas l'istogramma n viene sommato a n+1,
c e il quadrato del suo valore viene sommato a n+2,
c L'istogramma n viene anche usato alla fine per combinare
c i totali dei vari contributi sig0,sig2, etc. mentre
c l'istogramma n+3 viene usato per combinare gli
c errori dei vari contributi sig0,sig2, etc.
c Non si vuole che n e n+3 vengano salvati o riesumati.
c Cambiando il tag in N,N+3, questo non avviene (si guardi
c in mbook e save/restart e anche mclear in questo programma.
c
      call mbook(n,  string,del,xl,xu)
      call mbook(n+1,'tmp ',del,xl,xu)
      call mbook(n+2,'tmp square',del,xl,xu)
      call mbook(n+3,'error ',del,xl,xu)
      call puttag(n,'YST')
      call puttag(n+3,'NST')
      return
      end

      subroutine accum(inclde)
      implicit real * 8 (a-h,o-z)
      include 'dbook.inc'
      PARAMETER (NMB=NPLOTS)
      character * 3 tag
      logical inclde
c
c     Accumula i valori e i valori al quadrato per l'analisi statistica,
c     e svuota l'istogramma di accumulo.
c
      do j=1,nmb-3
         call gettag(j,tag)
         if(tag.eq.'YST') then
             if (inclde) then
c Sum the results in histos 'j' onto 'j+1' and the squares to 'j+2'
c This also empties histograms 'j'
                call mopera(j,'A',j+1,j+2,dum,dum)
             else
c Only empty the histograms 'j'.
                call mopera(j,'X',j,j,dum,dum)
             endif
         endif
      enddo
      return
      end

      subroutine mclear
      implicit real * 8 (a-h,o-z)
      character * 70 files(20), filn
      data nfil/0/
c
c sum up files
      call sumfil(files,nfil)
      nfil=0
      return
      entry addfil(filn)
c adds file filn to the list of save files.
      nfil = nfil+1
      if(nfil.gt.20) then
         write(*,*) 'mclear: too many files.'
         stop
      endif
      files(nfil) = filn
      end

      subroutine sumfil(files,nfil)
      implicit real * 8 (a-h,o-z)
      include 'dbook.inc'
      PARAMETER (NMB=NPLOTS)
      external dummyinit
      character * 3 tag
      character * 70 files(*)
c
      do j=1,nfil
         call resume(files(j),dummyinit,'NO')
c
c completa l'analisi statistica
c
         do k=1,nmb-3
            call gettag(k,tag)
            if(tag.eq.'YST') then
               call addup(k)
            endif
         enddo
      enddo
      end

      subroutine dummyinit()
      return
      end

      subroutine addup(j)
      implicit none
      integer j
      real*8 dum
c
c accumula j+1 riscalato in j e pone in j+2 la stima dell'errore
c
      call mopera(j+1,'E',j+2,j,dum,dum)
c
c accumula l'errore in quadratura
c
      call mopera(j+2,'Q',j+3,j+3,dum,dum)
      end



