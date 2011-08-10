C**********************************************************************
C    SIMPLE HISTOGRAMMING PACKAGE --  SIMPLIFIED VERSION OF HBOOK
C    BY Michelangelo Mangano    NOVEMBER 1988
C    LAST REVISED NOVEMBER 9, 1988
C    LAST REVISED JUNE 12, 1989  (ADD SCATTER PLOTS)
C    LAST REVISED oct 1990 (Add multi-plots on one page, routines MULTITOP,
C         			MTFILL,...)
C    LAST REVISED Jun 2003 by FM: passed to double precision AND 
C    included the date for linux      
C    LAST REVISED Jun 2008 by OM: add inverse function 
C    LAST REVISED Oct 2009 by AP/OM: possibility of multiple independent 
C                        set of histo.
C**********************************************************************
C
C Fills up to 200 histograms with up to 100 bins. 
C Gives a data file (to be specified in the calling program by assigning 
C a file name to unit 98) and a topdrawer file (to be specified in the 
C calling program by assigning a file name to unit 99).
C
C INITIALIZATION:
C Call once INIHIST; this just resets a few counters and logicals
C Call MBOOK(M,N,'TITLE',DEL,XMIN,XMAX) for each histogram to be booked.
C M (an integer) is the label of the histogram object (by default M=1)
C N (an integer) is the label of the histogram;
C 'TITLE' is the name of the histogram (no more then 100 characters);
C DEL (real*8) is the bin size;
C XMIN (real*8) is the lower limit of the first bin;
C XMAX (real*8)is the upper limit of the last  bin
C Example:
C      call mbook(1,2,'pt distribution',1.,10,70)
C This call initializes histogram number 2, called 'pt distribution';
C The bin size will be 1. (possibly GeV, if that's what you want), the
C first bin being  10<x<11. and the last one being 69.<x<70
C
C FILLING:
C When it's time, call MFILL(M,N,X,Y); this will add Y (real*8) to the bin 
C in which X (real*8) happens to be, within histogram (M,N). 
C
C PLAYING AROUND:
C At the end of the day you may want to sum, divide, cancel, etc.etc.
C various histograms (bin by bin). Then you call MOPERA(M,I,'O',J,K,X,Y). 
C The 1-character string O can take the following values:
C +  : sums       X*(hist I) with Y*(hist J) and puts the result in hist K;
C -  : subtracts  X*(hist I) with Y*(hist J) and puts the result in hist K;
C *  : multiplies X*(hist I) with Y*(hist J) and puts the result in hist K;
C /  : divides    X*(hist I) with Y*(hist J) and puts the result in hist K;
C I  : inverses hist I and put the results [X*(1/I)]  in hist K;
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
C Now we can finalize our histograms; MFINAL(M,N) will calculate the integral
C of the histogram (M,N), the mean value of the X variable and its RMS.
C If we now want to renormalize the hist's, we can call MNORM(M,N,X), which
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
C MTOP(M,N,L,'X','Y','SCALE'). 
c The points of the plot will be taken from histogram
C (M,N), the error bars from histogram (M,L). 'SCALE', character*(*), determines
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
C   BOOK(N,M),      Three-letter character-string: 'NO' if histogram was not 
C		  Booked, 'YES' otherwise
C   TITLE(N,M),     Title of the histogram
C
C   HMIN(N,M),      Min value of x range
C   HMAX(N,M),      Max value of x range
C   HDEL(N,M),      Bin width
C   NBIN(N,M),      Total number of bins
C   USCORE(N,M),    Total integral of underscores with x < HMIN(N,M)
C   OSCORE(N,M),    Total integral of onderscores with x > HMAX(N,M)
C   IUSCORE(N,M),   Number of entries with x < HMIN(N,M)
C   IOSCORE(N,M),   Number of entries with x > HMAX(N,M)
C   IENT(N,M),      Total number of entries within x range HMIN(N,M)<x<HMAX(N,M)
C   HINT(N,M),      Integral of the histogram within HMIN(N,M)<x<HMAX(N,M)
C   HAVG(N,M),      Average value of x, weighted over the x range of the histo
C   HSIG(N,M),      Quadratic dispersion of x around the average
C   HIST(N,L,M),    Value of bin L-th
C   XHIS(N,L,M),    Central x value of bin L-th
C   IHIS(N,L,M),    Number of entries within bin L-th
C   NHIST(M)         Total number of booked histograms
C
      SUBROUTINE INIHIST
C*************************************************************
c     initialization
      IMPLICIT NONE
      include 'dbook.inc'
c
c     LOCAL
c
      INTEGER I,J
C
C     START
C
      DO J=1, NHistoVar
      NHist(J)=0
      DO  I=1,NHistoByVar             
      BOOK(I,J)=' NO'
      ENDDO  
      ENDDO
      END

      SUBROUTINE MBOOK(M,N,TIT,DEL,XMIN,XMAX)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N,M
      CHARACTER*20 TIT
      REAL*8 DEL,XMIN,XMAX
C
C     GLOBAL
C
      include 'dbook.inc'
C
C     LOCAL
C
      INTEGER I,J
C
C     START
C
      IF(M.gt.NHistovar.or.abs(N).gt.NHistobyvar) THEN
         Write(*,*) 'histogram input exceeded definition range'
         Write(*,*) 'input',M,N,' for limit ',Nhistovar,NHistoByVar
         STOP
      endif
      NHIST(M) = MAX(N,NHIST(M))
      IF(BOOK(N,M)(1:1).EQ.'Y') THEN
	 CALL MWARN('MBOOK')
         WRITE(*,*) 'Histogram',N,M,TITLE(N,M),' already in use. '
         WRITE(*,*) 'superseded by ',TIT
      ENDIF
      BOOK(N,M) = 'YES'
      TITLE(N,M) = ' '//TIT
1     HDEL(N,M) = DEL
      NBIN(N,M) = INT((XMAX-XMIN)/DEL+0.00001d0)
      IF(NBIN(N,M).GT.NHistoBin) THEN
	WRITE(*,*) 'TOO MANY BINS (',NBIN(N,M),') REQUIRED IN HIST ',N,M
	WRITE(*,*) 'RE-ENTER BIN SIZE DELTA (OLD BIN = ',DEL,' ):'
	READ(*,*) DEL
	GO TO 1
      ENDIF
      HMIN(N,M) = XMIN
      HMAX(N,M) = NBIN(N,M)*DEL+XMIN
      IF(abs(HMAX(N,M)-XMAX).gt.0.001d0) THEN
	 CALL MWARN('MBOOK')
         WRITE(*,*)
     #'Histogram ', TIT, ' Change of upper limit:',xmax,'-->',HMAX(N,M)
      ENDIF
      IENT(N,M) = 0
      IUSCORE(N,M) = 0
      IOSCORE(N,M) = 0
      USCORE(N,M) = 0
      OSCORE(N,M) = 0
      HAVG(N,M) = 0
      HINT(N,M) = 0
      HSIG(N,M) = 0
      DO I=1,NBIN(N,M)
         XHIS(N,I,M)=HMIN(N,M)+HDEL(N,M)*(DFLOAT(I)-0.5d0)
         IHIS(N,I,M)=0
         HIST(N,I,M)=0
      ENDDO
      END

      SUBROUTINE MFILL(M,N,X,Y)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER N,M
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
      IF(X.LT.HMIN(N,M)) THEN
         USCORE(N,M) = USCORE(N,M) + Y
         IUSCORE(N,M) = IUSCORE(N,M) + 1
      ELSEIF(X.GT.HMAX(N,M)) THEN
         OSCORE(N,M) = OSCORE(N,M) + Y
         IOSCORE(N,M) = IOSCORE(N,M) + 1
      ELSE
         XI=((X-HMIN(N,M))/HDEL(N,M))+1
         I=INT(XI)
         IENT(N,M)=IENT(N,M)+1
         IHIS(N,I,M)=IHIS(N,I,M)+1
         HIST(N,I,M)=HIST(N,I,M)+Y
      ENDIF
      END


      SUBROUTINE MINTEG(MN,NIN,NOUT,IDIR,IPOW)
C*************************************************************
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
      INTEGER NIN,NOUT,IDIR,IPOW,MN
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
      M = NBIN(NIN,MN)                                           
      I = (IDIR + 3)/2
      IF(NOUT.NE.NIN) THEN
      	CALL MBOOK(MN,NOUT,TITLE(NIN,MN)//C(I), 
     &                HDEL(NIN,MN),HMIN(NIN,MN),HMAX(NIN,MN))
      ENDIF
      IF(IDIR.EQ.1) THEN
         HIST(NOUT,1,MN) = SUMPOW(HIST(NIN,1,MN),USCORE(NIN,MN),IPOW)
         IHIS(NOUT,1,MN) = IHIS(NIN,1,MN) + IUSCORE(NIN,MN)
         XHIS(NOUT,1,MN) = XHIS(NIN,1,MN) + HDEL(NIN,MN)/2
         DO L=2,M                      
            HIST(NOUT,L,MN) = SUMPOW(HIST(NIN,L,MN),HIST(NOUT,L-1,MN),IPOW)
            IHIS(NOUT,L,MN) = IHIS(NIN,L,MN) + IHIS(NOUT,L-1,MN) 
            XHIS(NOUT,L,MN) = XHIS(NIN,L,MN) + HDEL(NIN,MN)/2
         ENDDO
         OSCORE(NOUT,MN) = SUMPOW(OSCORE(NIN,MN),HIST(NIN,M,MN),IPOW)
         IOSCORE(NOUT,MN) = IOSCORE(NIN,MN) + IHIS(NIN,M,MN)
      ELSEIF(IDIR.EQ.-1) THEN
         HIST(NOUT,M,MN) = SUMPOW(HIST(NIN,M,MN),OSCORE(NIN,MN),IPOW)
         IHIS(NOUT,M,MN) = IHIS(NIN,M,MN) + IOSCORE(NIN,MN)
         XHIS(NOUT,M,MN) = XHIS(NIN,M,MN) - HDEL(NIN,MN)/2
         DO L=M-1,1,-1                        
            HIST(NOUT,L,MN) = SUMPOW(HIST(NIN,L,MN),HIST(NOUT,L+1,MN),IPOW)
            IHIS(NOUT,L,MN) = IHIS(NIN,L,MN) + IHIS(NOUT,L+1,MN)
            XHIS(NOUT,L,MN) = XHIS(NIN,L,MN) - HDEL(NIN,MN)/2
         ENDDO
         USCORE(NOUT,MN) = SUMPOW(USCORE(NIN,MN),HIST(NIN,MN,1),IPOW)
         IUSCORE(NOUT,MN) = IUSCORE(NIN,MN)+IHIS(NIN,MN,1)
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

      SUBROUTINE MOPERA(M,I,OPER,J,K,X,Y)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,I,J,K
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
      IF(NBIN(I,M).NE.NBIN(J,M).AND.(OPER.EQ.'+'.OR.OPER.EQ.'-'.OR.OPER.EQ.
     &    '*'.OR.OPER.EQ.'/'.OR.OPER.EQ.'M'.OR.OPER.EQ.'A')) THEN
	  CALL MWARN('MOPERA')
          WRITE(*,*) M,I,J                               
  20      FORMAT(' ****** INCOMPATIBLE OPERATION HIST ',I2,' &',I2,
     &    '*******'/)
          RETURN
      ENDIF
      IF(OPER.EQ.'E') THEN
c If I contains the accumulated weights, J the accumulated squares of the
c weights and IHIS(J,1) the number of accumulated entries, 'E' will add
c the average value of I to K and will put in J the quadratic dispersion.
         IF(IHIS(J,1,M).NE.0) THEN
            XXX = 1./IHIS(J,1,M)
         ELSE
            XXX = 0
         ENDIF
         DO L=1,NBIN(I,M)
            XSUM   = HIST(I,L,M)
            XSUMSQ = HIST(J,L,M)
            HIST(K,L,M)=HIST(K,L,M) + XXX*HIST(I,L,M)
            HIST(J,L,M)=XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
         ENDDO
         IENT(K,M)=IENT(K,M)+IENT(I,M)
         XSUM = USCORE(I,M)
         XSUMSQ = USCORE(J,M)
         USCORE(K,M) = USCORE(K,M)+XXX*XSUM
         USCORE(J,M) = XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
         XSUM = OSCORE(I,M)
         XSUMSQ = OSCORE(J,M)
         OSCORE(K,M) = OSCORE(K,M)+XXX*XSUM
         OSCORE(J,M) = XXX*DSQRT(ABS(XSUMSQ-XSUM**2*XXX))
      ELSEIF(OPER.EQ.'Q') THEN
         DO L=1,NBIN(I,M)
            HIST(K,L,M) = SQRT(HIST(J,L,M)**2+HIST(I,L,M)**2)
         ENDDO
         USCORE(K,M) = SQRT(USCORE(J,M)**2+USCORE(I,M)**2)
         OSCORE(K,M) = SQRT(OSCORE(J,M)**2+OSCORE(I,M)**2)
      ELSEIF(OPER.EQ.'A') THEN
         DO L=1,NBIN(I,M)
            HIST(J,L,M) = HIST(J,L,M) + HIST(I,L,M)
            IHIS(J,L,M) = IHIS(J,L,M) + IHIS(I,L,M)
            HIST(K,L,M) = HIST(K,L,M) + HIST(I,L,M)**2
            IHIS(K,L,M) = IHIS(K,L,M) + 1
            IENT(K,M) = IENT(K,M)+1
            HIST(I,L,M) = 0
            IHIS(I,L,M) = 0
         ENDDO
         IENT(J,M) = IENT(J,M)+IENT(I,M)
         IUSCORE(J,M) = IUSCORE(J,M) + IUSCORE(I,M)
         USCORE(J,M) = USCORE(J,M) + USCORE(I,M)
         IOSCORE(J,M) = IOSCORE(J,M) + IOSCORE(I,M)
         OSCORE(J,M) = OSCORE(J,M) + OSCORE(I,M)
         IUSCORE(K,M) = IUSCORE(K,M) + 1
         USCORE(K,M) = USCORE(K,M) + USCORE(I,M)**2
         IOSCORE(K,M) = IOSCORE(K,M) + 1
         OSCORE(K,M) = OSCORE(K,M) + OSCORE(I,M)**2
         IENT(I,M) = 0
         IUSCORE(I,M) = 0
         IOSCORE(I,M) = 0
         USCORE(I,M) = 0
         OSCORE(I,M) = 0
      ELSE
        DO L=1,NBIN(I,M)
      	IF(OPER.EQ.'+') THEN
       	  HIST(K,L,M)=X*HIST(I,L,M) + Y*HIST(J,L,M)
      	ELSEIF(OPER.EQ.'-') THEN
      	  HIST(K,L,M)=X*HIST(I,L,M) - Y*HIST(J,L,M)
      	ELSEIF(OPER.EQ.'*') THEN
      	  HIST(K,L,M)=X*HIST(I,L,M) * Y*HIST(J,L,M)
      	ELSEIF(OPER.EQ.'/') THEN
          IF(Y.EQ.0..OR.HIST(J,L,M).EQ.0.) THEN
            HIST(K,L,M)=0.
          ELSE
            HIST(K,L,M)=X*HIST(I,L,M) / (Y*HIST(J,L,M))
          ENDIF
      	ELSEIF(OPER.EQ.'I') THEN
          IF(HIST(I,L,M).EQ.0.) THEN
            HIST(K,L,M)=0.
          ELSE
            HIST(K,L,M)=X/HIST(I,L,M) 
          ENDIF
       	ELSEIF(OPER.EQ.'F') THEN
      	  HIST(K,L,M)=X*HIST(I,L,M)
      	ELSEIF(OPER.EQ.'R') THEN
          IF(HIST(I,L,M).GT.0.) THEN
            HIST(K,L,M)=X*DSQRT(HIST(I,L,M))
          ELSE                           
            HIST(K,L,M)=0.
          ENDIF
      	ELSEIF(OPER.EQ.'S') THEN
          HIST(K,L,M)=X*HIST(I,L,M)**2
      	ELSEIF(OPER.EQ.'L') THEN  
          IF(HIST(I,L,M).EQ.0..OR.J.EQ.0.) THEN
             HIST(K,L,M)=0.
           ELSE
             HIST(K,L,M)=X*LOG10(Y*HIST(I,L,M))
           ENDIF
      	ELSEIF(OPER.EQ.'M') THEN
           IF(I.NE.J) XNORM=HIST(I,L,M)
           IF(I.EQ.J) XNORM=DFLOAT(IHIS(J,L,M))
           IF(XNORM.NE.0.) THEN
             XAVG=HIST(J,L,M)/XNORM
             HIST(K,L,M)=
     &       DSQRT(ABS(-XAVG**2+HIST(K,L,M)/XNORM)/DFLOAT(IHIS(I,L,M)))
             HIST(J,L,M)=XAVG 
           ELSE 
             HIST(K,L,M)=0.
             HIST(J,L,M)=0.
           ENDIF
      	ELSEIF(OPER.EQ.'V') THEN                 
           XAVG=HIST(I,L,M)*X
           XSQAVG=HIST(J,L,M)*X
           XNORM=DFLOAT(IHIS(I,L,M))*X
           IF(XNORM.NE.0.) THEN
              HIST(K,L,M)=DSQRT(ABS(XSQAVG-XAVG**2)/XNORM)
              HIST(I,L,M)=XAVG
           ELSE  
              HIST(K,L,M)=0.
           ENDIF 
      	ELSE 
	 CALL MWARN('MOPERA')
         WRITE(*,*) OPER
   5     FORMAT(' ****** OPERATION ="',A1,'" UNKNOWN ********'/)
         RETURN
        ENDIF
        ENDDO
      	IF(OPER.EQ.'+') THEN
       	  USCORE(K,M)=X*USCORE(I,M) + Y*USCORE(J,M)  
       	  OSCORE(K,M)=X*OSCORE(I,M) + Y*OSCORE(J,M)  
      	ELSEIF(OPER.EQ.'-') THEN     
      	  USCORE(K,M)=X*USCORE(I,M) - Y*USCORE(J,M)
      	  OSCORE(K,M)=X*OSCORE(I,M) - Y*OSCORE(J,M)
      	ELSEIF(OPER.EQ.'*') THEN     
      	  USCORE(K,M)=X*USCORE(I,M) * Y*USCORE(J,M)
      	  OSCORE(K,M)=X*OSCORE(I,M) * Y*OSCORE(J,M)
      	ELSEIF(OPER.EQ.'/') THEN     
          IF(Y.EQ.0..OR.USCORE(J,M).EQ.0.) THEN
            USCORE(K,M)=0.
          ELSE
            USCORE(K,M)=X*USCORE(I,M) / (Y*USCORE(J,M))
          ENDIF
          IF(Y.EQ.0..OR.OSCORE(J,M).EQ.0.) THEN
            OSCORE(K,M)=0.
          ELSE
            OSCORE(K,M)=X*OSCORE(I,M) / (Y*OSCORE(J,M))
          ENDIF
      	ELSEIF(OPER.EQ.'I') THEN     
          IF(X.EQ.0..OR.USCORE(I,M).EQ.0.) THEN
            USCORE(K,M)=0.
          ELSE
            USCORE(K,M)=X / (USCORE(I,M))
          ENDIF
          IF(X.EQ.0..OR.OSCORE(I,M).EQ.0.) THEN
            OSCORE(K,M)=0.
          ELSE
            OSCORE(K,M)=X / (OSCORE(I,M))
          ENDIF
       	ELSEIF(OPER.EQ.'F') THEN
      	  USCORE(K,M)=X*USCORE(I,M)
      	  OSCORE(K,M)=X*OSCORE(I,M)
      	ELSEIF(OPER.EQ.'R') THEN
          IF(USCORE(I,M).GT.0.) THEN
            USCORE(K,M)=X*DSQRT(USCORE(I,M))
          ELSE                           
            USCORE(K,M)=0.
          ENDIF     
          IF(OSCORE(I,M).GT.0.) THEN
            OSCORE(K,M)=X*DSQRT(OSCORE(I,M))
          ELSE                           
            OSCORE(K,M)=0.
          ENDIF     
      	ELSEIF(OPER.EQ.'S') THEN
          USCORE(K,M)=X*USCORE(I,M)**2
          OSCORE(K,M)=X*OSCORE(I,M)**2
      	ELSEIF(OPER.EQ.'L') THEN  
          IF(USCORE(I,M).EQ.0..OR.J.EQ.0.) THEN
             USCORE(K,M)=0.
           ELSE
             USCORE(K,M)=X*LOG10(Y*USCORE(I,M))
           ENDIF                         
          IF(OSCORE(I,M).EQ.0..OR.J.EQ.0.) THEN
             OSCORE(K,M)=0.
           ELSE
             OSCORE(K,M)=X*LOG10(Y*OSCORE(I,M))
           ENDIF                         
        ENDIF
      ENDIF
      RETURN
      END

      SUBROUTINE MZERO(M,N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,N 
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
      BOOK(N,M)='RES'
      IENT(N,M)=0
      IUSCORE(N,M)=0
      IOSCORE(N,M)=0
      HAVG(N,M)=0.
      HINT(N,M)=0.
      DO 1 I=1,NBIN(N,M)
   1  HIST(N,I,M)=0.
      END

      SUBROUTINE MRESET(M,N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,N
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     START
C
      BOOK(N,M)='RES'
      END

      SUBROUTINE PUTTAG(M,J,NAME)
C*************************************************************
c Per marcare un istogramma
      IMPLICIT NONE
C
C     ARGUMENTS
C
      CHARACTER*20 NAME
      INTEGER M,J
C
C     GLOBAL
C
      INCLUDE 'dbook.inc'
C
C     LOCAL
C
      CHARACTER*20 TAG
C
C     START
C
      BOOK(J,M) = NAME
      RETURN
      ENTRY GETTAG(M,J,TAG)
      TAG = BOOK(J,M)
      END

      SUBROUTINE MFINAL(M,N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,N
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
      AVG=0
      XIN=0                                  
      SIG=0
      IF=0
      DO J=1,NBIN(N,M)
         X=HIST(N,J,M)
 	 AVG=AVG+X*XHIS(N,J,M)
         XIN=XIN+X
	 IF(X.NE.0) IF=1
      ENDDO             
      IF(XIN.EQ.0) GO TO 10
      AVG = AVG/XIN
      DO J=1,NBIN(N,M)
         SIG=HIST(N,J,M)*(XHIS(N,J,M)-AVG)**2+SIG
      ENDDO
      SIG=DSQRT(ABS(SIG/XIN))
 10   CONTINUE
      HINT(N,M) = XIN
      HAVG(N,M) = AVG
      HSIG(N,M) = SIG
      IF(IF.EQ.0) BOOK(N,M)='RES'
      END               

      SUBROUTINE MNORM(M,N,X)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,N
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
      IF(BOOK(N,M)(:1).NE.'Y')RETURN
      IF(HINT(N,M).EQ.0.) THEN
	CALL MWARN('MNORM')
	WRITE(*,*)' INTEGRAL HIST ',N,M,' IS ZERO: CANNOT RENORMALIZE'
	RETURN               
      ELSE
	Y=X/HINT(N,M)
      ENDIF
      DO 1, I=1,NBIN(N,M)
    1 HIST(N,I,M)=HIST(N,I,M)*Y
      HINT(N,M)=X            
      OSCORE(N,M)=OSCORE(N,M)*Y
      USCORE(N,M)=USCORE(N,M)*Y
      END                  

      SUBROUTINE MPRINT(M,N)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER M,N
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
      IF(BOOK(N,M)(:1).NE.'Y') RETURN
C      WRITE(98,7) N,IYEAR,IMON,IDAY,CTIME(1:5)
      WRITE(98,*) TITLE(N,M)
      DO 1 J=1,NBIN(N,M)
      IF(HIST(N,J,M).GT.-1d-99.and.HIST(N,J,M).LT.1d-99) GO TO 1
      WRITE(98,'(3X,F10.4,2X,E15.4)')  
     &                            XHIS(N,J,M),HIST(N,J,M)
    1 CONTINUE
      WRITE(98,15) HAVG(N,M),HSIG(N,M),HINT(N,M)
      WRITE(98,20) IENT(N,M),IUSCORE(N,M),IOSCORE(N,M)
C    7 FORMAT(4X,'HIST = ',I3,'   19',I2,'-',I2,'-',I2,1X,A5/)
   10 FORMAT(4X,2E10.3)
   15 FORMAT(/' AVG =',E10.3,4X,' RMS =',E10.3,' INTEGRAL =',E10.3,/)
   20 FORMAT(' ENTRIES=',I5,4X,'UNDERSCORE=',I5,4x,'OVERSCORE=',I5,//)
      END


      SUBROUTINE MTOP(V,N,M,BTIT,LTIT,SCALE)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER V,N,M
      CHARACTER*20 LTIT,BTIT
      CHARACTER*3 SCALE
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
      IF(BOOK(N,V)(:1).NE.'Y') RETURN
      WRITE(99,100) TITLE(N,V),BTIT,LTIT,SCALE,HMIN(N,V),HMAX(N,V)
  100 FORMAT( /1x,                               
     &' SET WINDOW Y 2.5 TO 9.'/,1X,
     &' SET WINDOW X 2.5 TO 10.'/,1X,
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
      DO J=1,NBIN(N,V)
         maxbin=max(maxbin,abs(HIST(N,J,V)))
      ENDDO
      DO 1 J=1,NBIN(N,V)
c         IF(abs(HIST(N,J)).le.maxbin*10d-7) GO TO 1
      IF(HIST(N,J,V).lt.1d-99.and.HIST(N,J,V).gt.-1d-99) THEN
      WRITE(99,'(3X,F10.4,2(2X,E15.4))')
     &                            XHIS(N,J,V), 0d0, HIST(M,J,V)
      ELSE
      WRITE(99,'(3X,F10.4,2(2X,E15.4))')  
     &                            XHIS(N,J,V),HIST(N,J,V),HIST(M,J,V)
      ENDIF
    1 CONTINUE
      WRITE(99,200)
  200 FORMAT('   HISTO')
      WRITE(99,300) HINT(N,V),HAVG(N,V),HSIG(N,V),IENT(N,V),IUSCORE(N,V)
     &   ,IOSCORE(N,V)
  300 FORMAT( /1x,                               
     &' BOX 6.25 1.0 SIZE 7.5 0.8'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 2.8 1.2 "INT =',1PE10.3,'   AVG =',1PE10.3,
     &             '   RMS =',1PE10.3,'"',/1X,
     &' TITLE 2.8 0.9 "Entries =',I8,2x,'Undersc =',I6,2X
     &                                 ,'Oversc =',I6,'"',/1X,
     &' SET TITLE SIZE -2')                            
      WRITE(99,400)
  400 FORMAT('   NEW PLOT')
      END



      SUBROUTINE MULTITOP(V,NH,NE,N,M,BTIT,LTIT,SCA)
C*************************************************************
      IMPLICIT NONE
C
C     ARGUMENTS
C
      INTEGER NH,NE,N,M,V
      CHARACTER*20 LTIT,BTIT
      CHARACTER*3 SCA
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
      DATA TIT0,LAB0,TIC0/-3,-3,0.06/
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
      IF(BOOK(NH,V)(:1).NE.'Y') RETURN
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
      FMN=MAX(HINT(NH,V)*NBIN(NH,V),1.E12)
      FMX=-FMN                        
      XMX=0.
      DO IBIN=1,NBIN(NH,V)
	X=HIST(NH,IBIN,V)
      	IF(X.NE.0.) FMX=MAX(FMX,X)
      	IF(X.NE.0.) FMN=MIN(FMN,X)
	XMX=MAX(XMX,ABS(X))
      ENDDO                
      IF(XMX.EQ.0.) GO TO 203
      SCALE=SCA
50    IF(SCALE.EQ.'LIN') THEN
	IF(FMN.GE.0.)	FMIN=0.
	IF(FMN.LT.0.)	FMIN=FMN*1.3
	IF(FMX.GT.0.)	FMAX=FMX*1.3
	IF(FMX.LT.0.)	FMAX=0.
      ELSEIF(SCALE.EQ.'LOG') THEN 
        IF(FMN.LE.0.) THEN
	     	SCALE='LIN'
		GO TO 50
	ENDIF
	FMIN=LOG10(FMN)
	FMAX=LOG10(FMX)
       	FEXP=AINT(FMAX+(FMAX-FMIN)*0.2+1)
	FMIN=10.**AINT(LOG10(FMN))
	IF(FMIN.GT.FMX) FMIN=FMIN/10.
	FMAX=10.**(FEXP)                 
      ENDIF                         
      WRITE(99,100) TITS,LABS,TICS,XL,XU,YL,YU
100   FORMAT(2X,'SET FONT DUPLEX',/,                           
     *       2X,'SET TITLE SIZE ',F8.4,/,
     *       2X,'SET LABEL SIZE ',F8.4,/,
     *       2X,'SET TICKS TOP OFF SIZE ',F8.4,/,
     *       2X,'SET WINDOW X ',F8.4,' TO ',F8.4,/,
     *       2X,'SET WINDOW Y ',F8.4,' TO ',F8.4)
      XTIT=XL+XTIT0
      YTIT=YU+YTIT0
      WRITE(99,101) XL,YTIT,TITLE(NH,V)(1:20)
101   FORMAT('  TITLE ',2(F8.4,1X),'"',A,'"')                  
      YTIT=YTIT-2.*YTIT0
      WRITE(99,102) XTIT,YTIT,HINT(NH,V)
102   FORMAT('  TITLE ',2(F8.4,1X),'" INT=',1PE10.3,'"')                  
      YTIT=YTIT-YTIT0
      WRITE(99,103) XTIT,YTIT,IENT(NH,V)
103   FORMAT('  TITLE ',2(F8.4,1X),'" ENT=',I9,'"')                  
      YTIT=YTIT-YTIT0                         
      IF(USCORE(NH,V).NE.0.) THEN
        WRITE(99,104) XTIT,YTIT,USCORE(NH,V)
104     FORMAT('  TITLE ',2(F8.4,1X),'" UFL=',1PE10.3,'"')                  
        YTIT=YTIT-YTIT0                      
      ENDIF
      IF(OSCORE(NH,V).NE.0.) THEN
        WRITE(99,105) XTIT,YTIT,OSCORE(NH,V)
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
     * HMIN(NH,V),HMAX(NH,V),FMIN,FMAX
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
      ENTRY MTFILL(V,NH,NE,N,M,BTIT,LTIT,SCA)
      IP=IP+1                             
      IF(IP.GT.4) IP=1
      WRITE(99,110) TITLE(NH,V),HINT(NH,V),IENT(NH,V)
110   FORMAT(' ( ',A,/,' ( INT=',1PE10.3,'  ENTRIES=',I12)
      DO 200 IBIN=1,NBIN(NH,V)           
      IF(HIST(NH,IBIN,V).EQ.0..AND.HIST(NE,IBIN,V).EQ.0.) GO TO 200
      WRITE(99,'(3X,F10.4,2(2X,E15.4))')  
     &          XHIS(NH,IBIN,V),HIST(NH,IBIN,V),HIST(NE,IBIN,V)
200   CONTINUE                                           
      WRITE(99,201)  PLOT(IP)
      IF(BOOK(NE,V).NE.'NO')   WRITE(99,*)  '  PLOT'
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
