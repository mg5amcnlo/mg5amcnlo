C####
C#### YOU WILL NOT NEED WHAT FOLLOWS IN YOUR OWN ANALYSIS
C####
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
C**********************************************************************
C
C Fills up to 100 histograms with up to 100 bins. 
C Gives a data file (to be specified in the calling program by assigning 
C a file name to unit 98) and a topdrawer file (to be specified in the 
C calling program by assigning a file name to unit 99).
C
C INITIALIZATION:
C Call once INIHIST; this just resets a few counters and logicals
C Call MBOOK(N,'TITLE',DEL,XMIN,XMAX) for each histogram to be booked.
C N (an integer) is the label of the histogram;
C 'TITLE' is the name of the histogram (no more then 100 characters);
C DEL (real*4) is the bin size;
C XMIN (real*4) is the lower limit of the first bin;
C XMAX (real*4)is the upper limit of the last  bin
C Example:
C      call mbook(2,'pt distribution',1.,10,70)
C This call initializes histogram number 2, called 'pt distribution';
C The bin size will be 1. (possibly GeV, if that's what you want), the
C first bin being  10<x<11. and the last one being 69.<x<70
C
C FILLING:
C When it's time, call MFILL(N,X,Y); this will add Y (real*4) to the bin 
C in which X (real*4) happens to be, within histogram N. 
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
C		  Booked, 'RES' if it was resetted, 'YES' otherwise
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
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE*50,BOOK*3
      CHARACTER TITLE2*50,BOOK2*3
      DO 1, I=1,NMB             
   1  BOOK(I)=' NO'
      DO 2 I=1,50
   2  BOOK2(I)=' NO'
      END  
    
      SUBROUTINE MBOOK(N,TIT,DEL,XMIN,XMAX)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST
      CHARACTER TITLE*50,BOOK*3
      CHARACTER*(*) TIT
      NHIST = MAX(N,NHIST)
      IF(BOOK(N)(1:1).EQ.'Y') THEN
	 CALL MBWARN('MBOOK')
         WRITE(*,*) 'Histogram',N,TITLE(N),' already in use. '
         WRITE(*,*) 'superseded by ',TIT
      ENDIF
      BOOK(N) = 'YES'
      TITLE(N) = ' '//TIT
1     HDEL(N) = DEL
      NBIN(N) = INT((XMAX-XMIN)/(DEL*0.9999))
      IF(NBIN(N).GT.100) THEN
	WRITE(*,*) 'TOO MANY BINS (',NBIN(N),') REQUIRED IN HIST ',N
	WRITE(*,*) 'RE-ENTER BIN SIZE DELTA (OLD BIN = ',DEL,' ):'
	READ(*,*) DEL
	GO TO 1
      ENDIF
      HMIN(N) = XMIN
      HMAX(N) = NBIN(N)*DEL+XMIN
      IF(ABS(HMAX(N)-XMAX).GT.1.D-3*DEL) THEN
	 CALL MBWARN('MBOOK')
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
         XHIS(N,I)=HMIN(N)+HDEL(N)*(FLOAT(I)-0.5)
         IHIS(N,I)=0
         HIST(N,I)=0
      ENDDO
      END

      SUBROUTINE MFILL(N,X,Y)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      REAL*8 SUM
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
C SF, 10/3/2010 -- avoids numerical problems
         SUM=HIST(N,I)+Y
         IF(ABS(SUM).LT.1.d-19)THEN
           HIST(N,I)=0.e0
         ELSE
           HIST(N,I)=SNGL(SUM)
         ENDIF
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
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      CHARACTER*14  C
      DIMENSION C(2) 
      DATA C/' INTEG BELOW X',' INTEG ABOVE X'/
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
         CALL MBWARN('MINTEG')
         WRITE(*,*) 'Wrong idir in minteg: OPERATION NOT PERFORMED'
         RETURN
      ENDIF
      END

      FUNCTION SUMPOW(X,Y,IPOW)
      IF(IPOW.EQ.1) THEN
         SUMPOW = X + Y
      ELSEIF(IPOW.EQ.2) THEN
         SUMPOW = SQRT(X**2+Y**2)
      ELSEIF(IPOW.EQ.0) THEN
         CALL MBWARN('SUMPOW')
         WRITE(*,*)'Error: IPOW=0 not allowed in SUMPOW'
      ELSE
         SUMPOW = (X**IPOW+Y**IPOW)**(1./IPOW)
      ENDIF
      END

      SUBROUTINE MOPERA(I,OPER,J,K,X,Y)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      CHARACTER OPER*1
      IF(NBIN(I).NE.NBIN(J).AND.(OPER.EQ.'+'.OR.OPER.EQ.'-'.OR.OPER.EQ.
     &    '*'.OR.OPER.EQ.'/'.OR.OPER.EQ.'M'.OR.OPER.EQ.'A')) THEN
	  CALL MBWARN('MOPERA')
          WRITE(*,*) I,J                               
  20      FORMAT(' ****** INCOMPATIBLE OPERATION HIST ',I2,' &',I2,
     &    '*******'/)
          RETURN
      ENDIF
      IF(BOOK(I)(1:1).NE.'Y'.OR.BOOK(J)(1:1).NE.'Y') RETURN
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
            HIST(J,L)=XXX*SQRT(ABS(XSUMSQ-XSUM**2*XXX))
         ENDDO
         IENT(K)=IENT(K)+IENT(I)
         XSUM = USCORE(I)
         XSUMSQ = USCORE(J)
         USCORE(K) = USCORE(K)+XXX*XSUM
         USCORE(J) = XXX*SQRT(ABS(XSUMSQ-XSUM**2*XXX))
         XSUM = OSCORE(I)
         XSUMSQ = OSCORE(J)
         OSCORE(K) = OSCORE(K)+XXX*XSUM
         OSCORE(J) = XXX*SQRT(ABS(XSUMSQ-XSUM**2*XXX))
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
            IENT(K) = IENT(K)+1
            HIST(I,L) = 0
            IHIS(I,L) = 0
         ENDDO
         IENT(J) = IENT(J)+IENT(I)
         IUSCORE(J) = IUSCORE(J) + IUSCORE(I)
         USCORE(J) = USCORE(J) + USCORE(I)
         IOSCORE(J) = IOSCORE(J) + IOSCORE(I)
         OSCORE(J) = OSCORE(J) + OSCORE(I)
         IUSCORE(K) = IUSCORE(K) + 1
         USCORE(K) = USCORE(K) + USCORE(I)**2
         IOSCORE(K) = IOSCORE(K) + 1
         OSCORE(K) = OSCORE(K) + OSCORE(I)**2
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
            HIST(K,L)=X*SQRT(HIST(I,L))
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
           IF(I.EQ.J) XNORM=FLOAT(IHIS(J,L))
           IF(XNORM.NE.0.) THEN
             XAVG=HIST(J,L)/XNORM
             HIST(K,L)=
     &       SQRT(ABS(-XAVG**2+HIST(K,L)/XNORM)/FLOAT(IHIS(I,L)))
             HIST(J,L)=XAVG 
           ELSE 
             HIST(K,L)=0.
             HIST(J,L)=0.
           ENDIF
      	ELSEIF(OPER.EQ.'V') THEN                 
           XAVG=HIST(I,L)*X
           XSQAVG=HIST(J,L)*X
           XNORM=FLOAT(IHIS(I,L))*X
           IF(XNORM.NE.0.) THEN
              HIST(K,L)=SQRT(ABS(XSQAVG-XAVG**2)/XNORM)
              HIST(I,L)=XAVG
           ELSE  
              HIST(K,L)=0.
           ENDIF 
      	ELSE 
	 CALL MBWARN('MOPERA')
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
            USCORE(K)=X*SQRT(USCORE(I))
          ELSE                           
            USCORE(K)=0.
          ENDIF     
          IF(OSCORE(I).GT.0.) THEN
            OSCORE(K)=X*SQRT(OSCORE(I))
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

      SUBROUTINE MCOPY(NIN,NOUT)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
C
C If the histogram is not booked or resetted, do not copy it
      IF(BOOK(NIN)(1:1).NE.'Y'.AND.BOOK(NIN)(1:1).NE.'R') RETURN
C If the histogram was resetted without having being booked, do not copy it.
C This may happen if MFINAL is called prior to MCOPY
      IF(BOOK(NIN)(1:1).EQ.'R'.AND.HMIN(NIN).EQ.HMAX(NIN)) RETURN
      IF(BOOK(NOUT)(1:1).NE.'Y'.AND.BOOK(NOUT)(1:1).NE.'R') 
     &  CALL MBOOK(NOUT,TITLE(NIN)(2:),HDEL(NIN),HMIN(NIN),HMAX(NIN))
      IF(BOOK(NIN).EQ.'RES')BOOK(NOUT)='RES'
      HDEL(NOUT)=HDEL(NIN)
      HMIN(NOUT)=HMIN(NIN)
      HMAX(NOUT)=HMAX(NIN)
      NBIN(NOUT)=NBIN(NIN)
      IENT(NOUT)=IENT(NIN)
      IUSCORE(NOUT)=IUSCORE(NIN)
      IOSCORE(NOUT)=IOSCORE(NIN)
      USCORE(NOUT)=USCORE(NIN)
      OSCORE(NOUT)=OSCORE(NIN)
      HAVG(NOUT)=HAVG(NIN)
      HINT(NOUT)=HINT(NIN)
      HSIG(NOUT)=HSIG(NIN)
      DO I=1,NBIN(NOUT)
        HIST(NOUT,I)=HIST(NIN,I)
        XHIS(NOUT,I)=XHIS(NIN,I)
        IHIS(NOUT,I)=IHIS(NIN,I)
      ENDDO
      END

      SUBROUTINE MZERO(N)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      BOOK(N)='RES'
      IENT(N)=0
      IUSCORE(N)=0
      IOSCORE(N)=0
      HAVG(N)=0.
      HINT(N)=0.
      DO 1 I=1,NBIN(N)
   1  HIST(N,I)=0.
      END                   

      SUBROUTINE MDUMP(N)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      DIMENSION HISTL(100),XHISL(100),IHISL(100)
      CHARACTER TITLE*50,BOOK*3,TITLEL*50,BOOKL*3
      SAVE HISTL,XHISL,HDELL,HMINL,HMAXL,USCOREL,OSCOREL,
     &NBINL,IHISL,IUSCOREL,IOSCOREL,IENTL,HAVGL,HINTL,HSIGL,
     &BOOKL,TITLEL
      BOOKL=BOOK(N)                                                           
      TITLEL=TITLE(N)                   
      HDELL=HDEL(N)  
      HMINL=HMIN(N)
      HMAXL=HMAX(N)
      USCOREL=USCORE(N)
      OSCOREL=OSCORE(N)
      NBINL=NBIN(N)
      HSIGL=HSIG(N)
      IENTL=IENT(N)     
      IUSCOREL=IUSCORE(N)
      IOSCOREL=IOSCORE(N)      
      HAVGL=HAVG(N)         
      HINTL=HINT(N)   
      DO 1 I=1,NBIN(N)
      XHISL(I)=XHIS(N,I)
      IHISL(I)=IHIS(N,I)
   1  HISTL(I)=HIST(N,I)
      RETURN
      ENTRY MFETCH(N)
      BOOK(N)=BOOKL                   
      TITLE(N)=TITLEL                   
      HDEL(N)=HDELL  
      HMIN(N)=HMINL
      HMAX(N)=HMAXL
      USCORE(N)=USCOREL
      OSCORE(N)=OSCOREL
      NBIN(N)=NBINL
      HSIG(N)=HSIGL
      IENT(N)=IENTL     
      IUSCORE(N)=IUSCOREL
      IOSCORE(N)=IOSCOREL      
      HAVG(N)=HAVGL         
      HINT(N)=HINTL   
      DO 2 I=1,NBINL
      XHIS(N,I)=XHISL(I)
      IHIS(N,I)=IHISL(I)
   2  HIST(N,I)=HISTL(I)
      END               

      SUBROUTINE MRESET(N)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      BOOK(N)='RES'
      END

      SUBROUTINE PUTTAG(J,NAME)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
c Per marcare un istogramma
      CHARACTER * (*) NAME, TAG
      BOOK(J) = NAME
      RETURN
      ENTRY GETTAG(J,TAG)
      TAG = BOOK(J)
      END

      SUBROUTINE MFINAL(N)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
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
      SIG=SQRT(ABS(SIG/XIN))
 10   CONTINUE
      HINT(N) = XIN
      HAVG(N) = AVG
      HSIG(N) = SIG
      IF(IF.EQ.0) BOOK(N)='RES'
      END               

      SUBROUTINE MFINAL3(N)
C Identical to MFINAL, except for the fact that if the histogram is void
C but was previously booked, BOOK is left as it is and not set to RES
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3
      IF(BOOK(N)(1:1).NE.'Y') RETURN
      AVG=0
      XIN=0                                  
      SIG=0
      IF=0
      IF(HMIN(N).NE.HMAX(N))IF=1
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
      SIG=SQRT(ABS(SIG/XIN))
 10   CONTINUE
      HINT(N) = XIN
      HAVG(N) = AVG
      HSIG(N) = SIG
      IF(IF.EQ.0) BOOK(N)='RES'
      END               

      SUBROUTINE MNORM(N,X)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST
      CHARACTER TITLE*50,BOOK*3,CTIME*10,CDATE*10
      IF(BOOK(N).NE.'YES')RETURN
      IF(HINT(N).EQ.0.) THEN
c        CALL MBKWRN('MNORM')
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
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3,CTIME*10
      DATA INI/0/
      IF(INI.EQ.0) THEN
      CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      INI=1
      ENDIF
      IF(BOOK(N)(:1).NE.'Y') RETURN
      WRITE(98,7) N,IYEAR,IMON,IDAY,CTIME(1:5)
      WRITE(98,*) TITLE(N)
      DO 1 J=1,NBIN(N)
      IF(HIST(N,J).EQ.0.) GO TO 1
      WRITE(98,'(3X,F10.4,2X,E15.4)')  
     &                            XHIS(N,J),HIST(N,J)
    1 CONTINUE
      WRITE(98,15) HAVG(N),HSIG(N),HINT(N)
      WRITE(98,20) IENT(N),IUSCORE(N),IOSCORE(N)
    7 FORMAT(4X,'HIST = ',I3,'   19',I2,'-',I2,'-',I2,1X,A5/)
   10 FORMAT(4X,2E10.3)
   15 FORMAT(/' AVG =',E10.3,4X,' RMS =',E10.3,' INTEGRAL =',E10.3,/)
   20 FORMAT(' ENTRIES=',I5,4X,'UNDERSCORE=',I5,4x,'OVERSCORE=',I5,//)
      END

      SUBROUTINE MTOP(N,M,BTIT,LTIT,SCALE)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3,CTIME*10
      CHARACTER*(*) LTIT,BTIT,SCALE
      DATA INI/0/
      IF(INI.EQ.0) THEN
      CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      INI=1
      ENDIF
      IF(BOOK(N)(:1).NE.'Y') RETURN
      WRITE(99,100) TITLE(N),BTIT,LTIT,SCALE,HMIN(N),HMAX(N)
  100 FORMAT( /1x,                               
     &' SET WINDOW Y 2.5 TO 9.'/,1X,
     &' SET WINDOW X 2.5 TO 10.'/,1X,
     &' SET FONT DUPLEX '/1X, 
     &' SET SYMBOL 5O SIZE 1.8'/,1X,
     &' TITLE TOP ','"',A,'"',/1X,
     &' TITLE BOTTOM ','"',A,'"',/1X,
     &' TITLE LEFT ','"',A,'"',/1X,
     &' SET SCALE Y ',A,/1X,
     &' (SET TICKS TOP OFF)   '/1x,     
     &' SET LIMITS X ',F10.5,' ',F10.5,/1X,
     &' SET ORDER X Y DY ')
      DO 1 J=1,NBIN(N)
      WRITE(99,'(3X,F10.4,2(2X,E15.4))')  
     &                            XHIS(N,J),HIST(N,J),HIST(M,J)
    1 CONTINUE
      WRITE(99,200)
  200 FORMAT('   PLOT')
      WRITE(99,300) HINT(N),HAVG(N),HSIG(N),IENT(N),IUSCORE(N)
     &   ,IOSCORE(N),USCORE(N),OSCORE(N),IMON,IDAY,IYEAR,CTIME(1:5)
  300 FORMAT( /1x,                               
     &' BOX 6.25 0.9 SIZE 7.5 1.2'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 2.8 1.2 "INT =',1PE10.3,'   AVG =',1PE10.3,
     &             '   RMS =',1PE10.3,'"',/1X,
     &' TITLE 2.8 0.9 "Entries =',I8,2x,'Undersc =',I6,2X
     &                                 ,'Oversc =',I6,'"',/1X,
     &' TITLE 2.8 0.6 "Intgr ufloat=',1PE10.3,'  Intgr ofloat=',
     &      1PE10.3,'"',/1X,
     &' TITLE 7.5 0.6 "',I2,'-',I2,', 19',I2,2X,A5,'"',/1X,                
     &' SET TITLE SIZE -2')                            
      WRITE(99,400)
  400 FORMAT('   NEW PLOT')
      END
 
      SUBROUTINE MNRTOP(N,M,BTIT,LTIT,SCALE)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      CHARACTER TITLE*50,BOOK*3,CTIME*10
      CHARACTER*(*) LTIT,BTIT,SCALE
      DATA INI/0/
      IF(INI.EQ.0) THEN
         CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
         INI=1
      ENDIF
      CALL MFINAL(N)
      WRITE(99,100) TITLE(N)(1:20),BTIT,LTIT,SCALE,HMIN(N),HMAX(N)
  100 FORMAT( /1x,                               
     &' SET WINDOW Y 2.5 TO 7.'/,1X,
     &' SET WINDOW X 2.5 TO 10.'/,1X,
     &' SET FONT DUPLEX '/1X, 
     &' SET SYMBOL 5O SIZE 1.8'/,1X,
     &' TITLE TOP ','"',A,'"',/1X,
     &' TITLE BOTTOM ','"',A,'"',/1X,
     &' TITLE LEFT ','"',A,'"',/1X,
     &' SET SCALE Y ',A,/1X,
     &' (SET TICKS TOP OFF)   '/1x,     
     &' SET LIMITS X ',F10.5,' ',F10.5,/1X,
     &' SET ORDER X Y DY ')
      DO 1 J=1,NBIN(N)
      IF(HIST(N,J).EQ.0) GO TO 1
      WRITE(99,'(3X,F10.4,2(2X,E10.3))')  
     &                            XHIS(N,J),HIST(N,J),HIST(M,J)
    1 CONTINUE
      WRITE(99,200)
  200 FORMAT('   PLOT')
      WRITE(99,300) HINT(N),HAVG(N),HSIG(N),IENT(N),IUSCORE(N)
     &   ,IOSCORE(N),USCORE(N),OSCORE(N),IMON,IDAY,IYEAR,CTIME(1:5)
  300 FORMAT( /1x,                               
     &' BOX 6.25 0.9 SIZE 7.5 1.2'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 2.8 1.2 "INT =',1PE10.3,'   AVG =',1PE10.3,
     &             '   RMS =',1PE10.3,'"',/1X,
     &' TITLE 2.8 0.9 "Entries =',I8,2x,'Undersc =',I6,2X
     &                                 ,'Oversc =',I6,'"',/1X,
     &' TITLE 2.8 0.6 "Intgr ufloat=',1PE10.3,'  Intgr ofloat=',
     &      1PE10.3,'"',/1X,
     &' TITLE 7.5 0.6 "',I2,'-',I2,', 19',I2,2X,A5,'"',/1X,                
     &' SET TITLE SIZE -2')
      WRITE(99,400)
  400 FORMAT('   NEW PLOT')
      END


      SUBROUTINE MBOOK2(N,TIT,DELX,XMIN,XMAX,DELY,YMIN,YMAX)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
      CHARACTER*(*) TIT
      NHIST2=MAX(N,NHIST2)
      IF(BOOK2(N)(1:1).EQ.'Y') THEN
	 CALL MBWARN('MBOOK2')
         WRITE(*,*) 'Histogram',N,TITLE2(N),' already in use. '
         WRITE(*,*) 'superseded by ',TIT
      ENDIF
      TITLE2(N)='   '//TIT                     
      BOOK2(N)='YES'
C-- setup x-boundaries
      HDEL2(1,N)=DELX
      HMIN2(1,N)=XMIN  
      HMAX2(1,N)=XMAX
      NBIN2(1,N)=INT((XMAX-XMIN)/(DELX*0.9999))
C-- setup y-boundaries
      HDEL2(2,N)=DELY
      HMIN2(2,N)=YMIN  
      HMAX2(2,N)=YMAX
      NBIN2(2,N)=INT((YMAX-YMIN)/(DELY*0.9999))
      IENT2(N)=0                      
      IOSCORE2(N)=0
      HAVG2(1,N)=0.
      HAVG2(2,N)=0.
      HINT2(N)=0.
      DO 1 I=1,NBIN2(1,N)
      XHIS2(N,I)=HMIN2(1,N)+HDEL2(1,N)*(FLOAT(I)-0.5)
   1  CONTINUE
      DO 2 I=1,NBIN2(2,N)
      YHIS2(N,I)=HMIN2(2,N)+HDEL2(2,N)*(FLOAT(I)-0.5)
   2  CONTINUE
      DO 3 I=1,NBIN2(1,N)
      DO 3 J=1,NBIN2(2,N)
      HIST2(N,I,J)=0.                   
   3  CONTINUE
      END        

      SUBROUTINE MFILL2(N,X,Y,WGT)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
C Modified by SF on 10/8/2005; now similar to mfill()
      IF( X.LT.HMIN2(1,N) .OR. X.GE.HMAX2(1,N) .OR.
     #    Y.LT.HMIN2(2,N) .OR. Y.GE.HMAX2(2,N) )THEN
         IOSCORE2(N)=IOSCORE2(N)+1 
      ELSE
         XI=((X-HMIN2(1,N))/HDEL2(1,N))+1.
         YI=((Y-HMIN2(2,N))/HDEL2(2,N))+1.
         I=INT(XI)
         J=INT(YI)
         IF( I.GT.0.AND.I.LE.NBIN2(1,N) .AND.
     #       J.GT.0.AND.J.LE.NBIN2(2,N) )THEN
           IENT2(N)=IENT2(N)+1
           IHIS2(N,I,J)=IHIS2(N,I,J)+1
           HIST2(N,I,J)=HIST2(N,I,J)+WGT
         ELSE
           IOSCORE2(N)=IOSCORE2(N)+1 
         ENDIF
      ENDIF
      END


      SUBROUTINE MOPERA2(I,OPER,J,K,X,Y)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
      CHARACTER OPER*1
      IF(NBIN2(1,I).NE.NBIN2(1,J).OR.NBIN2(2,I).NE.NBIN2(2,J).
     &AND.(OPER.EQ.'+'.OR.OPER.EQ.'-'.OR.OPER.EQ.         
     &'*'.OR.OPER.EQ.'/'.OR.OPER.EQ.'M')) GO TO 10
      DO L1=1,NBIN2(1,I)
      DO L2=1,NBIN2(2,I)
      IF(OPER.EQ.'+') THEN
      HIST2(K,L1,L2)=X*HIST2(I,L1,L2) + Y*HIST2(J,L1,L2)
      ELSEIF(OPER.EQ.'-') THEN
      HIST2(K,L1,L2)=X*HIST2(I,L1,L2) - Y*HIST2(J,L1,L2)
      ELSEIF(OPER.EQ.'*') THEN
      HIST2(K,L1,L2)=X*HIST2(I,L1,L2) * Y*HIST2(J,L1,L2)
      ELSEIF(OPER.EQ.'/') THEN
        IF(Y.EQ.0..OR.HIST2(J,L1,L2).EQ.0.) THEN
          HIST2(K,L1,L2)=0.
          ELSE
          HIST2(K,L1,L2)=X*HIST2(I,L1,L2) / (Y*HIST2(J,L1,L2))
        ENDIF
      ELSEIF(OPER.EQ.'F') THEN
      HIST2(K,L1,L2)=X*HIST2(I,L1,L2)
      ELSEIF(OPER.EQ.'R') THEN
        IF(HIST2(I,L1,L2).GT.0.) THEN
        HIST2(K,L1,L2)=X*SQRT(HIST2(I,L1,L2))
        ELSE
        HIST2(K,L1,L2)=0.
        ENDIF
      ELSEIF(OPER.EQ.'S') THEN
      HIST2(K,L1,L2)=X*HIST2(I,L1,L2)**2
      ELSEIF(OPER.EQ.'L') THEN
        IF(HIST2(I,L1,L2).EQ.0..OR.J.EQ.0.) THEN
             HIST2(K,L1,L2)=0.
             ELSE
             HIST2(K,L1,L2)=X*LOG10(Y*HIST2(I,L1,L2))
        ENDIF
      ELSE
      WRITE(98,5) OPER
   5  FORMAT(' ****** OPERATION ="',A1,'" UNKNOWN ********'/)
      RETURN
      ENDIF
      END DO
      ENDDO
      RETURN
  10  WRITE(98,20) I,J
  20  FORMAT(' ****** INCOMPATIBLE OPERATION HIST2 ',I2,' &',I2,
     &                                                   '*******'/)
      END
     

      SUBROUTINE MCOPY2(NIN,NOUT)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
C
C If the histogram is not booked or resetted, do not copy it
      IF(BOOK2(NIN)(1:1).NE.'Y'.AND.BOOK2(NIN)(1:1).NE.'R') RETURN
C If the histogram was resetted without having being booked, do not copy it.
C This may happen if MFINAL2 is called prior to MCOPY2
      IF(BOOK2(NIN)(1:1).EQ.'R'.AND.
     &   HMIN2(1,NIN).EQ.HMAX2(1,NIN)) RETURN
      IF(BOOK2(NOUT)(1:1).NE.'Y'.AND.BOOK2(NOUT)(1:1).NE.'R') 
     &  CALL MBOOK2(NOUT,TITLE2(NIN),
     &              HDEL2(1,NIN),HMIN2(1,NIN),HMAX2(1,NIN),
     &              HDEL2(2,NIN),HMIN2(2,NIN),HMAX2(2,NIN))
      IF(BOOK2(NIN).EQ.'RES')BOOK2(NOUT)='RES'
      HDEL2(1,NOUT)=HDEL2(1,NIN)
      HMIN2(1,NOUT)=HMIN2(1,NIN)
      HMAX2(1,NOUT)=HMAX2(1,NIN)
      NBIN2(1,NOUT)=NBIN2(1,NIN)
      HDEL2(2,NOUT)=HDEL2(2,NIN)
      HMIN2(2,NOUT)=HMIN2(2,NIN)
      HMAX2(2,NOUT)=HMAX2(2,NIN)
      NBIN2(2,NOUT)=NBIN2(2,NIN)
      IENT2(NOUT)=IENT2(NIN)
      IOSCORE2(NOUT)=IOSCORE2(NIN)
      HAVG2(1,NOUT)=HAVG2(1,NIN)
      HAVG2(2,NOUT)=HAVG2(2,NIN)
      HINT2(NOUT)=HINT2(NIN)
      HSIG2(1,NOUT)=HSIG2(1,NIN)
      HSIG2(2,NOUT)=HSIG2(2,NIN)
      DO I=1,NBIN2(1,NOUT)
        XHIS2(NOUT,I)=XHIS2(NIN,I)
        XPROJ(NOUT,I)=XPROJ(NIN,I)
      ENDDO
      DO I=1,NBIN2(2,NOUT)
        YHIS2(NOUT,I)=YHIS2(NIN,I)
        YPROJ(NOUT,I)=YPROJ(NIN,I)
      ENDDO
      DO I=1,NBIN2(1,NOUT)
        DO J=1,NBIN2(2,NOUT)
          HIST2(NOUT,I,J)=HIST2(NIN,I,J)
          IHIS2(NOUT,I,J)=IHIS2(NIN,I,J)
        ENDDO
      ENDDO
      END


      SUBROUTINE MFINAL2(N)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
      IF(BOOK2(N)(:1).NE.'Y') RETURN
      XIN=0.                                  
C-- projection on the x-axis
      DO 2 I=1,NBIN2(1,N)
        DO 1 J=1,NBIN2(2,N)
   1    XPROJ(N,I)=XPROJ(N,I)+HIST2(N,I,J)
   2  XIN=XIN+XPROJ(N,I)
      IF(XIN.EQ.0.) GO TO 10
C-- projection on the y-axis
      DO 3 J=1,NBIN2(2,N)
        DO 3 I=1,NBIN2(1,N)
   3    YPROJ(N,J)=YPROJ(N,J)+HIST2(N,I,J)
      HINT2(N)=XIN
      RETURN
  10  BOOK2(N)=' NO'
      END               

      SUBROUTINE MFINAL32(N)
C Identical to MFINAL3, except for the fact that if the histogram is void
C but was previously booked, BOOK is left as it is and not set to RES, 
C and for the fact that XPROJ and YPROJ are initialized to zero, which 
C might prevent errors in the computation of the integral in the case that 
C the histograms are plotted several times during a single run (via MCOPY2)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
      IF(BOOK2(N)(1:1).NE.'Y') RETURN
      XIN=0.                                  
      DO I=1,NBIN2(1,N)
        XPROJ(N,I)=0.
      ENDDO
      DO J=1,NBIN2(2,N)
        YPROJ(N,J)=0.
      ENDDO
C-- projection on the x-axis
      DO 2 I=1,NBIN2(1,N)
        DO 1 J=1,NBIN2(2,N)
   1    XPROJ(N,I)=XPROJ(N,I)+HIST2(N,I,J)
   2  XIN=XIN+XPROJ(N,I)
      IF(XIN.EQ.0.) GO TO 10
C-- projection on the y-axis
      DO 3 J=1,NBIN2(2,N)
        DO 3 I=1,NBIN2(1,N)
   3    YPROJ(N,J)=YPROJ(N,J)+HIST2(N,I,J)
      HINT2(N)=XIN
      RETURN
  10  BOOK2(N)=' NO'
      END               

      SUBROUTINE PROHIS(N,M,IAX)
C-- projects the scatter plot N onto the IAX axis (x,y->IAX=1,2) and
C   put the contents in histogram M, after automatically booking it.
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST                                                      
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE*50,BOOK*3
      CHARACTER TITLE2*50,BOOK2*3
      BOOK(M)='YES'      
      NBIN(M)=NBIN2(IAX,N)
      HDEL(M)=HDEL2(IAX,N)
      HMIN(M)=HMIN2(IAX,N)
      HMAX(M)=HMAX2(IAX,N)
      NHIST=MAX(NHIST,M)
      TITLE(M)=TITLE2(N)//'(PROJ)'
      DO I=1,NBIN(M)
      IF(IAX.EQ.1)      THEN
      HIST(M,I)=XPROJ(N,I)
      XHIS(M,I)=XHIS2(N,I)
      ELSEIF(IAX.EQ.2)      THEN
      HIST(M,I)=YPROJ(N,I)
      XHIS(M,I)=YHIS2(N,I)
      ENDIF
      ENDDO
      END                             


      SUBROUTINE MNORM2(N,X)    
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3
      IF(BOOK2(N)(:1).NE.'Y')RETURN
      DO 1, I=1,NBIN2(1,N)
      DO 1, J=1,NBIN2(2,N)
    1 HIST2(N,I,J)=HIST2(N,I,J)/HINT2(N)*X
      HINT2(N)=X                      
      END  

      SUBROUTINE MPRINT2(N)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3,CTIME*10
      DATA INI/0/               
      IF(INI.EQ.0) THEN
      CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      INI=1
      ENDIF
      IF(BOOK2(N)(:1).NE.'Y') RETURN
      WRITE(98,7) N,IYEAR,IMON,IDAY,CTIME(1:5)
      WRITE(98,*) TITLE2(N)
      WRITE(98,10) ((XHIS2(N,I),YHIS2(N,J),HIST2(N,I,J),J=1,NBIN2(2,N))
     *                            ,I=1,NBIN2(1,N))
      WRITE(98,20) IENT2(N),HINT2(N),IOSCORE2(N)
    7 FORMAT(4X,'HIST = ',I3,'   19',I2,'-',I2,'-',I2,1X,A5/)
   10 FORMAT(4X,3E10.3)
   20 FORMAT(' ENTRIES=',I5,4X,' INTEGRAL =',E10.3,4x,
     *              'OVERSCORE=',I5,//)
      END


      SUBROUTINE MTOP2(N,BTIT,LTIT,PLOPT)
      COMMON/HISTO2/HIST2(50,100,100),XHIS2(50,100),YHIS2(50,100)
     &,HDEL2(2,50),XPROJ(50,100),YPROJ(50,100),HMIN2(2,50)
     &,HMAX2(2,50),NBIN2(2,50),IHIS2(50,100,100),IOSCORE2(50)
     &,IENT2(50),NHIST2,HAVG2(2,50),HINT2(50),HSIG2(2,50),BOOK2(50),
     & TITLE2(50)
      CHARACTER TITLE2*50,BOOK2*3,CTIME*10
      CHARACTER*(*) LTIT,BTIT,PLOPT
      DOUBLE PRECISION RANXX
      DATA INI/0/                  
C Modified by SF on 26/10/2007, for a better treatment of negative entries.
C When BOX is chosen, negative bins are represented by circles rather
C than boxes. SCMAX need be redefined as the maximum of the absolute
C values of all entries. When DOT is chosen, the number of dots is now
C proportional to the absolute value of the bin entry.
C The layout of the BOX option has been changed. The size of boxes (or squares)
C is now such that, if the cross section is flat, the boxes approximately
C cover the whole plane without overlapping and without leaving empty spaces.
C This assumes that the length of the axes is about 5
      IF(INI.EQ.0) THEN
      ISEED=2143567+2*INT(SECNDS(0.0))
      CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      INI=1
      ENDIF
      IF(BOOK2(N)(:1).NE.'Y') RETURN
      WRITE(99,100) TITLE2(N),BTIT,LTIT,HMIN2(1,N),HMAX2(1,N),
     *   HMIN2(2,N),HMAX2(2,N)
  100 FORMAT( /1x,                               
     &' SET WINDOW Y 2.5 TO 9.'/,1X,
     &' SET WINDOW X 2.5 TO 10.'/,1X,
     &' SET FONT DUPLEX '/1X, 
     &' TITLE TOP ','"',A,'"',/1X,
     &' TITLE BOTTOM ','"',A,'"',/1X,
     &' TITLE LEFT ','"',A,'"',/1X,
     &' (SET TICKS TOP OFF)   '/1x,     
     &' SET LIMITS X ',F10.5,' ',F10.5,/1X,
     &' SET LIMITS Y ',F10.5,' ',F10.5,/1X,
     &' PLOT AXES',/1X,
     &' SET ORDER X Y ')

C-- squares with area proportional to the bin weight
      IF(PLOPT.EQ.'BOX') THEN
      SCMAX=0
      DO 1, I=1,NBIN2(1,N)
      DO 1, J=1,NBIN2(2,N)
      SCMAX=MAX(SCMAX,ABS(HIST2(N,I,J)))
  1   CONTINUE
      IF(SCMAX.EQ.0.) GO TO 211
      DO 2, I=1,NBIN2(1,N)
      DO 2, J=1,NBIN2(2,N)
      IF(HIST2(N,I,J).EQ.0.) GO TO 2
      IF(HIST2(N,I,J).GT.0.)THEN
        WRITE(99,200) XHIS2(N,I),YHIS2(N,J),
     #    HIST2(N,I,J)/SCMAX*5/NBIN2(1,N)*ABS(HMAX2(1,N)-HMIN2(1,N)),
     #    HIST2(N,I,J)/SCMAX*5/NBIN2(2,N)*ABS(HMAX2(2,N)-HMIN2(2,N))
      ELSE
        WRITE(99,210) XHIS2(N,I),YHIS2(N,J),
     #    -HIST2(N,I,J)/SCMAX*5/NBIN2(1,N)*ABS(HMAX2(1,N)-HMIN2(1,N)),
     #    -HIST2(N,I,J)/SCMAX*5/NBIN2(2,N)*ABS(HMAX2(2,N)-HMIN2(2,N))
      ENDIF
  2   CONTINUE
 211  CONTINUE
 200  FORMAT(2X,'BOX',F14.8,2X,F14.8,3X,'DATA SIZE',F14.8,1X,F14.8)
 210  FORMAT(2X,'CIRCLE',F14.8,2X,F14.8,3X,'DATA SIZE',F14.8,1X,F14.8)
      WRITE(99,300)
  300 FORMAT('   PLOT')
      WRITE(99,400) HINT2(N),IENT2(N)
     &   ,IOSCORE2(N),IMON,IDAY,IYEAR,CTIME(1:5)
  400 FORMAT( /1x,                               
     &' BOX 6.25 1.0 SIZE 7.5 1.'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 2.8 1.2 "INT =',1PE10.3,'    Entries =',I8,2x
     &                                 ,'Oversc =',I6,'"',/1X,
     &' TITLE 2.8 0.8 "',I2,'-',I2,', 19',I2,2X,A5,'"'/1X,
     &' SET TITLE SIZE -2')
      WRITE(99,500)
  500 FORMAT('   NEW PLOT')

C-- dots - as many as the value of HINT2(n), distributed according to wgt
      ELSEIF(PLOPT.EQ.'DOT') THEN                                        
      XINT=0.
      DO 11, I=1,NBIN2(1,N)
      DO 11, J=1,NBIN2(2,N)
      XINT=XINT+HIST2(N,I,J)    
 11   CONTINUE
      WRITE(99,*) '  SET INTENSITY 3'
      DO 3, I=1,NBIN2(1,N)          
      DO 3, J=1,NBIN2(2,N)
      IF(HIST2(N,I,J).EQ.0..OR.XINT.EQ.0.) GO TO 3
      NDOTS=INT(500*ABS(HIST2(N,I,J))/XINT)
C      RN=SNGL(RANXX(ISEED))
C      REST=HIST2(N,I,J)-FLOAT(NDOTS)
C      IF(RN.LT.REST) NDOTS=NDOTS+1
      DO ND=1,NDOTS
      X=XHIS2(N,I)+(2.*SNGL(RANXX(ISEED))-1.)*HDEL2(1,N)
      Y=YHIS2(N,J)+(2.*SNGL(RANXX(ISEED))-1.)*HDEL2(2,N)
      WRITE(99,*) X,Y
      ENDDO
      NTOT=NTOT+NDOTS
  3   CONTINUE       
      WRITE(99,*) '  SET INTENSITY 1'
      WRITE(99,300)                 
      WRITE(99,600) HINT2(N),IENT2(N),NTOT
     &   ,IMON,IDAY,IYEAR,CTIME(1:5)
  600 FORMAT( /1x,                               
     &' BOX 6.25 1.0 SIZE 7.5 1.'/,1X,
     &' SET WINDOW Y 0. TO 2.'/,1X,
     &' SET TITLE SIZE -1.5'/1X,
     &' SET FONT DUPLEX '/1X,
     &' TITLE 2.8 1.2 "INT =',1PE10.3,'    Entries =',I8,2x
     &                                 ,'Points =',I6,'"',/1X,
     &' TITLE 2.8 0.8 "',I2,'-',I2,', 19',I2,2X,A5,'"'/1X,
     &' SET TITLE SIZE -2')
      WRITE(99,500)

      ENDIF
      END

            
      FUNCTION RANXX(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      DOUBLE PRECISION MINV,RANXX
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      RANXX = SEED*MINV
      END


      SUBROUTINE MULTITOP(NH,NE,N,M,BTIT,LTIT,SCA)
      PARAMETER (NMB=1000)
      COMMON/HISTO/HIST(NMB,100),XHIS(NMB,100),HDEL(NMB),HMIN(NMB)
     &,HMAX(NMB),USCORE(NMB),OSCORE(NMB)
     &,NBIN(NMB),IHIS(NMB,100),IUSCORE(NMB),IOSCORE(NMB)
     &,IENT(NMB),HAVG(NMB),HINT(NMB),HSIG(NMB),BOOK(NMB),TITLE(NMB)
     &,NHIST
      REAL LAB0,LABS
      CHARACTER TITLE*50,BOOK*3,CTIME*10,CDATE*10,SCALE*3
      CHARACTER*(*) LTIT,BTIT,SCA
      CHARACTER*7  PLOT(4)
      DATA PLOT/'SOLID','DASHES','DOTS','DOTDASH'/
C  PLOT SIZE, CORNERS
      DATA WIDTH,HEIGHT/11.5,8.5/,XCORN,YCORN/1.5,1./
C  PLOT VERSUS TEXT FRACTION
      DATA XPFRAC,YPFRAC/0.75,0.75/,XTFRAC,YTFRAC/0.25,0.25/
C  DEFAULT SIZES
      DATA TIT0,LAB0,TIC0/3.,3.,0.06/
      DATA INI/0/
      IF(INI.EQ.0) THEN
      CALL MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      IFRAME=0
      WRITE(99,71) CDATE,CTIME(1:5)
   71 FORMAT(4X,' ( ',A10,1X,A5/)
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
        WRITE(99,1) CDATE,CTIME(1:5)
  1     FORMAT(' ( SET FONT DUPLEX',/,'  SET TITLE SIZE 2',/,
     +      ' TITLE 12.8 9 ANGLE -90 ','" MLM   ',A10,1X,A5,'"')
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
      XD=WIDTH/FLOAT(N)
      SRED=SQRT(FLOAT(N*M))
      TITS=TIT0/SRED
      LABS=LAB0/SRED
      TICS=TIC0/SRED
      XTIT0=0.55*XPFRAC*XD
      NOLD=N
10    IF(M.EQ.MOLD) GO TO 20
      YD=HEIGHT/FLOAT(M)
      YTIT0=0.06*YD
      MOLD=M
20    CONTINUE
      XL=(I-1)*XD + XCORN
      YL=(M-J)*YD + YCORN
      XU=XL+XD*XPFRAC
      YU=YL+YD*YPFRAC
      IP=0
c inizio modifiche
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
C fine modifiche
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
      WRITE(99,151) TITLE(NH)
151   FORMAT(' (  TITLE    TOP "',A,'"')
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

      SUBROUTINE NEWPLOT
      WRITE(99,202) 
202   FORMAT('   NEW PLOT',/,/)
      CALL MULTITOP(1,1,1,1,' ',' ','REF')
      END                         

C*******************************************************************
C     END OF THE HISTOGRAMMING PACKAGE
C*******************************************************************


C***** GENERIC UTILITIES *********************************
C
C   VSUM(P,Q,K)    returns K(I)=P(I)+Q(I) , I=1,4
C   DOT(P,Q)       Lorentz scalar product (+---)
C   RNDINT(N,INDX) returns a random ordering of 1,2,..,N
C   UTSORT(A,N,K,IOPT):
C     Sort A(N) into ascending order
C     IOPT = 1 : return sorted A and index array K
C     IOPT = 2 : return index array K only

      SUBROUTINE VSUM(P1,P2,Q)
      REAL P1(4),P2(4),Q(4)
      DO I=1,4
      Q(I)=P1(I)+P2(I)
      END DO
      END

      SUBROUTINE RNDINT(N,INDX)
C   returns a random ordering of the string of integers (1,2,..,N)
      COMMON/RNDM/ISEED
      DOUBLE PRECISION RANXX
      DIMENSION X(500),INDX(N)
      REAL X
      INTEGER INDX
      IF(N.GT.500) WRITE(6,*) 'WARNING**RNDINT'
      DO I=1,N
      X(I)=SNGL(RANXX(ISEED))
      ENDDO
      CALL UTSORT(X,N,INDX,2)
      END

      SUBROUTINE UTSORT(A,N,K,IOPT)
C     Sort A(N) into ascending order
C     IOPT = 1 : return sorted A and index array K
C     IOPT = 2 : return index array K only
C------------------------------------------------------------------------
      DIMENSION A(N),K(N),B(500),IL(500),IR(500)
      IF (N.GT.500) WRITE(6,*) 'WARNING**UTSORT'
      IL(1)=0
      IR(1)=0
      DO 10 I=2,N
      IL(I)=0
      IR(I)=0
      J=1
   2  IF(A(I).GT.A(J)) GO TO 5
   3  IF(IL(J).EQ.0) GO TO 4
      J=IL(J)
      GO TO 2
   4  IR(I)=-J
      IL(J)=I
      GO TO 10
   5  IF(IR(J).LE.0) GO TO 6
      J=IR(J)
      GO TO 2
   6  IR(I)=IR(J)
      IR(J)=I
  10  CONTINUE
      I=1
      J=1
      GO TO 8
  20  J=IL(J)
   8  IF(IL(J).GT.0) GO TO 20
   9  K(I)=J
      B(I)=A(J)
      I=I+1
      IF(IR(J)) 12,30,13
  13  J=IR(J)
      GO TO 8
  12  J=-IR(J)
      GO TO 9
  30  IF(IOPT.EQ.2) RETURN
      DO 31 I=1,N
  31  A(I)=B(I)
 999  END

      SUBROUTINE MBWARN(ROUT)
      CHARACTER*(*) ROUT
      WRITE(*,*) '***********************************************'
      WRITE(*,*) '***** WARNING CALLED FROM ROUTINE ',ROUT,':'
      END


      SUBROUTINE MYDATE(IMON,IDAY,IYEAR,CTIME,CDATE)
      DIMENSION IVAL(3)
      CHARACTER*10 CTIME,CDATE
      CHARACTER*10 CTIME2,CDATE2
      CALL IDATE(IVAL)
      IDAY=IVAL(1)
      IMON=IVAL(2)
      IYEAR=IVAL(3)
      CALL DATE_AND_TIME(TIME=CTIME2)
      CALL DATE_AND_TIME(DATE=CDATE2)
      CTIME=CTIME2(1:2)//":"//CTIME2(3:4)
      CDATE=CDATE2(1:4)//"-"//CDATE2(5:6)//"-"//CDATE2(7:8)
      END
