      PROGRAM DRIVER
C**************************************************************************
C     THIS IS THE DRIVER FOR CHECKING THE STANDALONE MATRIX ELEMENT
C     INCLUDING SUBTRACTION TERMS IN THE SOFT/COLINEAR LIMITS.
C     IT USES A SIMPLE PHASE SPACE GENERATOR
C     Fabio Maltoni - 3rd Febraury 2007
C     Updated by Nicolas Greiner and Rikkert Frederix for the
C     dipole terms - July 2008
C**************************************************************************
      IMPLICIT NONE
C     
C     CONSTANTS  
C     
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
C     
C     INCLUDE FILES
C     
C---  the include file with the values of the parameters and masses	
      INCLUDE "coupl.inc"
C---  integer nexternal ! number particles (incoming+outgoing) in the me 
      INCLUDE "nexternal.inc" 
C---  particle masses
      REAL*8 PMASS(NEXTERNAL)	
C---  integer    n_max_cg
      INCLUDE "ngraphs.inc"     !how many diagrams (could be useful to know...)

C     
C     LOCAL
C     
      INTEGER I,J,K,l,m,l1,m1
      REAL*8 P(0:3,NEXTERNAL)      ! four momenta. Energy is the zeroth component.
      REAL*8 SQRTS                 ! sqrt(s)= center of mass energy 
      REAL*8 ME,SUBTRACT,SUBTRACT2 ! Matrix Element and subtraction terms
      real*8 s(nexternal,nexternal)! Final state invariants
      real*8 start,number          ! Starting value for invariants and the number of phase space points
      logical cut                  ! Cut to prevent going into double limits
      real*8 sum
C     
C     EXTERNAL
C     
      REAL*8 DOT
      EXTERNAL DOT
      
C-----
C     BEGIN CODE
C-----
C     
C---  INITIALIZATION CALLS
C     
c---  Call to initialize the values of the couplings, masses and widths 
c     used in the evaluation of the matrix element. The primary parameters of the
c     models are read from Cards/param_card.dat. The secondary parameters are calculated
c     in Source/MODEL/couplings.f. The values are stored in common blocks that are listed
c     in coupl.inc .

      call setpara('param_card.dat',.true.)  !first call to setup the paramaters
      include "pmass.inc"                    !set up masses

      SQRTS=1000d0              !CMS energy in GEV
      number=1000000            !Number of phase space points



c********************************************************************************************
C First do the collinear limits:
c********************************************************************************************
      write (*,*) 'Check of all the collinear limits:'
c Loop over all possible limits of final sate 
      do l=1,nexternal-1
         do m=l+1,nexternal

            start=100d0
            write (*,*) ' '
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'
            write (*,*) ' Limit: p(',l,').p(',m,') goes to zero'
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'
            write (*,11) 'p(',l,').p(',m,')/s(1,2) ,',' sqrt(s(',l
     .           ,',',m,')) ,','|M|^2 ,'
     .           ,'|Sub.term|^2 ,','|M|^2/|Sub.term|^2'
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'


c     Loop over phase space points
            do i=1,number

c---  Now use a simple multipurpose PS generator (RAMBO) just to get a 
c     RANDOM set of four momenta of given masses pmass(i) to be used to evaluate 
c     the madgraph matrix-element and the subtraction terms.
c     	
               CALL GET_MOMENTA(SQRTS,PMASS,P)	

c     Calculate all the final state invariants
               do l1=1,nexternal-1
                  do m1=l1+1,nexternal
                     s(l1,m1)=dot(p(0,l1),p(0,m1))
                  enddo
               enddo
               
               cut =.false.
c     Only continue if we are closer to the limit then the starting value.
c     If we pass the if statement, update 'start' with the new value.
               if(s(l,m)/s(1,2) .lt. start) then
                  
c     Make sure that we are only looking in a single collinear limit. Double
c     logarithms are not cancelled by the dipoles. Do this by cutting
c     away the phase-space point if any of the other final state invariants
c     is smaller than 10% of the CMS energy.
                  do j=1,nexternal-1
                     do k=j+1,nexternal
                        if((j.ne.l).or.(k.ne.m)) then
                           if(s(j,k)/s(1,2).lt.0.01d0) then
                              cut=.true.
                           endif
                        endif
                     enddo
                  enddo

c     If we found a point closer to the limit, update 'start' and
c     calculate the matrix element and subtraction terms.
                  if(.not.cut) then
                     start=s(l,m)/s(1,2)
                     CALL SMATRIX(P,ME)
                     Call DIPOLSUM(P,SUBTRACT)
c     Uncomment the following 2 lines to include the possible non-divergent dipoles,
c     i.e. dipoles for which the unresolved is massive (= in particles.dat the mass
c     is set to something different from 'ZERO').
c     Call DIPOLSUMFINITE(P,SUBTRACT2)
c     SUBTRACT=SUBTRACT+SUBTRACT2

c     Write out the results
                     Write (*,10)s(l,m)/s(1,2),' ,',dsqrt(abs(
     &                    (p(0,l)+p(0,m))**2-(p(1,l)+p(1,m))**2
     &                    -(p(2,l)+p(2,m))**2-(p(3,l)+p(3,m))**2))
     &                    ,' ,',ME,' ,',SUBTRACT,' ,',ME/SUBTRACT
                  endif
               endif
            enddo
         enddo
      enddo







c********************************************************************************
C Now check also the soft limits:
c********************************************************************************
      do i=1,5
         write (*,*) ' '
      enddo
      write (*,*) 'Check off all the soft limits:'
c loop over all final state particles:
      do l=3, nexternal

            start=100d0
            write (*,*) ' '
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'
            write (*,*) ' Limit: p(',l,') goes soft'
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'
            write (*,13) 'p(0,',l,')^2/s(1,2) ,','|M|^2 ,'
     .           ,'|Sub.term|^2 ,','|M|^2/|Sub.term|^2'
            write (*,'(2A)')'-----------------------------------------',
     .          '------------------------------------------------------'

c     Loop over phase space points
            do i=1,number

c     Get the random set of 4-momenta
               CALL GET_MOMENTA(SQRTS,PMASS,P)	

c     Calculate all the final state invariants
               do l1=1,nexternal-1
                  do m1=l1+1,nexternal
                     s(l1,m1)=dot(p(0,l1),p(0,m1))
                  enddo
               enddo
               
               cut =.false.
               sum = 0d0
c     Only continue if we are closer to the limit then the starting value.
c     If we pass the if statement, update 'start' with the new value.
               if(p(0,l)**2/s(1,2) .lt. start) then
                  
c     Make sure that we are only looking at the soft limit. Do this by cutting
c     away the phase-space point if any of the other final state invariants
c     is smaller than 10% of the CMS energy.
                  do j=1,nexternal-1
                     do k=j+1,nexternal
                        if((j.ne.l).and.(k.ne.l)) then
                           if(s(j,k)/s(1,2).lt.0.01d0) then
                              cut=.true.
                           endif
                        endif
                        if((j.eq.l).or.(k.eq.l))then
                           sum=sum+s(j,k)
                        endif
                     enddo
                  enddo
c     cut away points that are collinear by forcing that all the invariants
c     with the soft particle contribute at least 2*nexternal'th to the sum. This
c     makes sure that all these invariants go to zero at the same time.              
                  do j=1,nexternal-1
                     do k=j+1,nexternal
                        if(((j.eq.l).or.(k.eq.l)).and.
     .                       (s(j,k).lt.sum/real(2*nexternal)))then
                           cut=.true.
                        endif
                     enddo
                  enddo

c     If we found a point closer to the limit, update 'start' and
c     calculate the matrix element and subtraction terms.
                  if(.not.cut) then
                     start=p(0,l)**2/s(1,2)
                     CALL SMATRIX(P,ME)
                     Call DIPOLSUM(P,SUBTRACT)
c     Uncomment the following 2 lines to include the possible non-divergent dipoles,
c     i.e. dipoles for which the unresolved is massive (= in particles.dat the mass
c     is set to something different from 'ZERO').
c     Call DIPOLSUMFINITE(P,SUBTRACT2)
c     SUBTRACT=SUBTRACT+SUBTRACT2

c     Write out the results
                     Write (*,14) p(0,l)/s(1,2),
     .                    ' ,',ME,' ,',SUBTRACT,' ,',ME/SUBTRACT
                  endif
               endif
            enddo

      enddo


 10   format(1X,e18.6,A2,e16.6,A2,e16.6,A2,e16.6,A2,e16.6)
 11   format(1X,A2,I2,A4,I2,A,1X,A,I2,A1,I2,A,A18,A18,A20)
 14   format(1X,e18.6,A2,e16.6,A2,e16.6,A2,e16.6)
 13   format(3X,A4,I2,A12,A18,A18,A20)

      end
	
	  
	  
	  
      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end


      SUBROUTINE GET_MOMENTA(ENERGY,PMASS,P)
C---- auxiliary function to change convention between madgraph and rambo
c---- four momenta. 	  
      IMPLICIT NONE
      INCLUDE "nexternal.inc"
C     ARGUMENTS
      REAL*8 ENERGY,PMASS(NEXTERNAL),P(0:3,NEXTERNAL),PRAMBO(4,10),WGT
C     LOCAL
      INTEGER I
      
      P(0,1)=energy/2
      P(1,1)=0d0
      P(2,1)=0d0
      P(3,1)=energy/2
      
      P(0,2)=energy/2
      P(1,2)=0d0
      P(2,2)=0d0
      P(3,2)=-energy/2

      call rambo(nexternal-2,energy,pmass(3),prambo,WGT)
      DO I=3, NEXTERNAL
         P(0,I)=PRAMBO(4,I-2)	
         P(1,I)=PRAMBO(1,I-2)
         P(2,I)=PRAMBO(2,I-2)
         P(3,I)=PRAMBO(3,I-2)	
      ENDDO

      RETURN
      END


      SUBROUTINE RAMBO(N,ET,XM,P,WT)
***********************************************************************
*                       RAMBO                                         *
*    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
*                                                                     *
*    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR                *
*    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING                 *
*    THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS                      *
*    -- ADJUSTED BY HANS KUIJF, WEIGHTS ARE LOGARITHMIC (20-08-90)    *
*                                                                     *
*    N  = NUMBER OF PARTICLES                                         *
*    ET = TOTAL CENTRE-OF-MASS ENERGY                                 *
*    XM = PARTICLE MASSES ( DIM=NEXTERNAL-2 )                         *
*    P  = PARTICLE MOMENTA ( DIM=(4,NEXTERNAL-2) )                    *
*    WT = WEIGHT OF THE EVENT                                         *
***********************************************************************
      IMPLICIT REAL*8(A-H,O-Z)
	  INCLUDE "nexternal.inc"
      DIMENSION XM(NEXTERNAL-2),P(4,NEXTERNAL-2)
      DIMENSION Q(4,NEXTERNAL-2),Z(NEXTERNAL-2),R(4),
     .   B(3),P2(NEXTERNAL-2),XM2(NEXTERNAL-2),
     .   E(NEXTERNAL-2),V(NEXTERNAL-2),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
*
* INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,10
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,10
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
*
* CHECK ON THE NUMBER OF PARTICLES
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
*
* CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP
*
* THE PARAMETER VALUES ARE NOW ACCEPTED
*
* GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE
  201 DO 202 I=1,N
         r1=rn(1)
      C=2.*r1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RN(2)
      r1=rn(3)
      r2=rn(4)
      Q(4,I)=-LOG(r1*r2)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
  202 Q(1,I)=Q(4,I)*S*SIN(F)
*
* CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 203 I=1,4
  203 R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
  204 R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
  205 B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
*
* TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(4,I)=X*(G*Q(4,I)+BQ)
*
* CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
  208 IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
*
* RETURN FOR WEIGHTED MASSLESS MOMENTA
  209 IF(NM.NE.0) GOTO 210
* RETURN LOG OF WEIGHT
      WT=WT
      RETURN
*
* MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
  210 XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
  301 P2(I)=P(4,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
  302 F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
  303 G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      PRINT 1006,ITMAX
      GOTO 305
  304 X=X-F0/(X*G0)
      GOTO 302
  305 DO 307 I=1,N
      V(I)=X*P(4,I)
      DO 306 K=1,3
  306 P(K,I)=X*P(K,I)
  307 P(4,I)=E(I)
*
* CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
  308 WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
*
* RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
  309 IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
* RETURN LOG OF WEIGHT
  310 WT=WT
      RETURN
*
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',
     . ' SMALLER THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE',
     . ' DESIRED ACCURACY =',D15.6)
      END

      FUNCTION RN(IDUMMY)
      REAL*8 RN,RAN
      SAVE INIT
      DATA INIT /1/
      IF (INIT.EQ.1) THEN
        INIT=0
        CALL RMARIN(1802,9373)
      END IF
*
  10  CALL RANMAR(RAN)
      IF (RAN.LT.1D-16) GOTO 10
      RN=RAN
*
      END



      SUBROUTINE RANMAR(RVEC)
*     -----------------
* Universal random number generator proposed by Marsaglia and Zaman
* in report FSU-SCRI-87-50
* In this version RVEC is a double precision variable.
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
      UNI = RANU(IRANMR) - RANU(JRANMR)
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RANU(IRANMR) = UNI
      IRANMR = IRANMR - 1
      JRANMR = JRANMR - 1
      IF(IRANMR .EQ. 0) IRANMR = 97
      IF(JRANMR .EQ. 0) JRANMR = 97
      RANC = RANC - RANCD
      IF(RANC .LT. 0D0) RANC = RANC + RANCM
      UNI = UNI - RANC
      IF(UNI .LT. 0D0) UNI = UNI + 1D0
      RVEC = UNI
      END
 
      SUBROUTINE RMARIN(IJ,KL)
*     -----------------
* Initializing routine for RANMAR, must be called before generating
* any pseudorandom numbers with RANMAR. The input values should be in
* the ranges 0<=ij<=31328 ; 0<=kl<=30081
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
* This shows correspondence between the simplified input seeds IJ, KL
* and the original Marsaglia-Zaman seeds I,J,K,L.
* To get the standard values in the Marsaglia-Zaman paper (i=12,j=34
* k=56,l=78) put ij=1802, kl=9373
      I = MOD( IJ/177 , 177 ) + 2
      J = MOD( IJ     , 177 ) + 2
      K = MOD( KL/169 , 178 ) + 1
      L = MOD( KL     , 169 )
      DO 300 II = 1 , 97
        S =  0D0
        T = .5D0
        DO 200 JJ = 1 , 24
          M = MOD( MOD(I*J,179)*K , 179 )
          I = J
          J = K
          K = M
          L = MOD( 53*L+1 , 169 )
          IF(MOD(L*M,64) .GE. 32) S = S + T
          T = .5D0*T
  200   CONTINUE
        RANU(II) = S
  300 CONTINUE
      RANC  =   362436D0 / 16777216D0
      RANCD =  7654321D0 / 16777216D0
      RANCM = 16777213D0 / 16777216D0
      IRANMR = 97
      JRANMR = 33
      END






