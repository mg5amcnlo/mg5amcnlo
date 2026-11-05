      PROGRAM CHECK_SA_FKS
C     ******************************************************************
C     Standalone driver for the FKS Born building blocks.
C     For a single phase-space point it prints, per flavour
C     configuration:
C       - the Born                       B
C       - the spin-correlated Born       BORNTILDE
C       - the color/charge-linked Borns  B_ij
C     The link topology (which (m,n) leg pairs are real links) is read
C     at run time from born_links.dat, written by the exporter.
C     ******************************************************************
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      REAL*8 P(0:3,NEXTERNAL-1)
      REAL*8 PMASS(NEXTERNAL-1)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)
      INCLUDE 'coupl.inc'
      INTEGER I,J,ILINK,NLINKS,M,N,ICALL
      INTEGER PDG_M,PDG_N,COL_M,COL_N,ITYPE
      REAL*8 BORN, WGT, SQRTS, BORNTILDE, TOTMASS
C     optional run-time controls read from the command line:
C       arg1 = sqrt(s) of the phase-space point (<=0 -> built-in default)
C       arg2 = number of Born re-evaluations (for the launch --timings mode)
      INTEGER NARGS, NCALLS, IARGC
      REAL*8 USER_ENERGY
      CHARACTER*100 ARG
C     the link topology is read once into these arrays, so the timing loop
C     re-evaluates the Born building blocks without re-reading the file.
      INTEGER MAXLINK
      PARAMETER (MAXLINK=1000)
      INTEGER MLIST(MAXLINK), NLIST(MAXLINK)
      REAL*8 WGTLIST(MAXLINK)
C     COLOFLEG(i) is the soft-correlation quantum number of Born leg i, taken
C     from born_links.dat (colour rep for [QCD], colour singlet=1 for the
C     charged-but-colourless legs of a [QED] run); it stays 0 iff the leg
C     never appears in a link, i.e. is uncorrelated. HASDIAG flags whether the
C     leg's diagonal self-link is already present. Both drive the rebuild of
C     the missing massless diagonals below.
      INTEGER COLOFLEG(NEXTERNAL), NTOT, ITMP
      LOGICAL HASDIAG(NEXTERNAL)
      REAL*8 LINKSUM, RTMP
      COMPLEX*16 ANS_CNT(2,NSPLITORDERS)
      COMMON /C_BORN_CNT/ ANS_CNT
      LOGICAL NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      COMMON /C_NEED_LINKS/ NEED_COLOR_LINKS, NEED_CHARGE_LINKS
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      LOGICAL SPLIT_TYPE_USED(NSPLITORDERS)
      COMMON/TO_SPLIT_TYPE_USED/SPLIT_TYPE_USED
      DOUBLE PRECISION PARTICLE_CHARGE_BORN(NEXTERNAL-1)
      COMMON /C_CHARGES_BORN/PARTICLE_CHARGE_BORN
      INCLUDE 'nFKSconfigs.inc'
      INCLUDE 'fks_info.inc'

C     the FKS bookkeeping common blocks must be initialised before the
C     Born is evaluated (NFKSPROCESS selects the IDEN/IJ data, and
C     SPLIT_TYPE_USED enables the counterterm/spin-correlation pieces).
C     Only the split orders that are actually perturbed in some FKS
C     configuration may be enabled: forcing all of them on makes the
C     spin-correlation lookup ask for an amp_split order that does not
C     exist for a mixed-coupling Born (e.g. z > q q~ has a QED order).
C     This mirrors the production fill_needed_splittings().
      NFKSPROCESS=1
      DO I=1,NSPLITORDERS
        SPLIT_TYPE_USED(I)=.FALSE.
      ENDDO
      DO J=1,FKS_CONFIGS
        DO I=1,NSPLITORDERS
          SPLIT_TYPE_USED(I)=SPLIT_TYPE_USED(I).OR.SPLIT_TYPE_D(J,I)
        ENDDO
      ENDDO

C     parse the optional command-line arguments (energy, #re-evaluations)
      USER_ENERGY=0D0
      NCALLS=1
      NARGS=IARGC()
      IF (NARGS.GE.1) THEN
        CALL GETARG(1,ARG)
        READ(ARG,*) USER_ENERGY
      ENDIF
      IF (NARGS.GE.2) THEN
        CALL GETARG(2,ARG)
        READ(ARG,*) NCALLS
      ENDIF
      IF (NCALLS.LT.1) NCALLS=1

      CALL SETPARA('param_card.dat')
      CALL PRINTOUT()

      INCLUDE 'born_pmass.inc'

C     the per-leg electric charges enter the charge-linked Born ([QED]
C     soft-photon links); they are written by the exporter and must be
C     filled before SBORN_SF is called on a charge link.
      INCLUDE 'born_charges.inc'

C     pick a center-of-mass energy comfortably above threshold
      TOTMASS=0D0
      DO I=1,NEXTERNAL-1
        TOTMASS=TOTMASS+PMASS(I)
      ENDDO
      SQRTS=1000D0
      IF (USER_ENERGY.GT.0D0) SQRTS=USER_ENERGY
C     keep the point above threshold so RAMBO never fails, even if the user
C     asks for an energy below the sum of the final-state masses
      IF (4D0*TOTMASS.GT.SQRTS) SQRTS=4D0*TOTMASS
      CALL GET_MOMENTA(SQRTS,PMASS,P)

C     whether the soft links of this configuration are colour links (a
C     gluon goes soft, [QCD]) or charge links (a photon goes soft, [QED])
C     is read from the generated data: sborn_sf takes a different branch
C     for each, so getting this from need_*_links_d (rather than forcing
C     colour) is what makes the [QED] building blocks come out right.
      NEED_COLOR_LINKS=NEED_COLOR_LINKS_D(NFKSPROCESS)
      NEED_CHARGE_LINKS=NEED_CHARGE_LINKS_D(NFKSPROCESS)

C     ---- one flavour configuration is available today; the loop is
C     ---- kept explicit so the flavour-merging extension only has to
C     ---- grow the upper bound and reset the relevant common blocks.
      WRITE(*,*) '==== FLAVOUR CONFIGURATION', 1, '===='

C     read the link topology once into MLIST/NLIST so the (optional) timing
C     loop below does not pay the file I/O on every re-evaluation
      DO I=1,NEXTERNAL-1
        COLOFLEG(I)=0
        HASDIAG(I)=.FALSE.
      ENDDO
      OPEN(UNIT=78,FILE='born_links.dat',STATUS='OLD')
      READ(78,*) NLINKS
      DO ILINK=1,NLINKS
        READ(78,*) M,N,PDG_M,PDG_N,COL_M,COL_N,ITYPE
        MLIST(ILINK)=M
        NLIST(ILINK)=N
C       remember the colour rep of each leg (recoverable from any link it
C       appears in) and whether its diagonal self-link is already present
        COLOFLEG(M)=COL_M
        COLOFLEG(N)=COL_N
        IF (M.EQ.N) HASDIAG(M)=.TRUE.
      ENDDO
      CLOSE(78)

C     evaluate the Born building blocks NCALLS times (NCALLS>1 only for the
C     launch --timings mode); the values are identical, so we keep the last
C     ones and print them once below.
      DO ICALL=1,NCALLS
        CALL SBORN(P,BORN)
        BORNTILDE=0D0
        DO J=1,NSPLITORDERS
          BORNTILDE=BORNTILDE+DBLE(ANS_CNT(2,J))
        ENDDO
        DO ILINK=1,NLINKS
          CALL SBORN_SF(P,MLIST(ILINK),NLIST(ILINK),WGT)
          WGTLIST(ILINK)=WGT
        ENDDO
      ENDDO

C     Rebuild the diagonal soft self-links B_ii that the link generator skips
C     for massless legs: find_color_links drops the leg1==leg2 pair when the
C     leg is massless (only massive emitters keep an explicit diagonal link),
C     so e.g. g g > t t~ has B_ij 3 3 / 4 4 for the tops but no 1 1 / 2 2 for
C     the gluons, and u u~ > w+ w- has the W charge diagonals but not the u
C     ones. They are recovered from the conservation Ward identity, which for
C     colour is sum_j T_i.T_j = 0 and for charge is sum_j Q_j = 0; both give
C     the diagonal as minus half the sum of the off-diagonal links touching
C     leg i:  B_ii = -1/2 * sum_{j/=i} B_ij.  This is convention independent
C     and reproduces the explicit massive diagonals (the MadFKS 1/2 and the
C     colour-basis / charge normalisation included) to machine precision in
C     both the [QCD] and [QED] cases, so HASDIAG skips the legs that already
C     carry an explicit diagonal to avoid double counting.
      NTOT=NLINKS
      DO I=1,NEXTERNAL-1
        IF (COLOFLEG(I).NE.0 .AND. .NOT.HASDIAG(I)) THEN
          LINKSUM=0D0
          DO ILINK=1,NLINKS
            IF (MLIST(ILINK).NE.NLIST(ILINK) .AND.
     &          (MLIST(ILINK).EQ.I .OR. NLIST(ILINK).EQ.I)) THEN
              LINKSUM=LINKSUM+WGTLIST(ILINK)
            ENDIF
          ENDDO
          NTOT=NTOT+1
          MLIST(NTOT)=I
          NLIST(NTOT)=I
          WGTLIST(NTOT)=-0.5D0*LINKSUM
        ENDIF
      ENDDO

C     order the links by (m,n) so the reconstructed diagonals sit next to the
C     leg's other links instead of being tacked on at the end: B_ij 1 1 then
C     1 2 ... rather than the off-diagonals first and the diagonals last. The
C     key m*NEXTERNAL+n is injective since leg numbers run 1..NEXTERNAL-1.
      DO I=1,NTOT-1
        DO J=I+1,NTOT
          IF (MLIST(J)*NEXTERNAL+NLIST(J) .LT.
     &        MLIST(I)*NEXTERNAL+NLIST(I)) THEN
            ITMP=MLIST(I)
            MLIST(I)=MLIST(J)
            MLIST(J)=ITMP
            ITMP=NLIST(I)
            NLIST(I)=NLIST(J)
            NLIST(J)=ITMP
            RTMP=WGTLIST(I)
            WGTLIST(I)=WGTLIST(J)
            WGTLIST(J)=RTMP
          ENDIF
        ENDDO
      ENDDO

      WRITE(*,*) 'BORN       =', BORN
      WRITE(*,*) 'BORNTILDE  =', BORNTILDE
      DO ILINK=1,NTOT
        WRITE(*,*) 'B_ij ', MLIST(ILINK), NLIST(ILINK), WGTLIST(ILINK)
      ENDDO

      END

      SUBROUTINE GET_MOMENTA(ENERGY,PMASS,P)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER NBORN
      PARAMETER (NBORN=NEXTERNAL-1)
      REAL*8 ENERGY,PMASS(NBORN),P(0:3,NBORN),PRAMBO(4,10),WGT
      INTEGER I
      IF (NINCOMING.EQ.1) THEN
C       decay: the parent decays at rest, so the available energy is
C       fixed by its mass (ENERGY is ignored) and the NBORN-1 decay
C       products are distributed by RAMBO in the parent rest frame.
        P(0,1)=PMASS(1)
        P(1,1)=0d0
        P(2,1)=0d0
        P(3,1)=0d0
        CALL RAMBO(NBORN-1,PMASS(1),PMASS(2),PRAMBO,WGT)
        DO I=2,NBORN
          P(0,I)=PRAMBO(4,I-1)
          P(1,I)=PRAMBO(1,I-1)
          P(2,I)=PRAMBO(2,I-1)
          P(3,I)=PRAMBO(3,I-1)
        ENDDO
      ELSE
C       2 -> n: back-to-back massless initial states along z
        P(0,1)=ENERGY/2
        P(1,1)=0d0
        P(2,1)=0d0
        P(3,1)=ENERGY/2
        P(0,2)=ENERGY/2
        P(1,2)=0d0
        P(2,2)=0d0
        P(3,2)=-ENERGY/2
        CALL RAMBO(NBORN-2,ENERGY,PMASS(3),PRAMBO,WGT)
        DO I=3,NBORN
          P(0,I)=PRAMBO(4,I-2)
          P(1,I)=PRAMBO(1,I-2)
          P(2,I)=PRAMBO(2,I-2)
          P(3,I)=PRAMBO(3,I-2)
        ENDDO
      ENDIF
      RETURN
      END

      SUBROUTINE RAMBO(N,ET,XM,P,WT)
***********************************************************************
*                       RAMBO                                         *
*    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)                  *
*    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR                *
*    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING                 *
***********************************************************************
      IMPLICIT REAL*8(A-H,O-Z)
      INCLUDE "nexternal.inc"
      DIMENSION XM(*),P(4,*)
      DIMENSION Q(4,NEXTERNAL-NINCOMING),Z(NEXTERNAL-NINCOMING),R(4),
     .   B(3),P2(NEXTERNAL-NINCOMING),XM2(NEXTERNAL-NINCOMING),
     .   E(NEXTERNAL-NINCOMING),V(NEXTERNAL-NINCOMING),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN
      DATA ACC/1.D-14/,ITMAX/6/,IBEGIN/0/,IWARN/5*0/
      SAVE TWOPI, PO2LOG, Z
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,NEXTERNAL-NINCOMING
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,NEXTERNAL-NINCOMING
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP
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
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(4,I)=X*(G*Q(4,I)+BQ)
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
  208 IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
  209 IF(NM.NE.0) GOTO 210
      WT=WT
      RETURN
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
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
  308 WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
  309 IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
  310 WT=WT
      RETURN
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
  10  CALL RANMAR(RAN)
      IF (RAN.LT.1D-16) GOTO 10
      RN=RAN
      END

      SUBROUTINE RANMAR(RVEC)
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
      IMPLICIT REAL*8(A-H,O-Z)
      COMMON/ RASET1 / RANU(97),RANC,RANCD,RANCM
      COMMON/ RASET2 / IRANMR,JRANMR
      SAVE /RASET1/,/RASET2/
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

C     Local copy of get_lo2_orders so the Born-only check does not have
C     to drag in the EW-Sudakov object chain (sudakov_wrapper, ...).
C     Never exercised at run time for the [QCD] Born, but referenced by
C     sborn_onehel inside born.f and therefore needed at link time.
      SUBROUTINE GET_LO2_ORDERS(LO2_ORDERS)
      IMPLICIT NONE
      INCLUDE 'orders.inc'
      INTEGER LO2_ORDERS(NSPLITORDERS)
      LO2_ORDERS(:) = BORN_ORDERS(:)
      LO2_ORDERS(QCD_POS) = LO2_ORDERS(QCD_POS) - 2
      LO2_ORDERS(QED_POS) = LO2_ORDERS(QED_POS) + 2
      RETURN
      END
