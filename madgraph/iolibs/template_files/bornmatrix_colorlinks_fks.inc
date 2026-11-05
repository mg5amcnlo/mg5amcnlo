      SUBROUTINE SBORN(P,ANS_SUMMED)
C
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C
C
%(process_lines)s
C
C
C CONSTANTS
C
      implicit none
      include 'nexternal.inc'
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1)
      double precision ans_summed
C
C VARIABLES
C
      INTEGER I,J,K
      include 'has_ewsudakov.inc'
      integer nord_integrated_born
      include 'orders.inc'
      COMPLEX*16 ANS(2,0:NSQAMPSO), ANS_CNT(2, NSPLITORDERS)
      LOGICAL KEEP_ORDER(NSQAMPSO), KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO), FIRSTTIME
      data keep_order / NSQAMPSO * .TRUE. /
      logical keep_order_born(NSQAMPSO)
      common /c_keep_order_B/ keep_order_born
      common /c_keep_order_cnt/ keep_order_cnt
      common /c_born_cnt/ ans_cnt
      integer ord_subtract
      data firsttime / .TRUE. /
      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real

      double precision amp2b(%(ngraphs)d), jamp2b(0:%(ncolor)d,0:nampso)
      common/to_amps_born/  amp2b,       jamp2b
      Double Precision amp2(%(ngraphs)d), jamp2(0:%(ncolor)d)
      common/to_amps/  amp2,       jamp2
      logical split_type_used(nsplitorders)
      common/to_split_type_used/split_type_used
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C BEGIN CODE
C
C look for orders which match the born order constraint

force_ijglu_zero = .false.

if (firsttime) then
 do i = 1, nsqampso
C this is for the orders of the born to integrate
  do j = 1, nsplitorders
   if(GETORDPOWFROMINDEX_B(j, i) .gt. born_orders(j)) then
    keep_order(i) = .false.
    EXIT
   endif
  enddo
  keep_order_born = keep_order

C this is for the orders of the counterterms
  do j = 1, nsplitorders
  keep_order_cnt(j,i) = .true.
   do k = 1, nsplitorders
    if (j.eq.k) then
     ord_subtract=2
    else
     ord_subtract=0
    endif
    if(GETORDPOWFROMINDEX_B(k, i) .gt. nlo_orders(k)-ord_subtract) then
     keep_order_cnt(j,i) = .false.
     EXIT
    endif
   enddo
  enddo

  nord_integrated_born = 0
  if (keep_order(i)) then
   nord_integrated_born = nord_integrated_born + 1
   !write(*,*) 'BORN: keeping split order ', i
  !else
   !write(*,*) 'BORN: not keeping split order ', i
  endif
 enddo

C check that, if one include the sudakov corrections, there is at most one contribution to be integrated at LO
if (has_ewsudakov.and.nord_integrated_born.gt.1) then
 write(*,*) 'ERROR in born.f, cannot compute Sudakov EW corrections with more than one Born coupling combination to be integrated.Found', nord_integrated_born
 stop 1
endif

 do j = 1, nsplitorders
  write(*,*) 'counterterm S.O', j, ordernames(j)
  do i = 1, nsqampso
   if (keep_order_cnt(j,i)) then
    write(*,*) 'BORN: keeping split order', i
   else
    write(*,*) 'BORN: not keeping split order', i
   endif
  enddo
 enddo
 firsttime = .false.
endif

CALL SBORN_SPLITORDERS(P,ANS)

C the born to be integrated
ans_summed = 0d0
max_val = 0d0

C reset the amp_split array
amp_split(1:amp_split_size) = 0d0
amp_split_cnt(1:amp_split_size,1:2,1:nsplitorders) = dcmplx(0d0,0d0)

do i = 1, nsqampso
 max_val = max(max_val, abs(ans(1,I)))
enddo
do i = 1, nsqampso
 if (keep_order(i)) then
    ANS_SUMMED = ANS_SUMMED + ANS(1,I)
C   keep track of the separate pieces corresponding to different coupling combinations
    do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
    enddo
    if(abs(ans(1,i)).gt.max_val*tiny) amp_split(orders_to_amp_split_pos(amp_orders)) = ans(1,i)
 endif
enddo
C this is to avoid fake non-zero contributions
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

wgt_me_born=ans_summed

C fill the amp2 and jamp2 arrays
amp2(1:ngraphs)=amp2b(1:ngraphs)! amp2 just needs to be copied

do i = 0, int(jamp2b(0,0))
jamp2(i)=0d0
do j = 1, nampso
c here sum all, this may be refined later
jamp2(i)=jamp2(i)+jamp2b(i,j)
enddo
enddo

%(skip_amp_cnt)s
C quantities for the counterterms
do j = 1, nsplitorders
 ans_cnt(1:2,j) = (0d0, 0d0)
 if (.not.split_type_used(j)) cycle
 do i = 1, nsqampso
  if (keep_order_cnt(j,i)) then
   ANS_CNT(1,j) = ANS_CNT(1,J) + ANS(1,I)
   ANS_CNT(2,j) = ANS_CNT(2,J) + ANS(2,I)
C   keep track of the separate pieces also for counterterms
   do k = 1, nsplitorders
      amp_orders(k) = GETORDPOWFROMINDEX_B(k, i)
C take into account the fact that the counterterm for a given split order
C will be multiplied by the corresponding squared coupling
      if (k.eq.j) amp_orders(k) = amp_orders(k) + 2
   enddo
C this is to avoid fake non-zero contributions
   if (abs(ans(1,I)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),1,j) = ans(1,I)
   if (abs(ans(2,I)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),2,j) = ans(2,I)
  endif
 enddo
C this is to avoid fake non-zero contributions
 if (abs(ans_cnt(1,j)).lt.max_val*tiny) ans_cnt(1,j)=(0d0,0d0)
 if (abs(ans_cnt(2,j)).lt.max_val*tiny) ans_cnt(2,j)=(0d0,0d0)
enddo


 999   return
END





      SUBROUTINE SBORN_ONEHEL(P,NHEL,HELL,ANS_SUMMED)
C
%(info_lines)s
C
C
C     Return the sum of the split orders which are required in orders.inc (BORN_ORDERS)
C     It works for a single set of helicities, passed by NHEL
C     Also the values needed for the counterterms are stored in the C_BORN_CNT common block
C
C
%(process_lines)s
C
C
C CONSTANTS
C
      implicit none
      include 'nexternal.inc'
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      integer HELL
      double precision ans_summed
C
C VARIABLES
C
      INTEGER I,J,K
      include 'orders.inc'
      COMPLEX*16 ANS(2,NSQAMPSO)
      REAL*8 borns(2,0:NSQAMPSO)
      LOGICAL KEEP_ORDER(NSQAMPSO)
      common /c_keep_order_B/ keep_order

      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt

      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
      include 'ewsudakov_haslo.inc'
      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex amp_split_ewsud_LO2(amp_split_size)
      common /to_amp_split_ewsud_LO2/ amp_split_ewsud_LO2
      integer iden
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero
      %(den_factor_lines)s
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C
C     stuff for the LO2
      integer lo2_orders(nsplitorders)
      logical orders_equal
      external orders_equal
C
C BEGIN CODE
C
C look for orders which match the born order constraint

      iden=iden_values(nfksprocess)
force_ijglu_zero = .true.

CALL BORN(P,NHEL,HELL,0,ANS,borns)

ans_summed = 0d0
max_val = 0d0

C reset the amp_split array
amp_split_ewsud(1:amp_split_size) = dcmplx(0d0,0d0)
amp_split_ewsud_lo2(1:amp_split_size) = dcmplx(0d0,0d0)

do i = 1, nsqampso
 max_val = max(max_val, abs(ans(1,I)))
enddo

call get_lo2_orders(lo2_orders)

do i = 1, nsqampso
    if (keep_order(i)) then
     ANS_SUMMED = ans_summed + ANS(1,I) / iden
     do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
     enddo
     !amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(1,I)
     if(abs(ans(1,i)).gt.max_val*tiny) amp_split_ewsud(orders_to_amp_split_pos(amp_orders)) = ans(1,i) / iden
    endif
C the following is for the LO2
    if (has_lo2) then
     do j = 1, nsplitorders
      amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
     enddo
     if (orders_equal(amp_orders, lo2_orders)) then
      if(abs(ans(1,i)).gt.max_val*tiny) amp_split_ewsud_lo2(orders_to_amp_split_pos(amp_orders)) = ans(1,i) / iden
     endif
    endif
enddo

C this is to avoid fake non-zero contributions
if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

  return
END




SUBROUTINE SBORN_SPLITORDERS(P1,ANS)
C
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      include "nexternal.inc"
      include "born_nhel.inc"
      include "coupl.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C
C ARGUMENTS
C
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(2,0:NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER IHEL,IDEN,i,j,jj,glu_ij,BMODE
      REAL*8 borns(2,0:NSQAMPSO), BORNRHO(NSQAMPSO)
      COMPLEX*16 BORNTILDE
      INTEGER NTRY(%(nconfs)d)
      DATA NTRY /%(nconfs)d*0/
      COMPLEX*16 T(2,NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
%(den_factor_lines)s
%(ij_lines)s
C
C GLOBAL VARIABLES
C
      Double Precision amp2(ngraphs), jamp2(0:%(ncolor)d,0:NAMPSO)
      common/to_amps_born/  amp2,       jamp2
      DATA jamp2(0,0) /   %(ncolor)d/
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      logical cond_ij
      LOGICAL BORN_CACHE_OK
      INTEGER BORN_CACHE_NFKS
      COMMON /C_BORN_CACHE/ BORN_CACHE_OK, BORN_CACHE_NFKS
      COMPLEX*16 BORN_CACHE_ANS(2,0:NSQAMPSO)
      SAVE BORN_CACHE_ANS
      LOGICAL CHECK_BORN_AMPS, BORN_CHECKS_AMPS
      COMMON /C_BORN_AMPS_CHECK/ CHECK_BORN_AMPS, BORN_CHECKS_AMPS
      DOUBLE PRECISION SAVEMOM4(0:3,NEXTERNAL-1)
      DOUBLE COMPLEX SAVED_COUP(%(ncoup_check)d)
      SAVE SAVEMOM4, SAVED_COUP
C     gscale: the strong coupling SAVED_G of the cached amplitudes,
C     and the powers RPOW(K) = (G/SAVED_G)**K used by BORN_GSCALE
      INTEGER MAXQCDPOW
      PARAMETER (MAXQCDPOW=%(maxqcdpow)d)
      LOGICAL DO_GSCALE
      PARAMETER (DO_GSCALE=%(born_gscale)s)
      DOUBLE PRECISION GSCALE_TOL
      PARAMETER (GSCALE_TOL=1D-12)
      DOUBLE PRECISION SAVED_G, RG, RPOW(0:MAXQCDPOW)
      LOGICAL COUP_CHANGED, GSCALED
      SAVE SAVED_G
      DATA SAVED_G /0D0/
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
C     set_ren_scale (or sreal_deg) kept the cached amplitudes instead of
C     discarding them: discard them here unless the momenta are exactly
C     the ones they were computed with, and the couplings of the Born
C     too, or differ from them only through alpha_S (gscale below)
      if (calculatedBorn .and. check_born_amps) then
         coup_changed=.false.
%(coup_check_lines)s
         do j=1,nexternal-1
            do i=0,3
               if (p1(i,j).ne.savemom4(i,j)) calculatedBorn=.false.
            enddo
         enddo
         if (calculatedBorn .and. coup_changed) then
C     gscale: same momenta, other couplings. If only alpha_S
C     changed (set_alphaS, e.g. the Born of the MC counterterm
C     at the coupling of the real emission), every coupling
C     changed by RPOW(n), n its QCD order: then rescale each
C     cached amplitude by RPOW(n), n its own QCD order (each
C     split order gets its power), instead of recomputing
            gscaled=do_gscale .and. saved_g.ne.0d0
            if (gscaled) then
               rg=g/saved_g
               rpow(0)=1d0
               do i=1,maxqcdpow
                  rpow(i)=rpow(i-1)*rg
               enddo
%(coup_gscale_lines)s
            endif
            if (gscaled) then
               call born_gscale(rpow,born_cache_ans)
%(coup_record_lines)s
               saved_g=g
            else
               calculatedBorn=.false.
            endif
         endif
      endif
      check_born_amps=.false.
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write (*,*) "momenta not the same in Born"
	       stop
            endif
         enddo
      endif
C     A repeated call: the same momenta as the last evaluation (the
C     amplitudes are cached), the same FKS configuration, and nothing
C     has touched the cached amplitudes since (BORN_CACHE_OK, reset by
C     BORN for IMODE = 0). The result, the colour density RHO and
C     AMP2/JAMP2 are still those of the last evaluation.
      IF (calculatedBorn .AND. BORN_CACHE_OK .AND. nFKSprocess.EQ.BORN_CACHE_NFKS) THEN
        ANS(:,:) = BORN_CACHE_ANS(:,:)
        RETURN
      ENDIF
      NTRY(nFKSprocess)=NTRY(nFKSprocess)+1
      if (NTRY(nFKSprocess).lt.2) then
       if (glu_ij.eq.0) then
         skip(nfksprocess)=0
       else
         skip(nFKSprocess)=1
         do while(nhel(glu_ij ,skip(nFKSprocess)).ne.-NHEL(GLU_IJ ,1))
            skip(nFKSprocess)=skip(nFKSprocess)+1
         enddo
         skip(nFKSprocess)=skip(nFKSprocess)-1
       endif
      endif
      DO JJ=1,NGRAPHS
          amp2(jj)=0d0
      ENDDO
      DO I=0,NAMPSO
      DO JJ=1,int(jamp2(0,0))
          jamp2(jj,I)=0d0
      ENDDO
      ENDDO
      if (.not.calculatedBorn) then
         do j=1,nexternal-1
            savemom(j,1)=p1(0,j)
            savemom(j,2)=p1(3,j)
            savemom4(0:3,j)=p1(0:3,j)
         enddo
%(coup_record_lines)s
         saved_g=g
         do j=1,max_bhel
            do jj=1,ngraphs
               saveamp(jj,j)=(0d0,0d0)
            enddo
         enddo
      endif
      DO I=0,NSQAMPSO
          ANS(1,I) = 0D0
          ANS(2,I) = 0D0
      ENDDO
      hel_fac=1d0
C     The JAMPs of every helicity are summed into the colour density
C     RHO (BORN_RHO_ADD); the Born and the colour-linked Borns are then
C     traces of RHO with their colour matrix. The Born of each helicity
C     is only needed on the first call, to find the good helicities.
      CALL BORN_RHO_RESET()
      IF (NTRY(nFKSprocess).LT.2) THEN
        BMODE=1
      ELSE
        BMODE=2
      ENDIF
      DO IHEL=1,NCOMB
        ! the following lines are to avoid segfaults when glu_ij=0
        cond_ij=skip(nfksprocess).eq.0
        if (.not.cond_ij) cond_ij=cond_ij.or.nhel(glu_ij,ihel).EQ.NHEL(GLU_IJ,1)
        !if (nhel(glu_ij,ihel).EQ.NHEL(GLU_IJ,1).or.skip(nfksprocess).eq.0) then
        if (cond_ij) then
          IF ((GOODHEL(IHEL,nFKSprocess) .OR. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) ) THEN

            CALL BORN(P1,NHEL(1,IHEL),IHEL,BMODE,T,borns)
            DO I=1,NSQAMPSO
                ANS(2,I)=ANS(2,I)+T(2,I)
            ENDDO
            IF (BMODE.EQ.1) THEN
            if ( borns(1,0).ne.0d0 .AND. .NOT. GOODHEL(IHEL,nFKSprocess) ) then
              GOODHEL(IHEL,nFKSprocess)=.TRUE.
            endif
            if ( borns(2,0).ne.0d0 .AND. .NOT. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) ) then
              GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess)=.TRUE.
            endif
            ENDIF
	  ENDIF
        ENDIF
      ENDDO
      CALL BORN_RHO_TRACE(0,BORNRHO)
      DO I=1,NSQAMPSO
      ANS(1,I)=BORNRHO(I)/DBLE(IDEN)
      ANS(2,I)=ANS(2,I)/DBLE(IDEN)
      ANS(1,0)=ANS(1,0)+ANS(1,I)
      ANS(2,0)=ANS(2,0)+ANS(2,I)
      ENDDO
      calculatedBorn=.true.
      BORN_CACHE_ANS(:,:)=ANS(:,:)
      BORN_CACHE_NFKS=nFKSprocess
      BORN_CACHE_OK=.TRUE.
      END


      SUBROUTINE BORN_GSCALE(RPOW,CACHE_ANS)
C
%(info_lines)s
C
C Rescales what the Born keeps from its last evaluation, at the
C strong coupling G0, to the strong coupling G (gscale), with
C RPOW(K) = (G/G0)**K. Each saved amplitude (all helicities)
C is rescaled by RPOW(AMPQCD), AMPQCD its QCD order. If the
C cached SBORN result CACHE_ANS, the colour density RHO and
C AMP2/JAMP2 are those of these amplitudes (BORN_CACHE_OK),
C they are rescaled too, each split order with its own power:
C the squared split order K by RPOW(SQSOQCD(K)), the JAMP2 of
C the amplitude split order M by RPOW(2*AMPSOQCD(M)).
C Otherwise, or if a split order has no single QCD power
C (GSCALE_FAST false), BORN_CACHE_OK is set to false and SBORN
C redoes the colour sums from the rescaled amplitudes.
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      include "nexternal.inc"
      include "born_nhel.inc"
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
      INTEGER    NCOLOR, NCF
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
      INTEGER MAXQCDPOW
      PARAMETER (MAXQCDPOW=%(maxqcdpow)d)
C
C ARGUMENTS
C
      DOUBLE PRECISION RPOW(0:MAXQCDPOW)
      COMPLEX*16 CACHE_ANS(2,0:NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,K
      INTEGER AMPQCD(NGRAPHS), AMPSOQCD(NAMPSO), SQSOQCD(NSQAMPSO)
      LOGICAL GSCALE_FAST
C
C GLOBAL VARIABLES
C
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      Double Precision amp2(ngraphs), jamp2(0:ncolor,0:nampso)
      common/to_amps_born/  amp2,       jamp2
      DOUBLE PRECISION RHO(NCF,NSQAMPSO)
      COMMON /C_BORN_RHO/ RHO
      LOGICAL BORN_CACHE_OK
      INTEGER BORN_CACHE_NFKS
      COMMON /C_BORN_CACHE/ BORN_CACHE_OK, BORN_CACHE_NFKS
%(gscale_data_lines)s
C ----------
C BEGIN CODE
C ----------
      DO I=1,NGRAPHS
        IF (AMPQCD(I).NE.0) SAVEAMP(I,:)=SAVEAMP(I,:)*RPOW(AMPQCD(I))
      ENDDO
      IF (.NOT.(BORN_CACHE_OK.AND.GSCALE_FAST)) THEN
        BORN_CACHE_OK=.FALSE.
        RETURN
      ENDIF
      DO I=1,NGRAPHS
        AMP2(I)=AMP2(I)*RPOW(2*AMPQCD(I))
      ENDDO
      DO J=1,NAMPSO
        DO I=1,NCOLOR
          JAMP2(I,J)=JAMP2(I,J)*RPOW(2*AMPSOQCD(J))
        ENDDO
      ENDDO
      CACHE_ANS(:,0)=(0D0,0D0)
      DO K=1,NSQAMPSO
        CACHE_ANS(:,K)=CACHE_ANS(:,K)*RPOW(SQSOQCD(K))
        CACHE_ANS(:,0)=CACHE_ANS(:,0)+CACHE_ANS(:,K)
        RHO(:,K)=RHO(:,K)*RPOW(SQSOQCD(K))
      ENDDO
      END


      SUBROUTINE BORN(P,NHEL,HELL,IMODE,ANS,borns)
C
%(info_lines)s
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)
C
C The amplitudes are computed by GET_AMP_BORN and the Born JAMPs by
C GET_JAMP_BORN. IMODE selects what is done with the JAMPs:
C   IMODE = 0 : the Born of this helicity (BORN_COLOR_SQUARE with the
C               Born colour matrix) in ANS(1,:) and BORNS
C   IMODE = 1 : the same, and the JAMPs are also added to the colour
C               density RHO (BORN_RHO_ADD)
C   IMODE = 2 : the JAMPs are only added to RHO; ANS(1,:) and BORNS
C               are zero
C The Born and the colour-linked Borns summed over helicities are then
C traces of RHO (BORN_RHO_TRACE). The spin-correlated Born ANS(2,:)
C pairs the two helicities of the FKS gluon, so it is computed here for
C every IMODE.

%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      include "nexternal.inc"
      include "born_nhel.inc"
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1),borns(2,0:NSQAMPSO)
      INTEGER NHEL(NEXTERNAL-1), HELL, IMODE
      COMPLEX*16 ANS(2,NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,ihel,back_hel,glu_ij,ihsave,iflip
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO), jamph(2, ncolor,nampso)
      COMPLEX*16 RES(NSQAMPSO)
C
C GLOBAL VARIABLES
C
      Double Precision amp2(%(ngraphs)d), jamp2(0:%(ncolor)d,0:nampso)
      common/to_amps_born/  amp2,       jamp2
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      integer step_hel
      logical cond_ij
      logical force_ijglu_zero
      common /to_force_ijglu/ force_ijglu_zero
      LOGICAL BORN_CACHE_OK
      INTEGER BORN_CACHE_NFKS
      COMMON /C_BORN_CACHE/ BORN_CACHE_OK, BORN_CACHE_NFKS

%(ij_lines)s
C ----------
C BEGIN CODE
C ----------
C     a single-helicity call (SBORN_ONEHEL) adds to AMP2/JAMP2 and may
C     overwrite saved amplitudes: the cached SBORN result is stale
      IF (IMODE.EQ.0) BORN_CACHE_OK=.FALSE.
      borns(:,:) =0d0
      ans(:,:) = (0d0, 0d0)
      jamph(:,:,:) = (0d0, 0d0)

      glu_ij = ij_values(nfksprocess)
      if (force_ijglu_zero) glu_ij = 0
      if (glu_ij.ne.0) then
        back_hel = nhel(glu_ij)
        if (back_hel.ne.0) then
          step_hel=-2*back_hel
        else
          step_hel=1
        endif
      else
        back_hel=0
        step_hel=1
      endif
      DO IHEL=back_hel,-back_hel,step_hel
        if (glu_ij.ne.0) then
          cond_ij=IHEL.EQ.back_hel.OR.NHEL(GLU_IJ).NE.0
        else
          cond_ij=IHEL.EQ.back_hel
        endif
        IF (cond_ij) THEN
        if (glu_ij.ne.0) then
          if (nhel(glu_ij).ne.0) nhel(glu_ij) = ihel
        endif
C       helicity slot of the saved amplitudes: hell for the helicity
C       passed in, hell+skip for the flipped one of the FKS gluon
        if(ihel.eq.back_hel)then
          ihsave=hell
        elseif(ihel.eq.-back_hel)then
          ihsave=hell+skip(nFKSprocess)
        else
          write(*,*) "ERROR #1 in born.f"
          stop
        endif
        if (.not. calculatedBorn) then
          CALL GET_AMP_BORN(P,NHEL,AMP)
          saveamp(1:ngraphs,ihsave)=amp(1:ngraphs)
        else
          amp(1:ngraphs)=saveamp(1:ngraphs,ihsave)
        endif
        CALL GET_JAMP_BORN(AMP,JAMP)
        iflip=2-(1+back_hel*ihel)/2
        IF (IMODE.GE.1) CALL BORN_RHO_ADD(JAMP)
        IF (IMODE.LE.1) THEN
C         the Born of this helicity: the Born colour matrix
          CALL BORN_COLOR_SQUARE(0,JAMP,RES)
          DO I = 1, NSQAMPSO
            BORNS(iflip,I)=BORNS(iflip,I)+DBLE(RES(I))
          ENDDO
        ENDIF
        Do I = 1, NGRAPHS
          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
        Enddo
        do J = 1,NAMPSO
        Do I = 1, NCOLOR
          Jamp2(i,J)=Jamp2(i,J)+Jamp(i,J)*dconjg(Jamp(i,J))
          Jamph(iflip,i,J)=Jamp(i,J)
        Enddo
        ENDDO
      endif
      Enddo
      do i = 1, nsqampso
      borns(1,0)=borns(1,0)+borns(1,i)
      borns(2,0)=borns(2,0)+borns(2,i)
      ans(1,i) = borns(1,i) + borns(2,i)
      enddo
C     the spin-correlated Born: the Born colour matrix between the JAMPs
C     of the two helicities of the FKS gluon
      CALL BORN_COLOR_CONTRACT(0,JAMPH(2,:,:),JAMPH(1,:,:),RES)
      ANS(2,:) = RES(:)
      if (glu_ij.ne.0) nhel(glu_ij) = back_hel
      END


      SUBROUTINE GET_AMP_BORN(P,NHEL,AMP)
C
%(info_lines)s
C Returns the Born amplitudes AMP(NGRAPHS) for the helicity
C configuration NHEL (the HELAS calls)
%(process_lines)s
C
      USE ALOHA_OBJECT
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS,    NEIGEN
      PARAMETER (NGRAPHS=   %(ngraphs)d,NEIGEN=  1)
      INTEGER    NWAVEFUNCS
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "coupl.inc"
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1)
      COMPLEX*16 AMP(NGRAPHS)
C
C LOCAL VARIABLES
C
      INTEGER IC(NEXTERNAL-1),nmo
      parameter (nmo=nexternal-1)
      data ic /nmo*1/
      INTEGER FLAVOR(NEXTERNAL-1)
      DATA FLAVOR /NMO*1/
      type(aloha) W(NWAVEFUNCS)
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
      END


      SUBROUTINE GET_JAMP_BORN(AMP,JAMP)
C
%(info_lines)s
C Returns the Born JAMPs (colour-ordered amplitudes, in the Born colour
C basis) for each amplitude split order
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
C
C ARGUMENTS
C
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
C
C LOCAL VARIABLES
C
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
C ----------
C BEGIN CODE
C ----------
      jamp(:,:) = (0d0,0d0)
%(jamp_lines)s
      END


      SUBROUTINE BORN_COLOR_CONTRACT(ILINK,JAMPA,JAMPB,RES)
C
%(info_lines)s
C Contracts two sets of Born JAMPs with one of the colour matrices, all
C expressed in the Born colour basis:
C   ILINK = 0      : the Born colour matrix
C   ILINK = 1..NLINKS : the colour-linked Born for the ILINK-th link
C                    (the same numbering as sborn_sf and born_links.dat)
C For each squared split order it returns
C   RES = sum_ij S_ij (JAMPA(j,M) conj(JAMPB(i,N))
C                      + JAMPA(i,M) conj(JAMPB(j,N)))/2
C i.e. the symmetrised interference needed for the spin-correlated Born.
C For JAMPA = JAMPB use the cheaper BORN_COLOR_SQUARE, or, summed over
C helicities, BORN_RHO_TRACE. The matrices are symmetric: only the
C upper triangle is stored, with the off-diagonal entries doubled, as
C integers over DENOM(ILINK) (BLOCK DATA BORN_COLOR_DATA).
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR, NCF, NLINKS
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
      PARAMETER (NLINKS=%(nlinks)d)
C
C ARGUMENTS
C
      INTEGER ILINK
      COMPLEX*16 JAMPA(NCOLOR,NAMPSO), JAMPB(NCOLOR,NAMPSO)
      COMPLEX*16 RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N,CF_INDEX
      COMPLEX*16 ZTEMP
      INTEGER CF(NCF,0:NLINKS), DENOM(0:NLINKS)
      COMMON /C_BORN_COLOR_MATRICES/ CF, DENOM
      INTEGER SQSO(NAMPSO,NAMPSO)
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      SAVE SQSO, FIRSTTIME
C
C FUNCTION
C
      INTEGER SQSOINDEXB
C ----------
C BEGIN CODE
C ----------
C     the squared split-order index of each (M,N) pair, looked up once
      IF (FIRSTTIME) THEN
        DO M = 1, NAMPSO
          DO N = 1, NAMPSO
            SQSO(M,N) = SQSOINDEXB(M,N)
          ENDDO
        ENDDO
        FIRSTTIME = .FALSE.
      ENDIF
      RES(:) = (0d0,0d0)
      DO M = 1, NAMPSO
        DO N = 1, NAMPSO
          ZTEMP = (0d0,0d0)
          CF_INDEX = 0
          DO I = 1, NCOLOR
            DO J = I, NCOLOR
              CF_INDEX = CF_INDEX + 1
              IF (CF(CF_INDEX,ILINK).NE.0) ZTEMP = ZTEMP + CF(CF_INDEX,ILINK)*(JAMPA(J,M)*DCONJG(JAMPB(I,N))+JAMPA(I,M)*DCONJG(JAMPB(J,N)))
            ENDDO
          ENDDO
          RES(SQSO(M,N)) = RES(SQSO(M,N)) + ZTEMP
        ENDDO
      ENDDO
      RES(:) = RES(:)/(2d0*DENOM(ILINK))
      END


      SUBROUTINE BORN_COLOR_SQUARE(ILINK,JAMP,RES)
C
%(info_lines)s
C conj(JAMP).S.JAMP for each squared split order, S being the Born colour
C matrix (ILINK = 0) or the one of the ILINK-th colour link, all in the
C Born colour basis. Only the real part of RES is meaningful.
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR, NCF, NLINKS
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
      PARAMETER (NLINKS=%(nlinks)d)
C
C ARGUMENTS
C
      INTEGER ILINK
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N,CF_INDEX
      COMPLEX*16 ZTEMP, ZSUM, ZT(NCOLOR)
      INTEGER CF(NCF,0:NLINKS), DENOM(0:NLINKS)
      COMMON /C_BORN_COLOR_MATRICES/ CF, DENOM
      INTEGER SQSO(NAMPSO,NAMPSO)
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      SAVE SQSO, FIRSTTIME
C
C FUNCTION
C
      INTEGER SQSOINDEXB
C ----------
C BEGIN CODE
C ----------
C     the squared split-order index of each (M,N) pair, looked up once
      IF (FIRSTTIME) THEN
        DO M = 1, NAMPSO
          DO N = 1, NAMPSO
            SQSO(M,N) = SQSOINDEXB(M,N)
          ENDDO
        ENDDO
        FIRSTTIME = .FALSE.
      ENDIF
      RES(:) = (0d0,0d0)
      DO M = 1, NAMPSO
C       ZT = S.JAMP(:,M), using the packed upper triangle
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0d0,0d0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX + 1
            ZTEMP = ZTEMP + CF(CF_INDEX,ILINK)*JAMP(J,M)
          ENDDO
          ZT(I) = ZTEMP
        ENDDO
        DO N = 1, NAMPSO
          ZSUM = (0d0,0d0)
          DO I = 1, NCOLOR
            ZSUM = ZSUM + ZT(I)*DCONJG(JAMP(I,N))
          ENDDO
          RES(SQSO(M,N)) = RES(SQSO(M,N)) + ZSUM
        ENDDO
      ENDDO
      RES(:) = RES(:)/DENOM(ILINK)
      END


      SUBROUTINE BORN_RHO_RESET()
C
%(info_lines)s
C Sets the colour density RHO of the Born JAMPs (BORN_RHO_ADD) to zero.
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR, NCF
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
C
C GLOBAL VARIABLES
C
      DOUBLE PRECISION RHO(NCF,NSQAMPSO)
      COMMON /C_BORN_RHO/ RHO
C ----------
C BEGIN CODE
C ----------
      RHO(:,:) = 0d0
      END


      SUBROUTINE BORN_RHO_ADD(JAMP)
C
%(info_lines)s
C Adds the Born JAMPs of one helicity to the colour density
C   RHO(ij,K) = sum_hel sum_(M,N) Re(JAMP(j,M) conj(JAMP(i,N)))
C for i <= j (packed as the colour matrices), the sum running over the
C pairs of amplitude split orders (M,N) of the squared split order K.
C All the colour matrices S are real and symmetric, so the helicity sum
C of conj(JAMP).S.JAMP is the trace sum_ij S_ij RHO(ij,K)
C (BORN_RHO_TRACE): the sum over helicities is done once, for the Born
C and for all the colour-linked Borns.
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR, NCF
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
C
C ARGUMENTS
C
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N,K,CF_INDEX
      DOUBLE PRECISION RHO(NCF,NSQAMPSO)
      COMMON /C_BORN_RHO/ RHO
      INTEGER SQSO(NAMPSO,NAMPSO)
      LOGICAL FIRSTTIME
      DATA FIRSTTIME /.TRUE./
      SAVE SQSO, FIRSTTIME
C
C FUNCTION
C
      INTEGER SQSOINDEXB
C ----------
C BEGIN CODE
C ----------
C     the squared split-order index of each (M,N) pair, looked up once
      IF (FIRSTTIME) THEN
        DO M = 1, NAMPSO
          DO N = 1, NAMPSO
            SQSO(M,N) = SQSOINDEXB(M,N)
          ENDDO
        ENDDO
        FIRSTTIME = .FALSE.
      ENDIF
      DO M = 1, NAMPSO
C       (M,M): Re(JAMP(j,M) conj(JAMP(i,M))) is symmetric in i,j
        K = SQSO(M,M)
        CF_INDEX = 0
        DO I = 1, NCOLOR
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX + 1
            RHO(CF_INDEX,K) = RHO(CF_INDEX,K) + DBLE(JAMP(J,M))*DBLE(JAMP(I,M)) + DIMAG(JAMP(J,M))*DIMAG(JAMP(I,M))
          ENDDO
        ENDDO
C       (M,N) and (N,M) for N > M share the squared split order
        DO N = M+1, NAMPSO
          K = SQSO(M,N)
          CF_INDEX = 0
          DO I = 1, NCOLOR
            DO J = I, NCOLOR
              CF_INDEX = CF_INDEX + 1
              RHO(CF_INDEX,K) = RHO(CF_INDEX,K) + DBLE(JAMP(J,M))*DBLE(JAMP(I,N)) + DIMAG(JAMP(J,M))*DIMAG(JAMP(I,N)) + DBLE(JAMP(I,M))*DBLE(JAMP(J,N)) + DIMAG(JAMP(I,M))*DIMAG(JAMP(J,N))
            ENDDO
          ENDDO
        ENDDO
      ENDDO
      END


      SUBROUTINE BORN_RHO_TRACE(ILINK,RES)
C
%(info_lines)s
C The Born (ILINK = 0) or the ILINK-th colour-linked Born, summed over
C the helicities added to the colour density RHO since the last
C BORN_RHO_RESET, for each squared split order (not divided by IDEN):
C   RES(K) = sum_ij S_ij RHO(ij,K)
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NCOLOR, NCF, NLINKS
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
      PARAMETER (NLINKS=%(nlinks)d)
C
C ARGUMENTS
C
      INTEGER ILINK
      DOUBLE PRECISION RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER K,CF_INDEX
      DOUBLE PRECISION TEMP
      INTEGER CF(NCF,0:NLINKS), DENOM(0:NLINKS)
      COMMON /C_BORN_COLOR_MATRICES/ CF, DENOM
      DOUBLE PRECISION RHO(NCF,NSQAMPSO)
      COMMON /C_BORN_RHO/ RHO
C ----------
C BEGIN CODE
C ----------
      DO K = 1, NSQAMPSO
        TEMP = 0d0
        DO CF_INDEX = 1, NCF
          TEMP = TEMP + CF(CF_INDEX,ILINK)*RHO(CF_INDEX,K)
        ENDDO
        RES(K) = TEMP/DENOM(ILINK)
      ENDDO
      END


      BLOCK DATA BORN_COLOR_DATA
C The colour matrices in the Born colour basis: index 0 is the Born one,
C 1..NLINKS the colour links (upper triangle, off-diagonal entries doubled,
C integers over DENOM)
      IMPLICIT NONE
      INTEGER    NCOLOR, NCF, NLINKS
      PARAMETER (NCOLOR=%(ncolor)d, NCF=NCOLOR*(NCOLOR+1)/2)
      PARAMETER (NLINKS=%(nlinks)d)
      INTEGER I
      INTEGER CF(NCF,0:NLINKS), DENOM(0:NLINKS)
      COMMON /C_BORN_COLOR_MATRICES/ CF, DENOM
      LOGICAL BORN_CACHE_OK
      INTEGER BORN_CACHE_NFKS
      COMMON /C_BORN_CACHE/ BORN_CACHE_OK, BORN_CACHE_NFKS
      DATA BORN_CACHE_OK /.FALSE./, BORN_CACHE_NFKS /0/
      LOGICAL CHECK_BORN_AMPS, BORN_CHECKS_AMPS
      COMMON /C_BORN_AMPS_CHECK/ CHECK_BORN_AMPS, BORN_CHECKS_AMPS
      DATA CHECK_BORN_AMPS /.FALSE./
      DATA BORN_CHECKS_AMPS /%(born_checks_amps)s/
%(color_link_data_lines)s
      END


      SUBROUTINE SB_SF_LINK(P,ILINK,ANS_SUMMED)
C
%(info_lines)s
C
C     The ILINK-th colour-linked Born, summed over the split orders
C     required for the counterterms (the values per split order are
C     stored in AMP_SPLIT_CNT). It is the trace of the colour density
C     left by the last Born call with the colour matrix of the link
C     (BORN_RHO_TRACE), so SBORN must have been called before at the
C     same momenta.
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
      include 'nexternal.inc'
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL-1), ANS_SUMMED
      INTEGER ILINK
C
C VARIABLES
C
      INTEGER I,J,IDEN
      include 'orders.inc'
      REAL*8 ANS(0:NSQAMPSO), RES(NSQAMPSO)
      LOGICAL KEEP_ORDER_CNT(NSPLITORDERS, NSQAMPSO)
      common /c_keep_order_cnt/ keep_order_cnt
      integer amp_orders(nsplitorders)
      double precision tiny
      parameter (tiny = 1d-12)
      double precision max_val
%(den_factor_lines)s
C
C GLOBAL VARIABLES
C
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      LOGICAL CHECK_BORN_AMPS, BORN_CHECKS_AMPS
      COMMON /C_BORN_AMPS_CHECK/ CHECK_BORN_AMPS, BORN_CHECKS_AMPS
C
C     FUNCTIONS
C
      integer GETORDPOWFROMINDEX_B
      integer orders_to_amp_split_pos
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p(0,j) .or. savemom(j,2).ne.p(3,j)) then
               calculatedBorn=.false.
               write(*,*) 'Error in sb_sf: momenta not the same in the born'
               stop
            endif
         enddo
      else
         write(*,*) 'Error in sb_sf: color_linked borns should be called only with calculatedborn = true'
         stop
      endif
      if (check_born_amps) then
         write(*,*) 'Error in sb_sf: the Born must be called again first'
         stop
      endif
      ANS(:) = 0D0
      CALL BORN_RHO_TRACE(ILINK,RES)
      DO I=1,NSQAMPSO
        ANS(I)=RES(I)/DBLE(IDEN)
        ANS(0)=ANS(0)+ANS(I)
      ENDDO

C color-linked borns are called for QCD-type emissions
      ans_summed = 0d0
      max_val = 0d0

C reset the amp_split_cnt array
      amp_split_cnt(1:amp_split_size,1:2,1:nsplitorders) = dcmplx(0d0,0d0)

      do i = 1, nsqampso
        max_val = max(max_val, abs(ans(I)))
      enddo

      do i = 1, nsqampso
        if (keep_order_cnt(qcd_pos, i)) then
         ANS_SUMMED = ans_summed + ANS(I)
         do j = 1, nsplitorders
          amp_orders(j) = GETORDPOWFROMINDEX_B(j, i)
C take into account the fact that this is for QCD
          if (j.eq.qcd_pos) amp_orders(j) = amp_orders(j) + 2
         enddo
         if(abs(ans(i)).gt.max_val*tiny) amp_split_cnt(orders_to_amp_split_pos(amp_orders),1,qcd_pos) = ans(i)
        endif
      enddo

C this is to avoid fake non-zero contributions
      if (abs(ans_summed).lt.max_val*tiny) ans_summed=0d0

      return
      END


      BLOCK DATA GOODHELS
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      DATA GOODHEL/THEL*.FALSE./
      end



C
C Helper functions to deal with the split orders.
C

      INTEGER FUNCTION SQSOINDEXB(AMPORDERA,AMPORDERB)
C
C This functions plays the role of the interference matrix. It can be hardcoded or
C made more elegant using hashtables if its execution speed ever becomes a relevant
C factor. From two split order indices of the jamps, it return the corresponding
C index in the squared order canonical ordering.
C
C CONSTANTS
C
      implicit none
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER AMPORDERA, AMPORDERB
C
C LOCAL VARIABLES
C
      INTEGER I, SQORDERS(NSPLITORDERS)
      INTEGER AMPSPLITORDERS(NAMPSO,NSPLITORDERS)
	  %(ampsplitorders)s
C
C FUNCTION
C
      INTEGER SQSOINDEXB_FROM_ORDERS
C
C BEGIN CODE
C
      DO I=1,NSPLITORDERS
	    SQORDERS(I)=AMPSPLITORDERS(AMPORDERA,I)+AMPSPLITORDERS(AMPORDERB,I)
	  ENDDO
	  SQSOINDEXB=SQSOINDEXB_FROM_ORDERS(SQORDERS)
	  END



      INTEGER FUNCTION SQSOINDEXB_FROM_ORDERS(ORDERS)
C
C From a list of values for the split orders, this function returns the
c corresponding index in the squared orders canonical ordering.
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER ORDERS(NSPLITORDERS)
C
C LOCAL VARIABLES
C
      INTEGER I,J
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      DO I=1,NSQAMPSO
	    DO J=1,NSPLITORDERS
		  IF (ORDERS(J).NE.SQSPLITORDERS(I,J)) GOTO 1009
		ENDDO
		SQSOINDEXB_FROM_ORDERS = I
		RETURN
1009    CONTINUE
	  ENDDO

	  WRITE(*,*) 'ERROR:: Stopping function sqsoindex_from_orders'
	  WRITE(*,*) 'Could not find squared orders ',(ORDERS(I),I=1,NSPLITORDERS)
	  STOP

	  END



      INTEGER FUNCTION GETORDPOWFROMINDEX_B(IORDER, INDX)
C
C Return the power of the IORDER-th order appearing at position INDX
C in the split-orders output
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSPLITORDERS
	  PARAMETER (NSPLITORDERS=%(nSplitOrders)d)
C
C ARGUMENTS
C
	  INTEGER IORDER, INDX
C
C LOCAL VARIABLES
C
      integer i
      INTEGER SQSPLITORDERS(NSQAMPSO,NSPLITORDERS)
%(sqsplitorders)s
C
C BEGIN CODE
C
      IF (IORDER.GT.NSPLITORDERS.OR.IORDER.LT.1) THEN
      WRITE(*,*) "INVALID IORDER B", IORDER
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSPLITORDERS
      STOP
      ENDIF

      IF (INDX.GT.NSQAMPSO.OR.INDX.LT.1) THEN
      WRITE(*,*) "INVALID INDX B", INDX
      WRITE(*,*) "SHOULD BE BETWEEN 1 AND ", NSQAMPSO
      STOP
      ENDIF

      GETORDPOWFROMINDEX_B=SQSPLITORDERS(INDX, IORDER)
      END


      SUBROUTINE GET_NSQSO_B(NSQSO)
C
C     Simple subroutine returning the number of squared split order
C     contributions returned in ANS when calling SMATRIX_SPLITORDERS
C
      implicit none
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
	  INTEGER NSQSO

	  NSQSO=NSQAMPSO

      END


