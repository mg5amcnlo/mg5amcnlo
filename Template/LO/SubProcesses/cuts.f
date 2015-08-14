      logical function pass_point(p)
c************************************************************************
c     This function is called from sample to see if it needs to 
c     bother calculating the weight from all the different conficurations
c     You can either just return true, or have it call passcuts
c************************************************************************
      implicit none
c
c     Arguments
c
      double precision p
c
c     External
c
      logical passcuts
      external passcuts
c-----
c  Begin Code
c-----
      pass_point = .true.
c      pass_point = passcuts(p)
      end
C 
      LOGICAL FUNCTION PASSCUTS(P)
C**************************************************************************
C     INPUT:
C            P(0:3,1)           MOMENTUM OF INCOMING PARTON
C            P(0:3,2)           MOMENTUM OF INCOMING PARTON
C            P(0:3,3)           MOMENTUM OF ...
C            ALL MOMENTA ARE IN THE REST FRAME!!
C            COMMON/JETCUTS/   CUTS ON JETS
C     OUTPUT:
C            TRUE IF EVENTS PASSES ALL CUTS LISTED
C**************************************************************************
      IMPLICIT NONE
c
c     Constants
c
      include 'genps.inc'
      include 'nexternal.inc'
C
C     ARGUMENTS
C
      REAL*8 P(0:3,nexternal)

C
C     LOCAL
C
      LOGICAL FIRSTTIME,FIRSTTIME2,pass_bw,notgood,good,foundheavy
      LOGICAL DEBUG
      integer i,j,njets,nheavyjets,nleptons,hardj1,hardj2
      REAL*8 XVAR,ptmax1,ptmax2,htj,tmp,inclht
      real*8 ptemp(0:3), ptemp2(0:3)
      character*20 formstr
C
C     PARAMETERS
C
      real*8 PI
      parameter( PI = 3.14159265358979323846d0 )
C
C     EXTERNAL
C
      REAL*8 R2,DOT,ET,RAP,DJ,SumDot,pt,ALPHAS,PtDot
      logical cut_bw,setclscales
      external R2,DOT,ET,RAP,DJ,SumDot,pt,ALPHAS,cut_bw,setclscales,PtDot
C
C     GLOBAL
C
      include 'run.inc'
      include 'cuts.inc'

      double precision ptjet(nexternal)
      double precision ptheavyjet(nexternal)
      double precision ptlepton(nexternal)
      double precision temp

C VARIABLES TO SPECIFY JETS
      DOUBLE PRECISION PJET(NEXTERNAL,0:3)
      DOUBLE PRECISION PTMIN
      DOUBLE PRECISION PT1,PT2
      INTEGER K,J1,J2

C VARIABLES FOR KT CUT
      DOUBLE PRECISION PTNOW,COSTH,PABS1,PABS2
      DOUBLE PRECISION ETA1,ETA2,COSH_DETA,COS_DPHI,KT1SQ,KT2SQ, DPHI

      double precision etmin(nincoming+1:nexternal),etamax(nincoming+1:nexternal)
      double precision emin(nincoming+1:nexternal)
      double precision                    r2min(nincoming+1:nexternal,nincoming+1:nexternal)
      double precision s_min(nexternal,nexternal)
      double precision etmax(nincoming+1:nexternal),etamin(nincoming+1:nexternal)
      double precision emax(nincoming+1:nexternal)
      double precision r2max(nincoming+1:nexternal,nincoming+1:nexternal)
      double precision s_max(nexternal,nexternal)
      double precision ptll_min(nexternal,nexternal),ptll_max(nexternal,nexternal)
      double precision inclHtmin,inclHtmax
      common/to_cuts/  etmin, emin, etamax, r2min, s_min,
     $     etmax, emax, etamin, r2max, s_max, ptll_min, ptll_max, inclHtmin,inclHtmax

      double precision ptjmin4(4),ptjmax4(4),htjmin4(2:4),htjmax4(2:4)
      logical jetor
      common/to_jet_cuts/ ptjmin4,ptjmax4,htjmin4,htjmax4,jetor

      double precision ptlmin4(4),ptlmax4(4)
      common/to_lepton_cuts/ ptlmin4,ptlmax4

c
c     Special cuts
c

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw
C
C     SPECIAL CUTS
C
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL),IS_A_ONIUM(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      logical  do_cuts(nexternal)
      COMMON /TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_NU,IS_HEAVY,
     . IS_A_ONIUM, do_cuts
C
C     Keep track of whether cuts already calculated for this event
C
      LOGICAL CUTSDONE,CUTSPASSED
      COMMON/TO_CUTSDONE/CUTSDONE,CUTSPASSED
      DATA CUTSDONE,CUTSPASSED/.FALSE.,.FALSE./

C $B$ MW_NEW_DEF $E$ !this is a tag for MadWeight

      double precision xqcutij(nexternal,nexternal),xqcuti(nexternal)
      common/to_xqcuts/xqcutij,xqcuti

c jet cluster algorithm
      integer nQCD !,NJET,JET(nexternal)
c      double precision plab(0:3, nexternal)
      double precision pQCD(0:3,nexternal)!,PJET(0:3,nexternal)
c      double precision rfj,sycut,palg,fastjetdmerge
c      integer njet_eta
c     Photon isolation
      integer nph,nem,nin
      double precision ptg,chi_gamma_iso,iso_getdrv40
      double precision Etsum(0:nexternal)
      real drlist(nexternal)
      double precision pgamma(0:3,nexternal),pem(0:3,nexternal)
      logical alliso
C     Sort array of results: ismode>0 for real, isway=0 for ascending order
      integer ismode,isway,izero,isorted(nexternal)
      parameter (ismode=1)
      parameter (isway=0)
      parameter (izero=0)

      include 'coupl.inc'
C
C
c
      DATA FIRSTTIME,FIRSTTIME2/.TRUE.,.TRUE./

c put momenta in common block for couplings.f
      double precision pp(0:3,max_particles)
      common /momenta_pp/pp

      DATA DEBUG/.FALSE./

C-----
C  BEGIN CODE
C-----



      PASSCUTS=.TRUE.             !EVENT IS OK UNLESS OTHERWISE CHANGED
      IF (FIRSTTIME) THEN
         FIRSTTIME=.FALSE.
c      Preparation for reweighting by setting up clustering by diagrams
         call initcluster()
c
c
         write(formstr,'(a,i2.2,a)')'(a10,',nexternal,'i8)'
         write(*,formstr) 'Particle',(i,i=nincoming+1,nexternal)
         write(formstr,'(a,i2.2,a)')'(a10,',nexternal,'f8.1)'
         write(*,formstr) 'Et >',(etmin(i),i=nincoming+1,nexternal)
         write(*,formstr) 'E >',(emin(i),i=nincoming+1,nexternal)
         write(*,formstr) 'Eta <',(etamax(i),i=nincoming+1,nexternal)
         write(*,formstr) 'xqcut: ',(xqcuti(i),i=nincoming+1,nexternal)
         write(formstr,'(a,i2.2,a)')'(a,i2,a,',nexternal,'f8.1)'
         do j=nincoming+1,nexternal-1
            write(*,formstr) 'd R #',j,'  >',(-0.0,i=nincoming+1,j),
     &           (r2min(i,j),i=j+1,nexternal)
            do i=j+1,nexternal
               r2min(i,j)=r2min(i,j)*dabs(r2min(i,j))    !Since r2 returns distance squared
               r2max(i,j)=r2max(i,j)*dabs(r2max(i,j))
            enddo
         enddo
         do j=nincoming+1,nexternal-1
            write(*,formstr) 's min #',j,'>',
     &           (s_min(i,j),i=nincoming+1,nexternal)
         enddo
         do j=nincoming+1,nexternal-1
            write(*,formstr) 'xqcutij #',j,'>',
     &           (xqcutij(i,j),i=nincoming+1,nexternal)
         enddo

cc
cc     Set the strong coupling
cc
c         call set_ren_scale(P,scale)
c
cc     Check that the user funtions for setting the scales
cc     have been edited if the choice of an event-by-event
cc     scale choice has been made 
c
c         if(.not.fixed_ren_scale) then
c            if(scale.eq.0d0) then
c               write(6,*) 
c               write(6,*) '* >>>>>>>>>ERROR<<<<<<<<<<<<<<<<<<<<<<<*'
c               write(6,*) ' Dynamical renormalization scale choice '
c               write(6,*) ' selected but user subroutine' 
c               write(6,*) ' set_ren_scale not edited in file:setpara.f'
c               write(6,*) ' Switching to a fixed_ren_scale choice'
c               write(6,*) ' with scale=zmass'
c               scale=91.2d0
c               write(6,*) 'scale=',scale
c               fixed_ren_scale=.true.
c               call set_ren_scale(P,scale)
c            endif
c         endif
         
c         if(.not.fixed_fac_scale) then
c            call set_fac_scale(P,q2fact)
c            if(q2fact(1).eq.0d0.or.q2fact(2).eq.0d0) then
c               write(6,*) 
c               write(6,*) '* >>>>>>>>>ERROR<<<<<<<<<<<<<<<<<<<<<<<*'
c               write(6,*) ' Dynamical renormalization scale choice '
c               write(6,*) ' selected but user subroutine' 
c               write(6,*) ' set_fac_scale not edited in file:setpara.f'
c               write(6,*) ' Switching to a fixed_fac_scale choice'
c               write(6,*) ' with q2fact(i)=zmass**2'
c               fixed_fac_scale=.true.
c               q2fact(1)=91.2d0**2
c               q2fact(2)=91.2d0**2
c               write(6,*) 'scales=',q2fact(1),q2fact(2)
c            endif
c         endif

         if(fixed_ren_scale) then
            G = SQRT(4d0*PI*ALPHAS(scale))
            call update_as_param()
         endif

c     Put momenta in the common block to zero to start
         do i=0,3
            do j=1,max_particles
               pp(i,j) = 0d0
            enddo
         enddo
         
      ENDIF ! IF FIRSTTIME

      IF (CUTSDONE) THEN
         PASSCUTS=CUTSPASSED
         RETURN
      ENDIF
      CUTSDONE=.TRUE.
c      CUTSPASSED=.FALSE.

c
c     Make sure have reasonable 4-momenta
c
      if (p(0,1) .le. 0d0) then
         passcuts=.false.
         return
      endif

c     Also make sure there's no INF or NAN
      do i=1,nexternal
         do j=0,3
            if(p(j,i).gt.1d32.or.p(j,i).ne.p(j,i))then
               passcuts=.false.
               return
            endif
         enddo
      enddo
      
c
c     Limit S_hat
c
c      if (x1*x2*stot .gt. 500**2) then
c         passcuts=.false.
c         return
c      endif

C $B$ DESACTIVATE_CUT $E$ !This is a tag for MadWeight

      if(debug) write (*,*) '============================='
      if(debug) write (*,*) ' EVENT STARTS TO BE CHECKED  '
      if(debug) write (*,*) '============================='
c     
c     p_t min & max cuts
c     
      do i=nincoming+1,nexternal
         if(debug) write (*,*) 'pt(',i,')=',pt(p(0,i)),'   ',etmin(i),
     $        ':',etmax(i)
         notgood=(pt(p(0,i)) .lt. etmin(i)).or.
     &        (etmax(i).ge.0d0.and.pt(p(0,i)) .gt. etmax(i))
         if (notgood) then
            if(debug) write (*,*) i,' -> fails'
            passcuts=.false.
            return
         endif
      enddo
c
c    missing ET min & max cut + Invariant mass of leptons and neutrino 
c    nb: missing Et defined as the vector sum over the neutrino's pt
c
c-- reset ptemp(0:3)
      do j=0,3
         ptemp(j)=0 ! for the neutrino
         ptemp2(j)=0 ! for the leptons
      enddo
c-  sum over the momenta
      do i=nincoming+1,nexternal
         if(is_a_nu(i)) then            
         if(debug) write (*,*) i,' -> neutrino '
            do j=0,3
               ptemp(j)=ptemp(j)+p(j,i)
            enddo
         elseif(is_a_l(i)) then            
         if(debug) write (*,*) i,' -> lepton '
            do j=0,3
               ptemp2(j)=ptemp2(j)+p(j,i)
            enddo
         endif

      enddo
c-  check the et
      if(debug.and.ptemp(0).eq.0d0) write (*,*) 'No et miss in event'
      if(debug.and.ptemp(0).gt.0d0) write (*,*) 'Et miss =',pt(ptemp(0)),'   ',misset,':',missetmax
      if(debug.and.ptemp2(0).eq.0d0) write (*,*) 'No leptons in event'
      if(debug.and.ptemp(0).gt.0d0) write (*,*) 'Energy of leptons =',pt(ptemp2(0))
      if(ptemp(0).gt.0d0) then
         notgood=(pt(ptemp(0)) .lt. misset).or.
     &        (missetmax.ge.0d0.and.pt(ptemp(0)) .gt. missetmax)
         if (notgood) then
            if(debug) write (*,*) ' missing et cut -> fails'
            passcuts=.false.
            return
         endif
      endif
      if (mmnl.gt.0d0.or.mmnlmax.ge.0d0)then
         if(dsqrt(SumDot(ptemp,ptemp2,1d0)).lt.mmnl.or.mmnlmax.ge.0d0.and.dsqrt(SumDot(ptemp, ptemp2,1d0)).gt.mmnlmax) then
            if(debug) write (*,*) 'lepton invariant mass -> fails'
            passcuts=.false.
            return
         endif
      endif
c
c     pt cut on heavy particles
c     gives min(pt) for (at least) one heavy particle
c
      if(ptheavy.gt.0d0)then
         passcuts=.false.
         foundheavy=.false.
         do i=nincoming+1,nexternal
            if(is_heavy(i)) then            
               if(debug) write (*,*) i,' -> heavy '
               foundheavy=.true.
               if(pt(p(0,i)).gt.ptheavy) passcuts=.true.
            endif
         enddo
         
         if(.not.passcuts.and.foundheavy)then
            if(debug) write (*,*) ' heavy particle cut -> fails'
            return
         else
            passcuts=.true.
         endif
      endif
c     
c     E min & max cuts
c     
      do i=nincoming+1,nexternal
         if(debug) write (*,*) 'p(0,',i,')=',p(0,i),'   ',emin(i),':',emax(i)
         notgood=(p(0,i) .le. emin(i)).or.
     &        (emax(i).ge.0d0 .and. p(0,i) .gt. emax(i))
         if (notgood) then
            if(debug) write (*,*) i,' -> fails'
            passcuts=.false.
            return
         endif
      enddo
c     
c     Rapidity  min & max cuts
c     
      do i=nincoming+1,nexternal
         if(debug) write (*,*) 'abs(rap(',i,'))=',abs(rap(p(0,i))),'   ',etamin(i),':',etamax(i)
         notgood=(etamax(i).ge.0.and.abs(rap(p(0,i))) .gt. etamax(i)).or.
     &        (abs(rap(p(0,i))) .lt. etamin(i))
         if (notgood) then
            if(debug) write (*,*) i,' -> fails'
            passcuts=.false.
            return
         endif
      enddo
c     
c     DeltaR min & max cuts
c     
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            if(debug) write (*,*) 'r2(',i, ',' ,j,')=',dsqrt(r2(p(0,i),p(0,j)))
            if(debug) write (*,*) dsqrt(r2min(j,i)),dsqrt(r2max(j,i))
            if(r2min(j,i).gt.0.or.r2max(j,i).ge.0d0) then
               tmp=r2(p(0,i),p(0,j))
               notgood=(tmp .lt. r2min(j,i)).or.
     $              (r2max(j,i).ge.0d0 .and. tmp .gt. r2max(j,i))
               if (notgood) then
                  if(debug) write (*,*) i,j,' -> fails'
                  passcuts=.false.
                  return
               endif
            endif
         enddo
      enddo


c     s-channel min & max pt of sum of 4-momenta
c     
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            if(debug)write (*,*) 'ptll(',i,',',j,')=',PtDot(p(0,i),p(0,j))
            if(debug)write (*,*) ptll_min(j,i),ptll_max(j,i)
            if(ptll_min(j,i).gt.0.or.ptll_max(j,i).ge.0d0) then
               tmp=PtDot(p(0,i),p(0,j))
               notgood=(tmp .lt. ptll_min(j,i).or.
     $              ptll_max(j,i).ge.0d0 .and. tmp.gt.ptll_max(j,i))
               if (notgood) then
                  if(debug) write (*,*) i,j,' -> fails'
                  passcuts=.false.
                  return
               endif
            endif
         enddo
      enddo




c     
c     s-channel min & max invariant mass cuts
c     
      do i=nincoming+1,nexternal-1
         do j=i+1,nexternal
            if(debug) write (*,*) 's(',i,',',j,')=',Sumdot(p(0,i),p(0,j),+1d0)
            if(debug) write (*,*) s_min(j,i),s_max(j,i)
            if(s_min(j,i).gt.0.or.s_max(j,i).ge.0d0) then
               tmp=SumDot(p(0,i),p(0,j),+1d0)
               if(s_min(j,i).le.s_max(j,i) .or. s_max(j,i).lt.0d0)then
                  notgood=(tmp .lt. s_min(j,i).or.
     $                 s_max(j,i).ge.0d0 .and. tmp .gt. s_max(j,i)) 
                  if (notgood) then
                     if(debug) write (*,*) i,j,' -> fails'
                     passcuts=.false.
                     return
                  endif
               else
                  notgood=(tmp .lt. s_min(j,i).and.tmp .gt. s_max(j,i)) 
                  if (notgood) then
                     if(debug) write (*,*) i,j,' -> fails'
                     passcuts=.false.
                     return
                  endif
               endif
            endif
         enddo
      enddo
C     $B$DESACTIVATE_BW_CUT$B$ This is a Tag for MadWeight
c     
c     B.W. phase space cuts
c     
      pass_bw=cut_bw(p)
c     JA 4/8/11 always check pass_bw
      if ( pass_bw.and..not.CUTSPASSED) then
         passcuts=.false.
         if(debug) write (*,*) ' pass_bw -> fails'
         return
      endif
C     $E$DESACTIVATE_BW_CUT$E$ This is a Tag for MadWeight
        CUTSPASSED=.FALSE.
C     
C     maximal and minimal pt of the jets sorted by pt
c     
      njets=0
      nheavyjets=0

c- fill ptjet with the pt's of the jets.
      do i=nincoming+1,nexternal
         if(is_a_j(i)) then
            njets=njets+1
            ptjet(njets)=pt(p(0,i))
         endif
         if(is_a_b(i)) then
            nheavyjets=nheavyjets+1
            ptheavyjet(nheavyjets)=pt(p(0,i))
         endif

      enddo
      if(debug) write (*,*) 'not yet ordered ',njets,'   ',ptjet

C----------------------------------------------------------------------------
C     DURHAM_KT CUT
C----------------------------------------------------------------------------
      IF(NJETS.GT.0 .AND.KT_DURHAM.GT.0D0) THEN
C RESET JET MOMENTA
      njets=0
      DO I=1,NEXTERNAL
        DO J=0,3
          PJET(I,J) = 0E0
        ENDDO
      ENDDO

      do i=nincoming+1,nexternal
         if(is_a_j(i).and.do_cuts(i)) then
           njets=njets+1
           DO J=0,3
             PJET(NJETS,J) = P(J,I)
           ENDDO
         endif
      enddo

C DURHAM KT SEPARATION CUT


        PTMIN = EBEAM(1) + EBEAM(2)

        DO I=1,NJETS

C         PT WITH RESPECT TO Z AXIS FOR HADRONIC COLLISIONS
          IF ( (LPP(1).NE.0) .OR. (LPP(2).NE.0)) THEN
            PT1 = DSQRT(PJET(I,1)**2 + PJET(I,2)**2)
            PTMIN = MIN( PTMIN, PT1 )
          ENDIF

          DO J=I+1,NJETS
C           GET ANGLE BETWEEN JETS
            PABS1 = DSQRT(PJET(I,1)**2 + PJET(I,2)**2 + PJET(I,3)**2)
            PABS2 = DSQRT(PJET(J,1)**2 + PJET(J,2)**2 + PJET(J,3)**2)
C           CHECK IF 3-MOMENTA DO NOT VANISH
            IF(PABS1*PABS2 .NE. 0D0) THEN
              COSTH = ( PJET(I,1)*PJET(J,1) + PJET(I,2)*PJET(J,2) + PJET(I,3)*PJET(J,3) )/(PABS1*PABS2)
            ELSE
C           IF 3-MOMENTA VANISH, MAKE JET COSTH = 1D0 SO THAT JET MEASURE VANISHES
              COSTH = 1D0
            ENDIF
C           GET PT AND ETA OF JETS
            PT2 = DSQRT(PJET(J,1)**2 + PJET(J,2)**2)
            ETA1 = 0.5D0*LOG( (PJET(I,0) + PJET(I,3)) / (PJET(I,0) - PJET(I,3)) )
            ETA2 = 0.5D0*LOG( (PJET(J,0) + PJET(J,3)) / (PJET(J,0) - PJET(J,3)) )
C           GET COSH OF DELTA ETA, COS OF DELTA PHI
            COSH_DETA = DCOSH( ETA1 - ETA2 )
            COS_DPHI = ( PJET(I,1)*PJET(J,1) + PJET(I,2)*PJET(J,2) ) / (PT1*PT2)
            DPHI = DACOS( COS_DPHI )
            IF ( (LPP(1).EQ.0) .AND. (LPP(2).EQ.0)) THEN
C             KT FOR E+E- COLLISION
              PTNOW = DSQRT( 2D0*MIN(PJET(I,0)**2,PJET(J,0)**2)*( 1D0-COSTH ) )
             ELSE
C             HADRONIC KT, FASTJET DEFINITION
              PTNOW = DSQRT( MIN(PT1**2,PT2**2)*( (ETA1 - ETA2 )**2 + DPHI**2 )/(D_PARAMETER**2) )
            ENDIF

            PTMIN = MIN( PTMIN, PTNOW )

          ENDDO ! LOOP OVER NJET

        ENDDO ! LOOP OVER NJET

C CHECK COMPATIBILITY WITH CUT
        IF( (PTMIN .LT. KT_DURHAM)) THEN
          PASSCUTS = .FALSE.
          RETURN
        ENDIF
      ENDIF ! IF NJETS.GT. 0 .AND. DO_KT_DURHAM

C----------------------------------------------------------------------------
C----------------------------------------------------------------------------



c- check existance of jets if jet cuts are on
      if(njets.lt.1.and.(htjmin.gt.0.or.ptj1min.gt.0).or.
     $     njets.lt.2.and.ptj2min.gt.0.or.
     $     njets.lt.3.and.ptj3min.gt.0.or.
     $     njets.lt.4.and.ptj4min.gt.0)then
         if(debug) write (*,*) i, ' too few jets -> fails'
         passcuts=.false.
         return
      endif

c - sort jet pts
      do i=1,njets-1
         do j=i+1,njets
            if(ptjet(j).gt.ptjet(i)) then
               temp=ptjet(i)
               ptjet(i)=ptjet(j)
               ptjet(j)=temp
            endif
         enddo
      enddo
      if(debug) write (*,*) 'ordered ',njets,'   ',ptjet
c
c     Use "and" or "or" prescriptions 
c     
      inclht=0

      if(njets.gt.0) then

       notgood=.not.jetor
       if(debug) write (*,*) 'jetor :',jetor  
       if(debug) write (*,*) '0',notgood   
      
      do i=1,min(njets,4)
            if(debug) write (*,*) i,ptjet(i), '   ',ptjmin4(i),
     $        ':',ptjmax4(i)
         if(jetor) then     
c---  if one of the jets does not pass, the event is rejected
            notgood=notgood.or.(ptjmax4(i).ge.0d0 .and.
     $           ptjet(i).gt.ptjmax4(i)) .or.
     $           (ptjet(i).lt.ptjmin4(i))
            if(debug) write (*,*) i,' notgood total:', notgood   
         else
c---  all cuts must fail to reject the event
            notgood=notgood.and.(ptjmax4(i).ge.0d0 .and.
     $              ptjet(i).gt.ptjmax4(i) .or.
     $              (ptjet(i).lt.ptjmin4(i)))
            if(debug) write (*,*) i,' notgood total:', notgood   
         endif
      enddo


      if (notgood) then
         if(debug) write (*,*) i, ' multiple pt -> fails'
         passcuts=.false.
         return
      endif

c---------------------------
c      Ht cuts
C---------------------------
      htj=ptjet(1)

      do i=2,njets
         htj=htj+ptjet(i)
         if(debug) write (*,*) i, 'htj ',htj
         if(debug.and.i.le.4) write (*,*) 'htmin ',i,' ', htjmin4(i),':',htjmax4(i)
         if(i.le.4)then
            if(htj.lt.htjmin4(i) .or.
     $        htjmax4(i).ge.0d0.and.htj.gt.htjmax4(i)) then
            if(debug) write (*,*) i, ' ht -> fails'
            passcuts=.false.
            return
            endif
         endif
      enddo

      if(htj.lt.htjmin.or.htjmax.ge.0d0.and.htj.gt.htjmax)then
         if(debug) write (*,*) i, ' htj -> fails'
         passcuts=.false.
         return
      endif

      inclht=htj

      endif !if there are jets 

      if(nheavyjets.gt.0) then
         do i=1,nheavyjets
            inclht=inclht+ptheavyjet(i)
         enddo
      endif !if there are heavyjets

      if(inclht.lt.inclHtmin.or.
     $     inclHtmax.ge.0d0.and.inclht.gt.inclHtmax)then
         if(debug) write (*,*) ' inclhtmin=',inclHtmin,' -> fails'
         passcuts=.false.
         return
      endif

C     
C     maximal and minimal pt of the leptons sorted by pt
c     
      nleptons=0

      if(ptl1min.gt.0.or.ptl2min.gt.0.or.ptl3min.gt.0.or.ptl4min.gt.0.or.
     $     ptl1max.ge.0d0.or.ptl2max.ge.0d0.or.
     $     ptl3max.ge.0d0.or.ptl4max.ge.0d0) then

c     - fill ptlepton with the pt's of the leptons.
         do i=nincoming+1,nexternal
            if(is_a_l(i)) then
               nleptons=nleptons+1
               ptlepton(nleptons)=pt(p(0,i))
            endif
         enddo
         if(debug) write (*,*) 'not yet ordered ',njets,'   ',ptjet

c     - check existance of leptons if lepton cuts are on
         if(nleptons.lt.1.and.ptl1min.gt.0.or.
     $        nleptons.lt.2.and.ptl2min.gt.0.or.
     $        nleptons.lt.3.and.ptl3min.gt.0.or.
     $        nleptons.lt.4.and.ptl4min.gt.0)then
            if(debug) write (*,*) i, ' too few leptons -> fails'
            passcuts=.false.
            return
         endif

c     - sort lepton pts
         do i=1,nleptons-1
            do j=i+1,nleptons
               if(ptlepton(j).gt.ptlepton(i)) then
                  temp=ptlepton(i)
                  ptlepton(i)=ptlepton(j)
                  ptlepton(j)=temp
               endif
            enddo
         enddo
         if(debug) write (*,*) 'ordered ',nleptons,'   ',ptlepton

         if(nleptons.gt.0) then

            notgood = .false.
            do i=1,min(nleptons,4)
               if(debug) write (*,*) i,ptlepton(i), '   ',ptlmin4(i),':',ptlmax4(i)
c---  if one of the leptons does not pass, the event is rejected
               notgood=notgood.or.
     $              (ptlmax4(i).ge.0d0.and.ptlepton(i).gt.ptlmax4(i)).or.
     $              (ptlepton(i).lt.ptlmin4(i))
               if(debug) write (*,*) i,' notgood total:', notgood   
            enddo


            if (notgood) then
               if(debug) write (*,*) i, ' multiple pt -> fails'
               passcuts=.false.
               return
            endif
         endif
      endif
C>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>     
C     SPECIAL CUTS
C<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

C     REQUIRE AT LEAST ONE JET WITH PT>XPTJ
         
         IF(xptj.gt.0d0) THEN
            xvar=0
            do i=nincoming+1,nexternal
               if(is_a_j(i)) xvar=max(xvar,pt(p(0,i)))
            enddo
            if (xvar .lt. xptj) then
               passcuts=.false.
               return
            endif
         ENDIF

C     REQUIRE AT LEAST ONE PHOTON WITH PT>XPTA
         
         IF(xpta.gt.0d0) THEN
            xvar=0
            do i=nincoming+1,nexternal
               if(is_a_a(i)) xvar=max(xvar,pt(p(0,i)))
            enddo
            if (xvar .lt. xpta) then
               passcuts=.false.
               return
            endif
         ENDIF

C     REQUIRE AT LEAST ONE B  WITH PT>XPTB
         
         IF(xptb.gt.0d0) THEN
            xvar=0
            do i=nincoming+1,nexternal
               if(is_a_b(i)) xvar=max(xvar,pt(p(0,i)))
            enddo
            if (xvar .lt. xptb) then
               passcuts=.false.
               return
            endif
         ENDIF

C     REQUIRE AT LEAST ONE LEPTON  WITH PT>XPTL
         
         IF(xptl.gt.0d0) THEN
            xvar=0
            do i=nincoming+1,nexternal
               if(is_a_l(i)) xvar=max(xvar,pt(p(0,i)))
            enddo
            if (xvar .lt. xptl) then
               passcuts=.false.
               if(debug) write (*,*) ' xptl -> fails'
               return
            endif
         ENDIF
C
C     WBF CUTS: TWO TYPES
C    
C     FIRST TYPE:  implemented by FM
C
C     1. FIND THE 2 HARDEST JETS
C     2. REQUIRE |RAP(J)|>XETAMIN
C     3. REQUIRE RAP(J1)*ETA(J2)<0
C
C     SECOND TYPE : added by Simon de Visscher 1-08-2007
C
C     1. FIND THE 2 HARDEST JETS
C     2. REQUIRE |RAP(J1)-RAP(J2)|>DELTAETA
C     3. REQUIRE RAP(J1)*RAP(J2)<0
C
C
         hardj1=0
         hardj2=0
         ptmax1=0d0
         ptmax2=0d0

C-- START IF AT LEAST ONE OF THE CUTS IS ACTIVATED
         
         IF(XETAMIN.GT.0D0.OR.DELTAETA.GT.0D0) THEN
            
C-- FIND THE HARDEST JETS

            do i=nincoming+1,nexternal
               if(is_a_j(i)) then
c                  write (*,*) i,pt(p(0,i))
                  if(pt(p(0,i)).gt.ptmax1) then
                     hardj2=hardj1
                     ptmax2=ptmax1
                     hardj1=i
                     ptmax1=pt(p(0,i))
                  elseif(pt(p(0,i)).gt.ptmax2) then
                     hardj2=i
                     ptmax2=pt(p(0,i))
                  endif
c                  write (*,*) hardj1,hardj2,ptmax1,ptmax2
               endif
            enddo
            
            if (hardj2.eq.0) goto 21 ! bypass vbf cut since not enough jets

C-- NOW APPLY THE CUT I            

            if (abs(rap(p(0,hardj1))) .lt. xetamin
     &       .or.abs(rap(p(0,hardj2))) .lt. xetamin
     &       .or.rap(p(0,hardj1))*rap(p(0,hardj2)) .gt.0d0) then
             passcuts=.false.
             return
            endif

            
C-- NOW APPLY THE CUT II
            
            if (abs(rap(p(0,hardj1))-rap(p(0,hardj2))) .lt. deltaeta) then
             passcuts=.false.
             return
            endif
         
c            write (*,*) hardj1,hardj2,rap(p(0,hardj1)),rap(p(0,hardj2))
         
         ENDIF

c Begin photon isolation
c NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
c     Use is made of parton cm frame momenta. If this must be
c     changed, pQCD used below must be redefined
c NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE
c If we do not require a mimimum jet energy, there's no need to apply
c jet clustering and all that.
      if (ptgmin.ne.0d0) then

c Put all (light) QCD partons in momentum array for jet clustering.
c From the run_card.dat, maxjetflavor defines if b quark should be
c considered here (via the logical variable 'is_a_jet').  nQCD becomes
c the number of (light) QCD partons at the real-emission level (i.e. one
c more than the Born).

      nQCD=0
      do j=nincoming+1,nexternal
         if (is_a_j(j)) then
            nQCD=nQCD+1
            do i=0,3
               pQCD(i,nQCD)=p(i,j) ! Use C.o.M. frame momenta
            enddo
         endif
      enddo

        nph=0
        do j=nincoming+1,nexternal
          if (is_a_a(j)) then
            nph=nph+1
            do i=0,3
              pgamma(i,nph)=p(i,j) ! Use C.o.M. frame momenta
            enddo
          endif
        enddo
        if(nph.eq.0) goto 444

        if(isoEM)then
          nem=nph
          do k=1,nem
            do i=0,3
              pem(i,k)=pgamma(i,k)
            enddo
          enddo
          do j=nincoming+1,nexternal
            if (is_a_l(j)) then
              nem=nem+1
              do i=0,3
                pem(i,nem)=p(i,j) ! Use C.o.M. frame momenta
              enddo
            endif
          enddo
        endif

        alliso=.true.

        j=0
        dowhile(j.lt.nph.and.alliso)
c Loop over all photons
          j=j+1

          ptg=pt(pgamma(0,j))
          if(ptg.lt.ptgmin)then
            passcuts=.false.
            return
          endif

c Isolate from hadronic energy
          do i=1,nQCD
            drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pQCD(0,i)))
          enddo
          call sortzv(drlist,isorted,nQCD,ismode,isway,izero)
          Etsum(0)=0.d0
          nin=0
          do i=1,nQCD
            if(dble(drlist(isorted(i))).le.R0gamma)then
              nin=nin+1
              Etsum(nin)=Etsum(nin-1)+pt(pQCD(0,isorted(i)))
            endif
          enddo
          do i=1,nin
            alliso=alliso .and.
     #        Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     #                                  R0gamma,xn,epsgamma,ptg)
          enddo

c Isolate from EM energy
          if(isoEM.and.nem.gt.1)then
            do i=1,nem
              drlist(i)=sngl(iso_getdrv40(pgamma(0,j),pem(0,i)))
            enddo
            call sortzv(drlist,isorted,nem,ismode,isway,izero)
c First of list must be the photon: check this, and drop it
            if(isorted(1).ne.j.or.drlist(isorted(1)).gt.1.e-4)then
              write(*,*)'Error #1 in photon isolation'
              write(*,*)j,isorted(1),drlist(isorted(1))
              stop
            endif
            Etsum(0)=0.d0
            nin=0
            do i=2,nem
              if(dble(drlist(isorted(i))).le.R0gamma)then
                nin=nin+1
                Etsum(nin)=Etsum(nin-1)+pt(pem(0,isorted(i)))
              endif
            enddo
            do i=1,nin
              alliso=alliso .and.
     #          Etsum(i).le.chi_gamma_iso(dble(drlist(isorted(i))),
     #                                    R0gamma,xn,epsgamma,ptg)
            enddo

          endif

c End of loop over photons
        enddo

        if(.not.alliso)then
          passcuts=.false.
          return
        endif
      endif

 444    continue
c End photon isolation


C...Set couplings if event passed cuts

 21   if(.not.fixed_ren_scale) then
         call set_ren_scale(P,scale)
         if(scale.gt.0) G = SQRT(4d0*PI*ALPHAS(scale))
      endif

      if(.not.fixed_fac_scale) then
         call set_fac_scale(P,q2fact)
      endif

c
c     Here we cluster event and reset factorization and renormalization
c     scales on an event-by-event basis, as well as check xqcut for jets
c
c     Note the following condition is the first line of setclscales
c      if(xqcut.gt.0d0.or.ickkw.gt.0.or.scale.eq.0.or.q2fact(1).eq.0)then
c     Do not duplicate it since some variable are set for syscalc in the fct
        if(.not.setclscales(p))then
         if(debug) write (*,*) ' setclscales -> fails'
         passcuts=.false.
         return
       endif
c      endif

c     Set couplings in model files
      if(.not.fixed_ren_scale.or..not.fixed_couplings) then
         if (.not.fixed_couplings)then
            do i=0,3
               do j=1,nexternal
                  pp(i,j)=p(i,j)
               enddo
            enddo
         endif
         call update_as_param()
      endif

      IF (FIRSTTIME2) THEN
        FIRSTTIME2=.FALSE.
        write(6,*) 'alpha_s for scale ',scale,' is ', G**2/(16d0*atan(1d0))
      ENDIF

      if(debug) write (*,*) '============================='
      if(debug) write (*,*) ' EVENT PASSED THE CUTS       '
      if(debug) write (*,*) '============================='

      CUTSPASSED=.TRUE.

      RETURN
      END


C
C     FUNCTION FOR ISOLATION
C

      function iso_getdrv40(p1,p2)
      implicit none
      real*8 iso_getdrv40,p1(0:3),p2(0:3)
      real*8 iso_getdr
c
      iso_getdrv40=iso_getdr(p1(0),p1(1),p1(2),p1(3),
     #                       p2(0),p2(1),p2(2),p2(3))
      return
      end


      function iso_getdr(en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2)
      implicit none
      real*8 iso_getdr,en1,ptx1,pty1,pl1,en2,ptx2,pty2,pl2,deta,dphi,
     # iso_getpseudorap,iso_getdelphi
c
      deta=iso_getpseudorap(en1,ptx1,pty1,pl1)-
     #     iso_getpseudorap(en2,ptx2,pty2,pl2)
      dphi=iso_getdelphi(ptx1,pty1,ptx2,pty2)
      iso_getdr=sqrt(dphi**2+deta**2)
      return
      end


      function iso_getpseudorap(en,ptx,pty,pl)
      implicit none
      real*8 iso_getpseudorap,en,ptx,pty,pl,tiny,pt,eta,th
      parameter (tiny=1.d-5)
c
      pt=sqrt(ptx**2+pty**2)
      if(pt.lt.tiny.and.abs(pl).lt.tiny)then
        eta=sign(1.d0,pl)*1.d8
      else
        th=atan2(pt,pl)
        eta=-log(tan(th/2.d0))
      endif
      iso_getpseudorap=eta
      return
      end


      function iso_getdelphi(ptx1,pty1,ptx2,pty2)
      implicit none
      real*8 iso_getdelphi,ptx1,pty1,ptx2,pty2,tiny,pt1,pt2,tmp
      parameter (tiny=1.d-5)
c
      pt1=sqrt(ptx1**2+pty1**2)
      pt2=sqrt(ptx2**2+pty2**2)
      if(pt1.ne.0.d0.and.pt2.ne.0.d0)then
        tmp=ptx1*ptx2+pty1*pty2
        tmp=tmp/(pt1*pt2)
        if(abs(tmp).gt.1.d0+tiny)then
          write(*,*)'Cosine larger than 1'
          stop
        elseif(abs(tmp).ge.1.d0)then
          tmp=sign(1.d0,tmp)
        endif
        tmp=acos(tmp)
      else
        tmp=1.d8
      endif
      iso_getdelphi=tmp
      return
      end

      function chi_gamma_iso(dr,R0,xn,epsgamma,pTgamma)
c Eq.(3.4) of Phys.Lett. B429 (1998) 369-374 [hep-ph/9801442]
      implicit none
      real*8 chi_gamma_iso,dr,R0,xn,epsgamma,pTgamma
      real*8 tmp,axn
c
      axn=abs(xn)
      tmp=epsgamma*pTgamma
      if(axn.ne.0.d0)then
        tmp=tmp*( (1-cos(dr))/(1-cos(R0)) )**axn
      endif
      chi_gamma_iso=tmp
      return
      end


*
* $Id: sortzv.F,v 1.1.1.1 1996/02/15 17:49:50 mclareni Exp $
*
* $Log: sortzv.F,v $
* Revision 1.1.1.1  1996/02/15 17:49:50  mclareni
* Kernlib
*
*
c$$$#include "kerngen/pilot.h"
      SUBROUTINE SORTZV (A,INDEX,N1,MODE,NWAY,NSORT)
C
C CERN PROGLIB# M101    SORTZV          .VERSION KERNFOR  3.15  820113
C ORIG. 02/10/75
C
      DIMENSION A(N1),INDEX(N1)
C
C
      N = N1
      IF (N.LE.0)            RETURN
      IF (NSORT.NE.0) GO TO 2
      DO 1 I=1,N
    1 INDEX(I)=I
C
    2 IF (N.EQ.1)            RETURN
      IF (MODE)    10,20,30
   10 CALL SORTTI (A,INDEX,N)
      GO TO 40
C
   20 CALL SORTTC(A,INDEX,N)
      GO TO 40
C
   30 CALL SORTTF (A,INDEX,N)
C
   40 IF (NWAY.EQ.0) GO TO 50
      N2 = N/2
      DO 41 I=1,N2
      ISWAP = INDEX(I)
      K = N+1-I
      INDEX(I) = INDEX(K)
   41 INDEX(K) = ISWAP
   50 RETURN
      END
*     ========================================
      SUBROUTINE SORTTF (A,INDEX,N1)
C
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTI (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTC (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF(ICMPCH(AI,A(I22)))3,3,21
   21 INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (ICMPCH(A(I22),A(I222))) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (ICMPCH(AI,A(I22))) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      FUNCTION ICMPCH(IC1,IC2)
C     FUNCTION TO COMPARE TWO 4 CHARACTER EBCDIC STRINGS - IC1,IC2
C     ICMPCH=-1 IF HEX VALUE OF IC1 IS LESS THAN IC2
C     ICMPCH=0  IF HEX VALUES OF IC1 AND IC2 ARE THE SAME
C     ICMPCH=+1 IF HEX VALUES OF IC1 IS GREATER THAN IC2
      I1=IC1
      I2=IC2
      IF(I1.GE.0.AND.I2.GE.0)GOTO 40
      IF(I1.GE.0)GOTO 60
      IF(I2.GE.0)GOTO 80
      I1=-I1
      I2=-I2
      IF(I1-I2)80,70,60
 40   IF(I1-I2)60,70,80
 60   ICMPCH=-1
      RETURN
 70   ICMPCH=0
      RETURN
 80   ICMPCH=1
      RETURN
      END

