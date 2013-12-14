c++++++++++++++++++++++
c  CONTENT
c++++++++++++++++++++++
 
c      real*8  function PROJ_ETA(P,NHEL,IC)
c      real*8 function PROJ_psi(P,NHEL,IC)
c      real*8 function PROJ_PSI_REL(P,NHEL,IC)
c      real*8 function PROJ_CHI(P,NHEL,IC,J_qn)
c      real*8 function PROJ_H(P,NHEL,IC)
c      real*8 function PROJ_PSILEPT(P,NHEL,IC)

c      subroutine spin_projection(P,NHEL,IC,TEMP)
c      subroutine spin_singlet_proj(P,NHEL,IC,TEMP)
c      subroutine pseudoscalar(P1,P2,LAMBDA1,LAMBDA2,MC,PS)
c      subroutine pseudoscalar0(P1,P2,LAMBDA1,LAMBDA2,M1,M2,PS)
c      subroutine vector0(P1,P2,LAMBDA1,LAMBDA2,M1,M2,jio)
c      subroutine VECTOR(P1,P2,LAMBDA1,LAMBDA2,MQ,jio)
c      subroutine vectorBc(P1,P2,LAMBDA1,LAMBDA2,M1,M2,jio)
c      subroutine leptonic_current(k1,k2,l1,l2,mq, jio)
c      subroutine genmom_lept(P,x,k1,k2,three_k1)
c      SUBROUTINE polariz(P,EPS)
c      subroutine get_helicity(P,eps)
c      subroutine get_helicity2(P,eps)



      real*8  function PROJ_ETA(P,NHEL,IC)
C
C******************************************************************************
C     THIS FUNCTION PROJECTS THE AMPLITUDE ONTO A CC~(3S1) STATE               * 
C     AND SQUARE IT                                                           *
C     THE ROUTINE IS UNIVERSAL                                                *
C******************************************************************************
C
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                    ! number of  particles
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      double precision  P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C     LOCAL VARIABLES
C
C         1. for the projection
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K,n
      DOUBLE COMPLEX PS
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3)  ! MOMENTA OF c AND c~ 
     & , weight1,weight2
C
      COMPLEX*16 JAMP(NCOLOR)
      DOUBLE COMPLEX PROJAMP(NCOLOR)
C
C         2.  to  square the amplitude
      COMPLEX*16 ZTEMP
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass      
C
C----
C  Begin code
C----
C     SPECIFYING THE MOMENTA OF C,C~ + THE TOTAL MOMENTUM
C

      weight1=qmass(1)/(qmass(1)+qmass(2))
      weight2=qmass(2)/(qmass(1)+qmass(2))

      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      ENDDO
      P_AMP(L1,next_amp-1)=P(L1,nexternal)*weight1
      P_AMP(L1,next_amp)=P(L1,nexternal)*weight2
      P1(L1)=P(L1,nexternal)*weight1
      P2(L1)=P(L1,nexternal)*weight2
      ENDDO
C
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 
C
C     INITIALIZATION
C
      DO K=1,NCOLOR
       projamp(K)=(0.0D0,0.0D0)
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2 
c      write(*,*) 'P1',P1
c      write(*,*) 'P2',P2
c      write(*,*) 'L1',L1
c      write(*,*) 'L2',L2
      CALL pseudoscalar0(P1,P2,L1,L2,qmass(1),
     & qmass(2),PS)
c      write(*,*) 'PS',PS
      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)
c      write(*,*) 'Jamp',Jamp
C 
      DO K=1, NCOLOR    ! INDEX OF JAMP
          projamp(K)=projamp(K)+JAMP(K)*PS  ! THE COLOR FACTOR IS INCLUDED IN SQAMP
      ENDDO
C  
      ENDDO
      ENDDO
C       
C************************************ 
C    SQUARE THE AMPLITUDE
C***********************************
C
      PROJ_ETA = 0.D0
      DO n = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,n)*PROJAMP(J)
          ENDDO
          PROJ_ETA =PROJ_ETA
     &         +dble(ZTEMP*DCONJG(PROJAMP(n))/DENOM(n))
      ENDDO
      RETURN 
      END 
C
      real*8 function PROJ_psi(P,NHEL,IC)
C
C******************************************************************************
C     THIS FUNCTION PROJECT THE AMPLITUDE ONTO A QQ~(3S1) STATE               * 
C     AND SQUARE IT                                                           *
C     THE ROUTINE IS UNIVERSAL                                                *
C******************************************************************************
C
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                   
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C     LOCAL VARIABLES
C
C         1. for the projection
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K
      DOUBLE COMPLEX JIO(6),TEMP(4,NCOLOR)
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3),PTOT(0:3)   ! MOMENTA OF c AND c~ AND TOTAL MOMENTUM
      DOUBLE complex EPS(0:3,-1:1)
C
      COMPLEX*16 JAMP(NCOLOR)
      DOUBLE COMPLEX PROJAMP(NCOLOR)
C
C         2.  to  square the amplitude
      COMPLEX*16 ZTEMP
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass      
      DOUBLE PRECISION weight1,weight2
C
C----
C  Begin code
C----
C     SPECIFYING THE MOMENTA OF C,C~ + THE TOTAL MOMENTUM
C

      weight1=qmass(1)/(qmass(1)+qmass(2))
      weight2=qmass(2)/(qmass(1)+qmass(2))

      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      ENDDO
      P_AMP(L1,next_amp-1)=P(L1,nexternal)*weight1
      P_AMP(L1,next_amp)=P(L1,nexternal)*weight2
      P1(L1)=P(L1,nexternal)*weight1
      P2(L1)=P(L1,nexternal)*weight2
      PTOT(L1)=P1(L1)+P2(L1)
      ENDDO

C
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 
C
C     INITIALIZATION
C
      DO J=1,4,1
      DO K=1,NCOLOR
       TEMP(J,K)=(0.0D0,0.0D0)
      ENDDO
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2  
      CALL VECTOR0(P1,P2,L1,L2,qmass(1),qmass(2),JIO)


      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)
C 
C 
      DO K=1, NCOLOR    ! INDEX OF JAMP
          DO J=1,4,1   ! PAY ATTENTION TO THE RANGE : J IN [1,4]
          TEMP(J,K)=TEMP(J,K)+JAMP(K)*JIO(J)  ! THE COLOR FACTOR IS INCLUDED IN SQAMP
          ENDDO
      ENDDO
C 
      ENDDO
      ENDDO
C       
c       CALL polariz(PTOT,EPS)
       CALL get_helicity2(PTOT,EPS,(qmass(1)+qmass(2)))

C
      DO K=1,NCOLOR
      PROJAMP(K)=TEMP(1,K)*EPS(0,NHEL(nexternal))-
     &  TEMP(2,K)*EPS(1,NHEL(nexternal))-
     & TEMP(3,K)*EPS(2,NHEL(nexternal))- 
     & TEMP(4,K)*EPS(3,NHEL(nexternal))
      ENDDO
C     

C************************************ 
C    SQUARE THE AMPLITUDE
C***********************************
C
      PROJ_PSI = 0.D0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*PROJAMP(J)
          ENDDO
          PROJ_PSI =PROJ_PSI
     &         +DBLE(ZTEMP*DCONJG(PROJAMP(I)))/DENOM(I)
      ENDDO
      RETURN 
      END 
C
      real*8 function PROJ_PSI_REL(P,NHEL,IC)
C
C******************************************************************************
C     THIS FUNCTION PROJECT THE AMPLITUDE ONTO A CC~(3S1) STATE               * 
C     AND SQUARE IT                                                           *
C     THE ROUTINE IS UNIVERSAL                                                *
C******************************************************************************
C
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                    ! number of  particles
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C     LOCAL VARIABLES
C
C         1. for the projection
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K
      DOUBLE COMPLEX JIO(6),TEMP(4,NCOLOR)
      DOUBLE COMPLEX JIOSTAR(6),TEMPSTAR(4,NCOLOR)
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3),PTOT(0:3)   ! MOMENTA OF c AND c~ AND TOTAL MOMENTUM
      DOUBLE PRECISION P_AMPSTAR(0:3,next_amp),
     &                  P1STAR(0:3),P2STAR(0:3)
      DOUBLE PRECISION EPS(0:3,-1:1)
C
      COMPLEX*16 JAMP(NCOLOR)
      DOUBLE COMPLEX PROJAMP(NCOLOR)
      COMPLEX*16 JAMPSTAR(NCOLOR)
      DOUBLE COMPLEX PROJAMPSTAR(NCOLOR)
C
C         2.  to  square the amplitude
      COMPLEX*16 ZTEMP
      complex*16 proj_matrix_temp
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass      
C
C     global
C
      double precision prel1(0:3),prel2(0:3)
      double precision BOUND, RHOSQ          ! BOUND IS THE UPPER BOUND FOR q_{REL}^2
      common /to_rel_mom/prel1,prel2,BOUND, RHOSQ
C
C----
C  Begin code
C----
C     SPECIFYING THE MOMENTA OF C,C~ + THE TOTAL MOMENTUM
C
      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      P_AMPSTAR(L1,L2)=P(L1,L2)
      ENDDO
C                 in the amplitude
      P_AMP(L1,next_amp-1)=P(L1,nexternal)/2+prel1(L1)
      P_AMP(L1,next_amp)=P(L1,nexternal)/2-prel1(L1)
      P1(L1)=P(L1,nexternal)/2+prel1(L1)
      P2(L1)=P(L1,nexternal)/2-prel1(L1)
C                 in the compl. conj. amplitude 
      P_AMPSTAR(L1,next_amp-1)=P(L1,nexternal)/2+prel2(L1)
      P_AMPSTAR(L1,next_amp)=P(L1,nexternal)/2-prel2(L1)
      P1STAR(L1)=P(L1,nexternal)/2+prel2(L1)
      P2STAR(L1)=P(L1,nexternal)/2-prel2(L1)
c
      PTOT(L1)=P(L1,nexternal)
      ENDDO
C
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 

C
C     INITIALIZATION
C
      DO J=1,4,1
      DO K=1,NCOLOR
       TEMP(J,K)=(0.0D0,0.0D0)
       TEMPSTAR(J,K)=(0.0D0,0.0D0)
      ENDDO
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2  
      CALL VECTOR(P1,P2,L1,L2,qmass(1),JIO)              ! for amp.
      CALL VECTOR(P1STAR,P2STAR,L1,L2,qmass(1),JIOSTAR)  ! for amp*

      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)         ! for amp.
      CALL matrix(P_AMPSTAR,NHEL_AMP,IC_AMP,JAMPSTAR) ! for amp*
C 
c       
      DO K=1, NCOLOR    ! INDEX OF JAMP
          DO J=1,4,1   ! PAY ATTENTION TO THE RANGE : J IN [1,4]
C                  THE COLOR FACTOR IS INCLUDED IN SQAMP
          TEMP(J,K)=TEMP(J,K)+JAMP(K)*JIO(J)  ! for amp
          TEMPSTAR(J,K)=TEMPSTAR(J,K)+JAMPSTAR(K)*JIOSTAR(J)  ! for amp*
          ENDDO
      ENDDO
C 
C  
      ENDDO
      ENDDO
C       
      CALL polariz(PTOT,EPS)
C
      DO K=1,NCOLOR
c                          for amp.
      PROJAMP(K)=TEMP(1,K)*EPS(0,NHEL(nexternal))-
     &  TEMP(2,K)*EPS(1,NHEL(nexternal))-
     & TEMP(3,K)*EPS(2,NHEL(nexternal))- 
     & TEMP(4,K)*EPS(3,NHEL(nexternal))
C                           for amp*
      PROJAMPSTAR(K)=TEMPSTAR(1,K)*EPS(0,NHEL(nexternal))-
     &  TEMPSTAR(2,K)*EPS(1,NHEL(nexternal))-
     & TEMPSTAR(3,K)*EPS(2,NHEL(nexternal))-
     & TEMPSTAR(4,K)*EPS(3,NHEL(nexternal))
      ENDDO
C     

C************************************ 
C    SQUARE THE AMPLITUDE
C***********************************
C
      PROJ_MATRIX_TEMP = (0.0D0,0.0D0)
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*PROJAMP(J)
          ENDDO
          PROJ_MATRIX_TEMP =PROJ_MATRIX_TEMP
     &         +ZTEMP*DCONJG(PROJAMPSTAR(I))/DENOM(I)
      ENDDO
      
      PROJ_PSI_REL = DBLE(proj_matrix_temp)
      RETURN 
      END 
C
      real*8 function PROJ_CHI(PP,NHEL,IC,J_qn)
C
C******************************************************************************
C     THIS FUNCTION PROJECT THE AMPLITUDE ONTO A P wave CC~ STATE             * 
C     AND SQUARE IT                                                           *
C     THE ROUTINE IS UNIVERSAL.                                               *     
C     MOMENTA MUST BE GIVEN IN THE QUARKONIUM REST FRAME                      *
C     UNDER WORK         !!                                                   *
C******************************************************************************
C
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     ARGUMENTS
C
      REAL*8 PP(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL),J_qn
C
C     LOCAL VARIABLES
C
C         1. for the projection
      DOUBLE PRECISION Pboost(0:3)  !,Pboostnd(0:3)
      Double precision P(0:3,NEXTERNAL),Z(0:3)
      double complex eps(1:3,-1:1)
c      double precision Ponium(0:3),Poniumnd(0:3)
C
      DOUBLE COMPLEX TEMP(4,NCOLOR),
     & TEMP1(4,NCOLOR),TEMP2(4,NCOLOR),TEMP3(4,NCOLOR)
C
      DOUBLE COMPLEX AMP(3,3,NCOLOR)
      DOUBLE COMPLEX hel_AMP(NCOLOR)
      DOUBLE COMPLEX hel_AMPJm1(NCOLOR,3),spin_mat(NCOLOR,3,-1:1)
C
C         2.  to  square the amplitude
      double precision sqm2,sqm6,sqm3
      COMPLEX*16 ZTEMP
      integer j,k,l,ia,ib,mu
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass
C
C     global
C
      double precision prel(0:3)
      common /to_pWAVE/prel
c
      double precision pnd(0:3,nexternal),delta

      double complex rho11,rho00,rho1m1,rhom1m1,rhom11
      double precision plam,pnu
      common /to_pol_param/rho11,rho00,rho1m1,rhom1m1,rhom11,plam,pnu

C
C----
C  Begin code
C----
      delta=0.00001d0
      
c
c   go to the quarkonium rest frame
c


       pboost(0)=pp(0,nexternal)
       do k=1,3
       pboost(k)=-pp(k,nexternal)
       enddo
       do j=1,nexternal
       call boostx(PP(0,j),pboost,P(0,j))
       do mu=0,3
       Pnd(mu,j)=P(mu,j)
       enddo
       enddo
       Pnd(0,nexternal)=dsqrt(P(0,nexternal)**2+delta**2)
c
c    define the tensor projected amplitude with UPPER indices
c
c    PAY ATTENTION : d/dq_i = - d/d_q^j
c
c      a) no relative momentum
c
       do k=0,3
       prel(K)=0d0
       enddo
c
       prel(1)=delta/1000000d0
       call spin_projection(P,NHEL,IC,TEMP)
c
c       a) component q_1
c
       prel(1)=delta

       call spin_projection(Pnd,NHEL,IC,TEMP1)
c
c       b) component q_2
c
       prel(1)=0d0
       prel(2)=delta

       call spin_projection(Pnd,NHEL,IC,TEMP2)
c
c       c) component q_3
c
       prel(2)=0d0
       prel(3)=delta

       call spin_projection(Pnd,NHEL,IC,TEMP3)
c
c      computing derivatives
c
       do k=1,NCOLOR
         do j=1,3
c       amp(0,j,k)=(0d0, 0d0)
       amp(1,j,k)=-(temp1(j+1,k)-temp(j+1,k))/dcmplx(delta)
       amp(2,j,k)=-(temp2(j+1,k)-temp(j+1,k))/dcmplx(delta)
       amp(3,j,k)=-(temp3(j+1,k)-temp(j+1,k))/dcmplx(delta)
         enddo
       enddo

C************************************ 
C    Select helicity state
C***********************************
C

      call get_helicity(PP(0,nexternal),eps)

       sqm6=1d0/dsqrt(6d0)
       sqm2=1d0/dsqrt(2d0)
       sqm3=1d0/dsqrt(3d0)

      if(j_qn.eq.0) then

      do L=1,Ncolor
      hel_amp(L)=(0d0,0d0)

          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+ amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,-1)
     &+eps(Ia,-1)*eps(Ib,1) -eps(Ia,0)*eps(Ib,0))*sqm3 
          enddo
          enddo
      enddo !L


      elseif(j_qn.eq.1) then

      do L=1,Ncolor
      hel_amp(L)=(0d0,0d0)

        if (nhel(nexternal).eq.1 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+ 
     & amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,0)-eps(Ia,0)*eps(Ib,1))*sqm2 
          enddo
          enddo

        elseif (nhel(nexternal).eq.0 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+ 
     & amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,-1)-eps(Ia,-1)*eps(Ib,1))*sqm2 
          enddo
          enddo

        elseif (nhel(nexternal).eq.-1 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+
     & amp(Ia,Ib,L)*(eps(Ia,0)*eps(Ib,-1)-eps(Ia,-1)*eps(Ib,0))*sqm2
          enddo
          enddo
        endif

      enddo !L

      elseif(j_qn.eq.2) then

      do L=1,Ncolor
      hel_amp(L)=(0d0,0d0)

        if (nhel(nexternal).eq.2 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+ 
     & amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,1)) 
          enddo
          enddo

        elseif (nhel(nexternal).eq.1 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+ 
     & amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,0)+eps(Ia,0)*eps(Ib,1))*sqm2 
          enddo
          enddo

        elseif (nhel(nexternal).eq.0 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+
     & amp(Ia,Ib,L)*(eps(Ia,1)*eps(Ib,-1)+eps(Ia,-1)*eps(Ib,1)+
     & 2d0*eps(Ia,0)*eps(Ib,0))*sqm6
          enddo
          enddo

        elseif (nhel(nexternal).eq.-1 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+
     & amp(Ia,Ib,L)*(eps(Ia,0)*eps(Ib,-1)+eps(Ia,-1)*eps(Ib,0))*sqm2
          enddo
          enddo

        elseif (nhel(nexternal).eq.-2 ) then
          do Ia=1,3
          do IB=1,3
           hel_amp(l)=hel_amp(l)+
     & amp(Ia,Ib,L)*eps(Ia,-1)*eps(Ib,-1)
          enddo
          enddo
        endif
     
c      write(*,*) 'hel_amp for hel,col',nhel(nexternal),L,hel_amp(l)
      enddo !L


      elseif(j_qn.eq.-1) then
c     here is a special option: 
c       * I average over the angular momentum degrees of freedom
c       * I keep track of the spin of the QQ~ pair
c     This is used for polarization studies involving transitions 3PJ[8]> J/psi
      do L=1,Ncolor
      do Ia=1,3
      hel_ampJm1(L,Ia)=(0d0,0d0)
c
          do IB=1,3
           hel_ampJm1(l,Ia)=hel_ampJm1(l,Ia)+
     & amp(Ia,Ib,L)*eps(Ib,nhel(nexternal))
          enddo

      enddo !angular momentum
      enddo !color

      ZTEMP = (0.D0,0.D0)
      do Ia=1,3 !angular momentum
      DO L = 1, NCOLOR
          DO K = 1, NCOLOR
          ZTEMP = ZTEMP+CF(K,L)*hel_ampJm1(K,Ia)*dconjg(hel_ampJm1(L,Ia))/DENOM(L)
          ENDDO
      ENDDO
      ENDDO
      PROJ_CHI = DBLE(ztemp)
      return
c
      elseif(j_qn.eq.-7) then
c     here is another special option:
c       * I average over the angular momentum degrees of freedom
c       * I compute the spin density matrix
c     This is used for polarization studies involving transitions 3PJ[8]> J/psi
      do L=1,Ncolor
      do Ia=1,3
        hel_ampJm1(L,Ia)=(0d0,0d0)
        spin_mat(l,Ia,-1)=(0d0,0d0)
        spin_mat(l,Ia,1)=(0d0,0d0)
        spin_mat(l,Ia,0)=(0d0,0d0)
c
          do IB=1,3

           spin_mat(l,Ia,-1)=spin_mat(l,Ia,-1)+
     & amp(Ia,Ib,L)*eps(Ib,-1)
           spin_mat(l,Ia,1)=spin_mat(l,Ia,1)+
     & amp(Ia,Ib,L)*eps(Ib,1)
           spin_mat(l,Ia,0)=spin_mat(l,Ia,0)+
     & amp(Ia,Ib,L)*eps(Ib,0)
          enddo
          enddo
      enddo !L

      ZTEMP = (0.D0,0.D0)
      do Ia=1,3
      do Ib=1,3
      DO L = 1, NCOLOR
          DO K = 1, NCOLOR
          ZTEMP = ZTEMP+CF(K,L)*amp(Ia,Ib,K)*dconjg(amp(Ia,Ib,L))/DENOM(L)
          ENDDO
      ENDDO
      ENDDO
      ENDDO
      PROJ_CHI = DBLE(ztemp)

      do Ia=1,3
      DO L = 1, NCOLOR
          DO K = 1, NCOLOR
          rho11 = rho11+CF(K,L)*spin_mat(K,Ia,1)*dconjg(spin_mat(L,Ia,1))/DENOM(L)
          rho00 = rho00+CF(K,L)*spin_mat(K,Ia,0)*dconjg(spin_mat(L,Ia,0))/DENOM(L)
          rho1m1 = rho1m1+CF(K,L)*spin_mat(K,Ia,1)*dconjg(spin_mat(L,Ia,-1))/DENOM(L)
          rhom11 = rhom11+CF(K,L)*spin_mat(K,Ia,-1)*dconjg(spin_mat(L,Ia,1))/DENOM(L)
          rhom1m1 = rhom1m1+CF(K,L)*spin_mat(K,Ia,-1)*dconjg(spin_mat(L,Ia,-1))/DENOM(L)
          ENDDO
      ENDDO
      ENDDO

      return


      endif ! end if J_qn= ...

C************************************ 
C     sum over colored amplitudes
C***********************************

      ZTEMP = (0.D0,0.D0)
      DO L = 1, NCOLOR
          DO K = 1, NCOLOR
          ZTEMP = ZTEMP+CF(K,L)*hel_amp(K)*dconjg(hel_amp(L))/DENOM(L)
          ENDDO
c          PROJ_chi =proj_chi+ZTEMP*DCONJG(JAMP(L))/DENOM(L)
      ENDDO
      PROJ_CHI = DBLE(ztemp)
c      write(*,*) 'proj_chi',proj_chi
c      pause 
 
c
c
c      Here I directly sum over the chi_c polarization (not used anymore)
c
c      ztemp=(0d0,0d0)
c      do Ia=2,4
c      do IB=2,4
c      DO K=1,ncolor
c      DO L=1,ncolor
c      if (j_qn.eq.0) then
c      ztemp=ztemp+amp(Ia,Ia,K)*DCONJG(amp(Ib,Ib,L))/3d0
c     &  *CF(K,L)/DENOM(L)
c       elseif (j_qn.eq.1) then
c      ztemp=ztemp+(amp(Ia,Ib,K)*DCONJG(amp(Ia,Ib,L))-
c     &     amp(Ia,Ib,K)*DCONJG(amp(Ib,Ia,L)))
c     &  *CF(K,L)/DENOM(L)/2d0
c      elseif (j_qn.eq.2) then
c      ztemp=ztemp+((amp(Ia,Ib,K)*DCONJG(amp(Ia,Ib,L))+
c     &     amp(Ia,Ib,K)*DCONJG(amp(Ib,Ia,L)))/2d0-
c     &   amp(Ia,Ia,K)*DCONJG(amp(Ib,Ib,L))/3d0)
c     &  *CF(K,L)/DENOM(L)
c      endif
c      enddo
c      enddo
c      enddo
c      enddo
c
c      PROJ_CHI = DBLE(ztemp)

      RETURN 
      END 
C



      subroutine spin_projection(P,NHEL,IC,TEMP)
c***********************************************************
c     This subroutine performs the spin projection.
c     The relative momentum is recorded in common block
c     NHEL runs from 1 to nexternal-2, since the sum over 
c     helicities of the bound state is performed in proj_matrix
c***********************************************************
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                    ! number of  particles
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
c      double precision Ponium(0:3), Pboost(0:3)
C
C     LOCAL VARIABLES
C
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K
      DOUBLE COMPLEX JIO(6),TEMP(4,NCOLOR)
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3),PTOT(0:3)   ! MOMENTA OF c AND c~ AND TOTAL MOMENTUM
      double precision Pthree2,Eq2_quark1,Eq2_quark2
C
      COMPLEX*16 JAMP(NCOLOR)
      DOUBLE PRECISION WEIGHT1, WEIGHT2
c      DOUBLE COMPLEX PROJAMP(NCOLOR)
C
C     global
C
      double precision prel(0:3)
      common /to_pWAVE/prel

      double precision qmass(2)
      COMMON/to_qmass/qmass

c      include 'qmass.inc'

      Pthree2=P(1,nexternal)**2
     & +P(2,nexternal)**2+P(3,nexternal)**2
      if(Pthree2.gt.0.001d0) then
      write(*,*) 'warning : we are not in the Quarkonium rest frame'
      endif
      Eq2_quark1=prel(1)**2+prel(2)**2+prel(3)**2+qmass(1)**2
      Eq2_quark2=prel(1)**2+prel(2)**2+prel(3)**2+qmass(2)**2
      PTOT(0)=dsqrt( Eq2_quark1)+dsqrt( Eq2_quark2)
      PTOT(1)=0d0
      PTOT(2)=0d0
      PTOT(3)=0d0

      weight1=dsqrt(Eq2_quark1)/(dsqrt(Eq2_quark1)+dsqrt(Eq2_quark2))
      weight2=dsqrt(Eq2_quark2)/(dsqrt(Eq2_quark1)+dsqrt(Eq2_quark2))

      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      ENDDO
C                 in the amplitude
      P_AMP(L1,next_amp-1)=PTOT(L1)*weight1+prel(L1)
      P_AMP(L1,next_amp)=PTOT(L1)*weight2-prel(L1)
      P1(L1)=PTOT(L1)*weight1+prel(L1)
      P2(L1)=PTOT(L1)*weight2-prel(L1)
c
      ENDDO
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 
C
C     INITIALIZATION
C
      DO J=1,4,1
      DO K=1,NCOLOR
       TEMP(J,K)=(0.0D0,0.0D0)
      ENDDO
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2  
      CALL VECTOR0(P1,P2,L1,L2,qmass(1),qmass(2),JIO)! for amp.

      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)         ! for amp.
C 
C 

      DO K=1, NCOLOR    ! INDEX OF JAMP
          DO J=1,4,1   ! PAY ATTENTION TO THE RANGE : J IN [1,4]
C                  THE COLOR FACTOR IS INCLUDED IN SQAMP
          TEMP(J,K)=TEMP(J,K)+JAMP(K)*JIO(J)
          ENDDO
      ENDDO
C 
C  
      ENDDO
      ENDDO
c
c     above, J is the spin index, K is the color index
c
      return
      end
      real*8 function PROJ_H(PP,NHEL,IC)
C
C******************************************************************************
C     THIS FUNCTION PROJECTS THE AMPLITUDE ONTO A P wave CC~ STATE             * 
C     AND SQUARE IT                                                           *
C     THE ROUTINE IS UNIVERSAL.                                               *     
C     MOMENTA MUST BE GIVEN IN THE QUARKONIUM REST FRAME                      *
C******************************************************************************
C
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     ARGUMENTS
C
      REAL*8 PP(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C
C     LOCAL VARIABLES
C
C         1. for the projection
      DOUBLE PRECISION Pboost(0:3) !,Pboostnd(0:3)
      REAL*8 P(0:3,NEXTERNAL)
c      double precision Ponium(0:3),Poniumnd(0:3)
C
      DOUBLE COMPLEX TEMP(NCOLOR),
     & TEMP1(NCOLOR),TEMP2(NCOLOR),TEMP3(NCOLOR)
C
      DOUBLE COMPLEX AMP(3,NCOLOR)
      DOUBLE COMPLEX hel_AMP(NCOLOR),eps(3,-1:1)
C
C         2.  to  square the amplitude
      COMPLEX*16 ZTEMP
      integer j,k,l,ia,mu
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass      
C
C     global
C
      double precision prel(0:3)
      common /to_pWAVE/prel

      double precision pnd(0:3,nexternal),delta
c      common/modif_p/pnd,delta
C
C----
C  Begin code
C----

      delta=0.0001d0      
c
c   go to the quarkonium rest frame
c

       pboost(0)=pp(0,nexternal)
c       pboostnd(0)=pnd(0,nexternal)
       do k=1,3
       pboost(k)=-pp(k,nexternal)
c       pboostnd(k)=-pnd(k,nexternal)
       enddo
       do j=1,nexternal
       call boostx(PP(0,j),pboost,P(0,j))
c       call boostx(Pnd(0,j),pboostnd,Pnd(0,j))
        do mu=0,3
        Pnd(mu,j)=P(mu,j)
        enddo
        enddo
 
        Pnd(0,nexternal)=dsqrt(P(0,nexternal)**2+delta**2)

c
c      a) no relative momentum
c
       do k=0,3
       prel(K)=0d0
       enddo
c

       prel(1)=delta/10000d0
       call spin_singlet_proj(P,NHEL,IC,TEMP)

c
c       a) component q_1
c
       prel(1)=delta

       call spin_singlet_proj(Pnd,NHEL,IC,TEMP1)
c
c       b) component q_2
c
       prel(1)=0d0
       prel(2)=delta

       call spin_singlet_proj(Pnd,NHEL,IC,TEMP2)
c
c       c) component q_3
c
       prel(2)=0d0
       prel(3)=delta

       call spin_singlet_proj(Pnd,NHEL,IC,TEMP3)
c
c      computing derivatives
c
       do k=1,NCOLOR
       amp(1,k)=-(temp1(k)-temp(k))/dcmplx(delta)
       amp(2,k)=-(temp2(k)-temp(k))/dcmplx(delta)
       amp(3,k)=-(temp3(k)-temp(k))/dcmplx(delta)
       enddo

C************************************
C    Selecting helicity state
C    and squaring the amplitude
C************************************

      call get_helicity(PP(0,nexternal),eps)

      do L=1,ncolor
        hel_amp(l)=(0d0,0d0)
        do Ia=1,3
          hel_amp(l)=hel_amp(l)+amp(Ia,l)*eps(Ia,nhel(nexternal))
        enddo
      enddo
 
      ztemp=(0d0,0d0)
      DO K=1,ncolor
      DO L=1,ncolor
      ztemp=ztemp+hel_amp(K)
     &*DCONJG(hel_amp(L)) 
     &  *CF(K,L)/DENOM(L)
      enddo
      enddo

      PROJ_H = DBLE(ztemp)
      RETURN 
      END 
C



      subroutine spin_singlet_proj(P,NHEL,IC,TEMP)
c***********************************************************
c     This subroutine performs the spin projection.
c     The relative momentum is recorded in common block
c     NHEL runs from 1 to nexternal-1, since the sum over 
c     helicities of the bound state is performed in proj_matrix
c***********************************************************
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                    ! number of  particles
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
c      double precision Ponium(0:3), Pboost(0:3)
C
C     LOCAL VARIABLES
C
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K
      DOUBLE COMPLEX PS,TEMP(NCOLOR)
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3),PTOT(0:3)   ! MOMENTA OF c AND c~ AND TOTAL MOMENTUM
      double precision Pthree2,Eq2_quark1,Eq2_quark2,weight1,weight2
C
      COMPLEX*16 JAMP(NCOLOR)
c      DOUBLE COMPLEX PROJAMP(NCOLOR)
C
C     global
C

      double precision prel(0:3)
      common /to_pWAVE/prel

      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass



      Pthree2=P(1,nexternal)**2+P(2,nexternal)**2
     & +P(3,nexternal)**2
      if(Pthree2.gt.0.001d0) then
      write(*,*) 'warning : we are not in the Quarkonium rest frame'
      endif
      Eq2_quark1=prel(1)**2+prel(2)**2+prel(3)**2+qmass(1)**2
      Eq2_quark2=prel(1)**2+prel(2)**2+prel(3)**2+qmass(2)**2
      PTOT(0)=dsqrt(Eq2_quark1)+dsqrt(Eq2_quark2)
      PTOT(1)=0d0
      PTOT(2)=0d0
      PTOT(3)=0d0
      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      ENDDO

      weight1=dsqrt(Eq2_quark1)/(dsqrt(Eq2_quark1)+dsqrt(Eq2_quark2))
      weight2=dsqrt(Eq2_quark2)/(dsqrt(Eq2_quark1)+dsqrt(Eq2_quark2))
C                 in the amplitude
      P_AMP(L1,next_amp-1)=PTOT(L1)*weight1+prel(L1)
      P_AMP(L1,next_amp)=PTOT(L1)*weight2-prel(L1)
      P1(L1)=PTOT(L1)*weight1+prel(L1)
      P2(L1)=PTOT(L1)*weight2-prel(L1)
c
      ENDDO
c
c      boost
c
c      do k=1,next_amp
c      call boostx(p_amp(0,k),Ponium,p_amp(0,k))
c      enddo
      do L1=0,3
      P1(L1)=p_amp(L1,next_amp-1)
      P2(L1)=p_amp(L1,next_amp)
      enddo
C
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 

C
C     INITIALIZATION
C
      DO K=1,NCOLOR
       TEMP(K)=(0.0D0,0.0D0)
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2  
      CALL pseudoscalar0(P1,P2,L1,L2,
     &    qmass(1),qmass(2),PS)              ! for amp.

      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)         ! for amp.
C 
c       
C 

      DO K=1, NCOLOR    ! INDEX OF JAMP
C                  THE COLOR FACTOR IS INCLUDED IN SQAMP
          TEMP(K)=TEMP(K)+JAMP(K)*PS
      ENDDO
C 
      ENDDO
      ENDDO
c
c     above, J is the spin index, K is the color index
c
      
c
c     go back to the rest frame
c
c      do k=1,ncolor
c      call boostx(temp(1,k),pboost,temp(1,k))
c      enddo
c
      return
      end


      real*8 function PROJ_PSILEPT(P,NHEL,IC)
C
C******************************************************************************
C     THIS FUNCTION PROJECT THE AMPLITUDE ONTO A CC~(3S1) STATE               * 
C     AND SQUARE IT. The quarkonium is decayed into leptons.                  *
C     The routine is universal.                                               *
C******************************************************************************
C
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'color_data.inc'
C
C     PARAMETERS
C
      INTEGER  NEXT_AMP                    ! initial number of particles
      PARAMETER (NEXT_AMP=NEXTERNAL+1)     ! # of external particles before the projection
C
C     ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXT_amp), IC(NEXTERNAL)    ! number of helicities =nexternal  !
C
C     LOCAL VARIABLES
C
C         1. for the projection
      INTEGER NHEL_AMP(next_amp),IC_AMP(next_amp), L1,L2,J,K
      DOUBLE COMPLEX JIO(6),TEMP(4,NCOLOR), jio_lept(6)
      DOUBLE PRECISION P_AMP(0:3,next_amp),
     &                  P1(0:3),P2(0:3),PTOT(0:3)   ! MOMENTA OF c AND c~ AND TOTAL MOMENTUM

      DOUBLE PRECISION WEIGHT1, WEIGHT2
c      DOUBLE PRECISION EPS(0:3,-1:1)
C
      COMPLEX*16 JAMP(NCOLOR)
      DOUBLE COMPLEX PROJAMP(NCOLOR)
C
C         3.  to  square the amplitude
      COMPLEX*16 ZTEMP
c
      double precision k1(0:3),k2(0:3),three_k1(3)
      common /lepton_stuff/ k1,k2,three_k1
C
c--masses
      DOUBLE PRECISION       qmass(2)
      COMMON/to_qmass/qmass      
C
C     SPECIFYING THE MOMENTA OF C,C~ + THE TOTAL MOMENTUM
C

      weight1=qmass(1)/(qmass(1)+qmass(2))
      weight2=qmass(2)/(qmass(1)+qmass(2))

      DO L1=0,3
      DO L2=1, next_amp-2
      P_AMP(L1,L2)=P(L1,L2)
      ENDDO
      P_AMP(L1,next_amp-1)=P(L1,nexternal)*WEIGHT1
      P_AMP(L1,next_amp)=P(L1,nexternal)*WEIGHT2
      P1(L1)=P(L1,nexternal)*WEIGHT1
      P2(L1)=P(L1,nexternal)*WEIGHT2
      PTOT(L1)=P1(L1)+P2(L1)
      ENDDO

C
C     RECORD THE POLARIZATIONS
C
      DO J=1,next_amp-2
      NHEL_AMP(J)=NHEL(J)
      IC_AMP(J)=IC(J)
      ENDDO
      IC_AMP(next_amp-1)=1
      IC_AMP(next_amp)=1
C
C************************************* 
C     START THE SUM   
C************************************* 
C
C     INITIALIZATION
C
      DO J=1,4,1
      DO K=1,NCOLOR
       TEMP(J,K)=(0.0D0,0.0D0)
      ENDDO
      ENDDO

      DO L1=-1,1,2  ! SUM OVER HELICITIES
      DO L2=-1,1,2
      NHEL_AMP(next_amp-1)=L1
      NHEL_AMP(next_amp)=L2  
      CALL VECTOR0(P1,P2,L1,L2,qmass(1),qmass(2),JIO)


      CALL matrix(P_AMP,NHEL_AMP,IC_AMP,JAMP)
C 
c       
      DO K=1, NCOLOR    ! INDEX OF JAMP
          DO J=1,4,1   ! PAY ATTENTION TO THE RANGE : J IN [1,4]
          TEMP(J,K)=TEMP(J,K)+JAMP(K)*JIO(J)  ! THE COLOR FACTOR IS INCLUDED IN SQAMP
          ENDDO
      ENDDO
C 
      ENDDO
      ENDDO
C       
      l1=nhel(next_amp-1)
      l2=nhel(next_amp)
      call leptonic_current(k1,k2,l1,l2,qmass(1), jio_lept)
C
      DO K=1,NCOLOR
      PROJAMP(K)=TEMP(1,K)*jio_lept(1)-
     &  TEMP(2,K)*jio_lept(2)-
     & TEMP(3,K)*jio_lept(3)- 
     & TEMP(4,K)*jio_lept(4)
      ENDDO
C     

C************************************ 
C    SQUARE THE AMPLITUDE
C***********************************
C
      PROJ_PSILEPT = 0.D0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*PROJAMP(J)
          ENDDO
          PROJ_PSILEPT =PROJ_PSILEPT
     &         +DBLE(ZTEMP*DCONJG(PROJAMP(I)))/DENOM(I)
      ENDDO
c      write(*,*) 'result ', projected_matrix
      RETURN 
      END 
C
      subroutine pseudoscalar(P1,P2,LAMBDA1,LAMBDA2,MC,PS)
c
c This subroutine computes the relevant current for s=0 projection.
c
c    input : P1 = momentum of c
c            P2 = momentum of c~
c            LAMBDA1 = helicity of c
c            LAMBDA2 = helicity of c~
c
c    output : PS = the scalar current
c           
c   
      implicit none
      double complex fi(6),fo(6),PS, MOMENTUM(2) 
      double precision q(0:3),MC,Norm,q2     

      DOUBLE COMPLEX MAT1(1:4,1:4),GAM5(1:4,1:4),MATRIX(1:4,1:4) 
c
c     "MAT1" IS THE MATRIX   SL(q)+2 E
c     "MATRIX" IS THE MATRIX    GAMMA^5 * MAT1
c 
      DOUBLE PRECISION E
      integer J1, J2,K,LAMBDA1,LAMBDA2                         

      DOUBLE PRECISION P1(0:3), P2(0:3)       
      double complex cImag, cZero, cOne
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ),
     & cOne=(1.0D0,0.0D0) )

      CALL IXXXXX(P1,MC,LAMBDA1,1,FI)     
      CALL OXXXXX(P2,MC,LAMBDA2,-1,FO)

      MOMENTUM(1) = -fo(5)+fi(5)
      MOMENTUM(2) = -fo(6)+fi(6)

      q(0) = dble( MOMENTUM(1))   
      q(1) = dble( MOMENTUM(2))   
      q(2) = dimag(MOMENTUM(2))   
      q(3) = dimag(MOMENTUM(1))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      E=dsqrt(q2)/2   
C
C      DEFINITION OF MAT1
C 
      MAT1(1,1)=2*E
      MAT1(1,2)=cZero
      MAT1(1,3)=q(0)-q(3)
      MAT1(1,4)=-q(1)+cImag*q(2)
      MAT1(2,1)=cZero
      MAT1(2,2)=2*E
      MAT1(2,3)=-q(1)-cImag*q(2)
      MAT1(2,4)=q(0)+q(3)
      MAT1(3,1)=q(0)+q(3)
      MAT1(3,2)=q(1)-cImag*q(2)
      MAT1(3,3)=2*E
      MAT1(3,4)=cZero
      MAT1(4,1)=q(1)+cImag*q(2)
      MAT1(4,2)=q(0)-q(3)
      MAT1(4,3)=cZero
      MAT1(4,4)=2*E 
C
C     INITIALIZATION OF GAMMA MATRICES
C 
      DO J1=1,4,1
        DO J2=1,4,1
          GAM5(J1,J2)=cZero
        ENDDO
      ENDDO
C
C     DEFINITION OF GAMMA 5 MATRIX
C     CHIRAL REPRESENTATION
C:$

c     gamma 5

      GAM5(1,1)=-cOne
      GAM5(2,2)=-cOne
      GAM5(3,3)=+cOne
      GAM5(4,4)=+cOne

c
c     DEFINITION OF MAT2 = GAMMA^MU (SL(q)+2E)

      DO J1=1,4,1
        DO J2=1,4,1
          MATRIX(J1,J2)=cZero
          DO K=1,4,1 
            MATRIX(J1,J2)=MATRIX(J1,J2)+GAM5(J1,K)*MAT1(K,J2)
          ENDDO
        ENDDO
      ENDDO
C

C    NORMALIZATION - does not contain the factor 1/sqrt(E) from the wave function

      Norm=1.0D0/(4*E*DSQRT(2.0D0)*(E+MC))

c    DEFINITION OF THE CURRENT


      PS=0
        DO J1=1,4,1
          DO J2 =1,4,1
          PS=PS+Norm*FO(J1)*MATRIX(J1,J2)*FI(J2)
          ENDDO
        ENDDO
      return
      end
      subroutine pseudoscalar0(P1,P2,LAMBDA1,LAMBDA2,M1,M2,PS)
c
c This subroutine computes the relevant current for s=0 projection.
c
c    input : P1 = momentum of c
c            P2 = momentum of c~
c            LAMBDA1 = helicity of c
c            LAMBDA2 = helicity of c~
c
c    output : PS = the scalar current
c           
c   
      implicit none
      double complex fi(6),fo(6),PS
      double precision M1,M2,N 
      integer lambda1,lambda2

      DOUBLE PRECISION P1(0:3), P2(0:3)       
      double complex cImag, cZero, cOne
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ),
     & cOne=(1.0D0,0.0D0) )

      CALL IXXXXX(P1,M1,LAMBDA1,1,FI)
      CALL OXXXXX(P2,M2,LAMBDA2,-1,FO)

      N=dsqrt(8.0d0*M1*M2)

      PS = (-fo(1)*fi(1)-fo(2)*fi(2)+fo(3)*fi(3)+fo(4)*fi(4))/N
      return
      end
      subroutine vector0(P1,P2,LAMBDA1,LAMBDA2,M1,M2,jio)
c
c This subroutine computes an off-shell vector current from an external
c fermion pair.  The vector boson propagator is given in Feynman gauge
c for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of OUTPUT vector v
c       real    vwidth         : width of OUTPUT vector v
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c     
      implicit none
      double complex fi(6),fo(6),gc(2),jio(6)
      double precision q(0:3),q2,p1(0:3),p2(0:3),N
      integer lambda1,lambda2
      double precision m1,m2
      
      double complex cImag
      parameter( cImag = ( 0.0d0, 1.0d0 ) )

      gc(1)=(1.0d0,0.0d0)
      gc(2)=(1.0d0,0.0d0)

      CALL IXXXXX(P1,M1,LAMBDA1,1,FI)     
      CALL OXXXXX(P2,M2,LAMBDA2,-1,FO)

      N=dsqrt(8.0d0*M1*M2)
      jio(5) = -fo(5)+fi(5)
      jio(6) = -fo(6)+fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)




            jio(1) = ( gc(1)*( fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)+fo(2)*fi(4)) )/N
            jio(2) = (-gc(1)*( fo(3)*fi(2)+fo(4)*fi(1))
     &                +gc(2)*( fo(1)*fi(4)+fo(2)*fi(3)) )/N
            jio(3) = ( gc(1)*( fo(3)*fi(2)-fo(4)*fi(1))
     &                +gc(2)*(-fo(1)*fi(4)+fo(2)*fi(3)))
     &               *cImag/N
            jio(4) = ( gc(1)*(-fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)-fo(2)*fi(4)) )/N

c 
      return
      end
      subroutine VECTOR(P1,P2,LAMBDA1,LAMBDA2,MQ,jio)
c
c This subroutine computes the relevant current for s=1 projection.
c
c    input : P1 = momentum of c
c            P2 = momentum of c~
c            LAMBDA1 = helicity of c
c            LAMBDA2 = helicity of c~
c
c    output : jio(1->4) = the current,
c             jio(5->6) = the total momentum q=p1+p2  
c   
      implicit none
      double complex fi(6),fo(6),jio(6)  
      double precision q(0:3),MQ,Norm,q2     

      DOUBLE COMPLEX MAT1(1:4,1:4),GAM(1:4,1:4,0:3),MATRIX(1:4,1:4,0:3) 
c
c     "MAT1" IS THE MATRIX   SL(q)+2 E
c     "MATRIX" IS THE MATRIX    GAMMA^\MU * MAT1
c 
      DOUBLE PRECISION E
      integer J1, J2,K,L,LAMBDA1,LAMBDA2                         
      DOUBLE PRECISION P1(0:3), P2(0:3)       
      double complex cImag, cZero, cOne
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ),
     & cOne=(1.0D0,0.0D0) )

      CALL IXXXXX(P1,MQ,LAMBDA1,1,FI)     
      CALL OXXXXX(P2,MQ,LAMBDA2,-1,FO)

      jio(5) = -fo(5)+fi(5)
      jio(6) = -fo(6)+fi(6)

      q(0) = dble( jio(5))   
      q(1) = dble( jio(6))   
      q(2) = dimag(jio(6))   
      q(3) = dimag(jio(5))   
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)
      E=dsqrt(q2)/2   
C
C      DEFINITION OF MAT1
C 
      MAT1(1,1)=2*E
      MAT1(1,2)=cZero
      MAT1(1,3)=q(0)-q(3)
      MAT1(1,4)=-q(1)+cImag*q(2)
      MAT1(2,1)=cZero
      MAT1(2,2)=2*E
      MAT1(2,3)=-q(1)-cImag*q(2)
      MAT1(2,4)=q(0)+q(3)
      MAT1(3,1)=q(0)+q(3)
      MAT1(3,2)=q(1)-cImag*q(2)
      MAT1(3,3)=2*E
      MAT1(3,4)=cZero
      MAT1(4,1)=q(1)+cImag*q(2)
      MAT1(4,2)=q(0)-q(3)
      MAT1(4,3)=cZero
      MAT1(4,4)=2*E 
C
C     INITIALIZATION OF GAMMA MATRICES
C 
      DO J1=1,4,1
        DO J2=1,4,1
          DO L=0,3,1
          GAM(J1,J2,L)=cZero
          ENDDO
        ENDDO
      ENDDO

C
C     DEFINITION OF GAMMA MATRICES
C     CHIRAL REPRESENTATION
C
c     gamma 0

      GAM(1,3,0)=cOne
      GAM(2,4,0)=cOne
      GAM(3,1,0)=cOne
      GAM(4,2,0)=cOne

c
c     gamma 1
c

      GAM(1,4,1)=cOne
      GAM(2,3,1)=cOne
      GAM(3,2,1)=-cOne
      GAM(4,1,1)=-cOne
c
c     gamma 2
c

      GAM(1,4,2)=-cImag
      GAM(2,3,2)=cImag
      GAM(3,2,2)=cImag
      GAM(4,1,2)=-cImag
c
c
c     gamma 3
c

      GAM(1,3,3)=cOne
      GAM(2,4,3)=-cOne
      GAM(3,1,3)=-cOne
      GAM(4,2,3)=cOne

c
c     DEFINITION OF MAT2 = GAMMA^MU (SL(q)+2E)

      DO L=0,3,1
        DO J1=1,4,1
          DO J2=1,4,1
          MATRIX(J1,J2,L)=cZero
            DO K=1,4,1
            MATRIX(J1,J2,L)=MATRIX(J1,J2,L)+GAM(J1,K,L)*MAT1(K,J2)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
C

C    NORMALIZATION - does not contain the factor 1/sqrt(E) from the wave function

      Norm=1.0D0/(4*E*DSQRT(2.0D0)*(E+MQ))

c    DEFINITION OF THE CURRENT


      DO L=0,3,1
        jio(L+1)=0
        DO J1=1,4,1
          DO J2 =1,4,1
          jio(L+1)=jio(L+1)+Norm*FO(J1)*MATRIX(J1,J2,L)*FI(J2)
          ENDDO
        ENDDO
      ENDDO


      return
      end
      subroutine vectorBc(P1,P2,LAMBDA1,LAMBDA2,M1,M2,jio)
c
c This subroutine computes an off-shell vector current from an external
c fermion pair.  The vector boson propagator is given in Feynman gauge
c for a massless vector and in unitary gauge for a massive vector.
c
c input:
c       complex fi(6)          : flow-in  fermion                   |fi>
c       complex fo(6)          : flow-out fermion                   <fo|
c       complex gc(2)          : coupling constants                  gvf
c       real    vmass          : mass  of OUTPUT vector v
c       real    vwidth         : width of OUTPUT vector v
c
c output:
c       complex jio(6)         : vector current          j^mu(<fo|v|fi>)
c     
      implicit none
      double complex fi(6),fo(6),gc(2),jio(6)
      double precision q(0:3),q2,p1(0:3),p2(0:3),N
      integer lambda1,lambda2
      double precision m1,m2
      
      double precision rZero, rOne
      parameter( rZero = 0.0d0, rOne = 1.0d0 )
      double complex cImag, cZero
      parameter( cImag = ( 0.0d0, 1.0d0 ), cZero = ( 0.0d0, 0.0d0 ) )

      gc(1)=(1.0d0,0.0d0)
      gc(2)=(1.0d0,0.0d0)

      CALL IXXXXX(P1,M1,LAMBDA1,1,FI)     
      CALL OXXXXX(P2,M2,LAMBDA2,-1,FO)

      N=dsqrt(2d0*2d0*M1*2d0*M2)
      jio(5) = -fo(5)+fi(5)
      jio(6) = -fo(6)+fi(6)

      q(0) = dble( jio(5))
      q(1) = dble( jio(6))
      q(2) = dimag(jio(6))
      q(3) = dimag(jio(5))
      q2 = q(0)**2-(q(1)**2+q(2)**2+q(3)**2)




            jio(1) = ( gc(1)*( fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)+fo(2)*fi(4)) )/N
            jio(2) = (-gc(1)*( fo(3)*fi(2)+fo(4)*fi(1))
     &                +gc(2)*( fo(1)*fi(4)+fo(2)*fi(3)) )/N
            jio(3) = ( gc(1)*( fo(3)*fi(2)-fo(4)*fi(1))
     &                +gc(2)*(-fo(1)*fi(4)+fo(2)*fi(3)))
     &               *cImag/N
            jio(4) = ( gc(1)*(-fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)-fo(2)*fi(4)) )/N

c 
      return
      end
      subroutine leptonic_current(k1,k2,l1,l2,mq, jio)
c
      implicit none
c
c     arguments
c
      double precision k1(0:3),k2(0:3),mq
      integer l1,l2
      double complex jio(6)
c
c     local
c
      double complex fi(6),fo(6),gc(2)
      double precision N
c
      double precision rZero,  pi
      parameter( rZero = 0.0d0, PI=3.1415926d0 )
      double complex cImag
      parameter( cImag = ( 0.0d0, 1.0d0 ))
c
      N=dsqrt(3.0d0)/(8.0d0*MQ*dsqrt(pi))
c
      call oxxxxx(k1,rZero,l1,1,FO)
      call ixxxxx(k2,rZero,l2,-1,FI)
c
      gc(1)=(1.0d0,0.0d0)
      gc(2)=(1.0d0,0.0d0)
c
      jio(5) = fo(5)-fi(5)
      jio(6) = fo(6)-fi(6)
c
            jio(1) = ( gc(1)*( fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)+fo(2)*fi(4)) )*N
            jio(2) = (-gc(1)*( fo(3)*fi(2)+fo(4)*fi(1))
     &                +gc(2)*( fo(1)*fi(4)+fo(2)*fi(3)) )*N
            jio(3) = ( gc(1)*( fo(3)*fi(2)-fo(4)*fi(1))
     &                +gc(2)*(-fo(1)*fi(4)+fo(2)*fi(3)))
     &               *N*cImag
            jio(4) = ( gc(1)*(-fo(3)*fi(1)+fo(4)*fi(2))
     &                +gc(2)*( fo(1)*fi(3)-fo(2)*fi(4)) )*N
c
      return
      end

      subroutine genmom_lept(P,x,k1,k2,three_k1)
C************************************************************************
C     GIVEN X(2) AND P SETS UP THE MOMENTUM k1-k2
C     AND ALSO SETS UP THE APPROPRIATE
C     JACOBIAN VALUE = JAC AND PHASESPACE WEIGTH PSWGT
C     X(1) = COSTHETA OF FIRST DECAY psi -> mu mu
C     X(2) = PHI OF FIRST DECAY psi -> mu mu
C************************************************************************
      IMPLICIT NONE
C     
C     CONSTANTS
C
      REAL*8     ZERO    ,      PI
      PARAMETER (ZERO=0D0,  PI=3.1415926d0 )
C
C     ARGUMENTS
C
      REAL*8 P(0:3),k1(0:3),k2(0:3)
      REAL*8 three_k1(3)
      REAL*8 X(2)
C
C     LOCAL
C
      REAL*8 RSH,SH,COSTH,PHI
      integer j
C
C     EXTERNAL
C
      REAL*8 DOT
C
C     GLOBAL
C
      DOUBLE PRECISION              S,X1,X2,PSWGT,JAC
      COMMON /PHASESPACE/ S,X1,X2,PSWGT,JAC
c
      SH = DOT(P,P)
      RSH = dsqrt(SH)
C
C       PICK VALUE OF THETA AND PHI FOR FIRST DECAY SH->M1+S1
C
        COSTH=-1.D0+2.D0*X(1)
        PHI = 2D0*PI*X(2)
C
C       DETERMINE JACOBIAN FOR THETA AND PHI
C     
        JAC =  JAC * 4D0*PI
C
C       CALCULATE COMPONENTS OF MOMENTUM FOUR-VECTORS
C       OF THE FINAL STATE MASSLESS PARTONS IN THEIR CM FRAME
C
        CALL MOM2CX(RSH,zero,zero,COSTH,PHI,k1,k2)
c
c       record three_k1
c
        do j=1,3
        three_k1(j)=k1(j)
        enddo
c
c       boost the momenta in the lab frame
c
        call boostx(k1,P,k1)
        call boostx(k2,P,k2)

      RETURN
      END
C
C*********************************************************************
      SUBROUTINE polariz(P,EPS)
C*********************************************************************
C 
C     ARGUMENTS 
      IMPLICIT NONE
C 
      DOUBLE PRECISION P(0:3),EPS(0:3,-1:1)
      
C
C     LOCAL
C 
      DOUBLE PRECISION E1(1:3),E2(1:3),E3(1:3)
      DOUBLE PRECISION J1(1:3),J2(1:3)
      DOUBLE PRECISION ALPHA(3),BETA(2)
      INTEGER J
      DOUBLE PRECISION SQP, SQNORM_P
      DOUBLE PRECISION EPS_SQMIN,EPS_SQMAX,j2_epsmin,norm1,norm2
C 
C     EXTERNAL
C     
      DOUBLE PRECISION DOT
      EXTERNAL DOT
C
C      PARAMETER
C
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D0)
C
C     POLARIZATIONS OF J=1 STATE
C 
      E1(1)=1.0D0
      E1(2)=0.0D0
      E1(3)=0.0D0
C 
      E2(1)=0.0D0
      E2(2)=1.0D0
      E2(3)=0.0D0
C 
      E3(1)=0.0D0
      E3(2)=0.0D0
      E3(3)=1.0D0

C
C     IF THE STATE IS AT REST, THE SUBROUTINE RETURNS 
C     THE THREE VECTORS ABOVE
C
      IF (P(1).EQ.zero.AND.P(2).eq.zero.AND.P(3).eq.zero) THEN
      DO J=1,3,1
      EPS(J,-1)=E1(J)
      EPS(J,0)=E2(J)
      EPS(J,1)=E3(J)
      EPS(0,-1)=ZERO
      EPS(0,0)=ZERO
      EPS(0,1)=ZERO
      ENDDO
      RETURN
      ENDIF
C 
C     PROJECTIONS
C

      ALPHA(1)=P(1)
      ALPHA(2)=P(2)
      ALPHA(3)=P(3)

      IF (dabs(ALPHA(1)).LT.dabs(ALPHA(3)).AND.dabs(ALPHA(2))
     &.LT.dabs(ALPHA(3))) THEN
      DO J=1,3,1
      J1(J)=E1(J)
      J2(J)=E2(J)
      ENDDO
      BETA(1)=ALPHA(1)
      BETA(2)=ALPHA(2)
      ELSEIF (dabs(ALPHA(1)).LT.dabs(ALPHA(2)).AND.dabs(ALPHA(3))
     & .LT.dabs(ALPHA(2))) THEN
      DO J=1,3,1
      J1(J)=E1(J)
      J2(J)=E3(J)
      ENDDO
      BETA(1)=ALPHA(1)
      BETA(2)=ALPHA(3)
      ELSEIF (dabs(ALPHA(2)).LT.dabs(ALPHA(1)).AND.dabs(ALPHA(3)).LT.
     & dabs(ALPHA(1))) THEN
      DO J=1,3,1
      J1(J)=E2(J)
      J2(J)=E3(J)
      ENDDO 
      BETA(1)=ALPHA(2)
      BETA(2)=ALPHA(3)
      ENDIF
C
      SQNORM_P=P(1)**2+P(2)**2+P(3)**2
C
      EPS_SQMIN=0d0
      EPS_SQMAX=0d0
      j2_epsmin=0d0
      DO J=1,3,1
      EPS(J,-1)=J1(J)-BETA(1)*P(J)/SQNORM_P
C
      EPS_SQMIN=EPS_SQMIN+EPS(J,-1)*EPS(J,-1)
      j2_epsmin=j2_epsmin+EPS(j,-1)*J2(J) 
      ENDDO
C
      NORM1=DSQRT(EPS_SQMIN)
      j2_epsmin=j2_epsmin/norm1
C
      DO J=1,3
      EPS(J,-1)=EPS(J,-1)/norm1
      EPS(J,1)=J2(J)-BETA(2)*P(J)/SQNORM_P-j2_epsmin*EPS(J,-1)
      EPS_SQMAX=EPS_SQMAX+EPS(J,1)*EPS(J,1)
      ENDDO
      NORM2=DSQRT(EPS_SQMAX)

      EPS(0,-1)=0d0
      EPS(0,1)=0d0

      SQP=P(0)**2-SQNORM_P
      DO J=1,3,1
      EPS(J,1)=EPS(J,1)/norm2
      EPS(J,0)=P(0)*P(J)/DSQRT(SQNORM_P*SQP)
      ENDDO 
      EPS(0,0)=DSQRT(SQNORM_P/SQP)
      RETURN
      END

      subroutine get_helicity(P,eps)
c
c     Given P (momentum in the lab), this subroutine gets the polarization vector 
c     Frame: px=py=pz=0 (rest frame of the particle)
c     Quantization axis: P

c
c     arguments
c
      double precision P(0:3)
      double complex eps(3,-1:1)
c
c      local:
c
      double precision pt, pp,sqh
c---
c Begin code     
c---
      pt=dsqrt(p(1)**2+p(2)**2)
      pp=dsqrt(p(1)**2+p(2)**2+p(3)**2)
      sqh=dsqrt(1d0/2d0)
c
      eps(1,0)=dcmplx(p(1)/pp)
      eps(2,0)=dcmplx(p(2)/pp)
      eps(3,0)=dcmplx(p(3)/pp)
c
      eps(1,1)=dcmplx(-p(2)/pt*sqh , p(3)*p(1)/(pt*pp)*sqh)
      eps(2,1)=dcmplx(p(1)/pt*sqh , p(3)*p(2)/(pt*pp)*sqh)
      eps(3,1)=dcmplx( 0d0 , -pt/pp*sqh )
c
      eps(1,-1)=dcmplx( p(2)/pt*sqh , p(3)*p(1)/(pt*pp)*sqh)
      eps(2,-1)=dcmplx(-p(1)/pt*sqh , p(3)*p(2)/(pt*pp)*sqh)
      eps(3,-1)=dcmplx( 0d0 , -pt/pp*sqh )
c
      return
      end

      subroutine get_helicity2(P,eps,m)
c
c     Given P (momentum in the lab), this subroutine gets the polarization vector
c     Frame: lab
c     Quantization axis: P

c
c     arguments
c
      double precision P(0:3),m
      double complex eps(0:3,-1:1)
c
c      local:
c
      double precision pt, pp,sqh
c---
c Begin code
c---
      pt=dsqrt(p(1)**2+p(2)**2)
      pp=dsqrt(p(1)**2+p(2)**2+p(3)**2)
      sqh=dsqrt(1d0/2d0)
c 
      if (pt.ne.0d0) then
      eps(0,0)=dcmplx(pp/m)
      eps(1,0)=dcmplx(p(1)/pp*P(0)/m)
      eps(2,0)=dcmplx(p(2)/pp*P(0)/m)
      eps(3,0)=dcmplx(p(3)/pp*P(0)/m)
c
      eps(0,1)=dcmplx(0d0)
      eps(1,1)=dcmplx(-p(2)/pt*sqh , p(3)*p(1)/(pt*pp)*sqh)
      eps(2,1)=dcmplx(p(1)/pt*sqh , p(3)*p(2)/(pt*pp)*sqh)
      eps(3,1)=dcmplx( 0d0 , -pt/pp*sqh )
c
      eps(0,-1)=dcmplx(0d0)
      eps(1,-1)=dcmplx( p(2)/pt*sqh , p(3)*p(1)/(pt*pp)*sqh)
      eps(2,-1)=dcmplx(-p(1)/pt*sqh , p(3)*p(2)/(pt*pp)*sqh)
      eps(3,-1)=dcmplx( 0d0 , -pt/pp*sqh )
c
      else
  
      eps(0,0)=dcmplx(p(3)/m)
      eps(1,0)=dcmplx(0d0)
      eps(2,0)=dcmplx(0d0)
      eps(3,0)=dcmplx(P(0)/m)
c
      eps(0,1)=dcmplx(0d0)
      eps(1,1)=dcmplx(-sqh , 0d0)
      eps(2,1)=dcmplx( 0d0 , sign(sqh,p(3)))
      eps(3,1)=dcmplx( 0d0  )
c
      eps(0,-1)=dcmplx(0d0)
      eps(1,-1)=dcmplx( sqh , 0d0)
      eps(2,-1)=dcmplx(0d0 , sign(sqh,p(3)))
      eps(3,-1)=dcmplx( 0d0  )
      endif

      return
      end
