module photoabsorption
  ! The cross section of gamma+A -> A* in terms of the energy of gamma
  ! where the energy of gamma Etarget is in the rest frame of the target A
  ! In the frame of A with its energy per nucleon as En, the energy of gamma is
  ! Elab. Then, we have Etarget=(En+sqrt(En**2-mn**2))/mn*Elab, where mn is the  ! average nucleon mass. It can be approximated as
  ! Etarget = 2*Elab*En/mn*(1-mn**2/En**2/4+O(mn**4/En**4))
  implicit none
  ! it will calculate cross sections by increasing all parameters by XSIGMA sigma.
  REAL(KIND(1d0))::XSIGMA=0d0
  ! 1: use parameteric uncertainty to estimate error
  ! 2: use 2sigma uncertainty to estimate error
  ! When NEUTRON_ERROR_ESTIMATE=2, XSIGMA > 0.5d0 (upper limit)
  !                                       < -0.5d0 (lower limit)
  !                                       otherwise (central value)
  INTEGER::NEUTRON_ERROR_ESTIMATE=2
contains

  ! absorption cross section (mb)
  FUNCTION GAMMAABS(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1d0))::GAMMAABS
!    GAMMAABS=GAMMAABS_NC(E_gamma,A,Z)
!    RETURN
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       ! Pb208
       GAMMAABS=SIGMAABS_Pb208(E_gamma)
    ELSEIF(A.EQ.197.AND.Z.EQ.79)THEN
       ! Au197
       GAMMAABS=SIGMAABS_Au197(E_gamma)
    ELSEIF(A.EQ.238.AND.Z.EQ.92)THEN
       ! U238
       GAMMAABS=SIGMAABS_U238(E_gamma)
    ELSEIF(A.EQ.16.AND.Z.EQ.8)THEN
       ! O16
       GAMMAABS=SIGMAABS_O16(E_gamma)
    ELSE
       WRITE(*,*) 'Only Pb208, Au197, U238, O16 are supported for now'
       WRITE(*,*) 'A=',A,'Z=',Z
       STOP
    ENDIF
    return
  END FUNCTION GAMMAABS

  ! 1n cross section (mb)
  FUNCTION GAMMATO1N(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1d0))::GAMMATO1N
!    GAMMATO1N=GAMMATO1N_NC(E_gamma,A,Z)
!    RETURN
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       ! Pb208
       GAMMATO1N=SIGMA1N_Pb208(E_gamma)
    ELSEIF(A.EQ.197.AND.Z.EQ.79)THEN
       ! Au197
       GAMMATO1N=SIGMA1N_Au197(E_gamma)
    ELSEIF(A.EQ.238.AND.Z.EQ.92)THEN
       ! U238
       GAMMATO1N=SIGMA1N_U238(E_gamma)
    ELSEIF(A.EQ.16.AND.Z.EQ.8)THEN
       ! O16
       GAMMATO1N=SIGMA1N_O16(E_gamma)
    ELSE
       WRITE(*,*) 'Only Pb208, Au197, U238, O16 are supported for now'
       WRITE(*,*) 'A=',A,'Z=',Z
       STOP
    ENDIF
    return
  END FUNCTION GAMMATO1N

  ! 2n cross section (mb)
  FUNCTION GAMMATO2N(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1d0))::GAMMATO2N
!    GAMMATO2N=GAMMATO2N_NC(E_gamma,A,Z)
!    RETURN
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       ! Pb208
       GAMMATO2N=SIGMA2N_Pb208(E_gamma)
    ELSEIF(A.EQ.197.AND.Z.EQ.79)THEN
       ! Au197
       GAMMATO2N=SIGMA2N_Au197(E_gamma)
    ELSEIF(A.EQ.238.AND.Z.EQ.92)THEN
       ! U238
       GAMMATO2N=SIGMA2N_U238(E_gamma)
    ELSEIF(A.EQ.16.AND.Z.EQ.8)THEN
       ! O16 (use >=2n approximation)
       GAMMATO2N=SIGMAX2N_O16(E_gamma)
    ELSE
       WRITE(*,*) 'Only Pb208, Au197, U238, O16 are supported for now'
       WRITE(*,*) 'A=',A,'Z=',Z
       STOP
    ENDIF
    return
  END FUNCTION GAMMATO2N

  ! 3n cross section (mb)
  FUNCTION GAMMATO3N(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1d0))::GAMMATO3N
!    GAMMATO3N=GAMMATO3N_NC(E_gamma,A,Z)
!    RETURN
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       ! Pb208
       GAMMATO3N=SIGMA3N_Pb208(E_gamma)
    ELSEIF(A.EQ.197.AND.Z.EQ.79)THEN
       ! Au197
       GAMMATO3N=SIGMA3N_Au197(E_gamma)
    ELSEIF(A.EQ.238.AND.Z.EQ.92)THEN
       ! U238
       GAMMATO3N=SIGMA3N_U238(E_gamma)
    ELSE
       WRITE(*,*) 'Only Pb208, Au197, U238 are supported for now'
       WRITE(*,*) 'A=',A,'Z=',Z
       STOP
    ENDIF
    return
  END FUNCTION GAMMATO3N

  ! 4n cross section (mb)
  FUNCTION GAMMATO4N(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1d0))::GAMMATO4N
!    GAMMATO4N=GAMMATO4N_NC(E_gamma,A,Z)
!    RETURN
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       ! Pb208
       GAMMATO4N=SIGMA4N_Pb208(E_gamma)
    ELSEIF(A.EQ.197.AND.Z.EQ.79)THEN
       ! Au197
       GAMMATO4N=SIGMA4N_Au197(E_gamma)
    ELSEIF(A.EQ.238.AND.Z.EQ.92)THEN
       ! U238
       GAMMATO4N=SIGMA4N_U238(E_gamma)
    ELSE
       WRITE(*,*) 'Only Pb208, Au197, U238 are supported for now'
       WRITE(*,*) 'A=',A,'Z=',Z
       STOP
    ENDIF
    return
  END FUNCTION GAMMATO4N

  FUNCTION DENLAN(X)
    !FUNCTION FROM CERNLIB G110
    !Ref:K.S. Kölbig and B. Schorr, A program package for the Landau distribution, Computer Phys. Comm. 31 (1984) 97--111.
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::X
    REAL(KIND(1D0))::DENLAN
    REAL(KIND(1D0))::U, V
    REAL(KIND(1D0))::P1(0:4), P2(0:4), P3(0:4), P4(0:4), P5(0:4), P6(0:4)
    REAL(KIND(1D0))::Q1(0:4), Q2(0:4), Q3(0:4), Q4(0:4), Q5(0:4), Q6(0:4)
    REAL(KIND(1D0))::A1(1:3), A2(1:2)

    DATA P1/0.4259894875D0, -0.1249762550D0, 0.3984243700D-1, -0.6298287635D-2, 0.1511162253D-2/
    DATA Q1/1.0, -0.3388260629D0, 0.9594393323D-1, -0.1608042283D-1, 0.3778942063D-2/
    
    DATA P2/0.1788541609D0, 0.1173957403D0, 0.1488850518D-1, -0.1394989411D-2, 0.1283617211D-3/
    DATA Q2/1.0, 0.7428795082D0, 0.3153932961D0, 0.6694219548D-1, 0.8790609714D-2/

    DATA P3/0.1788544503D0, 0.9359161662D-1, 0.6325387654D-2, 0.6611667319D-4, -0.2031049101D-5/
    DATA Q3/1.0, 0.6097809921D0, 0.2560616665D0, 0.4746722384D-1, 0.6957301675D-2/

    DATA P4/0.9874054407D0, 0.1186723273D3, 0.8492794360D3, -0.7437792444D3, 0.4270262186D3/
    DATA Q4/1.0, 0.1068615961D3, 0.3376496214D3, 0.2016712389D4, 0.1597063511D4/

    DATA P5/0.1003675074D1, 0.1675702434D3, 0.4789711289D4, 0.2121786767D5, -0.2232494910D5/
    DATA Q5/1.0, 0.1569424537D3, 0.3745310488D4, 0.9834698876D4, 0.6692428357D5/

    DATA P6/0.1000827619D1, 0.6649143136D3, 0.6297292665D5, 0.4755546998D6, -0.5743609109D7/
    DATA Q6/1.0, 0.651411098D3, 0.5697473333D5, 0.1659174725D6, -0.2815759939D7/

    DATA A1/0.4166666667D-1, -0.1996527778D-1, 0.2709538966D-1/
    DATA A2/-0.1845568670D1, -0.4284640743D1/
    V = X
    IF (V .LT. -5.5D0) THEN
       U = DEXP(V + 1.0D0)
       DENLAN = 0.3989422803D0*(DEXP(-1d0/U)/DSQRT(U))* &
            (1d0 + (A1(1) + (A1(2) + A1(3)*U)*U)*U)
    ELSEIF (V .LT. -1D0) THEN
       U = DEXP(-V - 1d0)
       DENLAN = DEXP(-U)*DSQRT(U)* &
            (P1(0) + (P1(1) + (P1(2) + (P1(3) + P1(4)*V)*V)*V)*V)/ &
            (Q1(0) + (Q1(1) + (Q1(2) + (Q1(3) + Q1(4)*V)*V)*V)*V)
    ELSEIF (V .LT. 1D0) THEN
       DENLAN = (P2(0) + (P2(1) + (P2(2) + (P2(3) + P2(4)*V)*V)*V)*V)/ &
            (Q2(0) + (Q2(1) + (Q2(2) + (Q2(3) + Q2(4)*V)*V)*V)*V)
    ELSEIF (V .LT. 5D0) THEN
       DENLAN = (P3(0) + (P3(1) + (P3(2) + (P3(3) + P3(4)*V)*V)*V)*V)/ &
            (Q3(0) + (Q3(1) + (Q3(2) + (Q3(3) + Q3(4)*V)*V)*V)*V)
    ELSEIF (V .LT. 12D0) THEN
       U = 1D0/V
       DENLAN = U**2*(P4(0) + (P4(1) + (P4(2) + (P4(3) + P4(4)*U)*U)*U)*U)/ &
            (Q4(0) + (Q4(1) + (Q4(2) + (Q4(3) + Q4(4)*U)*U)*U)*U)
    ELSEIF (V .LT. 50D0) THEN
       U = 1D0/V
       DENLAN = U**2*(P5(0) + (P5(1) + (P5(2) + (P5(3) + P5(4)*U)*U)*U)*U)/ &
            (Q5(0) + (Q5(1) + (Q5(2) + (Q5(3) + Q5(4)*U)*U)*U)*U)
    ELSEIF (V .LT. 300D0) THEN
       U = 1D0/V
       DENLAN = U**2*(P6(0) + (P6(1) + (P6(2) + (P6(3) + P6(4)*U)*U)*U)*U)/ &
            (Q6(0) + (Q6(1) + (Q6(2) + (Q6(3) + Q6(4)*U)*U)*U)*U)
    ELSE
       U = 1D0/(V - V*DLOG(V)/(V + 1D0))
       DENLAN = U**2*(1D0 + (A2(1) + A2(2)*U)*U)
    END IF
    RETURN
  END FUNCTION DENLAN

  FUNCTION LANDAU(X, CONSTANT, MPV, SIGMA)
    IMPLICIT NONE
    !Rewrite of DENLAN function in order to match the 3 parameters ROOT implementation
    REAL(KIND(1D0)), INTENT(IN) :: X, CONSTANT, MPV, SIGMA
    REAL(KIND(1D0)) :: LANDAU
    LANDAU = CONSTANT*DENLAN((X - MPV)/SIGMA)
    return
  END FUNCTION LANDAU

  FUNCTION LANDAUAPPROX(X, CONSTANT, MPV, SIGMA1, SIGMA2)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::X,CONSTANT,MPV,SIGMA1,SIGMA2
    REAL(KIND(1d0))::LANDAUAPPROX
    REAL(KIND(1d0))::z1,z2,exponent
    z1=(X-MPV)/SIGMA1
    z2=(X-MPV)/SIGMA2
    exponent=-0.5d0*(z1+DEXP(-z2))
    LANDAUAPPROX=CONSTANT*DEXP(exponent)
    return
  END FUNCTION LANDAUAPPROX
  
  !Usefull functions
  FUNCTION LORENTZ(X, CONSTANT, MEAN, GAMMA)
    IMPLICIT NONE
    REAL(KIND(1D0)), INTENT(IN) :: X, CONSTANT, MEAN, GAMMA
    REAL(KIND(1D0)) :: LORENTZ
    LORENTZ = CONSTANT*(X*GAMMA)**2/((X**2 - MEAN**2)**2 + (X*GAMMA)**2)
    return
  END FUNCTION LORENTZ
  
  FUNCTION MODIFIED_LORENTZ(X, CONSTANT, MEAN, GAMMA)
    IMPLICIT NONE
    REAL(KIND(1D0)), INTENT(IN) :: X, CONSTANT, MEAN, GAMMA
    REAL(KIND(1D0)) :: MODIFIED_LORENTZ
    MODIFIED_LORENTZ = CONSTANT*GAMMA**2/MEAN*X**3/((X**2 - MEAN**2)**2 + X**4*GAMMA**2/MEAN**2)
    return
  END FUNCTION MODIFIED_LORENTZ

  FUNCTION GAUSS(X, CONSTANT, MEAN, SIGMA)
    IMPLICIT NONE
    REAL(KIND(1D0)), INTENT(IN) :: X, CONSTANT, MEAN, SIGMA
    REAL(KIND(1D0)) :: GAUSS
    GAUSS = CONSTANT*DEXP(-0.5d0*((X - MEAN)/SIGMA)**2)
    return
  END FUNCTION GAUSS

  FUNCTION SIGMOID(X, X0, a)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::X,X0,a
    REAL(KIND(1d0))::SIGMOID
    SIGMOID=1d0/(1d0+DEXP(-2d0*a*(X-X0)))
    return
  END FUNCTION SIGMOID

  ! Asymptotic high-energy tail of gamma+p with Reggeon ansatz fit from Zeus (hep-ex/0202034)
  FUNCTION SIGMA_p_Reggon(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMA_p_Reggon
    REAL(KIND(1d0)),PARAMETER::mp=0.9382720813d0 ! proton mass (in GeV)
    REAL(KIND(1d0))::a,b,eps,eta,W2gap
    ! eqs.(5-6) in hep-ex/0202034
    a=57d0+XSIGMA*5d0
    b=121d0+XSIGMA*13d0
    eps=0.093d0+XSIGMA*0.002d0
    eta=0.358d0+XSIGMA*0.015d0
    W2gap=(2d0*E_gamma/1000+mp)*mp
    SIGMA_p_Reggon=a*W2gap**eps+b*W2gap**(-eta)
    SIGMA_p_Reggon=SIGMA_p_Reggon/1000d0 ! from mu b to mb
    return
  END FUNCTION SIGMA_p_Reggon

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! Fit O16 by Luca Maxia                                                         !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! absorption cross section (mb)
  FUNCTION SIGMAABS_O16(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAABS_O16
    SIGMAABS_O16=SIGMA1N_O16(E_gamma)+SIGMAXN_HE_O16(E_gamma)
    RETURN
  END FUNCTION SIGMAABS_O16

  ! high-energy tail of the absorption cross section
  FUNCTION SIGMAXN_HE_O16(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAXN_HE_O16
    REAL(KIND(1d0))::p0,a,x0
    REAL(KIND(1d0))::constlor,meanlor,gammalor
    REAL(KIND(1d0))::u,u0
    ! from which energy (MeV) use A*Rshadow*Reggon
    REAL(KIND(1d0)),PARAMETER::EXnregg=81900d0
    ! the shadowing factor in the HE tail
    REAL(KIND(1d0)),PARAMETER::Rshadow=0.82d0
    INTEGER,PARAMETER::AAA=16
    REAL(KIND(1d0))::rescalingfact=1d0
    SIGMAXN_HE_O16=0d0
    ! S2n threshold
    IF(E_gamma.LT.28.89d0)RETURN
    IF(E_gamma.LE.EXnregg)THEN
       IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
          IF(E_gamma.LE.185d0)THEN
             rescalingfact=-1d0
          ELSE
             rescalingfact=1d0
          ENDIF
          p0=1.47782584d0+rescalingfact*XSIGMA*0.01590469d0
          a=0.01380006d0+rescalingfact*XSIGMA*0.00364337d0
          x0=158.03854494d0+rescalingfact*XSIGMA*21.15543269d0
          constlor=5.05725353d0+rescalingfact*XSIGMA*0.30859858d0
          meanlor=351.97783343d0+rescalingfact*XSIGMA*7.61077610d0
          gammalor=347.39629465d0+rescalingfact*XSIGMA*25.18670363d0
       ELSE
          IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
             p0=1.47782584d0
             a=0.01380006d0
             x0=158.03854494d0
             constlor=5.05725353d0
             meanlor=351.97783343d0
             gammalor=347.39629465d0
          ELSEIF(XSIGMA.GT.0.5d0)THEN
             p0=1.58902095d0
             a=1.94024226d-2
             x0=1.10582818d2
             constlor=5.68366979d0
             meanlor=3.50392869d2
             gammalor=3.21442549d2
          ELSE
             p0=1.35435312d0
             a=5.89076734d-2
             x0=1.90076441d2
             constlor=4.48793411d0
             meanlor=3.61038457d2
             gammalor=3.51539781d2
          ENDIF
       ENDIF
       u=a*(E_gamma-x0)
       u0=a*x0
       SIGMAXN_HE_O16=p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
       SIGMAXN_HE_O16=SIGMAXN_HE_O16&
            +MODIFIED_LORENTZ(E_gamma,constlor,meanlor,gammalor)
    ELSE
       ! we use the reggeon one (we take the central value only)
       SIGMAXN_HE_O16=SIGMA_p_Reggon(E_gamma)*AAA*Rshadow
    ENDIF
    SIGMAXN_HE_O16=MAX(SIGMAXN_HE_O16,0d0)
    RETURN
  END FUNCTION SIGMAXN_HE_O16
  
  FUNCTION SIGMA1N_O16(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA1N_O16
    REAL(KIND(1d0))::constlor1,meanlor1,gammalor1
    REAL(KIND(1d0))::constlan1,meanlan1,sigmalan1
    REAL(KIND(1d0))::constlor2,meanlor2,gammalor2
    REAL(KIND(1d0))::constlan2,meanlan2,sigmalan2
    REAL(KIND(1d0))::constlan3,meanlan3,sigmalan3
    REAL(KIND(1d0))::constgaus,meangaus,sigmagaus
    SIGMA1N_O16=0d0
    ! Sn threshold
    IF(E_gamma.LT.15.66d0)RETURN
    IF(E_gamma.GT.140d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       constlor1=1.53841359d0+XSIGMA*0.12217653d0
       meanlor1=17.33471911d0+XSIGMA*0.01619269d0
       gammalor1=0.45948392d0+XSIGMA*0.06056148d0
       constlan1=1.81255372d0+XSIGMA*0.20499132d0
       meanlan1=19.12889083d0+XSIGMA*0.03488850d0
       sigmalan1=0.22285157d0+XSIGMA*0.03566147d0
       constlor2=6.99746328d0+XSIGMA*0.27195205d0
       meanlor2=22.08228205d0+XSIGMA*0.01960723d0
       gammalor2=1.02929524d0+XSIGMA*0.05906911d0
       constlan2=9.87132029d0+XSIGMA*0.31557737d0
       meanlan2=24.35781354d0+XSIGMA*0.07103115d0
       sigmalan2=1.17815356d0+XSIGMA*0.05403314d0
       constlan3=1.71511428d0+XSIGMA*0.43538579d0
       meanlan3=34.32932490d0+XSIGMA*0.99277220d0
       sigmalan3=6.84525905d0+XSIGMA*1.54804718d0
       constgaus=1.18465809d0+XSIGMA*0.15926411d0
       meangaus=61.40869615d0+XSIGMA*5.50341919d0
       sigmagaus=16.51718901d0+XSIGMA*6.23231339d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          constlor1=1.53841359d0
          meanlor1=17.33471911d0
          gammalor1=0.45948392d0
          constlan1=1.81255372d0
          meanlan1=19.12889083d0
          sigmalan1=0.22285157d0
          constlor2=6.99746328d0
          meanlor2=22.08228205d0
          gammalor2=1.02929524d0
          constlan2=9.87132029d0
          meanlan2=24.35781354d0
          sigmalan2=1.17815356d0
          constlan3=1.71511428d0
          meanlan3=34.32932490d0
          sigmalan3=6.84525905d0
          constgaus=1.18465809d0
          meangaus=61.40869615d0
          sigmagaus=16.51718901d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          constlor1=1.71322474d0
          meanlor1=17.33116683d0
          gammalor1=0.41263637d0
          constlan1=1.97211689d0
          meanlan1=19.10518491d0
          sigmalan1=0.22311056d0
          constlor2=7.00096657d0
          meanlor2=22.07213509d0
          gammalor2=1.06449215d0
          constlan2=10.05121334d0
          meanlan2=24.36359051d0
          sigmalan2=1.21478079d0
          constlan3=1.50064548d0
          meanlan3=34.93090032d0
          sigmalan3=6.55360203d0
          constgaus=1.50813284d0
          meangaus=61.93527598d0
          sigmagaus=21.67203647d0
       ELSE
          constlor1=1.43143873d0
          meanlor1=17.3396131d0
          gammalor1=0.37825016d0
          constlan1=1.54728123d0
          meanlan1=19.14489153d0
          sigmalan1=0.19487433d0
          constlor2=6.86301947d0
          meanlor2=22.08997585d0
          gammalor2=0.9627984d0
          constlan2=9.59772391d0
          meanlan2=24.34282394d0
          sigmalan2=1.15503399d0
          constlan3=1.82683549d0
          meanlan3=34.3237094d0
          sigmalan3=7.2150568d0
          constgaus=0.89050919d0
          meangaus=61.33371569d0
          sigmagaus=10.69839029d0
       ENDIF
    ENDIF
    SIGMA1N_O16=LORENTZ(E_gamma,constlor1,meanlor1,gammalor1)
    SIGMA1N_O16=SIGMA1N_O16+LANDAUAPPROX(E_gamma,constlan1,meanlan1,sigmalan1,sigmalan1)
    SIGMA1N_O16=SIGMA1N_O16+LORENTZ(E_gamma,constlor2,meanlor2,gammalor2)
    SIGMA1N_O16=SIGMA1N_O16+LANDAUAPPROX(E_gamma,constlan2,meanlan2,sigmalan2,sigmalan2)
    SIGMA1N_O16=SIGMA1N_O16+GAUSS(E_gamma,constlan3,meanlan3,sigmalan3)
    SIGMA1N_O16=SIGMA1N_O16+GAUSS(E_gamma,constgaus,meangaus,sigmagaus)
    SIGMA1N_O16=MAX(SIGMA1N_O16,0d0)
    return
  END FUNCTION SIGMA1N_O16

  ! >= 2n cross section
  FUNCTION SIGMAX2N_O16(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX2N_O16
    SIGMAX2N_O16=0d0
    ! S2n threshold
    IF(E_gamma.LT.28.89d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    SIGMAX2N_O16=SIGMAXN_HE_O16(E_gamma)
    return
  END FUNCTION SIGMAX2N_O16

  ! >= 1n cross section
  FUNCTION SIGMAX1N_O16(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX1N_O16
    SIGMAX1N_O16=SIGMAX2N_O16(E_gamma)+SIGMA1N_O16(E_gamma)
    return
  END FUNCTION SIGMAX1N_O16

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! Fit Au197 by Luca Maxia                                                       !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! absorption cross section (mb)
  FUNCTION SIGMAABS_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAABS_Au197
    SIGMAABS_Au197=SIGMA1N_Au197(E_gamma)+SIGMA2N_Au197(E_gamma)&
         +SIGMA3N_Au197(E_gamma)+SIGMA4N_Au197(E_gamma)+SIGMAXN_HE_Au197(E_gamma)
    RETURN
  END FUNCTION SIGMAABS_Au197

  ! high-energy tail of the absorption cross section
  FUNCTION SIGMAXN_HE_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAXN_HE_Au197
    REAL(KIND(1d0))::p0,a,x0
    REAL(KIND(1d0))::const,mean,gamma
    REAL(KIND(1d0))::u,u0
    ! from which energy (MeV) use A*Rshadow*Reggon
    REAL(KIND(1d0)),PARAMETER::EXnregg=81900d0
    ! the shadowing factor in the HE tail
    REAL(KIND(1d0)),PARAMETER::Rshadow=0.72d0
    INTEGER,PARAMETER::AAA=197
    SIGMAXN_HE_Au197=0d0
    ! S5n threshold
    IF(E_gamma.LT.38.73d0)RETURN
    IF(E_gamma.LE.EXnregg)THEN
       IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
          p0=6.26975066d0+XSIGMA*0.10158607d0
          a=0.00285338d0+XSIGMA*0.00061583d0
          x0=1113.15581374d0+XSIGMA*56.74519478d0
          const=71.74968521d0+XSIGMA*2.63957277d0
          mean=375.84028844d0+XSIGMA*4.77743035d0
          gamma=439.70436525d0+XSIGMA*28.89017985d0
       ELSE
          IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
             p0=6.26975066d0
             a=0.00285338d0
             x0=1113.15581374d0
             const=71.74968521d0
             mean=375.84028844d0
             gamma=439.70436525d0
          ELSEIF(XSIGMA.GT.0.5d0)THEN
             p0=6.43430436d0
             a=3.07698825d-3
             x0=1.09997857d3
             const=7.61612763d1
             mean=3.70499892d2
             gamma=4.23976068d2
          ELSE
             p0=6.10312293d0
             a=2.68267336d-3
             x0=1.13422848d3
             const=6.73486055d1
             mean=3.82155002d2
             gamma=4.58835999d2
          ENDIF
       ENDIF
       u=a*(E_gamma-x0)
       u0=a*x0
       SIGMAXN_HE_Au197=SIGMAX5N_Au197(E_gamma)
       SIGMAXN_HE_Au197=SIGMAXN_HE_Au197&
            +p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
       SIGMAXN_HE_Au197=SIGMAXN_HE_Au197&
            +MODIFIED_LORENTZ(E_gamma,const,mean,gamma)
    ELSE
       ! we use the reggeon one (we take the central value only)
       SIGMAXN_HE_Au197=SIGMA_p_Reggon(E_gamma)*AAA*Rshadow
    ENDIF
    SIGMAXN_HE_Au197=MAX(SIGMAXN_HE_Au197,0d0)
    RETURN
  END FUNCTION SIGMAXN_HE_Au197
  
  FUNCTION SIGMA1N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA1N_Au197
    REAL(KIND(1d0))::constlor1,meanlor1,gammalor1
    REAL(KIND(1d0))::meangauss,sigmagauss
    REAL(KIND(1d0))::constlor2,meanlor2,sigmalor2
    SIGMA1N_Au197=0d0
    ! Sn threshold
    IF(E_gamma.LT.8.07d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       constlor1=749.58846550d0+XSIGMA*24.13949089d0
       meanlor1=14.16503172d0+XSIGMA*0.05096954d0
       gammalor1=4.10826491d0+XSIGMA*0.10265047d0
       meangauss=10.28799832d0+XSIGMA*0.25006213d0
       sigmagauss=4.59183356d0+XSIGMA*0.25195422d0
       constlor2=13.43523511d0+XSIGMA*2.34589988d0
       meanlor2=25.70317632d0+XSIGMA*0.95996896d0
       sigmalor2=10.04604873d0+XSIGMA*3.22749664d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          constlor1=749.58846550d0
          meanlor1=14.16503172d0
          gammalor1=4.10826491d0
          meangauss=10.28799832d0
          sigmagauss=4.59183356d0
          constlor2=13.43523511d0
          meanlor2=25.70317632d0
          sigmalor2=10.04604873d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          constlor1=767.30553617d0
          meanlor1=14.22633112d0
          gammalor1=4.2363338d0
          meangauss=10.43971254d0
          sigmagauss=4.4207009d0
          constlor2=16.62447815d0
          meanlor2=25.44114089d0
          sigmalor2=11.51530209d0
       ELSE
          constlor1=734.10425462d0
          meanlor1=14.10893042d0
          gammalor1=3.9827791d0
          meangauss=10.12908127d0
          sigmagauss=4.75501747d0
          constlor2=10.45123565d0
          meanlor2=26.02255608d0
          sigmalor2=7.94419924d0
       ENDIF
    ENDIF
    SIGMA1N_Au197=MODIFIED_LORENTZ(E_gamma,constlor1,meanlor1,gammalor1)*&
         GAUSS(E_gamma,1d0,meangauss,sigmagauss)
    SIGMA1N_Au197=SIGMA1N_Au197+MODIFIED_LORENTZ(E_gamma,constlor2,meanlor2,sigmalor2)
    SIGMA1N_Au197=MAX(SIGMA1N_Au197,0d0)
    return
  END FUNCTION SIGMA1N_Au197

  FUNCTION SIGMA2N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA2N_Au197
    REAL(KIND(1d0))::p0,p2,p4,mpv,sigma1,sigma2
    REAL(KIND(1d0))::constgauss,meangauss,sigmagauss
    ! additional rescaling factor to be applied
    REAL(KIND(1d0)),PARAMETER::rescalingfact=0.5d0
    SIGMA2N_Au197=0d0
    ! S2n threshold
    IF(E_gamma.LT.14.71d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=232.02774513d0+rescalingfact*XSIGMA*14.81413800d0
       p2=-0.37290448d0+rescalingfact*XSIGMA*0.04800320d0
       p4=0.00015705d0+rescalingfact*XSIGMA*0.00003068d0
       mpv=16.04127399d0+rescalingfact*XSIGMA*0.07179598d0
       sigma1=4.18359994d0+rescalingfact*XSIGMA*0.23751454d0
       sigma2=0.61823840d0+rescalingfact*XSIGMA*0.05633909d0
       constgauss=16.34516425d0+rescalingfact*XSIGMA*2.87014296d0
       meangauss=24.35594069d0+rescalingfact*XSIGMA*0.23992731d0
       sigmagauss=1.53810657d0+rescalingfact*XSIGMA*0.33861400d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=232.02774513d0
          p2=-0.37290448d0
          p4=0.00015705d0
          mpv=16.04127399d0
          sigma1=4.18359994d0
          sigma2=0.61823840d0
          constgauss=16.34516425d0
          meangauss=24.35594069d0
          sigmagauss=1.53810657d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=2.32496844d2
          p2=-3.64886253d-1
          p4=1.64493631d-4
          mpv=1.59817297d1
          sigma1=4.33751068d0
          sigma2=6.24736193d-1
          constgauss=1.72037118d1
          meangauss=2.44361656d1
          sigmagauss=1.47358918d0
       ELSE
          p0=2.31216819d2
          p2=-3.78108360d-1
          p4=1.44964317d-4
          mpv=1.61006299d1
          sigma1=4.00384897d0
          sigma2=6.10207590d-1
          constgauss=1.55492470d1
          meangauss=2.42511570d1
          sigmagauss=1.63977258d0
       ENDIF
    ENDIF
    SIGMA2N_Au197=LANDAUAPPROX(E_gamma,1d0,mpv,sigma1,sigma2)&
         *(p0+p2*E_gamma**2+p4*E_gamma**4)
    SIGMA2N_Au197=SIGMA2N_Au197+GAUSS(E_gamma,constgauss,meangauss,sigmagauss)
    SIGMA2N_Au197=MAX(SIGMA2N_Au197,0d0)
    return
  END FUNCTION SIGMA2N_Au197

  FUNCTION SIGMA3N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA3N_Au197
    REAL(KIND(1d0))::p0,mpv,sigma
    SIGMA3N_Au197=0d0
    ! S3n threshold
    IF(E_gamma.LT.23.08d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=37.079297001687d0+XSIGMA*1.065754102759d0
       mpv=29.314436297842d0+XSIGMA*0.108149921600d0
       sigma=2.458105060946d0+XSIGMA*0.078402823792d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=37.079297001687d0
          mpv=29.314436297842d0
          sigma=2.458105060946d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=39.01246725d0
          mpv=29.36793104d0
          sigma=2.54813324d0
       ELSE
          p0=35.16336901d0
          mpv=29.25630214d0
          sigma=2.36084983d0
       ENDIF
    ENDIF
    SIGMA3N_Au197=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA3N_Au197=MAX(SIGMA3N_Au197,0d0)
    return
  END FUNCTION SIGMA3N_Au197

  FUNCTION SIGMA4N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA4N_Au197
    REAL(KIND(1d0))::p0,mpv,sigma
    SIGMA4N_Au197=0d0
    ! S4n threshold
    IF(E_gamma.LT.30.04d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=17.399669899307d0+XSIGMA*1.000106951622d0
       mpv=39.406441478882d0+XSIGMA*0.326474623884d0
       sigma=3.356697687837d0+XSIGMA*0.237739152385d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=17.399669899307d0
          mpv=39.406441478882d0
          sigma=3.356697687837d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=19.2201983d0
          mpv=39.53662556d0
          sigma=3.62385044d0
       ELSE
          p0=15.61667046d0
          mpv=39.25849437d0
          sigma=3.04843829d0
       ENDIF
    ENDIF
    SIGMA4N_Au197=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA4N_Au197=MAX(SIGMA4N_Au197,0d0)
    return
  END FUNCTION SIGMA4N_Au197

  ! >= 5n cross section
  FUNCTION SIGMAX5N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX5N_Au197
    REAL(KIND(1d0))::p0,a,x0,u,u0
    SIGMAX5N_Au197=0d0
    ! S5n threshold
    IF(E_gamma.LT.38.73d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=9.031507911726d0+XSIGMA*0.620289501385d0
       a=0.274160384720d0+XSIGMA*0.128815984784d0
       x0=45.161755558767d0+XSIGMA*0.833305126405d0
    ELSE
       ! no uncertainty when NEUTRON_ERROR_ESTIMATE=2
       p0=9.031507911726d0
       a=0.274160384720d0
       x0=45.161755558767d0
    ENDIF
    u=a*(E_gamma-x0)
    u0=a*x0
    SIGMAX5N_Au197=p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
    SIGMAX5N_Au197=MAX(SIGMAX5N_Au197,0d0)
    return
  END FUNCTION SIGMAX5N_Au197

  ! >= 4n cross section
  FUNCTION SIGMAX4N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX4N_Au197
    SIGMAX4N_Au197=SIGMAX5N_Au197(E_gamma)+SIGMA4N_Au197(E_gamma)
    return
  END FUNCTION SIGMAX4N_Au197

  ! >= 3n cross section
  FUNCTION SIGMAX3N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX3N_Au197
    SIGMAX3N_Au197=SIGMAX4N_Au197(E_gamma)+SIGMA3N_Au197(E_gamma)
    return
  END FUNCTION SIGMAX3N_Au197

  ! >= 2n cross section
  FUNCTION SIGMAX2N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX2N_Au197
    SIGMAX2N_Au197=SIGMAX3N_Au197(E_gamma)+SIGMA2N_Au197(E_gamma)
    return
  END FUNCTION SIGMAX2N_Au197

  ! >= 1n cross section
  FUNCTION SIGMAX1N_Au197(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX1N_Au197
    SIGMAX1N_Au197=SIGMAX2N_Au197(E_gamma)+SIGMA1N_Au197(E_gamma)
    return
  END FUNCTION SIGMAX1N_Au197
  
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! Fit Pb208 by Luca Maxia                                                       !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! absorption cross section (mb)
  FUNCTION SIGMAABS_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAABS_Pb208
    SIGMAABS_Pb208=SIGMA1N_Pb208(E_gamma)+SIGMA2N_Pb208(E_gamma)&
         +SIGMA3N_Pb208(E_gamma)+SIGMA4N_Pb208(E_gamma)+SIGMAXN_HE_Pb208(E_gamma)
    RETURN
  END FUNCTION SIGMAABS_Pb208

  ! high-energy tail of the absorption cross section
  FUNCTION SIGMAXN_HE_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAXN_HE_Pb208
    REAL(KIND(1d0))::p0,a,x0
    REAL(KIND(1d0))::const,mean,gamma
    REAL(KIND(1d0))::u,u0
    ! from which energy (MeV) use A*Rshadow*Reggon
    REAL(KIND(1d0)),PARAMETER::EXnregg=81900d0
    ! the shadowing factor in the HE tail
    REAL(KIND(1d0)),PARAMETER::Rshadow=0.65d0
    INTEGER,PARAMETER::AAA=208
    SIGMAXN_HE_Pb208=0d0
    ! S5n threshold
    IF(E_gamma.LT.37.3d0)RETURN
    IF(E_gamma.LE.EXnregg)THEN
       IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
          p0=4.87715571d0+XSIGMA*0.10142092d0
          a=0.00433266d0+XSIGMA*0.00151412d0
          x0=1077.91851015d0+XSIGMA*54.68761205d0
          const=73.54430802d0+XSIGMA*3.00140716d0
          mean=381.20497036d0+XSIGMA*5.68016890d0
          gamma=479.16675543d0+XSIGMA*35.23050896d0
       ELSE
          IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
             p0=4.87715571d0
             a=0.00433266d0
             x0=1077.91851015d0
             const=73.54430802d0
             mean=381.20497036d0
             gamma=479.16675543d0
          ELSEIF(XSIGMA.GT.0.5d0)THEN
             p0=5.06845845d0
             a=4.69998677d-3
             x0=1.05503058d3
             const=7.81789713d1
             mean=3.74935867d2
             gamma=4.60689945d2
          ELSE
             p0=4.68445498d0
             a=4.06113141d-3
             x0=1.10994461d3
             const=6.89264637d1
             mean=3.88566089d2
             gamma=5.01448836d2
          ENDIF
       ENDIF
       u=a*(E_gamma-x0)
       u0=a*x0
       SIGMAXN_HE_Pb208=SIGMAX5N_Pb208(E_gamma)
       SIGMAXN_HE_Pb208=SIGMAXN_HE_Pb208&
            +p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
       SIGMAXN_HE_Pb208=SIGMAXN_HE_Pb208&
            +MODIFIED_LORENTZ(E_gamma,const,mean,gamma)
    ELSE
       ! we use the reggeon one (we take the central value only)
       SIGMAXN_HE_Pb208=SIGMA_p_Reggon(E_gamma)*AAA*Rshadow
    ENDIF
    SIGMAXN_HE_Pb208=MAX(SIGMAXN_HE_Pb208,0d0)
    RETURN
  END FUNCTION SIGMAXN_HE_Pb208
  
  FUNCTION SIGMA1N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA1N_Pb208
    REAL(KIND(1d0))::constlor,meanlor,gammalor
    REAL(KIND(1d0))::meangauss1,sigmagauss1
    REAL(KIND(1d0))::constlor1,meanlor1,sigmalor1
    REAL(KIND(1d0))::constlor2,meanlor2,sigmalor2
    SIGMA1N_Pb208=0d0
    ! Sn threshold
    IF(E_gamma.LT.7.37d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       constlor=848.54599804d0+XSIGMA*55.83988892d0
       meanlor=14.16335892d0+XSIGMA*0.11517196d0
       gammalor=4.75621788d0+XSIGMA*0.17855457d0
       meangauss1=11.18294867d0+XSIGMA*0.12457364d0
       sigmagauss1=3.29767635d0+XSIGMA*0.17054524d0
       constlor1=18.08656200d0+XSIGMA*9.21660307d0
       meanlor1=20.36074947d0+XSIGMA*1.01629369d0
       sigmalor1=3.37200408d0+XSIGMA*2.75519828d0
       constlor2=7.28913825d0+XSIGMA*8.38392035d0
       meanlor2=31.99532030d0+XSIGMA*1.65265455d0
       sigmalor2=3.02555715d0+XSIGMA*5.11113578d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          constlor=848.54599804d0
          meanlor=14.16335892d0
          gammalor=4.75621788d0
          meangauss1=11.18294867d0
          sigmagauss1=3.29767635d0
          constlor1=18.08656200d0
          meanlor1=20.36074947d0
          sigmalor1=3.37200408d0
          constlor2=7.28913825d0
          meanlor2=31.99532030d0
          sigmalor2=3.02555715d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          constlor=924.71005966d0
          meanlor=14.26896418d0
          gammalor=4.7837305d0
          meangauss1=11.10960106d0
          sigmagauss1=3.21806384d0
          constlor1=28.51834112d0
          meanlor1=20.09633432d0
          sigmalor1=4.13575935d0
          constlor2=15.30848143d0
          meanlor2=31.96851556d0
          sigmalor2=3.07800022d0
       ELSE
          constlor=785.54629602d0
          meanlor=14.07482704d0
          gammalor=4.7336941d0
          meangauss1=11.28549398d0
          sigmagauss1=3.32098376d0
          constlor1=9.73132494d0
          meanlor1=20.6113977d0
          sigmalor1=1.57652344d0
          constlor2=-1.15827652d0
          meanlor2=31.56237721d0
          sigmalor2=5.44570316d0
       ENDIF
    ENDIF
    SIGMA1N_Pb208=MODIFIED_LORENTZ(E_gamma,constlor,meanlor,gammalor)*&
         GAUSS(E_gamma,1d0,meangauss1,sigmagauss1)
    SIGMA1N_Pb208=SIGMA1N_Pb208+MODIFIED_LORENTZ(E_gamma,constlor1,meanlor1,sigmalor1)
    SIGMA1N_Pb208=SIGMA1N_Pb208+MODIFIED_LORENTZ(E_gamma,constlor2,meanlor2,sigmalor2)
    SIGMA1N_Pb208=MAX(SIGMA1N_Pb208,0d0)
    return
  END FUNCTION SIGMA1N_Pb208

  FUNCTION SIGMA2N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA2N_Pb208
    REAL(KIND(1d0))::p0,mpv,sigma
    REAL(KIND(1d0))::constgauss,meangauss,sigmagauss
    REAL(KIND(1d0))::constlor,meanlor,sigmalor
    SIGMA2N_Pb208=0d0
    ! S2n threshold
    IF(E_gamma.LT.14.11d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=154.23071627d0+XSIGMA*6.29449187d0
       mpv=16.49136582d0+XSIGMA*0.10433356d0
       sigma=1.13827892d0+XSIGMA*0.07998836d0
       constgauss=39.61700774d0+XSIGMA*3.07403680d0
       meangauss=22.60755279d0+XSIGMA*0.36446014d0
       sigmagauss=3.11704789d0+XSIGMA*0.25175020d0
       constlor=4.22801657d0+XSIGMA*1.11355146d0
       meanlor=62.07740587d0+XSIGMA*11.24291967d0
       sigmalor=87.87650723d0+XSIGMA*68.08597745d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=154.23071627d0
          mpv=16.49136582d0
          sigma=1.13827892d0
          constgauss=39.61700774d0
          meangauss=22.60755279d0
          sigmagauss=3.11704789d0
          constlor=4.22801657d0
          meanlor=62.07740587d0
          sigmalor=87.87650723d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=160.29651286d0
          mpv=16.43653913d0
          sigma=1.15667886d0
          constgauss=40.99916458d0
          meangauss=22.58992864d0
          sigmagauss=3.20653925d0
          constlor=5.37498413d0
          meanlor=65.58844023d0
          sigmalor=116.2900465d0
       ELSE
          p0=148.29919386d0
          mpv=16.54759959d0
          sigma=1.11658901d0
          constgauss=38.28023722d0
          meangauss=22.62048112d0
          sigmagauss=3.02033628d0
          constlor=2.98507791d0
          meanlor=58.94401379d0
          sigmalor=60.11284257d0
       ENDIF
    ENDIF
    SIGMA2N_Pb208=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA2N_Pb208=SIGMA2N_Pb208+GAUSS(E_gamma,constgauss,meangauss,sigmagauss)
    SIGMA2N_Pb208=SIGMA2N_Pb208+MODIFIED_LORENTZ(E_gamma,constlor,meanlor,sigmalor)
    SIGMA2N_Pb208=MAX(SIGMA2N_Pb208,0d0)
    return
  END FUNCTION SIGMA2N_Pb208

  FUNCTION SIGMA3N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA3N_Pb208
    REAL(KIND(1d0))::p0,mpv,sigma
    SIGMA3N_Pb208=0d0
    ! S3n threshold
    IF(E_gamma.LT.22.2d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=39.288077638417d0+XSIGMA*0.839372807844d0
       mpv=29.167769062312d0+XSIGMA*0.097243544711d0
       sigma=2.662454235648d0+XSIGMA*0.056375203420d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=39.288077638417d0
          mpv=29.167769062312d0
          sigma=2.662454235648d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=40.89112135d0
          mpv=29.18227431d0
          sigma=2.71477648d0
       ELSE
          p0=37.68948558d0
          mpv=29.15369737d0
          sigma=2.6065876d0
       ENDIF
    ENDIF
    SIGMA3N_Pb208=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA3N_Pb208=MAX(SIGMA3N_Pb208,0d0)
    return
  END FUNCTION SIGMA3N_Pb208

  FUNCTION SIGMA4N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA4N_Pb208
    REAL(KIND(1d0))::p0,mpv,sigma
    SIGMA4N_Pb208=0d0
    ! S4n threshold
    IF(E_gamma.LT.28.9d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=18.210909748612d0+XSIGMA*0.754385015845d0
       mpv=39.332314221991d0+XSIGMA*0.225554225531d0
       sigma=3.358485085166d0+XSIGMA*0.140721364413d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=18.210909748612d0
          mpv=39.332314221991d0
          sigma=3.358485085166d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=19.67224774d0
          mpv=39.50045346d0
          sigma=3.51919103d0
       ELSE
          p0=16.77383903d0
          mpv=39.14444278d0
          sigma=3.17478362d0
       ENDIF
    ENDIF
    SIGMA4N_Pb208=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA4N_Pb208=MAX(SIGMA4N_Pb208,0d0)
    return
  END FUNCTION SIGMA4N_Pb208

  ! >= 5n cross section
  FUNCTION SIGMAX5N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX5N_Pb208
    REAL(KIND(1d0))::p0,a,x0,u,u0
    SIGMAX5N_Pb208=0d0
    ! S5n threshold
    IF(E_gamma.LT.37.3d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=9.470598638805d0+XSIGMA*0.630314227734d0
       a=0.277273681013d0+XSIGMA*0.126696250854d0
       x0=45.133614846275d0+XSIGMA*0.799636327694d0
    ELSE
       ! no uncertainty when NEUTRON_ERROR_ESTIMATE=2
       p0=9.470598638805d0
       a=0.277273681013d0
       x0=45.133614846275d0
    ENDIF
    u=a*(E_gamma-x0)
    u0=a*x0
    SIGMAX5N_Pb208=p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
    SIGMAX5N_Pb208=MAX(SIGMAX5N_Pb208,0d0)
    return
  END FUNCTION SIGMAX5N_Pb208

  ! >= 4n cross section
  FUNCTION SIGMAX4N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX4N_Pb208
    SIGMAX4N_Pb208=SIGMAX5N_Pb208(E_gamma)+SIGMA4N_Pb208(E_gamma)
    return
  END FUNCTION SIGMAX4N_Pb208

  ! >= 3n cross section
  FUNCTION SIGMAX3N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX3N_Pb208
    SIGMAX3N_Pb208=SIGMAX4N_Pb208(E_gamma)+SIGMA3N_Pb208(E_gamma)
    return
  END FUNCTION SIGMAX3N_Pb208

  ! >= 2n cross section
  FUNCTION SIGMAX2N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX2N_Pb208
    SIGMAX2N_Pb208=SIGMAX3N_Pb208(E_gamma)+SIGMA2N_Pb208(E_gamma)
    return
  END FUNCTION SIGMAX2N_Pb208

  ! >= 1n cross section
  FUNCTION SIGMAX1N_Pb208(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX1N_Pb208
    SIGMAX1N_Pb208=SIGMAX2N_Pb208(E_gamma)+SIGMA1N_Pb208(E_gamma)
    return
  END FUNCTION SIGMAX1N_Pb208

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! Fit U238 by Luca Maxia                                                        !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! absorption cross section (mb)
  FUNCTION SIGMAABS_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAABS_U238
    SIGMAABS_U238=SIGMA1N_U238(E_gamma)+SIGMA2N_U238(E_gamma)&
         +SIGMA3N_U238(E_gamma)+SIGMAXN_HE_U238(E_gamma)
    RETURN
  END FUNCTION SIGMAABS_U238

  ! high-energy tail of the absorption cross section
  FUNCTION SIGMAXN_HE_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMAXN_HE_U238
    REAL(KIND(1d0))::p0,a,x0
    REAL(KIND(1d0))::const,mean,gamma
    REAL(KIND(1d0))::u,u0
    ! from which energy (MeV) use A*Rshadow*Reggon
    REAL(KIND(1d0)),PARAMETER::EXnregg=81900d0
    ! the shadowing factor in the HE tail
    REAL(KIND(1d0)),PARAMETER::Rshadow=0.70d0
    INTEGER,PARAMETER::AAA=238
    SIGMAXN_HE_U238=0d0
    ! S5n threshold
    IF(E_gamma.LT.23.12d0)RETURN
    IF(E_gamma.LE.EXnregg)THEN
       IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
          p0=7.29252922d0+XSIGMA*0.13895533d0
          a=0.00338930d0+XSIGMA*0.00090291d0
          x0=998.58162812d0+XSIGMA*64.53314293d0
          const=82.83579233d0+XSIGMA*3.29748995d0
          mean=401.21655386d0+XSIGMA*5.45817549d0
          gamma=421.30445797d0+XSIGMA*29.01861515d0
       ELSE
          IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
             p0=7.29252922d0
             a=0.00338930d0
             x0=998.58162812d0
             const=82.83579233d0
             mean=401.21655386d0
             gamma=421.30445797d0
          ELSEIF(XSIGMA.GT.0.5d0)THEN
             p0=7.48683845d0
             a=3.52241881d-3
             x0=9.85093403d2
             const=8.73825190d1
             mean=3.96340384d2
             gamma=4.09789621d2
          ELSE
             p0=7.09736096d0
             a=3.25987014d-3
             x0=1.01653984d3
             const=7.83292705d1
             mean=4.06755217d2
             gamma=4.34492078d2
          ENDIF
       ENDIF
       u=a*(E_gamma-x0)
       u0=a*x0
       SIGMAXN_HE_U238=SIGMAX4N_U238(E_gamma)
       SIGMAXN_HE_U238=SIGMAXN_HE_U238&
            +p0*(1d0/(1d0+DEXP(-2d0*u))-1d0/(1d0+DEXP(2d0*u0)))
       SIGMAXN_HE_U238=SIGMAXN_HE_U238&
            +MODIFIED_LORENTZ(E_gamma,const,mean,gamma)
    ELSE
       ! we use the reggeon one (we take the central value only)
       SIGMAXN_HE_U238=SIGMA_p_Reggon(E_gamma)*AAA*Rshadow
    ENDIF
    SIGMAXN_HE_U238=MAX(SIGMAXN_HE_U238,0d0)
    RETURN
  END FUNCTION SIGMAXN_HE_U238
  
  FUNCTION SIGMA1N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA1N_U238
    REAL(KIND(1d0))::constlor,meanlor,gammalor
    REAL(KIND(1d0))::meangauss1,sigmagauss1
    REAL(KIND(1d0))::constlor1,meanlor1,sigmalor1
    REAL(KIND(1d0))::gammalor1,constlor2,meanlor2,gammalor2
    REAL(KIND(1d0)),PARAMETER::rescalingfact=0.5d0 ! to ensure the band width good
    SIGMA1N_U238=0d0
    ! Sn threshold (make it a bit lower)
    IF(E_gamma.LT.6.01d0)RETURN
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       constlor=737.88781361d0+rescalingfact*XSIGMA*197.28295518d0
       meanlor=11.66556337d0+rescalingfact*XSIGMA*0.10449796d0
       gammalor=2.79688077d0+rescalingfact*XSIGMA*0.39629312d0
       meangauss1=8.38235402d0+rescalingfact*XSIGMA*0.17689627d0
       sigmagauss1=2.18875124d0+rescalingfact*XSIGMA*0.26727411d0
       constlor1=81.71913660d0+rescalingfact*XSIGMA*15.63558357d0
       meanlor1=13.63467523d0+rescalingfact*XSIGMA*0.51281138d0
       sigmalor1=4.00830564d0+rescalingfact*XSIGMA*0.72089007d0
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          constlor=590.70039840d0
          meanlor=11.41237602d0
          gammalor=3.01440233d0
          meangauss1=7.64090662d0
          sigmagauss1=3.22026211d0
          constlor1=55.56084095d0
          meanlor1=14.09813488d0
          gammalor1=1.62539805d0
          constlor2=31.62613385d0
          meanlor2=18.77778506d0
          gammalor2=2.63550790d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          constlor=653.54699134d0
          meanlor=11.40263708d0
          gammalor=2.98232901d0
          meangauss1=7.2542746d0
          sigmagauss1=3.42028057d0
          constlor1=57.8910944d0
          meanlor1=14.16932125d0
          gammalor1=1.56209888d0
          constlor2=39.70033818d0
          meanlor2=18.79677749d0
          gammalor2=3.39045969d0
       ELSE
          constlor=536.8867734d0
          meanlor=11.42511688d0
          gammalor=3.04899168d0
          meangauss1=7.98859044d0
          sigmagauss1=3.02051905d0
          constlor1=53.18758366d0
          meanlor1=13.97820157d0
          gammalor1=1.67931384d0
          constlor2=23.91544551d0
          meanlor2=18.75118961d0
          gammalor2=1.74019717d0
       ENDIF
    ENDIF
    SIGMA1N_U238=MODIFIED_LORENTZ(E_gamma,constlor,meanlor,gammalor)*&
         GAUSS(E_gamma,1d0,meangauss1,sigmagauss1)
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       SIGMA1N_U238=SIGMA1N_U238+MODIFIED_LORENTZ(E_gamma,constlor1,meanlor1,sigmalor1)
    ELSE
       SIGMA1N_U238=SIGMA1N_U238+GAUSS(E_gamma,constlor1,meanlor1,gammalor1)
       SIGMA1N_U238=SIGMA1N_U238+MODIFIED_LORENTZ(E_gamma,constlor2, meanlor2, gammalor2)
    ENDIF
    SIGMA1N_U238=MAX(SIGMA1N_U238,0d0)
    return
  END FUNCTION SIGMA1N_U238

  FUNCTION SIGMA2N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA2N_U238
    REAL(KIND(1d0))::p0,mpv,sigma
    REAL(KIND(1d0))::constgauss,meangauss,sigmagauss
    REAL(KIND(1d0))::constlor,meanlor,sigmalor
    SIGMA2N_U238=0d0
    IF(E_gamma.GT.200d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       ! S2n threshold (lower it a bit)
       IF(E_gamma.LT.10.20d0)RETURN
       p0=281.02421578d0+XSIGMA*20.87414743d0
       mpv=13.51210766d0+XSIGMA*0.10799686d0
       sigma=1.02249288d0+XSIGMA*0.07532587d0
       constgauss=46.37341129d0+XSIGMA*17.49223241d0
       meangauss=15.63541196d0+XSIGMA*0.28480609d0
       sigmagauss=2.41634605d0+XSIGMA*0.24356455d0
       constlor=5.43697647d0+XSIGMA*1.46963626d0
       meanlor=62.16425623d0+XSIGMA*6.70721615d0
       sigmalor=52.55735354d0+XSIGMA*33.19823843d0
    ELSE
       ! S2n threshold
       IF(E_gamma.LT.11.28d0)RETURN
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=315.93894443d0
          mpv=13.56975308d0
          sigma=1.09076970d0
          constgauss=27.58555441d0
          meangauss=16.22206518d0
          sigmagauss=1.92070346d0
          constlor=5.44369940d0
          meanlor=62.15678318d0
          sigmalor=52.31607302d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=343.68042828d0
          mpv=13.78649306d0
          sigma=1.26951369d0
          constgauss=8.20657505d0
          meangauss=22.95466751d0
          sigmagauss=1.60029699d0
          constlor=7.56560309d0
          meanlor=62.73049761d0
          sigmalor=72.7083805d0
       ELSE
          p0=226.09333074d0
          mpv=13.29671842d0
          sigma=0.82708744d0
          constgauss=84.32136035d0
          meangauss=15.23618146d0
          sigmagauss=2.1083971d0
          constlor=3.37140269d0
          meanlor=63.18921953d0
          sigmalor=27.29224035d0
       ENDIF
    ENDIF
    SIGMA2N_U238=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma)
    SIGMA2N_U238=SIGMA2N_U238+GAUSS(E_gamma,constgauss,meangauss,sigmagauss)
    SIGMA2N_U238=SIGMA2N_U238+MODIFIED_LORENTZ(E_gamma,constlor,meanlor,sigmalor)
    SIGMA2N_U238=MAX(SIGMA2N_U238,0d0)
    return
  END FUNCTION SIGMA2N_U238
  
  FUNCTION SIGMA3N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMA3N_U238
    REAL(KIND(1d0))::p0,mpv,sigma,sigma2
    SIGMA3N_U238=0d0
    ! S3n threshold
    IF(E_gamma.LT.17.83d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       p0=59.102002803219d0+XSIGMA*14.654052791675d0
       mpv=26.078948834759d0+XSIGMA*0.814327833076d0
       sigma=3.322505462362d0+XSIGMA*0.426298064714d0
       sigma2=sigma
    ELSE
       IF(XSIGMA.GE.-0.5d0.AND.XSIGMA.LE.0.5d0)THEN
          p0=67.563186281215d0
          mpv=22.362876825969d0
          sigma=4.218312433615d0
          sigma2=1.781089008967d0
       ELSEIF(XSIGMA.GT.0.5d0)THEN
          p0=67.563186281215d0+2d0*13.81480811d0
          mpv=22.362876825969d0-2d0*0.89215524d0/5d0
          sigma=4.218312433615d0+0.70523308d0
          sigma2=1.781089008967d0
       ELSE
          p0=67.563186281215d0-2d0*13.81480811d0
          mpv=22.362876825969d0+2d0*0.89215524d0/5d0
          sigma=4.218312433615d0-0.70523308d0
          sigma2=1.781089008967d0
       ENDIF
    ENDIF
    SIGMA3N_U238=LANDAUAPPROX(E_gamma,p0,mpv,sigma,sigma2)
    SIGMA3N_U238=MAX(SIGMA3N_U238,0d0)
    return
  END FUNCTION SIGMA3N_U238

  FUNCTION SIGMA4N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1d0))::SIGMA4N_U238
    SIGMA4N_U238=0d0
    ! S4n
    IF(E_gamma.LT.23.12d0)RETURN
    SIGMA4N_U238=MAX(SIGMAX4N_U238(E_gamma)-SIGMAX5N_U238(E_gamma),0d0)
    return
  END FUNCTION SIGMA4N_U238

  ! >= 5n cross section
  FUNCTION SIGMAX5N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX5N_U238
    REAL(KIND(1d0))::p0,a,x0,u,u0
    SIGMAX5N_U238=0d0
    ! S5n threshold
    IF(E_gamma.LT.29.97d0)RETURN
    ! no error for >=5n
    p0=10.836556028407577d0
    a=0.3172542141382115d0
    x0=51.643259608533185d0
    u=a*(E_gamma-x0)
    SIGMAX5N_U238=p0*(1d0/(1d0+DEXP(-2d0*u)))
    return
  END FUNCTION SIGMAX5N_U238

  ! >= 4n cross section
  FUNCTION SIGMAX4N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX4N_U238
    REAL(KIND(1d0))::const,mean,sigma1
    SIGMAX4N_U238=0d0
    ! S4n
    IF(E_gamma.LT.23.12d0)RETURN
    IF(NEUTRON_ERROR_ESTIMATE.EQ.1)THEN
       const=19.666982872665d0+XSIGMA*2.282073757972d0
       mean=42.162280860636d0+XSIGMA*0.939062213589d0
       sigma1=4.742193582540d0+XSIGMA*0.728768182482d0
    ELSE
       ! No error when NEUTRON_ERROR_ESTIMATE=2
       const=19.666982872665d0
       mean=42.162280860636d0
       sigma1=4.742193582540d0
    ENDIF
    SIGMAX4N_U238=SIGMAX5N_U238(E_gamma)
    SIGMAX4N_U238=SIGMAX4N_U238+LANDAUAPPROX(E_gamma,const,mean,sigma1,sigma1)
    SIGMAX4N_U238=MAX(SIGMAX4N_U238,0d0)
    return
  END FUNCTION SIGMAX4N_U238

  ! >= 3n cross section
  FUNCTION SIGMAX3N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX3N_U238
    SIGMAX3N_U238=SIGMAX4N_U238(E_gamma)+SIGMA3N_U238(E_gamma)
    return
  END FUNCTION SIGMAX3N_U238

  ! >= 2n cross section
  FUNCTION SIGMAX2N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX2N_U238
    SIGMAX2N_U238=SIGMAX3N_U238(E_gamma)+SIGMA2N_U238(E_gamma)
    return
  END FUNCTION SIGMAX2N_U238

  ! >= 1n cross section
  FUNCTION SIGMAX1N_U238(E_gamma)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    REAL(KIND(1D0))::SIGMAX1N_U238
    SIGMAX1N_U238=SIGMAX2N_U238(E_gamma)+SIGMA1N_U238(E_gamma)
    return
  END FUNCTION SIGMAX1N_U238

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! Fit Pb208 by Nicolas Crepet (not used anymore)                                !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! Fit of Sigma^i for E_gamma > 38 MeV, defined in 10.1016/0375-9474(81)90516-9
  FUNCTION SIGMA_NC_UPS_1(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_NC_UPS_1
    REAL(KIND(1D0))::p0, p1
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       p0 = 23.4757d0
       p1 = -0.0500124d0
       IF (E_gamma .GT. 38d0) THEN
          SIGMA_NC_UPS_1 = p1*E_gamma + p0
       ELSE
          SIGMA_NC_UPS_1 = 0d0
       END IF
    ELSE
       WRITE (*,*) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_NC_UPS_1
   
  FUNCTION SIGMA_NC_UPS_2(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_NC_UPS_2
    REAL(KIND(1D0))::p0, p1
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       p0 = 20.7126d0
       p1 = -0.0776226d0
       IF(E_gamma.GT.38d0)THEN
          SIGMA_NC_UPS_2 = p1*E_gamma + p0
       ELSE
          SIGMA_NC_UPS_2 = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_NC_UPS_2

  FUNCTION SIGMA_NC_UPS_3(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_NC_UPS_3
    REAL(KIND(1D0))::p0, p1
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       p0 = 17.0026d0
       p1 = -0.0612244d0
       IF (E_gamma .GT. 38d0) THEN
          SIGMA_NC_UPS_3 = p1*E_gamma+p0
       ELSE
          SIGMA_NC_UPS_3 = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_NC_UPS_3

  FUNCTION SIGMA_NC_UPS_4(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma !  photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_NC_UPS_4, U
    REAL(KIND(1D0))::p(0:3)
    IF(A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.GT.38d0.AND.E_gamma.LT.140d0)THEN
          p = (/11.241d0,0.321185d0,5.43676d0,4.85818d0/)
          U = p(1)*E_gamma-p(0)
          SIGMA_NC_UPS_4 = p(2)*(1d0-DEXP(-2d0*U))/(1d0+DEXP(-2d0*U))+p(3)
       ELSE
          SIGMA_NC_UPS_4 = 0d0
       ENDIF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_NC_UPS_4

  FUNCTION SIGMA_NC_UPS_5(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_NC_UPS_5, U
    REAL(KIND(1D0))::p(0:3)
    IF(A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.GT.44.231d0.AND.E_gamma.LT.140d0)THEN
          p = (/17.4791,0.419003,40.6289,-31.8213/)
          U = p(1)*E_gamma - p(0)
          SIGMA_NC_UPS_5 = p(2)*(1 - EXP(-2*U))/(1 + EXP(-2*U)) + p(3)
       ELSE
          SIGMA_NC_UPS_5 = 0d0
       ENDIF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_NC_UPS_5

  !Fit of gamma + Pb208 > Pb(208-i) + in
  !Under 38 MeV, use a fit of data from https://arxiv.org/abs/2403.11547
  !Above, we use Sigma^(i) - Sigma^i+1
  FUNCTION GAMMATO1N_NC(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO1N_NC ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: C_LOR, MEAN_LOR, GAMMA_LOR
    REAL(KIND(1d0)):: C_GAUSS_1, MEAN_GAUSS_1, SIGMA_GAUSS_1
    REAL(KIND(1d0)):: C_GAUSS_2, MEAN_GAUSS_2, SIGMA_GAUSS_2
    REAL(KIND(1d0)):: C_GAUSS_3, MEAN_GAUSS_3, SIGMA_GAUSS_3
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF (E_gamma.LT.8d0) THEN
          GAMMATO1N_NC = GAMMAABS_NC(E_gamma, A, Z)
       ELSEIF (E_gamma.LT.40d0) THEN
          C_LOR = 774.5868828165748d0
          MEAN_LOR = 13.726763439097818d0
          GAMMA_LOR = 3.810921328797886d0
          C_GAUSS_1 = 1d0
          MEAN_GAUSS_1 = 11.876796587979078d0
          SIGMA_GAUSS_1 = 3.506124333022094d0
          C_GAUSS_2 = 14.744225279978867d0
          MEAN_GAUSS_2 = 20.690123048793087d0
          SIGMA_GAUSS_2 = 1.4458592441947116d0
          C_GAUSS_3 = 4.2147775560860525d0
          MEAN_GAUSS_3 = 31.711688417560136d0
          SIGMA_GAUSS_3 = 4.917031043720243d0
          GAMMATO1N_NC = LORENTZ(E_gamma, C_LOR, MEAN_LOR, GAMMA_LOR)* &
               GAUSS(E_gamma, C_GAUSS_1, MEAN_GAUSS_1, SIGMA_GAUSS_1) + &
               GAUSS(E_gamma, C_GAUSS_2, MEAN_GAUSS_2, SIGMA_GAUSS_2) + &
               GAUSS(E_gamma, C_GAUSS_3, MEAN_GAUSS_3, SIGMA_GAUSS_3)
       ELSE
          GAMMATO1N_NC = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO1N_NC

  FUNCTION GAMMATO2N_NC(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN) :: E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO2N_NC ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: pol3(0:3), land(0:2), gauss1(0:2), gauss2(0:2)
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.LT.14d0)THEN
          GAMMATO2N_NC=0.0d0
       ELSEIF(E_gamma.LE.38d0)THEN
          land = (/1.62394D-04, 1.49198D01, 8.82677D-01/)
          pol3 = (/-7.04950D08, 1.08715D08, -5.25967D06, 7.44231D04/)
          gauss1 = (/2.71970D02, 1.85051D01, 8.45083D00/)
          gauss2 = (/2.09272D01, 3.41246D01, 2.90501D00/)
          GAMMATO2N_NC = LANDAU(E_gamma, land(0), land(1), land(2))* &
               (pol3(0) + pol3(1)*E_gamma + pol3(2)*E_gamma**2 + pol3(3)*E_gamma**3) + &
               GAUSS(E_gamma, gauss1(0), gauss1(1), gauss1(2)) + &
               GAUSS(E_gamma, gauss2(0), gauss2(1), gauss2(2))
       ELSEIF(E_gamma.LE.140d0)THEN
          GAMMATO2N_NC = MAX(SIGMA_NC_UPS_2(E_gamma, A, Z) - SIGMA_NC_UPS_3(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO2N_NC = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO2N_NC

  FUNCTION GAMMATO3N_NC(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO3N_NC ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: land(0:2), pol2(0:2)
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       IF(E_gamma.LT.22.659d0)THEN
          GAMMATO3N_NC = 0.0d0
       ELSEIF(E_gamma.LE.38d0)THEN
          land = (/0.8510484278430387D0, 28.167213873909613D0, 2.5372945327923646D0/)
          pol2 = (/-1.8801037291951625D3, 1.271738540610852D2, -1.9506353726996741D0/)
          GAMMATO3N_NC = LANDAU(E_gamma, land(0), land(1), land(2))* &
               (pol2(0) + pol2(1)*E_gamma + pol2(2)*E_gamma**2)
       ELSEIF(E_gamma.LE.140d0)THEN
          GAMMATO3N_NC = MAX(SIGMA_NC_UPS_3(E_gamma, A, Z)-SIGMA_NC_UPS_4(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO3N_NC = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO3N_NC

  FUNCTION GAMMATO4N_NC(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO4N_NC ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: land(0:2)
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.LT.32d0) THEN
          GAMMATO4N_NC = 0.0d0
       ELSEIF(E_gamma.LT.44.5d0) THEN
          land = (/62.594589628136525, 40.06306462041548, 3.227147958419363/)
          GAMMATO4N_NC = LANDAU(E_gamma, land(0), land(1), land(2))
       ELSEIF(E_gamma.LT.140d0) THEN
          GAMMATO4N_NC = MAX(SIGMA_NC_UPS_4(E_gamma, A, Z) - SIGMA_NC_UPS_5(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO4N_NC = 0.0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO4N_NC

  !Fit of the total photoabsorption, list of data used available in the gammaUPC article.
  FUNCTION GAMMAABS_NC(E_gamma, A, Z)
    IMPLICIT NONE
    integer,intent(in)::A, Z ! atomic number and mass
    real(kind(1d0)), intent(in)::E_gamma ! photon energy (input) (MeV)
    real(kind(1d0)) GAMMAABS_NC ! photoabsorption cross section (output) (mb)
    real(kind(1d0))::p1(0:8)
    real(kind(1d0))::p2(0:6)
    real(kind(1d0))::p3(0:9)
    real(kind(1d0))::p4(0:4)
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       p1 = (/260.65d0, 12.20d0, 3.251d0, 526.13d0, 13.93d0, 3.06d0, 8.44d0, 25.56d0, 1.97d0/)
       p2 = (/-1.34605D08, 1.87177D07, -6.56819D05, 7.13574D03, -8.21024D01, 8.23842D00, 1.18073D01/)
       p3 = (/45.6262, 624.685, -1047.55, 14.3129, 61.4578, 300.694, 75.3295, -12.525, 653.116, 464.523/)
       p4 = (/50.2742, 99.0847, 0.0599226, 0.142449, 208*.65/)
       IF (E_gamma .LT. 4d1) THEN
          GAMMAABS_NC = MODIFIED_LORENTZ(E_gamma, p1(0), p1(1), p1(2)) + &
               MODIFIED_LORENTZ(E_gamma, p1(3), p1(4), p1(5)) + &
               MODIFIED_LORENTZ(E_gamma, p1(6), p1(7), p1(8))
       ELSEIF (E_gamma .LT. 117d0 .AND. E_gamma .GE. 4d1) THEN
          GAMMAABS_NC = (p2(0) + p2(1)*E_gamma + p2(2)*E_gamma**2 + p2(3)*E_gamma**3)* &
               DEXP(-(E_gamma - p2(4))/p2(5)) + p2(6)
       ELSEIF (E_gamma .LT. 50d3 .AND. E_gamma .GE. 117d0) THEN
          GAMMAABS_NC = MODIFIED_LORENTZ(E_gamma, p3(0), p3(1), p3(2)) + p3(3) + &
               GAUSS(E_gamma, p3(4), p3(5), p3(6)) + &
               GAUSS(E_gamma, p3(7), p3(8), p3(9))
       ELSE
          GAMMAABS_NC = p4(4)/1000*(p4(0)*(E_gamma/1000)**(2*p4(2)) + &
               p4(1)*(E_gamma/1000)**(-2*p4(3)))
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  end FUNCTION GAMMAABS_NC

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !                                                                               !
  ! End of fit Pb208 by Nicolas Crepet (not used anymore)                         !
  !                                                                               !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end module photoabsorption

