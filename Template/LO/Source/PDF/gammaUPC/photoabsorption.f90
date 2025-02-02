module photoabsorption
  ! The cross section of gamma+A -> A* in terms of the energy of gamma
  ! where the energy of gamma Etarget is in the rest frame of the target A
  ! In the frame of A with its energy per nucleon as En, the energy of gamma is
  ! Elab. Then, we have Etarget=(En+sqrt(En**2-mn**2))/mn*Elab, where mn is the  ! average nucleon mass. It can be approximated as
  ! Etarget = 2*Elab*En/mn*(1-mn**2/En**2/4+O(mn**4/En**4))
  implicit none
contains

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
    GAUSS = CONSTANT*EXP(-0.5*((X - MEAN)/SIGMA)**2)
    return
  END FUNCTION GAUSS

   !Fit of Sigma^i for E_gamma > 38 MeV, defined in 10.1016/0375-9474(81)90516-9

  FUNCTION SIGMA_UPS_1(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_UPS_1
    REAL(KIND(1D0))::p0, p1
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       p0 = 23.4757d0
       p1 = -0.0500124
       IF (E_gamma .GT. 38d0) THEN
          SIGMA_UPS_1 = p1*E_gamma + p0
       ELSE
          SIGMA_UPS_1 = 0d0
       END IF
    ELSE
       WRITE (*,*) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_UPS_1
   
  FUNCTION SIGMA_UPS_2(E_gamma,A,Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_UPS_2
    REAL(KIND(1D0))::p0, p1
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       p0 = 20.7126d0
       p1 = -0.0776226d0
       IF(E_gamma.GT.38d0)THEN
          SIGMA_UPS_2 = p1*E_gamma + p0
       ELSE
          SIGMA_UPS_2 = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_UPS_2

  FUNCTION SIGMA_UPS_3(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_UPS_3
    REAL(KIND(1D0))::p0, p1
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       p0 = 17.0026d0
       p1 = -0.0612244d0
       IF (E_gamma .GT. 38d0) THEN
          SIGMA_UPS_3 = p1*E_gamma+p0
       ELSE
          SIGMA_UPS_3 = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_UPS_3

  FUNCTION SIGMA_UPS_4(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma !  photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_UPS_4, U
    REAL(KIND(1D0))::p(0:3)
    IF(A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.GT.38d0.AND.E_gamma.LT.140d0)THEN
          p = (/11.241d0,0.321185d0,5.43676d0,4.85818d0/)
          U = p(1)*E_gamma-p(0)
          SIGMA_UPS_4 = p(2)*(1d0-DEXP(-2d0*U))/(1d0+DEXP(-2d0*U))+p(3)
       ELSE
          SIGMA_UPS_4 = 0d0
       ENDIF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_UPS_4

  FUNCTION SIGMA_UPS_5(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)):: SIGMA_UPS_5, U
    REAL(KIND(1D0))::p(0:3)
    IF(A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.GT.44.231d0.AND.E_gamma.LT.140d0)THEN
          p = (/17.4791,0.419003,40.6289,-31.8213/)
          U = p(1)*E_gamma - p(0)
          SIGMA_UPS_5 = p(2)*(1 - EXP(-2*U))/(1 + EXP(-2*U)) + p(3)
       ELSE
          SIGMA_UPS_5 = 0d0
       ENDIF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION SIGMA_UPS_5

  !Fit of gamma + Pb208 > Pb(208-i) + in
  !Under 38 MeV, use a fit of data from https://arxiv.org/abs/2403.11547
  !Above, we use Sigma^(i) - Sigma^i+1
  FUNCTION GAMMATO1N(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER, INTENT(IN):: A,Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO1N ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: C_LOR, MEAN_LOR, GAMMA_LOR
    REAL(KIND(1d0)):: C_GAUSS_1, MEAN_GAUSS_1, SIGMA_GAUSS_1
    REAL(KIND(1d0)):: C_GAUSS_2, MEAN_GAUSS_2, SIGMA_GAUSS_2
    REAL(KIND(1d0)):: C_GAUSS_3, MEAN_GAUSS_3, SIGMA_GAUSS_3
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF (E_gamma.LT.8d0) THEN
          GAMMATO1N = GAMMAABS(E_gamma, A, Z)
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
          GAMMATO1N = LORENTZ(E_gamma, C_LOR, MEAN_LOR, GAMMA_LOR)* &
               GAUSS(E_gamma, C_GAUSS_1, MEAN_GAUSS_1, SIGMA_GAUSS_1) + &
               GAUSS(E_gamma, C_GAUSS_2, MEAN_GAUSS_2, SIGMA_GAUSS_2) + &
               GAUSS(E_gamma, C_GAUSS_3, MEAN_GAUSS_3, SIGMA_GAUSS_3)
       ELSE
          GAMMATO1N = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO1N

  FUNCTION GAMMATO2N(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN) :: E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO2N ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: pol3(0:3), land(0:2), gauss1(0:2), gauss2(0:2)
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.LT.14d0)THEN
          GAMMATO2N=0.0d0
       ELSEIF(E_gamma.LE.38d0)THEN
          land = (/1.62394D-04, 1.49198D01, 8.82677D-01/)
          pol3 = (/-7.04950D08, 1.08715D08, -5.25967D06, 7.44231D04/)
          gauss1 = (/2.71970D02, 1.85051D01, 8.45083D00/)
          gauss2 = (/2.09272D01, 3.41246D01, 2.90501D00/)
          GAMMATO2N = LANDAU(E_gamma, land(0), land(1), land(2))* &
               (pol3(0) + pol3(1)*E_gamma + pol3(2)*E_gamma**2 + pol3(3)*E_gamma**3) + &
               GAUSS(E_gamma, gauss1(0), gauss1(1), gauss1(2)) + &
               GAUSS(E_gamma, gauss2(0), gauss2(1), gauss2(2))
       ELSEIF(E_gamma.LE.140d0)THEN
          GAMMATO2N = MAX(SIGMA_UPS_2(E_gamma, A, Z) - SIGMA_UPS_3(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO2N = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO2N

  FUNCTION GAMMATO3N(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A, Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO3N ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: land(0:2), pol2(0:2)
    IF(A.EQ.208.AND.Z.EQ.82)THEN
       IF(E_gamma.LT.22.659d0)THEN
          GAMMATO3N = 0.0d0
       ELSEIF(E_gamma.LE.38d0)THEN
          land = (/0.8510484278430387D0, 28.167213873909613D0, 2.5372945327923646D0/)
          pol2 = (/-1.8801037291951625D3, 1.271738540610852D2, -1.9506353726996741D0/)
          GAMMATO3N = LANDAU(E_gamma, land(0), land(1), land(2))* &
               (pol2(0) + pol2(1)*E_gamma + pol2(2)*E_gamma**2)
       ELSEIF(E_gamma.LE.140d0)THEN
          GAMMATO3N = MAX(SIGMA_UPS_3(E_gamma, A, Z) - SIGMA_UPS_4(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO3N = 0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO3N

  FUNCTION GAMMATO4N(E_gamma, A, Z)
    IMPLICIT NONE
    REAL(KIND(1D0)),INTENT(IN)::E_gamma ! photon energy (input) (MeV)
    INTEGER,INTENT(IN)::A,Z ! atomic number and mass
    REAL(KIND(1D0)) :: GAMMATO4N ! photoabsorption cross section (output) (mb)
    REAL(KIND(1D0)):: land(0:2)
    IF (A.EQ.208.AND.Z.EQ.82) THEN
       IF(E_gamma.LT.32d0) THEN
          GAMMATO4N = 0.0d0
       ELSEIF(E_gamma.LT.44.5d0) THEN
          land = (/62.594589628136525, 40.06306462041548, 3.227147958419363/)
          GAMMATO4N = LANDAU(E_gamma, land(0), land(1), land(2))
       ELSEIF(E_gamma.LT.140d0) THEN
          GAMMATO4N = MAX(SIGMA_UPS_4(E_gamma, A, Z) - SIGMA_UPS_5(E_gamma, A, Z), 0d0)
       ELSE
          GAMMATO4N = 0.0d0
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  END FUNCTION GAMMATO4N

  !Fit of the total photoabsorption, list of data used available in the gammaUPC article.
  FUNCTION GAMMAABS(E_gamma, A, Z)
    IMPLICIT NONE
    integer,intent(in)::A, Z ! atomic number and mass
    real(kind(1d0)), intent(in)::E_gamma ! photon energy (input) (MeV)
    real(kind(1d0)) GAMMAABS ! photoabsorption cross section (output) (mb)
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
          GAMMAABS = MODIFIED_LORENTZ(E_gamma, p1(0), p1(1), p1(2)) + &
               MODIFIED_LORENTZ(E_gamma, p1(3), p1(4), p1(5)) + &
               MODIFIED_LORENTZ(E_gamma, p1(6), p1(7), p1(8))
       ELSEIF (E_gamma .LT. 117d0 .AND. E_gamma .GE. 4d1) THEN
          GAMMAABS = (p2(0) + p2(1)*E_gamma + p2(2)*E_gamma**2 + p2(3)*E_gamma**3)* &
               DEXP(-(E_gamma - p2(4))/p2(5)) + p2(6)
       ELSEIF (E_gamma .LT. 50d3 .AND. E_gamma .GE. 117d0) THEN
          GAMMAABS = MODIFIED_LORENTZ(E_gamma, p3(0), p3(1), p3(2)) + p3(3) + &
               GAUSS(E_gamma, p3(4), p3(5), p3(6)) + &
               GAUSS(E_gamma, p3(7), p3(8), p3(9))
       ELSE
          GAMMAABS = p4(4)/1000*(p4(0)*(E_gamma/1000)**(2*p4(2)) + &
               p4(1)*(E_gamma/1000)**(-2*p4(3)))
       END IF
    ELSE
       WRITE (*, *) 'Only Pb208 supported for now'
       STOP
    END IF
    return
  end FUNCTION GAMMAABS
end module photoabsorption

