MODULE ElasticPhotonPhotonFlux
  USE OpticalGlauber_Geometry
  USE NINTLIB ! for multiple dimensional integrations
  USE interpolation
  IMPLICIT NONE
  PRIVATE
  REAL(KIND(1d0)),PUBLIC::alphaem_elasticphoton=0.0072992700729927005d0
  REAL(KIND(1d0)),PRIVATE::aqedup=0.0072992700729927005d0
  ! For UPCs photon-photon collisions
  LOGICAL,PUBLIC::USE_CHARGEFORMFACTOR4PHOTON=.FALSE.
  PUBLIC::PhotonPhotonFlux_pp,PhotonPhotonFlux_pp_eval
  PUBLIC::PhotonFlux_proton_nob
  PUBLIC::PhotonPhotonFlux_pA_hardsphere,PhotonPhotonFlux_pA_hardsphere_eval
  PUBLIC::PhotonPhotonFlux_pA_WoodsSaxon,PhotonPhotonFlux_pA_WoodsSaxon_eval
  PUBLIC::PhotonFlux_nucleus_nob
  PUBLIC::PhotonPhotonFlux_AB_hardsphere,PhotonPhotonFlux_AB_hardsphere_eval
  PUBLIC::PhotonPhotonFlux_AB_WoodsSaxon,PhotonPhotonFlux_AB_WoodsSaxon_eval
  PUBLIC::Lgammagamma_UPC,dLgammagammadW_UPC
  PUBLIC::PhotonNumberDensity ! Equivalent Photon Approximation (EPA)
  PUBLIC::PhotonNumberDensity_ChargeFormFactor_WS,PhotonNumberDensity_ChargeFormFactor_proton
  REAL(KIND(1d0)),PARAMETER,PRIVATE::LOWER_BFactor_Limit=1D-1
  REAL(KIND(1d0)),PARAMETER,PRIVATE::GeVm12fm=0.1973d0 ! GeV-1 to fm
  INTEGER,PARAMETER,PRIVATE::SUB_FACTOR=2
  LOGICAL,PRIVATE,SAVE::print_banner=.FALSE.
  INTEGER,PRIVATE,SAVE::nuclearA_beam1,nuclearA_beam2,nuclearZ_beam1,nuclearZ_beam2
  ! energy in GeV per nucleon in each beam
  REAL(KIND(1d0)),DIMENSION(2),PRIVATE,SAVE::ebeam_PN
CONTAINS
  FUNCTION PNOHAD_pp(bx,by,b0)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for pp collisions
    ! typical value of b0=19.8 GeV-2 at the LHC
    ! fitted by DdE is b0=9.7511+0.222796*log(s/GeV**2)+0.0179103*log(s/GeV**2)**2 GeV-2 from 10^1 to 10^5 GeV dsqrt(s)
    ! a new one (see 2207.03012) b0=9.81+0.211*log(s/GeV**2)+0.0185*log(s/GeV**2)**2 GeV-2
    ! bx and by should be in unit of GeV-1
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_pp
    REAL(KIND(1d0)),INTENT(IN)::bx,by,b0
    REAL(KIND(1d0))::b2,gammasb,exponent
    b2=bx**2+by**2
    exponent=b2/2d0/b0
    IF(exponent.GT.500d0)THEN
       PNOHAD_pp=1d0
    ELSE
       gammasb=DEXP(-exponent)
       PNOHAD_pp=DABS(1d0-gammasb)**2
    ENDIF
    RETURN
  END FUNCTION PNOHAD_pp

  FUNCTION PNOHAD_AB_hardsphere(bx,by,ABAB,RR,sigmaNN)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for AB collisions
    ! bx, by, RR should be in unit of fm
    ! sigmaNN should be in unit of fm^2
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_AB_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bx,by,ABAB
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR
    REAL(KIND(1d0)),INTENT(IN)::sigmaNN ! inelastic NN xs (in unit of fm^2)
    REAL(KIND(1d0))::TAB
    TAB=TABhat_hardsphere_grid(bx,by,RR)*ABAB
    IF(TAB*sigmaNN.GT.500d0)THEN
       PNOHAD_AB_hardsphere=0d0
    ELSE
       PNOHAD_AB_hardsphere=DEXP(-TAB*sigmaNN)
    ENDIF
    RETURN
  END FUNCTION PNOHAD_AB_hardsphere

  FUNCTION PNOHAD_AB_WoodsSaxon(bx,by,RR,w,aa,A,sigmaNN)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for AB collisions
    ! bx, by, RR should be in unit of fm
    ! sigmaNN should be in unit of fm^2
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_AB_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),INTENT(IN)::sigmaNN ! inelastic NN xs (in unit of fm^2)
    REAL(KIND(1d0))::TAB,ABAB
    ABAB=A(1)*A(2)
    TAB=TABhat_WoodsSaxon_grid(bx,by,RR,w,aa,A)*ABAB
    IF(TAB*sigmaNN.GT.500d0)THEN
       PNOHAD_AB_WoodsSaxon=0d0
    ELSE
       PNOHAD_AB_WoodsSaxon=DEXP(-TAB*sigmaNN)
    ENDIF
    RETURN
  END FUNCTION PNOHAD_AB_WoodsSaxon

  FUNCTION PNOHAD_pA_hardsphere(bx,by,RR,A,sigmaNN)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for pA or Ap collisions
    ! bx, by, RR should be in unit of fm
    ! sigmaNN should be in unit of fm^2
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_pA_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bx,by,RR,A
    REAL(KIND(1d0)),INTENT(IN)::sigmaNN ! inelastic NN xs (in unit of fm^2)
    REAL(KIND(1d0))::TA
    TA=TAhat_hardsphere(bx,by,RR)*A
    IF(TA*sigmaNN.GT.500d0)THEN
       PNOHAD_pA_hardsphere=0d0
    ELSE
       PNOHAD_pA_hardsphere=DEXP(-TA*sigmaNN)
    ENDIF
    RETURN
  END FUNCTION PNOHAD_pA_hardsphere

  FUNCTION PNOHAD_pA_WoodsSaxon(bx,by,RR,w,aa,A,sigmaNN)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for pA or Ap collisions
    ! bx, by, RR should be in unit of fm
    ! sigmaNN should be in unit of fm^2
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_pA_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by,RR,w,aa,A
    REAL(KIND(1d0)),INTENT(IN)::sigmaNN ! inelastic NN xs (in unit of fm^2)
    REAL(KIND(1d0))::TA
    TA=TAhat_WoodsSaxon(bx,by,RR,w,aa,A,1,.FALSE.)*A
    IF(TA*sigmaNN.GT.500d0)THEN
       PNOHAD_pA_WoodsSaxon=0d0
    ELSE
       PNOHAD_pA_WoodsSaxon=DEXP(-TA*sigmaNN)
    ENDIF
    RETURN
  END FUNCTION PNOHAD_pA_WoodsSaxon

  FUNCTION PhotonNumberDensity(b,Ega,gamma)
    ! It gives us the photon number density with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonNumberDensity
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma
    REAL(KIND(1d0)),PARAMETER::pi2=9.86960440108935861883449099988d0
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::xx,Egaoga
    REAL(KIND(1d0)),EXTERNAL::BESSK1,BESSK0
    Egaoga=Ega/gamma ! =x_gamma * mN
    xx=Egaoga*b
    PhotonNumberDensity=one/pi2*Egaoga**2*(BESSK1(xx)**2+one/gamma**2*BESSK0(xx)**2)
    RETURN
  END FUNCTION PhotonNumberDensity

  FUNCTION PhotonNumberDensity_ChargeFormFactor_proton(b,Ega,gamma)
    ! It gives us the photon number density with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonNumberDensity_ChargeFormFactor_proton
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma
    REAL(KIND(1d0)),PARAMETER::aa=1.1867816581938533d0 ! in unit of GeV-1 = 1/DSQRT(0.71 GeV^2)
    REAL(KIND(1d0))::btilde,atilde
    REAL(KIND(1d0))::Egaoga
    REAL(KIND(1d0)),EXTERNAL::BESSK1,BESSK0
    REAL(KIND(1d0))::integral,sqrtterm
    REAL(KIND(1d0)),PARAMETER::pi2=9.86960440108935861883449099988d0
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::logb
    REAL(KIND(1d0)),PARAMETER::eulergamma=0.577215664901532860606512090082d0
    REAL(KIND(1d0)),PARAMETER::logtwo=0.693147180559945309417232121458d0
    IF(b.EQ.0d0)THEN
       PhotonNumberDensity_ChargeFormFactor_proton=0d0
       RETURN
    ENDIF
    Egaoga=Ega/gamma ! = x_gamma*mN
    ! we use the dipole form of proton form factor
    ! ChargeFormFactor_dipole_proton(Q)
    ! the analytic form can be fully integrated out
    ! see eq.(7.18) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
    ! first, let us define tilde functions
    btilde=b*Egaoga
    atilde=aa*Egaoga
    sqrtterm=DSQRT(1d0+atilde**(-2))
    IF(btilde.GE.1d-3.OR.btilde/atilde.GE.1d-3)THEN
       integral=Egaoga*(BESSK1(btilde)-sqrtterm*BESSK1(btilde*sqrtterm)&
            -btilde/(2d0*atilde**2)*BESSK0(btilde*sqrtterm))
    ELSE
       logb=DLOG(btilde)
       ! it is better to use the Taylor expansion
       ! for log(btilde) terms we sum up to higher order
       integral=(btilde**3/(16d0*atilde**4)+btilde**5*(3D0*atilde**2+2D0)/(384d0*atilde**6)&
            +btilde**7*(6d0*atilde**4+8d0*atilde**2+3d0)/(18432d0*atilde**8))*logb
       integral=integral+btilde*(one-atilde**2*sqrtterm)/(4d0*atilde**2)
       integral=integral-btilde**3/(64d0*atilde**4)*(2D0*(atilde**4-one)*sqrtterm&
            -2D0*atilde**2+(3d0-4d0*eulergamma+4d0*logtwo))
       integral=integral*Egaoga
    ENDIF
    PhotonNumberDensity_ChargeFormFactor_proton=one/pi2*integral**2
    RETURN
  END FUNCTION PhotonNumberDensity_ChargeFormFactor_proton

  FUNCTION PhotonNumberDensity_ChargeFormFactor_WS(b,Ega,gamma,RR,w,aa,bcut,btilcut,ibeam,integ_method)
    ! It gives us the photon number density with Z=1 and alpha=1 
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER::ibeam
    REAL(KIND(1d0))::PhotonNumberDensity_ChargeFormFactor_WS
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma,RR,w,aa
    REAL(KIND(1d0)),INTENT(IN)::bcut ! if bcut > 0, when b > bcut*RR, it will simply use PhotonNumberDensity (not from form factor). 
                                     ! This might be necessary in order to improve the numerical efficiency
                                     ! A nominal value is 2-3.
    REAL(KIND(1d0)),INTENT(IN)::btilcut ! if btilcut > 0, when b*Ega/gamma > btilcut*RR, it will simply use  PhotonNumberDensity (necessary for numerical stability)
                                        ! A nominal value is 0.7d0.
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule; 2: modified W transform with simpson (a bit slow)
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::Egaoga,integral,xga,Egaoga2,xga2,Ega2
    REAL(KIND(1d0))::Egaoga_common,b_common,R_common,w_common,aa_common
    COMMON/PND_CFF_WS/Egaoga_common,b_common,R_common,w_common,aa_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,NYA,i,j,n,k,l
    SAVE init,NXA,NYA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA,YA
    REAL(KIND(1d0)),DIMENSION(:,:,:),ALLOCATABLE::ZA
    SAVE XA,YA,ZA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=10d0
    INTEGER,PARAMETER::NBMAX=2
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=12
    ! NXSEG for x_gamma from 10**(-n-1) to 10**(-n) 
    INTEGER,PARAMETER::NXSEG=8
    INTEGER::log10xmin,ilog10x
    REAL(KIND(1d0)),PARAMETER::XMIN=1D-8
    REAL(KIND(1d0))::log10x1
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::db,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_save,w_save,aa_save
    SAVE R_save,w_save,aa_save
    INTEGER,PARAMETER::n_interp=6
    REAL(KIND(1d0)),DIMENSION(n_interp)::XA2,YA2
    REAL(KIND(1d0)),PARAMETER::PIPI=3.14159265358979323846264338328d0
    INTEGER::iter,npoints
    REAL(KIND(1d0))::integ1,integ2,integ3,integ4
    INTEGER,PARAMETER::itermax=12
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(nxseg+1)::YD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp,nxseg+1)::ZD2
    IF(ibeam.GT.2.OR.ibeam.LT.1)THEN
       WRITE(*,*)"Error: ibeam=/=1,2 in PhotonNumberDensity_ChargeFormFactor_WS"
       STOP
    ENDIF
    Egaoga=Ega/gamma ! gamma=Ebeam/mN, xga=Ega/Ebeam, Egaoga=xga*mN
    xga=Egaoga/mN
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF
    IF(init(ibeam).EQ.0.OR.(integ_method6.LT.0))THEN
       ! first do a rescaling (this can be explicitly verified from the analytic expressions)
       R_common=RR*Egaoga
       w_common=w ! for w I do not need the scaling
       aa_common=aa*Egaoga
       R_save(ibeam)=RR
       w_save(ibeam)=w
       aa_save(ibeam)=aa
       b_common=b*Egaoga
       Egaoga_common=one
       IF(integ_method6.LT.0)THEN
          IF(b.EQ.0d0)THEN
             PhotonNumberDensity_ChargeFormFactor_WS=0d0
             RETURN
          ELSE
             IF(btilcut.GT.0d0.AND.b_common.GT.btilcut*RR)THEN
                PhotonNumberDensity_ChargeFormFactor_WS=PhotonNumberDensity(b,Ega,gamma)
             ELSE
                npoints=30000
                CALL trapezoid_integration(npoints,PND_ChargeFormFactor_WS_fxn,&
                     one,integral)
                integral=integral*Egaoga/2d0
                integ2=PhotonNumberDensity_AnalyticInt4Series_WS(b,Ega,gamma,RR,w,aa,-1,10)/PIPI
                PhotonNumberDensity_ChargeFormFactor_WS=(integral+integ2)**2
             ENDIF
             RETURN
          ENDIF
       ENDIF
       rescaling_bmax_save(ibeam)=MAX(b,bmaxoR*RR)
       IF(bcut.GT.0d0)THEN
          rescaling_bmax_save(ibeam)=MIN(rescaling_bmax_save(ibeam),bcut*RR)
       ENDIF
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       NYA=NXSEG*(-log10xmin)+1
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NYA))
       ENDIF
       IF(.NOT.ALLOCATED(ZA))THEN
          ALLOCATE(ZA(2,NXA,NYA))
       ENDIF
       db=9d0/DBLE(NBSEG)
       IF(init(3-ibeam).EQ.1.AND.rescaling_bmax_save(ibeam).EQ.rescaling_bmax_save(3-ibeam).AND.&
            R_save(ibeam).EQ.R_save(3-ibeam).AND.w_save(ibeam).EQ.w_save(3-ibeam).AND.&
            aa_save(ibeam).EQ.aa_save(3-ibeam))THEN
          DO k=1,NXA
             XA(ibeam,k)=XA(3-ibeam,k)
          ENDDO
          DO k=1,NYA
             YA(ibeam,k)=YA(3-ibeam,k)
          ENDDO
          DO i=1,NXA
             DO j=1,NYA
                ZA(ibeam,i,j)=ZA(3-ibeam,i,j)
             ENDDO
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of photon number density from charge form factor of beam=",ibeam
          WRITE(*,*)"INFO: it will take a few minutes !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1 (not multiplied Egaoga yet !)
                XA(ibeam,NXA-k+1)=(10d0**(-n-1))*(1d0+DBLE(NBSEG+1-i)*db)*rescaling_bmax_save(ibeam)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in PhotonNumberDensity_ChargeFormFactor_WS"
             STOP
          ENDIF
          XA(ibeam,1)=0d0
          K=0
          DO I=0,log10xmin+1,-1
             DO J=1,nxseg
                log10x1=-1d0/DBLE(nxseg)*DBLE(J-1)+DBLE(I)
                K=K+1
                YA(ibeam,K)=log10x1
             ENDDO
          ENDDO
          IF(K.NE.NYA-1)THEN
             WRITE(*,*)"ERROR: K != NYA-1"
             STOP
          ENDIF
          YA(ibeam,NYA)=DBLE(log10xmin)
          DO I=1,NXA
             CALL progress(I,NXA)
             IF(XA(ibeam,I).EQ.0d0)THEN
                ZA(ibeam,I,J)=0d0
                CYCLE
             ENDIF
             DO J=1,NYA
                xga2=10d0**(YA(ibeam,J)) ! x_gamma
                Egaoga2=xga2*mN
                Ega2=Egaoga2*gamma
                b_common=XA(ibeam,I)*Egaoga2 ! = b*x*mN
                R_common=RR*Egaoga2 ! =  R*x*mN
                aa_common=aa*Egaoga2 ! = aa*x*mN
                IF((btilcut.GT.0d0.AND.b_common.GT.btilcut*RR).OR.(xga2.LT.1D-4.AND.XA(ibeam,I).GT.RR))THEN
                   integ2=PhotonNumberDensity_AnalyticInt4Series_WS(XA(ibeam,I),Ega2,gamma,RR,w,aa,-1,10)/PIPI
                   integ3=PhotonNumberDensity(b_common/Egaoga2,Ega2,gamma)
                   integral=DSQRT(integ3)-integ2
                   ZA(ibeam,I,J)=integral
                ELSE
                   IF(integ_method6.EQ.2)THEN
                      ! use modified W transform (by Sidi) to calculate the integral
                      CALL mWT_integrate_PND_ChargeFormFactor_WS(integral)
                      integral=integral*Egaoga2/PIPI
                   ELSE
                      IF(xga2.LT.1D-7)THEN
                         npoints=500000
                      ELSE
                         npoints=10000
                      ENDIF
                      CALL trapezoid_integration(npoints,PND_ChargeFormFactor_WS_fxn,&
                           one,integral)
                      integral=integral*Egaoga2/2d0
                      integ4=integral
                      integ2=PhotonNumberDensity_AnalyticInt4Series_WS(XA(ibeam,I),Ega2,gamma,RR,w,aa,-1,10)/PIPI
                      integ1=(integral+integ2)**2
                      integ3=PhotonNumberDensity(b_common/Egaoga2,Ega2,gamma)
                      IF(b_common/Egaoga2.GE.2d0*RR.AND.(DABS(integ3/integ1).GT.1.5d0.OR.DABS(integ3/integ1).LT.0.67d0))THEN
                         ! when b_common/Egaoga > 2*RA, the EPA is expected to be good
                         iter=1
                         DO WHILE((DABS(integ3/integ1).GT.1.5d0.OR.DABS(integ3/integ1).LT.0.67d0.OR.&
                              DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)&
                              .AND.iter.LE.itermax)
                            integ4=integral
                            ! increase the points by a factor of 2
                            npoints=npoints*2
                            CALL trapezoid_integration(npoints,PND_ChargeFormFactor_WS_fxn,&
                                 one,integral)
                            integral=integral*Egaoga2/2d0
                            integ1=(integral+integ2)**2
                            iter=iter+1
                         END DO
                         IF(DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)THEN
                            WRITE(*,*)"WARNING: the integral is not stable (b,Ega,gamma,RA,wA,aA)=",&
                                 b_common/Egaoga2,Ega2,gamma,RR,w,aa
                            WRITE(*,*)"WARNING: integral in two iterations #1:",integ4,integral
                         ENDIF
                         IF(DABS(integ3/integ1).GT.1.5d0.OR.DABS(integ3/integ1).LT.0.67d0)THEN
                            WRITE(*,*)"WARNING: the EPA is not good at (b,Ega,gamma,RA,wA,aA)=",&
                                 b_common/Egaoga2,Ega2,gamma,RR,w,aa
                            WRITE(*,*)"WARNING: EPA, non-EPA #1:",integ3,integ1
                         ENDIF
                      ELSEIF(DABS(integ3/integ1).LT.0.67d0.AND.xga2.LT.0.2d0.AND.xga2.GE.1D-7)THEN
                         ! in general, we expect the charge form factor is smaller than EPA when xga is not too close to 1
                         iter=1
                         DO WHILE((DABS(integ3/integ1).LT.0.67d0.OR.&
                              DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)&
                              .AND.iter.LE.itermax)
                            integ4=integral
                            ! increase the points by a factor of 2
                            npoints=npoints*2
                            CALL trapezoid_integration(npoints,PND_ChargeFormFactor_WS_fxn,&
                                 one,integral)
                            integral=integral*Egaoga2/2d0
                            integ1=(integral+integ2)**2
                            iter=iter+1
                         END DO
                         IF(DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)THEN
                            WRITE(*,*)"WARNING: the integral is not stable (b,Ega,gamma,RA,wA,aA)=",&
                                 b_common/Egaoga2,Ega2,gamma,RR,w,aa
                            WRITE(*,*)"WARNING: integral in two iterations #2:",integ4,integral
                         ENDIF
                         IF(DABS(integ3/integ1).LT.0.67d0)THEN
                            WRITE(*,*)"WARNING: the EPA is not good at (b,Ega,gamma,RA,wA,aA)=",&
                                 b_common/Egaoga2,Ega2,gamma,RR,w,aa
                            WRITE(*,*)"WARNING: EPA, non-EPA #2:",integ3,integ1
                         ENDIF
                      ELSEIF(b_common.GT.1D-7.AND.xga2.GE.1D-7)THEN
                         ! we try to do some numerical improvement
                         iter=1
                         DO WHILE((iter.EQ.1.OR.&
                              DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)&
                              .AND.iter.LE.itermax)
                            integ4=integral
                            ! increase the points by a factor of 2
                            npoints=npoints*2
                            CALL trapezoid_integration(npoints,PND_ChargeFormFactor_WS_fxn,&
                                 one,integral)
                            integral=integral*Egaoga2/2d0
                            integ1=(integral+integ2)**2
                            iter=iter+1
                         END DO
                         IF(DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)THEN
                            WRITE(*,*)"WARNING: the integral is not stable (b,Ega,gamma,RA,wA,aA)=",&
                                 b_common/Egaoga2,Ega2,gamma,RR,w,aa
                            WRITE(*,*)"WARNING: integral in two iterations #3:",integ4,integral
                         ENDIF
                      ENDIF
                   ENDIF
                   IF(ISNAN(integral))THEN
                      WRITE(*,*)"ERROR: the integral is not stable (b,Ega,gamma,RA,wA,aA)=",&
                           b_common/Egaoga2,Ega2,gamma,RR,w,aa
                      STOP
                   ENDIF
                   ZA(ibeam,I,J)=integral
                ENDIF
             ENDDO
          ENDDO
       ENDIF
       init(ibeam)=1
    ENDIF
    IF(R_save(ibeam).NE.RR.OR.w_save(ibeam).NE.w.OR.aa_save(ibeam).NE.aa)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in PhotonNumberDensity_ChargeFormFactor_WS"
       WRITE(*,*)"INFO: ibeam=",ibeam
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_save(ibeam),w_save(ibeam),aa_save(ibeam)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR,w,aa
       STOP
    ENDIF
    IF(b.GT.rescaling_bmax_save(ibeam).OR.b.LE.0d0)THEN
       IF(bcut.LE.0d0.OR.b.LE.0d0)THEN
          PhotonNumberDensity_ChargeFormFactor_WS=0d0
       ELSE
          ! we simply use PhotonNumberDensity (EPA)
          PhotonNumberDensity_ChargeFormFactor_WS=PhotonNumberDensity(b,Ega,gamma)
       ENDIF
    ELSEIF((bcut.GT.0d0.AND.b.GT.bcut*RR).OR.(btilcut.GT.0d0.AND.b*Egaoga.GT.btilcut*RR).OR.&
         (xga.LT.1D-4.AND.b.GT.RR))THEN
       ! we simply use PhotonNumberDensity (EPA)
       PhotonNumberDensity_ChargeFormFactor_WS=PhotonNumberDensity(b,Ega,gamma)
    ELSE
       XI(1)=b
       YI(1)=DLOG10(xga)

       db=MIN(b/rescaling_bmax_save(ibeam),1d0)
       N=-FLOOR(DLOG10(db))-1 ! b is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          ! b=rescaling_bmax_save(ibeam)
          K=NXA-n_interp
          !integral=YA(ibeam,NXA)
       ELSE
          ! NXA=NBSEG*(nbmax+1)+1
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam,k-NBSEG).GT.b)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam,k).LT.b)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam,i).GE.b)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
             !DO j=1,n_interp
             !   XA2(j)=XA(ibeam,i-n_interp/2+1+j)
             !   YA2(j)=YA(ibeam,i-n_interp/2+1+j)
             !ENDDO
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
             !DO j=1,n_interp
             !   XA2(j)=XA(ibeam,j)
             !   YA2(j)=YA(ibeam,j)
             !ENDDO
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
             !DO j=1,n_interp
             !   XA2(n_interp-j+1)=XA(ibeam,NA+1-j)
             !   YA2(n_interp-j+1)=YA(ibeam,NA+1-j)
             !ENDDO
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
          !CALL SPLINE_INTERPOLATE(XA2,YA2,n_interp,bb,integral)
       ENDIF

       IF(YI(1).GE.0d0)THEN
          ilog10x=-1
       ELSE
          ilog10x=FLOOR(YI(1))
       ENDIF
       L=NXSEG*(-ilog10x-1)
       
       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam,K+I)
       ENDDO
       DO I=1,NXSEG+1
          YD2_1D(I)=YA(ibeam,L+I)
       ENDDO
       DO I=1,n_interp
          DO J=1,NXSEG+1
             ZD2(I,J)=ZA(ibeam,K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(n_interp-1,NXSEG,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
       integral=ZI(1)
       ! integral=integral*Egaoga/2d0 (the factor has been included in the grid)
       ! the series ones are fully known analytically
       ! Let us keep the first 10 terms
       ! If we want to keep all order terms for K1(btil), set NMIN > 0
       ! otherwise, set NMIN < 0
       integral=integral+PhotonNumberDensity_AnalyticInt4Series_WS(b,Ega,gamma,RR,w,aa,-1,10)/PIPI
       PhotonNumberDensity_ChargeFormFactor_WS=integral**2
    ENDIF
    !PhotonNumberDensity_ChargeFormFactor_WS=Egaoga**2/4d0*integral**2
    RETURN
  END FUNCTION PhotonNumberDensity_ChargeFormFactor_WS

  FUNCTION PND_ChargeFormFactor_WS_fxn(x)
    ! x = ArcTan[kT*gamma/Ega]*2/Pi
    IMPLICIT NONE
    REAL(KIND(1d0))::PND_ChargeFormFactor_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::x
    REAL(KIND(1d0))::kT,Q,CFF,pref,bkT
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    !REAL(KIND(1d0)),PARAMETER::PIo4=0.785398163397448309615660845820d0
    !REAL(KIND(1d0)),PARAMETER::sqrt2Pi=2.50662827463100050241576528481d0
    REAL(KIND(1d0)),EXTERNAL::BESSJ1
    REAL(KIND(1d0))::Egaoga_common,b_common,R_common,w_common,aa_common
    COMMON/PND_CFF_WS/Egaoga_common,b_common,R_common,w_common,aa_common
    IF(x.GE.1d0.OR.x.LE.0d0)THEN
       PND_ChargeFormFactor_WS_fxn=0d0
       RETURN
    ENDIF
    pref=DTAN(PIo2*x)
    kT=pref*Egaoga_common
    bkT=kT*b_common
    pref=pref**2
    Q=DSQRT(kT**2+Egaoga_common**2)
    ! 10 means including the series 10 terms
    !CFF=ChargeFormFactor_WoodsSaxon(Q,R_common,w_common,aa_common,10)
    ! Let us exclude the series terms, which can be integrated fully analytically
    CFF=ChargeFormFactor_WoodsSaxon(Q,R_common,w_common,aa_common,0)
    IF(ISNAN(CFF))THEN
       PRINT *, "ChargeFormFactor is NaN with ",Q, R_common, w_common, aa_common
       STOP
       CFF=0d0
    ENDIF
    !IF(bkT.LE.5d2)THEN
    PND_ChargeFormFactor_WS_fxn=pref*BESSJ1(bkT)*CFF
    !ELSE
    !   PND_ChargeFormFactor_WS_fxn=pref*CFF*(3d0/4d0/DSQRT(bkT)**3/sqrt2Pi*DSIN(PIo4+bkT)&
    !        -2d0/sqrt2Pi/DSQRT(bkT)*DCOS(PIo4+bkT))
    !ENDIF
    RETURN
  END FUNCTION PND_ChargeFormFactor_WS_fxn
  
  SUBROUTINE mWT_integrate_PND_ChargeFormFactor_WS(integral)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(OUT)::integral
    REAL(KIND(1d0)),EXTERNAL::ZEROJP ! zeros of the derivative of J_n
    REAL(KIND(1d0))::Egaoga_common,b_common,R_common,w_common,aa_common
    COMMON/PND_CFF_WS/Egaoga_common,b_common,R_common,w_common,aa_common
    REAL(KIND(1d0))::btil,Rtil,atil,kTtil,integ,kTtilmax
    REAL(KIND(1d0)),PARAMETER::PIPI=3.14159265358979323846264338328d0
    INTEGER::kmin,kmax,nmax,kk,nn,i,ninterval
    INTEGER::pmax,pmax_real
    INTEGER::pmax_save=-2
    SAVE pmax_save
    INTEGER,PARAMETER::PMAXMAX=20,PMINMIN=15
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XS,PSIS,FS
    !REAL(KIND(1d0)),DIMENSION(-1:PMAXMAX+2)::XS
    !REAL(KIND(1d0)),DIMENSION(-1:PMAXMAX+1)::PSIS,FS
    SAVE XS, PSIS, FS
    REAL(KIND(1d0))::xk,xn,Mp0,Np0
    btil=b_common*Egaoga_common
    Rtil=R_common*Egaoga_common
    atil=aa_common*Egaoga_common
    ! the integrand (charge form factor) is exponentially suppressed via e^(-pi*q*aA)
    ! we stop at e^(-15)
    kTtilmax=DSQRT((15d0/(PIPI*atil))**2-1d0)
    ! the zeros of sin(q*RA)
    ! they are DSQRT(((k*Pi)/Rtil)**2-1d0)
    kmin=MAX(CEILING(Rtil/PIPI),1)
    kmax=FLOOR(DSQRT(kTtilmax**2+1d0)*Rtil/PIPI)
    ! the zeros of J1(kT*b)
    ! the ith zeros of J1(x) is ZEROJP(0,i) with ZEROJP(0,1)=0
    ! The fact is that ZEROJP(0,i) is close to (i-1)*Pi
    ! (i-1)*Pi <= ZEROJP(0,i) < i*Pi
    nmax=FLOOR(kTtilmax/(PIPI*btil))+1
    IF(kmax.GE.kmin)THEN
       pmax=nmax+(kmax-kmin+1)-4
    ELSE
       pmax=nmax-4
    ENDIF
    pmax=MIN(pmax,PMAXMAX)
    pmax=MAX(pmax,PMINMIN)
    IF(pmax.LT.-1)THEN
       ninterval=1000
       ! integrate over f from [a,b] with n intervals
       CALL simpson(PND_ChargeFormFactor_WS_fxn2,0d0,kTtilmax,integral,ninterval)
    ELSE
       IF(pmax_save.LT.pmax)THEN
          ! first let us allocate the arrays
          IF(ALLOCATED(XS))THEN
             DEALLOCATE(XS)
          ENDIF
          IF(ALLOCATED(PSIS))THEN
             DEALLOCATE(PSIS)
          ENDIF
          IF(ALLOCATED(FS))THEN
             DEALLOCATE(FS)
          ENDIF
          ALLOCATE(XS(-1:pmax+2))
          ALLOCATE(PSIS(-1:pmax+1))
          ALLOCATE(FS(-1:pmax+1))
          pmax_save=pmax
       ENDIF
       kk=kmin
       nn=1
       xk=DSQRT(((kk*PIPI)/Rtil)**2-1d0)
       xn=ZEROJP(0,nn)
       pmax_real=pmax
       IF(kmin.LE.kmax)THEN
          DO i=-1,pmax+2
             IF(xn.LT.xk)THEN
                XS(i)=xn
                nn=nn+1
                xn=ZEROJP(0,nn)
             ELSEIF(xk.LT.xn)THEN
                XS(i)=xk
                kk=kk+1
                xk=DSQRT(((kk*PIPI)/Rtil)**2-1d0)
             ELSE
                XS(i)=xn
                nn=nn+1
                xn=ZEROJP(0,nn)
                kk=kk+1
                xk=DSQRT(((kk*PIPI)/Rtil)**2-1d0)
                pmax_real=pmax_real-1
             ENDIF
          ENDDO
       ELSE
          DO i=-1,pmax+2
             XS(i)=ZEROJP(0,i+2)
          ENDDO
       ENDIF
       DO i=-1,pmax_real+1
          ninterval=200
          CALL simpson(PND_ChargeFormFactor_WS_fxn2,XS(i),XS(i+1),PSIS(i),ninterval)
          IF(i.EQ.-1)THEN
             FS(i)=0d0
          ELSE
             FS(i)=FS(i-1)+PSIS(i-1)
          ENDIF
       ENDDO
       Mp0=mWT_Mfun(pmax_real,0,pmax_real+2,XS(-1:pmax_real+2),PSIS(-1:pmax_real+1),&
            FS(-1:pmax_real+1))
       Np0=mWT_Nfun(pmax_real,0,pmax_real+2,XS(-1:pmax_real+2),PSIS(-1:pmax_real+1),&
            FS(-1:pmax_real+1))
       integral=Mp0/Np0
    ENDIF
    RETURN
  END SUBROUTINE mWT_integrate_PND_ChargeFormFactor_WS

  FUNCTION PND_ChargeFormFactor_WS_fxn2(x)
    ! x = kTtil
    IMPLICIT NONE
    REAL(KIND(1d0))::PND_ChargeFormFactor_WS_fxn2
    REAL(KIND(1d0)),INTENT(IN)::x
    REAL(KIND(1d0))::kTtil,Qtil,CFF,bkT,Rtil,atil,btil
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),EXTERNAL::BESSJ1
    REAL(KIND(1d0))::Egaoga_common,b_common,R_common,w_common,aa_common
    COMMON/PND_CFF_WS/Egaoga_common,b_common,R_common,w_common,aa_common
    REAL(KIND(1d0)),PARAMETER::one=1d0
    IF(x.LE.0d0)THEN
       PND_ChargeFormFactor_WS_fxn2=0d0
       RETURN
    ENDIF
    kTtil=x ! kTtil=kT/Egaoga
    btil=b_common*Egaoga_common
    Rtil=R_common*Egaoga_common
    atil=aa_common*Egaoga_common
    bkT=kTtil*btil
    Qtil=DSQRT(kTtil**2+one)
    ! 10 means including the series 10 terms
    !CFF=ChargeFormFactor_WoodsSaxon(Q,R_common,w_common,aa_common,10)
    ! Let us exclude the series terms, which can be integrated fully analytically
    ! This is rescaling invariant by Egamma/gamma=x_gamma*mN 
    CFF=ChargeFormFactor_WoodsSaxon(Qtil,Rtil,w_common,atil,0)
    IF(ISNAN(CFF))THEN
       PRINT *, "ChargeFormFactor is NaN with ",Qtil, Rtil, w_common, atil
       STOP
       CFF=0d0
    ENDIF
    PND_ChargeFormFactor_WS_fxn2=BESSJ1(bkT)*CFF*kTtil**2/Qtil**2
    RETURN
  END FUNCTION PND_ChargeFormFactor_WS_fxn2

  ! Eq.(6) in 1607.06083 with Z=1 and alpha=1
  ! also see my notes OpticalGlauber.pdf
  FUNCTION NGAMMA(xi,gamma)
    IMPLICIT NONE
    REAL(KIND(1d0))::NGAMMA
    REAL(KIND(1d0)),INTENT(IN)::xi,gamma
    REAL(KIND(1d0)),EXTERNAL::BESSK1,BESSK0
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    NGAMMA=1d0/PIo2*(xi*BESSK0(xi)*BESSK1(xi)&
         -(1d0-1d0/gamma**2)*xi**2/2d0*(BESSK1(xi)**2-BESSK0(xi)**2))
    RETURN
  END FUNCTION NGAMMA

  FUNCTION PhotonFlux_proton_nob(x,gamma)
    ! set PNOHARD=1
    ! for proton
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_nob
    REAL(KIND(1d0)),INTENT(IN)::x,gamma
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0))::Rp,alpha,Z
    SAVE Rp,alpha,Z
    REAL(KIND(1d0))::xi
    IF(init.EQ.0)THEN
       Rp=Rproton/GeVm12fm ! from fm to GeV-1
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! for proton
       Z=1d0
       init=1
    ENDIF
    xi=mproton*x*Rp
    PhotonFlux_proton_nob=alpha/x*Z**2*NGAMMA(xi,gamma)
    RETURN
  END FUNCTION PhotonFlux_proton_nob

  FUNCTION PhotonFlux_nucleus_nob(x,gamma,Z,RA)
    ! set PNOHARD=1
    ! for proton
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_nucleus_nob
    REAL(KIND(1d0)),INTENT(IN)::x,gamma,Z,RA ! RA is the radius of nucleus (in unit of fm)
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::alpha
    SAVE alpha
    REAL(KIND(1d0))::xi
    IF(init.EQ.0)THEN
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       init=1
    ENDIF
    ! 0.1973 is from fm to GeV-1
    xi=mN*x*RA/GeVm12fm
    PhotonFlux_nucleus_nob=alpha/x*Z**2*NGAMMA(xi,gamma)
    RETURN
  END FUNCTION PhotonFlux_nucleus_nob

  FUNCTION PhotonPhotonFlux_pp(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pp
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x1,log10x2
    INTEGER::ilog10x1,ilog10x2
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,MY,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD
    SAVE MX,MY,XD_1D,YD_1D,ZD
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.0d0.OR.x2.LE.0d0.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pp=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1)THEN
       WRITE(*,*)"INFO: generate grid of photon-photon flux in pp (will take tens of seconds)"
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       !IF(log10xmin.NE.log10xmin_before.or.init.EQ.0)THEN
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(YD_1D))THEN
          DEALLOCATE(YD_1D)
       ENDIF
       ALLOCATE(YD_1D(MY+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1,MY+1))
       !ENDIF
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x1=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x1
             YD_1D(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       YD_1D(MY+1)=DBLE(log10xmin)
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pp_eval(xx1,xx2)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       XI(1)=DLOG10(x1)
       YI(1)=DLOG10(x2)
       ! CALL lagrange_interp_2d(MX,MY,XD_1D,YD_1D,ZD,1,XI,YI,ZI)
       IF(XI(1).GE.0d0)THEN
          ilog10x1=-1
       ELSE
          ilog10x1=FLOOR(XI(1))
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x2=-1
       ELSE
          ilog10x2=FLOOR(YI(1))
       ENDIF
       K=nseg*(-ilog10x1-1)
       DO I=1,nseg+1
          XD2_1D(I)=XD_1D(K+I)
       ENDDO
       L=nseg*(-ilog10x2-1)
       DO I=1,nseg+1
          YD2_1D(I)=YD_1D(L+I)
       ENDDO
       DO I=1,nseg+1
          DO J=1,nseg+1
             ZD2(I,J)=ZD(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_pp_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pp=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.(DABS(ZI(1)/pnohadval).GT.1D2.AND..NOT.USE_CHARGEFORMFACTOR4PHOTON))THEN
          PhotonPhotonFlux_pp=pnohadval
       ELSE
          PhotonPhotonFlux_pp=ZI(1)
       ENDIF
    ELSE
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pp=0d0
       ELSE
          PhotonPhotonFlux_pp=pnohadval
       ENDIF
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pp

  FUNCTION PhotonPhotonFlux_pp_eval(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_pp_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::b0_common ! in unit of GeV-2
    COMMON/PhotonPhoton_pp/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,b0_common
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),PARAMETER::TWOPI=6.28318530717958647692528676656d0
    REAL(KIND(1d0))::integral,Z1,Z2
    SAVE Z1,Z2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    SAVE aax,bbx,sub_num
    REAL(KIND(1d0))::bfact
    SAVE bfact
    REAL(KIND(1d0)),PARAMETER::bupper=2d0
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    REAL(KIND(1d0))::cmenergy
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonFlux_pp_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0)THEN
       IF(nb_proton(1).EQ.1.AND.nb_neutron(1).EQ.0)THEN
          nuclearA_beam1=0
          nuclearZ_beam1=0
       ELSE
          nuclearA_beam1=nb_proton(1)+nb_neutron(1)
          nuclearZ_beam1=nb_proton(1)
       ENDIF
       IF(nb_proton(2).EQ.1.AND.nb_neutron(2).EQ.0)THEN
          nuclearA_beam2=0
          nuclearZ_beam2=0
       ELSE
          nuclearA_beam2=nb_proton(2)+nb_neutron(2)
          nuclearZ_beam2=nb_proton(2)
       ENDIF
       ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
       ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
       gamma1_common=ebeam_PN(1)/mproton
       gamma2_common=ebeam_PN(2)/mproton
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! get b0 from the DdE fit
       cmenergy=2d0*DSQRT(ebeam_PN(1)*ebeam_PN(2)) ! in unit of GeV
       !b0_common=9.7511D0+0.222796D0*DLOG(cmenergy**2)&
       !     +0.0179103D0*DLOG(cmenergy**2)**2 ! in unit of GeV-2
       !a new one (see 2207.03012) b0=9.81+0.211*log(s/GeV**2)+0.0185*log(s/GeV**2)**2 GeV-2
       b0_common=9.81D0+0.211D0*DLOG(cmenergy**2)&
            +0.0185D0*DLOG(cmenergy**2)**2 ! in unit of GeV-2
       ! two Z are 1
       Z1=1d0
       Z2=1d0
       ! 0.1973 is from fm to GeV-1
       bfact=Rproton/GeVm12fm*mproton
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          bfact=bfact*LOWER_BFactor_Limit
       ENDIF
       bbx(1)=bupper
       bbx(2)=bupper
       aax(3)=0d0
       bbx(3)=TWOPI
       sub_num(1)=30
       sub_num(2)=30
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we should increase the number of segments
          sub_num(1)=sub_num(1)*SUB_FACTOR
          sub_num(2)=sub_num(2)*SUB_FACTOR
       ENDIF
       sub_num(3)=10
       init=1
    ENDIF
    x1_common=x1
    x2_common=x2
    E1_common=ebeam_PN(1)*x1
    E2_common=ebeam_PN(2)*x2
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       PhotonPhotonFlux_pp_eval=PhotonFlux_proton_nob(x1_common,gamma1_common)
       PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
            PhotonFlux_proton_nob(x2_common,gamma2_common)
       PhotonPhotonFlux_pp_eval=MAX(PhotonPhotonFlux_pp_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact*x1)
    ! b2*x2*mproton = Exp(bA(2))
    aax(2)=DLOG(bfact*x2)
    CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonFlux_pp_eval=PhotonFlux_proton_nob(x1_common,gamma1_common)
       PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
            PhotonFlux_proton_nob(x2_common,gamma2_common)
       PhotonPhotonFlux_pp_eval=MAX(PhotonPhotonFlux_pp_eval,0d0)
    ELSE
       PhotonPhotonFlux_pp_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pp_eval

  FUNCTION PhotonPhotonFlux_pp_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pp_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3
    ! 1/0.1973d0 from fm to GeV-1 for b 
    ! x1*b1*mproton=Exp(bA(1))
    ! x2*b2*mproton=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    !REAL(KIND(1d0)),PARAMETER::b0=19.8d0 ! in unit of GeV-2
    REAL(KIND(1d0))::b0_common ! in unit of GeV-2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    COMMON/PhotonPhoton_pp/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,b0_common
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_pp_fxn is not a three dimensional function"
       STOP
    ENDIF
    costh=DCOS(bA(3))
    ! in unit of GeV-1
    ! x1*b1*mproton=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mproton
    ! x2*b2*mproton=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mproton
    b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
    pnohad=PNOHAD_pp(b12,0d0,b0_common)
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_pp_fxn=0d0
       RETURN
    ENDIF
    IF(.NOT.USE_CHARGEFORMFACTOR4PHOTON)THEN
       Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
       Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    ELSE
       Ngamma1=PhotonNumberDensity_ChargeFormFactor_proton(b1,E1_common,gamma1_common)
       Ngamma2=PhotonNumberDensity_ChargeFormFactor_proton(b2,E2_common,gamma2_common)
    ENDIF
    PhotonPhotonFlux_pp_fxn=b1**2*b2**2*pnohad*Ngamma1*Ngamma2
    RETURN
  END FUNCTION PhotonPhotonFlux_pp_fxn

  FUNCTION PhotonPhotonFlux_pA_hardsphere(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x1,log10x2
    INTEGER::ilog10x1,ilog10x2
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,MY,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD
    SAVE MX,MY,XD_1D,YD_1D,ZD
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.0d0.OR.x2.LE.0d0.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pA_hardsphere=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1)THEN
       WRITE(*,*)"INFO: generate grid of photon-photon flux in pA or Ap (will take tens of seconds)"
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(YD_1D))THEN
          DEALLOCATE(YD_1D)
       ENDIF
       ALLOCATE(YD_1D(MY+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1,MY+1))
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x1=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x1
             YD_1D(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       YD_1D(MY+1)=DBLE(log10xmin)
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pA_hardsphere_eval(xx1,xx2)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       XI(1)=DLOG10(x1)
       YI(1)=DLOG10(x2)
       IF(XI(1).GE.0d0)THEN
          ilog10x1=-1
       ELSE
          ilog10x1=FLOOR(XI(1))
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x2=-1
       ELSE
          ilog10x2=FLOOR(YI(1))
       ENDIF
       K=nseg*(-ilog10x1-1)
       DO I=1,nseg+1
          XD2_1D(I)=XD_1D(K+I)
       ENDDO
       L=nseg*(-ilog10x2-1)
       DO I=1,nseg+1
          YD2_1D(I)=YD_1D(L+I)
       ENDDO
       DO I=1,nseg+1
          DO J=1,nseg+1
             ZD2(I,J)=ZD(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_pA_hardsphere_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pA_hardsphere=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.DABS(ZI(1)/pnohadval).GT.1D2)THEN
          PhotonPhotonFlux_pA_hardsphere=pnohadval
       ELSE
          PhotonPhotonFlux_pA_hardsphere=ZI(1)
       ENDIF
    ELSE
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pA_hardsphere=0d0
       ELSE
          PhotonPhotonFlux_pA_hardsphere=pnohadval
       ENDIF
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_hardsphere

  FUNCTION PhotonPhotonFlux_pA_hardsphere_eval(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_hardsphere_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, only evaluate with PNOHAD=1
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0))::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    COMMON/PhotonPhoton_pA_HS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),PARAMETER::TWOPI=6.28318530717958647692528676656d0
    REAL(KIND(1d0))::integral,Z1,Z2
    SAVE Z1,Z2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    SAVE aax,bbx,sub_num
    REAL(KIND(1d0))::bfact1,bfact2
    SAVE bfact1,bfact2
    REAL(KIND(1d0)),PARAMETER::bupper=3d0
    REAL(KIND(1d0))::aaVal,wVal
    CHARACTER(len=7)::Aname
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonFlux_pA_hardsphere_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0)THEN
       IF(nb_proton(1).EQ.1.AND.nb_neutron(1).EQ.0)THEN
          nuclearA_beam1=0
          nuclearZ_beam1=0
       ELSE
          nuclearA_beam1=nb_proton(1)+nb_neutron(1)
          nuclearZ_beam1=nb_proton(1)
       ENDIF
       IF(nb_proton(2).EQ.1.AND.nb_neutron(2).EQ.0)THEN
          nuclearA_beam2=0
          nuclearZ_beam2=0
       ELSE
          nuclearA_beam2=nb_proton(2)+nb_neutron(2)
          nuclearZ_beam2=nb_proton(2)
       ENDIF
       ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
       ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
       IF(nuclearA_beam1.NE.0)THEN
          gamma1_common=ebeam_PN(2)/mproton
          gamma2_common=ebeam_PN(1)/mN
       ELSE
          gamma1_common=ebeam_PN(1)/mproton
          gamma2_common=ebeam_PN(2)/mN
       ENDIF
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! proton Z is 1
       Z1=1d0
       ! read the nuclei information
       !nuclear_dir="./nuclear/"
       IF(nuclearA_beam1.NE.0)THEN
          Aname=GetASymbol(nuclearA_beam1,nuclearZ_beam1)
       ELSEIF(nuclearA_beam2.NE.0)THEN
          Aname=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
       ELSE
          WRITE(*,*)"ERROR: please set nuclearA_beam1 or nuclearA_beam2 nonzero first !"
          STOP
       ENDIF
       WRITE(*,*)"INFO: Two photon UPCs in p+"//TRIM(Aname)//" collisions"
       CALL GetNuclearInfo(Aname,A_common,Z2,RA_common,aaval,wval)
       ! read the inelastic NN cross section
       cmenergy=2d0*DSQRT(ebeam_PN(1)*ebeam_PN(2))
       sigNN_inel_common=sigma_inelastic(cmenergy)
       sigNN_inel_common=sigNN_inel_common*0.1d0 ! from mb to fm^2
       ! 0.1973 is from fm to GeV-1
       bfact1=Rproton/GeVm12fm*mproton
       bfact2=RA_common/GeVm12fm*mN
       bbx(1)=bupper
       bbx(2)=bupper
       aax(3)=0d0
       bbx(3)=TWOPI
       sub_num(1)=30
       sub_num(2)=30
       sub_num(3)=10
       init=1
    ENDIF
    IF(nuclearA_beam1.NE.0)THEN
       ! swap two beams
       x1_common=x2
       x2_common=x1
       E1_common=ebeam_PN(2)*x2
       E2_common=ebeam_PN(1)*x1
    ELSE
       x1_common=x1
       x2_common=x2
       E1_common=ebeam_PN(1)*x1
       E2_common=ebeam_PN(2)*x2
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonFlux_proton_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonPhotonFlux_pA_hardsphere_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common)
       PhotonPhotonFlux_pA_hardsphere_eval=MAX(PhotonPhotonFlux_pA_hardsphere_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2))
    aax(2)=DLOG(bfact2*x2_common)
    CALL ROMBERG_ND(PhotonPhotonFlux_pA_hardsphere_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          CALL ROMBERG_ND(PhotonPhotonFlux_pA_hardsphere_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonFlux_proton_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonPhotonFlux_pA_hardsphere_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common)
       PhotonPhotonFlux_pA_hardsphere_eval=MAX(PhotonPhotonFlux_pA_hardsphere_eval,0d0)
    ELSE
       PhotonPhotonFlux_pA_hardsphere_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_hardsphere_eval

  FUNCTION PhotonPhotonFlux_pA_hardsphere_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_hardsphere_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mproton=Exp(bA(1))
    ! x2*b2*mproton=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0))::RA_common, A_common      ! radius of nuclei and atom number of nuclei 
    COMMON/PhotonPhoton_pA_HS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_pA_hardsphere_fxn is not a three dimensional function"
       STOP
    ENDIF
    costh=DCOS(bA(3))
    ! in unit of GeV-1
    ! x1*b1*mproton=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mproton
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
    b12=b12*GeVm12fm ! from GeV-1 to fm
    pnohad=PNOHAD_pA_hardsphere(b12,0d0,RA_common,A_common,sigNN_inel_common)
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_pA_hardsphere_fxn=0d0
       RETURN
    ENDIF
    Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
    Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    PhotonPhotonFlux_pA_hardsphere_fxn=b1**2*b2**2*pnohad*Ngamma1*Ngamma2
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_hardsphere_fxn

  FUNCTION PhotonPhotonFlux_pA_WoodsSaxon(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x1,log10x2
    INTEGER::ilog10x1,ilog10x2
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,MY,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD
    SAVE MX,MY,XD_1D,YD_1D,ZD
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.0d0.OR.x2.LE.0d0.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pA_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1)THEN
       WRITE(*,*)"INFO: generate grid of photon-photon flux in pA or Ap (will take a few minutes)"
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(YD_1D))THEN
          DEALLOCATE(YD_1D)
       ENDIF
       ALLOCATE(YD_1D(MY+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1,MY+1))
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x1=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x1
             YD_1D(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       YD_1D(MY+1)=DBLE(log10xmin)
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pA_WoodsSaxon_eval(xx1,xx2)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       XI(1)=DLOG10(x1)
       YI(1)=DLOG10(x2)
       IF(XI(1).GE.0d0)THEN
          ilog10x1=-1
       ELSE
          ilog10x1=FLOOR(XI(1))
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x2=-1
       ELSE
          ilog10x2=FLOOR(YI(1))
       ENDIF
       K=nseg*(-ilog10x1-1)
       DO I=1,nseg+1
          XD2_1D(I)=XD_1D(K+I)
       ENDDO
       L=nseg*(-ilog10x2-1)
       DO I=1,nseg+1
          YD2_1D(I)=YD_1D(L+I)
       ENDDO
       DO I=1,nseg+1
          DO J=1,nseg+1
             ZD2(I,J)=ZD(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_pA_WoodsSaxon_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pA_WoodsSaxon=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.(DABS(ZI(1)/pnohadval).GT.1D2.AND..NOT.USE_CHARGEFORMFACTOR4PHOTON))THEN
          PhotonPhotonFlux_pA_WoodsSaxon=pnohadval
       ELSE
          PhotonPhotonFlux_pA_WoodsSaxon=ZI(1)
       ENDIF
    ELSE
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pA_WoodsSaxon=0d0
       ELSE
          PhotonPhotonFlux_pA_WoodsSaxon=pnohadval
       ENDIF
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_WoodsSaxon

  FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_eval(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_WoodsSaxon_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0))::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0))::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_pA_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,aaVal_common,wVal_common
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),PARAMETER::TWOPI=6.28318530717958647692528676656d0
    REAL(KIND(1d0))::integral,Z1,Z2
    SAVE Z1,Z2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    SAVE aax,bbx,sub_num
    REAL(KIND(1d0))::bfact1,bfact2
    SAVE bfact1,bfact2
    REAL(KIND(1d0)),PARAMETER::bupper=3d0
    CHARACTER(len=7)::Aname
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonFlux_pA_WoodsSaxon_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0)THEN
       IF(nb_proton(1).EQ.1.AND.nb_neutron(1).EQ.0)THEN
          nuclearA_beam1=0
          nuclearZ_beam1=0
       ELSE
          nuclearA_beam1=nb_proton(1)+nb_neutron(1)
          nuclearZ_beam1=nb_proton(1)
       ENDIF
       IF(nb_proton(2).EQ.1.AND.nb_neutron(2).EQ.0)THEN
          nuclearA_beam2=0
          nuclearZ_beam2=0
       ELSE
          nuclearA_beam2=nb_proton(2)+nb_neutron(2)
          nuclearZ_beam2=nb_proton(2)
       ENDIF
       ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
       ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
       IF(nuclearA_beam1.NE.0)THEN
          gamma1_common=ebeam_PN(2)/mproton
          gamma2_common=ebeam_PN(1)/mN
       ELSE
          gamma1_common=ebeam_PN(1)/mproton
          gamma2_common=ebeam_PN(2)/mN
       ENDIF
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! proton Z is 1
       Z1=1d0
       ! read the nuclei information
       !nuclear_dir="./nuclear/"
       IF(nuclearA_beam1.NE.0)THEN
          Aname=GetASymbol(nuclearA_beam1,nuclearZ_beam1)
       ELSEIF(nuclearA_beam2.NE.0)THEN
          Aname=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
       ELSE
          WRITE(*,*)"ERROR: please set nuclearA_beam1/nuclearZ_beam1 or nuclearA_beam2/nuclearZ_beam2 nonzero first !"
          STOP
       ENDIF
       WRITE(*,*)"INFO: Two photon UPCs in p+"//TRIM(Aname)//" collisions"
       CALL GetNuclearInfo(Aname,A_common,Z2,RA_common,aaVal_common,wVal_common)
       ! read the inelastic NN cross section
       cmenergy=2d0*DSQRT(ebeam_PN(1)*ebeam_PN(2))
       sigNN_inel_common=sigma_inelastic(cmenergy)
       sigNN_inel_common=sigNN_inel_common*0.1d0 ! from mb to fm^2
       ! 0.1973 is from fm to GeV-1
       bfact1=Rproton/GeVm12fm*mproton
       bfact2=RA_common/GeVm12fm*mN
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          bfact1=bfact1*LOWER_BFactor_Limit
          bfact2=bfact2*LOWER_BFactor_Limit
       ENDIF
       bbx(1)=bupper
       bbx(2)=bupper
       aax(3)=0d0
       bbx(3)=TWOPI
       sub_num(1)=30
       sub_num(2)=30
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we should increase the number of segments
          sub_num(1)=sub_num(1)*SUB_FACTOR
          sub_num(2)=sub_num(2)*SUB_FACTOR
       ENDIF
       sub_num(3)=10
       init=1
    ENDIF
    IF(nuclearA_beam1.NE.0)THEN
       ! swap two beams
       x1_common=x2
       x2_common=x1
       E1_common=ebeam_PN(2)*x2
       E2_common=ebeam_PN(1)*x1
    ELSE
       x1_common=x1
       x2_common=x2
       E1_common=ebeam_PN(1)*x1
       E2_common=ebeam_PN(2)*x2
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common)
       PhotonPhotonFlux_pA_WoodsSaxon_eval=MAX(PhotonPhotonFlux_pA_WoodsSaxon_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2))
    aax(2)=DLOG(bfact2*x2_common)
    CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common)
       PhotonPhotonFlux_pA_WoodsSaxon_eval=MAX(PhotonPhotonFlux_pA_WoodsSaxon_eval,0d0)
    ELSE
       PhotonPhotonFlux_pA_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_eval

  FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_fxn(dim_num,bA)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_WoodsSaxon_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mproton=Exp(bA(1))
    ! x2*b2*mproton=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0))::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0))::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_pA_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,aaVal_common,wVal_common
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::RR,aaa
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_pA_WoodsSaxon_fxn is not a three dimensional function"
       STOP
    ENDIF
    costh=DCOS(bA(3))
    ! in unit of GeV-1
    ! x1*b1*mproton=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mproton
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
    b12=b12*GeVm12fm ! from GeV-1 to fm
    pnohad=PNOHAD_pA_WoodsSaxon(b12,0d0,RA_common,wVal_common,aaVal_common,&
         A_common,sigNN_inel_common)
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_pA_WoodsSaxon_fxn=0d0
       RETURN
    ENDIF
    IF(.NOT.USE_CHARGEFORMFACTOR4PHOTON)THEN
       Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
       Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    ELSE
       ! converting from fm to GeV-1
       RR=RA_common/GeVm12fm
       aaa=aaVal_common/GeVm12fm
       IF(nuclearA_beam1.NE.0)THEN
          Ngamma1=PhotonNumberDensity_ChargeFormFactor_WS(b1,E1_common,gamma1_common,&
               RR,wVal_common,aaa,3d0,0.7d0,1)
          Ngamma2=PhotonNumberDensity_ChargeFormFactor_proton(b2,E2_common,gamma2_common)
       ELSE
          Ngamma1=PhotonNumberDensity_ChargeFormFactor_proton(b1,E1_common,gamma1_common)
          Ngamma2=PhotonNumberDensity_ChargeFormFactor_WS(b2,E2_common,gamma2_common,&
               RR,wVal_common,aaa,3d0,0.7d0,2)
       ENDIF
    ENDIF
    PhotonPhotonFlux_pA_WoodsSaxon_fxn=b1**2*b2**2*pnohad*Ngamma1*Ngamma2
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_fxn

  FUNCTION PhotonPhotonFlux_AB_hardsphere(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x1,log10x2
    INTEGER::ilog10x1,ilog10x2
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,MY,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD
    SAVE MX,MY,XD_1D,YD_1D,ZD
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.0d0.OR.x2.LE.0d0.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_AB_hardsphere=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1)THEN
       WRITE(*,*)"INFO: generate grid of photon-photon flux in AB (will take tens of seconds)"
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(YD_1D))THEN
          DEALLOCATE(YD_1D)
       ENDIF
       ALLOCATE(YD_1D(MY+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1,MY+1))
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x1=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x1
             YD_1D(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       YD_1D(MY+1)=DBLE(log10xmin)
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_AB_hardsphere_eval(xx1,xx2)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       XI(1)=DLOG10(x1)
       YI(1)=DLOG10(x2)
       IF(XI(1).GE.0d0)THEN
          ilog10x1=-1
       ELSE
          ilog10x1=FLOOR(XI(1))
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x2=-1
       ELSE
          ilog10x2=FLOOR(YI(1))
       ENDIF
       K=nseg*(-ilog10x1-1)
       DO I=1,nseg+1
          XD2_1D(I)=XD_1D(K+I)
       ENDDO
       L=nseg*(-ilog10x2-1)
       DO I=1,nseg+1
          YD2_1D(I)=YD_1D(L+I)
       ENDDO
       DO I=1,nseg+1
          DO J=1,nseg+1
             ZD2(I,J)=ZD(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_AB_hardsphere_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_AB_hardsphere=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.DABS(ZI(1)/pnohadval).GT.1D2)THEN
          PhotonPhotonFlux_AB_hardsphere=pnohadval
       ELSE
          PhotonPhotonFlux_AB_hardsphere=ZI(1)
       ENDIF
    ELSE
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_AB_hardsphere=0d0
       ELSE
          PhotonPhotonFlux_AB_hardsphere=pnohadval
       ENDIF
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_hardsphere

  FUNCTION PhotonPhotonFlux_AB_hardsphere_eval(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_hardsphere_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, only evaluate with PNOHAD=1
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    COMMON/PhotonPhoton_AB_HS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),PARAMETER::TWOPI=6.28318530717958647692528676656d0
    REAL(KIND(1d0))::integral,Z1,Z2
    SAVE Z1,Z2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    SAVE aax,bbx,sub_num
    REAL(KIND(1d0))::bfact1,bfact2
    SAVE bfact1,bfact2
    REAL(KIND(1d0)),PARAMETER::bupper=3d0
    REAL(KIND(1d0))::aaVal,wVal
    CHARACTER(len=7)::Aname1,Aname2
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonFlux_AB_hardsphere_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0)THEN
       IF(nb_proton(1).EQ.1.AND.nb_neutron(1).EQ.0)THEN
          nuclearA_beam1=0
          nuclearZ_beam1=0
       ELSE
          nuclearA_beam1=nb_proton(1)+nb_neutron(1)
          nuclearZ_beam1=nb_proton(1)
       ENDIF
       IF(nb_proton(2).EQ.1.AND.nb_neutron(2).EQ.0)THEN
          nuclearA_beam2=0
          nuclearZ_beam2=0
       ELSE
          nuclearA_beam2=nb_proton(2)+nb_neutron(2)
          nuclearZ_beam2=nb_proton(2)
       ENDIF
       ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
       ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
       IF(nuclearA_beam1.EQ.0.OR.nuclearA_beam2.EQ.0)THEN
          WRITE(*,*)"ERROR: Please set two beams as heavy ions first"
          STOP
       ENDIf
       gamma1_common=ebeam_PN(1)/mN
       gamma2_common=ebeam_PN(2)/mN
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! read the nuclei information
       !nuclear_dir="./nuclear/"
       Aname1=GetASymbol(nuclearA_beam1,nuclearZ_beam1)
       CALL GetNuclearInfo(Aname1,A_common(1),Z1,RA_common(1),aaval,wval)
       IF(nuclearA_beam2.NE.nuclearA_beam1.OR.nuclearZ_beam1.NE.nuclearZ_beam2)THEN
          Aname2=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
          CALL GetNuclearInfo(Aname2,A_common(2),Z2,RA_common(2),aaval,wval)
       ELSE
          Aname2=Aname1
          A_common(2)=A_common(1)
          Z2=Z1
          RA_common(2)=RA_common(1)
       ENDIF
       WRITE(*,*)"INFO: Two photon UPCs in "//TRIM(Aname1)//"+"//TRIM(Aname2)//" collisions"
       ! read the inelastic NN cross section
       cmenergy=2d0*DSQRT(ebeam_PN(1)*ebeam_PN(2))
       sigNN_inel_common=sigma_inelastic(cmenergy)
       sigNN_inel_common=sigNN_inel_common*0.1d0 ! from mb to fm^2
       ! 0.1973 is from fm to GeV-1
       bfact1=RA_common(1)/GeVm12fm*mN
       bfact2=RA_common(2)/GeVm12fm*mN
       bbx(1)=bupper
       bbx(2)=bupper
       aax(3)=0d0
       bbx(3)=TWOPI
       sub_num(1)=30
       sub_num(2)=30
       sub_num(3)=10
       init=1
    ENDIF
    x1_common=x1
    x2_common=x2
    E1_common=ebeam_PN(1)*x1
    E2_common=ebeam_PN(2)*x2
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonFlux_nucleus_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonPhotonFlux_AB_hardsphere_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common(2))
       PhotonPhotonFlux_AB_hardsphere_eval=MAX(PhotonPhotonFlux_AB_hardsphere_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mN = Exp(bA(1))
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2))
    aax(2)=DLOG(bfact2*x2_common)
    CALL ROMBERG_ND(PhotonPhotonFlux_AB_hardsphere_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          CALL ROMBERG_ND(PhotonPhotonFlux_AB_hardsphere_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1,x2
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonFlux_nucleus_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonPhotonFlux_AB_hardsphere_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common(2))
       PhotonPhotonFlux_AB_hardsphere_eval=MAX(PhotonPhotonFlux_AB_hardsphere_eval,0d0)
    ELSE
       PhotonPhotonFlux_AB_hardsphere_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_hardsphere_eval

  FUNCTION PhotonPhotonFlux_AB_hardsphere_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_hardsphere_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mN=Exp(bA(1))
    ! x2*b2*mN=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    COMMON/PhotonPhoton_AB_HS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_AB_hardsphere_fxn is not a three dimensional function"
       STOP
    ENDIF
    costh=DCOS(bA(3))
    ! in unit of GeV-1
    ! x1*b1*mN=Exp(bA(1))                                                                                                         
    b1=DEXP(bA(1))/x1_common/mN
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
    b12=b12*GeVm12fm ! from GeV-1 to fm
    pnohad=PNOHAD_AB_hardsphere(b12,0d0,A_common(1)*A_common(2),RA_common,&
         sigNN_inel_common)
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_AB_hardsphere_fxn=0d0
       RETURN
    ENDIF
    Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
    Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    PhotonPhotonFlux_AB_hardsphere_fxn=b1**2*b2**2*pnohad*Ngamma1*Ngamma2
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_hardsphere_fxn

  FUNCTION PhotonPhotonFlux_AB_WoodsSaxon(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x1,log10x2
    INTEGER::ilog10x1,ilog10x2
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,MY,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD
    SAVE MX,MY,XD_1D,YD_1D,ZD
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.0d0.OR.x2.LE.0d0.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_AB_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1)THEN
       WRITE(*,*)"INFO: generate grid of photon-photon flux in AB (will take a few minutes)"
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(YD_1D))THEN
          DEALLOCATE(YD_1D)
       ENDIF
       ALLOCATE(YD_1D(MY+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1,MY+1))
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x1=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x1
             YD_1D(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       YD_1D(MY+1)=DBLE(log10xmin)
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_AB_WoodsSaxon_eval(xx1,xx2)
           ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       XI(1)=DLOG10(x1)
       YI(1)=DLOG10(x2)
       IF(XI(1).GE.0d0)THEN
          ilog10x1=-1
       ELSE
          ilog10x1=FLOOR(XI(1))
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x2=-1
       ELSE
          ilog10x2=FLOOR(YI(1))
       ENDIF
       K=nseg*(-ilog10x1-1)
       DO I=1,nseg+1
          XD2_1D(I)=XD_1D(K+I)
       ENDDO
       L=nseg*(-ilog10x2-1)
       DO I=1,nseg+1
          YD2_1D(I)=YD_1D(L+I)
       ENDDO
       DO I=1,nseg+1
          DO J=1,nseg+1
             ZD2(I,J)=ZD(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_AB_WoodsSaxon=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.(DABS(ZI(1)/pnohadval).GT.1D2.AND..NOT.USE_CHARGEFORMFACTOR4PHOTON))THEN
          PhotonPhotonFlux_AB_WoodsSaxon=pnohadval
       ELSE
          !IF(DABS(ZI(1)/pnohadval).GT.1D2.OR.(DABS(ZI(1)/pnohadval).LT.1D-2))THEN
          !   PRINT *, "WARNING:",x1,x2, ZI(1), pnohadval
          !   !STOP
          !ENDIF
          PhotonPhotonFlux_AB_WoodsSaxon=ZI(1)
       ENDIF
    ELSE
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_AB_WoodsSaxon=0d0
       ELSE
          PhotonPhotonFlux_AB_WoodsSaxon=pnohadval
       ENDIF
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_WoodsSaxon

  FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_WoodsSaxon_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0)),DIMENSION(2)::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_AB_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,aaVal_common,wVal_common
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),PARAMETER::TWOPI=6.28318530717958647692528676656d0
    REAL(KIND(1d0))::integral,Z1,Z2
    SAVE Z1,Z2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    SAVE aax,bbx,sub_num
    REAL(KIND(1d0))::bfact1,bfact2
    SAVE bfact1,bfact2
    REAL(KIND(1d0)),PARAMETER::bupper=3d0
    CHARACTER(len=7)::Aname1,Aname2
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonFlux_AB_WoodsSaxon_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0)THEN
       IF(nb_proton(1).EQ.1.AND.nb_neutron(1).EQ.0)THEN
          nuclearA_beam1=0
          nuclearZ_beam1=0
       ELSE
          nuclearA_beam1=nb_proton(1)+nb_neutron(1)
          nuclearZ_beam1=nb_proton(1)
       ENDIF
       IF(nb_proton(2).EQ.1.AND.nb_neutron(2).EQ.0)THEN
          nuclearA_beam2=0
          nuclearZ_beam2=0
       ELSE
          nuclearA_beam2=nb_proton(2)+nb_neutron(2)
          nuclearZ_beam2=nb_proton(2)
       ENDIF
       ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
       ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
       IF(nuclearA_beam1.EQ.0.OR.nuclearA_beam2.EQ.0)THEN
          WRITE(*,*)"ERROR: Please set two beams as heavy ions first"
          STOP
       ENDIF
       gamma1_common=ebeam_PN(1)/mN
       gamma2_common=ebeam_PN(2)/mN
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
       ! read the nuclei information
       !nuclear_dir="./nuclear/"
       Aname1=GetASymbol(nuclearA_beam1,nuclearZ_beam1)
       CALL GetNuclearInfo(Aname1,A_common(1),Z1,RA_common(1),aaVal_common(1),wVal_common(1))
       IF(nuclearA_beam2.NE.nuclearA_beam1.OR.nuclearZ_beam1.NE.nuclearZ_beam2)THEN
          Aname2=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
          CALL GetNuclearInfo(Aname2,A_common(2),Z2,RA_common(2),aaVal_common(2),wVal_common(2))
       ELSE
          Aname2=Aname1
          A_common(2)=A_common(1)
          Z2=Z1
          RA_common(2)=RA_common(1)
          aaVal_common(2)=aaVal_common(1)
          wVal_common(2)=wVal_common(1)
       ENDIF
       WRITE(*,*)"INFO: Two photon UPCs in "//TRIM(Aname1)//"+"//TRIM(Aname2)//" collisions"
       ! read the inelastic NN cross section
       cmenergy=2d0*DSQRT(ebeam_PN(1)*ebeam_PN(2))
       sigNN_inel_common=sigma_inelastic(cmenergy)
       sigNN_inel_common=sigNN_inel_common*0.1d0 ! from mb to fm^2
       ! 0.1973 is from fm to GeV-1
       bfact1=RA_common(1)/GeVm12fm*mN
       bfact2=RA_common(2)/GeVm12fm*mN
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we can integrate b down to zero
          bfact1=bfact1*LOWER_BFactor_Limit
          bfact2=bfact2*LOWER_BFactor_Limit
       ENDIF
       bbx(1)=bupper
       bbx(2)=bupper
       aax(3)=0d0
       bbx(3)=TWOPI
       sub_num(1)=30
       sub_num(2)=30
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we should increase the number of segments
          sub_num(1)=sub_num(1)*SUB_FACTOR
          sub_num(2)=sub_num(2)*SUB_FACTOR
       ENDIF
       sub_num(3)=10
       init=1
    ENDIF
    x1_common=x1
    x2_common=x2
    E1_common=ebeam_PN(1)*x1
    E2_common=ebeam_PN(2)*x2
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common(2))
       PhotonPhotonFlux_AB_WoodsSaxon_eval=MAX(PhotonPhotonFlux_AB_WoodsSaxon_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mN = Exp(bA(1)) = b1*E_gamma1/gamma1 = b1tilde
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2)) = b2*E_gamma2/gamma2 = b2tilde
    aax(2)=DLOG(bfact2*x2_common)
    CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1,x2
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
            PhotonFlux_nucleus_nob(x2_common,gamma2_common,Z2,RA_common(2))
       PhotonPhotonFlux_AB_WoodsSaxon_eval=MAX(PhotonPhotonFlux_AB_WoodsSaxon_eval,0d0)
    ELSE
       PhotonPhotonFlux_AB_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_eval

  FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_WoodsSaxon_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mN=Exp(bA(1))
    ! x2*b2*mN=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0)),DIMENSION(2)::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_AB_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,aaVal_common,wVal_common
    REAL(KIND(1d0)),DIMENSION(2)::RR,aaa
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_AB_WoodsSaxon_fxn is not a three dimensional function"
       STOP
    ENDIF
    costh=DCOS(bA(3))
    ! in unit of GeV-1
    ! x1*b1*mN=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mN
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
    b12=b12*GeVm12fm ! from GeV-1 to fm

    pnohad=PNOHAD_AB_WoodsSaxon(b12,0d0,RA_common,wVal_common,aaVal_common,&
         A_common,sigNN_inel_common)
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_AB_WoodsSaxon_fxn=0d0
       RETURN
    ENDIF
    IF(.NOT.USE_CHARGEFORMFACTOR4PHOTON)THEN
       Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
       Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    ELSE
       ! converting from fm to GeV-1
       RR(1)=RA_common(1)/GeVm12fm
       RR(2)=RA_common(2)/GeVm12fm
       aaa(1)=aaVal_common(1)/GeVm12fm
       aaa(2)=aaVal_common(2)/GeVm12fm
       Ngamma1=PhotonNumberDensity_ChargeFormFactor_WS(b1,E1_common,gamma1_common,&
            RR(1),wVal_common(1),aaa(1),3d0,0.7d0,1)
       Ngamma2=PhotonNumberDensity_ChargeFormFactor_WS(b2,E2_common,gamma2_common,&
            RR(2),wVal_common(2),aaa(2),3d0,0.7d0,2)
    ENDIF
    PhotonPhotonFlux_AB_WoodsSaxon_fxn=b1**2*b2**2*pnohad*Ngamma1*Ngamma2
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_fxn

  ! photon-photon Luminosity
  ! see Eq.(7.13) in my notes OpticalGlauber.pdf
  FUNCTION Lgammagamma_UPC(scale,icoll,iprofile)
    ! icoll: 1 - pp; 2 - pA; 3 - AB
    ! iprofile: 0: P_{NOHAD}=1; 1 - Woods-Saxon; 2 - hard-sphere
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::Lgammagamma_UPC
    INTEGER,INTENT(IN)::icoll,iprofile
    REAL(KIND(1d0)),INTENT(IN)::scale ! shat=scale**2
    REAL(KIND(1d0))::tau_common,s,log1oxmax
    INTEGER::collision_type_common,profile_type_common
    COMMON/Lgammagamma_UPC_com/collision_type_common,profile_type_common,&
         tau_common
    ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
    ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
    s=4d0*ebeam_PN(1)*ebeam_PN(2)
    tau_common=scale**2/s
    collision_type_common=icoll
    profile_type_common=iprofile
    log1oxmax=DLOG(1d0/tau_common)
    CALL trapezoid_integration(1000,Lgammagamma_UPC_fxn,&
         log1oxmax,Lgammagamma_UPC)
    RETURN
  END FUNCTION Lgammagamma_UPC

  FUNCTION Lgammagamma_UPC_fxn(log1ox)
    IMPLICIT NONE
    REAL(KIND(1d0))::Lgammagamma_UPC_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1ox ! log(1/x)
    REAL(KIND(1d0))::x1,x2
    REAL(KIND(1d0))::tau_common
    INTEGER::collision_type_common,profile_type_common
    COMMON/Lgammagamma_UPC_com/collision_type_common,profile_type_common,&
         tau_common
    x1=DEXP(-log1ox)
    x2=tau_common/x1
    IF(collision_type_common.EQ.1)THEN
       ! pp
       IF(profile_type_common.EQ.0)THEN
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_pp(x1,x2,.TRUE.)
       ELSE
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_pp(x1,x2)
       ENDIF
    ELSEIF(collision_type_common.EQ.2)THEN
       ! pA or Ap
       IF(profile_type_common.EQ.1)THEN
          ! Woods-Saxon
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_pA_WoodsSaxon(x1,x2)
       ELSEIF(profile_type_common.EQ.2)THEN
          ! Hard-Sphere
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_pA_hardsphere(x1,x2)
       ELSEIF(profile_type_common.EQ.0)THEN
          ! P_{NOHAD}=1
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_pA_WoodsSaxon(x1,x2,.TRUE.)
       ELSE
          WRITE(*,*)"ERROR: do not know the profile type = ",profile_type_common
          STOP
       ENDIF
    ELSEIF(collision_type_common.EQ.3)THEN
       ! AB
       IF(profile_type_common.EQ.1)THEN
          ! Woods-Saxon
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_AB_WoodsSaxon(x1,x2)
       ELSEIF(profile_type_common.EQ.2)THEN
          ! Hard-Sphere
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_AB_hardsphere(x1,x2)
       ELSEIF(profile_type_common.EQ.0)THEN
          ! P_{NOHAD}=1
          Lgammagamma_UPC_fxn=PhotonPhotonFlux_AB_WoodsSaxon(x1,x2,.TRUE.)
       ELSE
          WRITE(*,*)"ERROR: do not know the profile type = ",profile_type_common
          STOP
       ENDIF
    ELSE
       WRITE(*,*)"ERROR: do not know the collision type = ",collision_type_common
       STOP
    ENDIF
    RETURN
  END FUNCTION Lgammagamma_UPC_fxn
  
  ! dL/dW at W=scale
  ! dL/dW=Lgammagamma*2W/s
  ! it is used in hep-ph/0112211
  FUNCTION dLgammagammadW_UPC(scale,icoll,iprofile)
    ! icoll: 1 - pp; 2 - pA; 3 - AB
    ! iprofile: 0; P_{NOHAD}=1; 1 - Woods-Saxon; 2 - hard-sphere
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::dLgammagammadW_UPC
    INTEGER,INTENT(IN)::icoll,iprofile
    REAL(KIND(1d0)),INTENT(IN)::scale ! scale=W
    REAL(KIND(1d0))::s
    IF(.NOT.print_banner)THEN
       WRITE(*,*)"==============================================================="
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|                       __    __  _______    ______           |"
       WRITE(*,*)"|                      |  \  |  \|       \  /      \          |"
       WRITE(*,*)"|   __      __         | $$  | $$| $$$$$$$\|  $$$$$$\         |"
       WRITE(*,*)"|  |  \    /  \ ______ | $$  | $$| $$__/ $$| $$   \$$         |"
       WRITE(*,*)"|   \$$ \/  $$ |      \| $$  | $$| $$    $$| $$               |"
       WRITE(*,*)"|     \$$  $$   \$$$$$$| $$  | $$| $$$$$$$ | $$   __          |"
       WRITE(*,*)"|      \$$$$           | $$__/ $$| $$      | $$__/  \         |"
       WRITE(*,*)"|      | $$             \$$    $$| $$       \$$    $$         |"
       WRITE(*,*)"|       \$$              \$$$$$$  \$$        \$$$$$$          |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    A library for exclusive photon-photon processes in       |"
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions            |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012                             |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    ebeam_PN(1)=ebeamMG5(1)/(nb_proton(1)+nb_neutron(1))
    ebeam_PN(2)=ebeamMG5(2)/(nb_proton(2)+nb_neutron(2))
    s=4d0*ebeam_PN(1)*ebeam_PN(2)
    dLgammagammadW_UPC=2d0*scale/s
    dLgammagammadW_UPC=dLgammagammadW_UPC*&
         Lgammagamma_UPC(scale,icoll,iprofile)
    RETURN
  END FUNCTION dLgammagammadW_UPC

  subroutine progress(j,nmax)
    implicit none
    integer,intent(in)::j,nmax
    integer::k
    character(:), allocatable :: bar, bar0
    character(5)::nmax_str
    !character(len=)::bar="???% |                                     |"
    integer::init=0
    save init,bar,bar0,nmax_str
    IF(init.EQ.0)THEN
       allocate(character(nmax+7) :: bar)
       allocate(character(nmax+7) :: bar0)
       bar(1:6)="???% |"
       do k=1,nmax
          bar(6+k:6+k)=" "
       enddo
       bar(nmax+7:nmax+7)="|"
       bar0=bar
       !bar="???% |"//repeat(' ',nmax)//"|"
       write(unit=nmax_str,fmt="(i5)") nmax+7
       nmax_str=adjustl(nmax_str)
       init=1
    ENDIF
    bar=bar0
    write(unit=bar(1:3),fmt="(i3)") INT(100*DBLE(j)/DBLE(nmax))
    do k=1, j
       bar(6+k:6+k)="*"
    enddo
    ! print the progress bar.
    write(unit=6,fmt="(a1,a"//trim(nmax_str)//")",advance="no") char(13), bar
    if (j.NE.nmax) then
       flush(unit=6)
    else
       write(unit=6,fmt=*)
    endif
    return
  end subroutine progress

END MODULE ElasticPhotonPhotonFlux
