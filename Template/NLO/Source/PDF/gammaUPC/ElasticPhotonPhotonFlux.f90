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
  PUBLIC::PhotonFlux_proton_EDFF_nob,PhotonFlux_proton_ChFF_nob
  PUBLIC::PhotonPhotonFlux_pA_hardsphere,PhotonPhotonFlux_pA_hardsphere_eval
  PUBLIC::PhotonPhotonFlux_pA_WoodsSaxon,PhotonPhotonFlux_pA_WoodsSaxon_eval
  PUBLIC::DeltaB_pp_at_x1x2,PhotonPhotonDeltaB_pp,PhotonPhotonDeltaB_pp_eval
  PUBLIC::DeltaB_pA_at_x1x2,PhotonPhotonDeltaB_pA_WoodsSaxon,PhotonPhotonDeltaB_pA_WoodsSaxon_eval
  PUBLIC::PhotonFlux_nucleus_EDFF_nob,PhotonFlux_nucleus_ChFF_nob
  PUBLIC::PhotonPhotonFlux_AB_hardsphere,PhotonPhotonFlux_AB_hardsphere_eval
  PUBLIC::PhotonPhotonFlux_AB_WoodsSaxon,PhotonPhotonFlux_AB_WoodsSaxon_eval
  PUBLIC::DeltaB_AB_at_x1x2,PhotonPhotonDeltaB_AB_WoodsSaxon,PhotonPhotonDeltaB_AB_WoodsSaxon_eval
  PUBLIC::Lgammagamma_UPC,dLgammagammadW_UPC,DeltaB_UPC_at_W
  PUBLIC::PhotonNumberDensity ! EDFF
  ! ChFF
  PUBLIC::PhotonNumberDensity_ChargeFormFactor_WS,PhotonNumberDensity_ChargeFormFactor_proton
  ! Forward neutron tagging
  ! tagging neutrons for two beams
  ! -2 (0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)
  ! For the proton beam, it will be ignored
  INTEGER,DIMENSION(2),PUBLIC::neutron_tagging=(/-2,-2/)
  ! If it is .TRUE. in A+B collisions and neutron_tagging(1)=!=neutron_tagging(2),
  ! it includes (neutron_tagging(1),neutron_tagging(2))+(neutron_tagging(2),neutron_tagging(1))
  LOGICAL,PUBLIC::neutron_conj_sum=.FALSE.
  PUBLIC::Pbreak_AB_WoodsSaxon,Pbreak_pA_WoodsSaxon
  PUBLIC::PbreakXn_LO_AB_WS,PbreakXn_LO_pA
  PUBLIC::Pbreak1n_LO_AB_WS,Pbreak1n_LO_pA
  PUBLIC::Pbreak2n_LO_AB_WS,Pbreak2n_LO_pA
  PUBLIC::Pbreak3n_LO_AB_WS,Pbreak3n_LO_pA
  PUBLIC::Pbreak4n_LO_AB_WS,Pbreak4n_LO_pA
  ! For generating virtualities of initial photons
  PUBLIC::generate_Q2_epa_proton_iWW,generate_Q2_epa_ion_ChFF
  PUBLIC::InitialMomentumReshuffle_PhotonPhoton
  PUBLIC::BOOSTTOECM,BOOSTFROMECM
  PUBLIC::GetASymbol,GetNuclearInfo
  REAL(KIND(1d0)),PARAMETER,PRIVATE::LOWER_BFactor_Limit=1D-1
  REAL(KIND(1d0)),PARAMETER,PRIVATE::GeVm12fm=0.1973d0 ! GeV-1 to fm
  INTEGER,PARAMETER,PRIVATE::SUB_FACTOR=2
  LOGICAL,PRIVATE,SAVE::print_banner=.FALSE.
  LOGICAL,PUBLIC::use_MC_Glauber=.FALSE.
  LOGICAL,PUBLIC::proton_FF_correction=.FALSE. ! when USE_CHARGEFORMFACTOR4PHOTON=.TRUE., 
                                               ! it will include the correction beyond proton dipole form factor
                                               ! when proton_FF_correction=.TRUE.
                                               ! it will impact both pp and pA UPCs
  INTEGER,PUBLIC::IEPSILON_EDFF=0  ! 0: bmin = RA; 1: bmin = RA+a
  LOGICAL,PRIVATE::CALC_DELTABQ=.FALSE. ! whether we want to calcuate DeltaB (.TRUE.) or not (.FALSE.)
  LOGICAL,PUBLIC::GENERATE_DELTAB_GRID=.FALSE. ! if true, it will always first generate the grid
  LOGICAL,PUBLIC::GENERATE_PhotonPhotonFlux_GRID=.TRUE. ! if true, it will alwyas first generate the grid for PhotonPhotonFlux
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

  FUNCTION PNOHAD_AB_WoodsSaxon(bx,by,RR,w,aa,A,Z,sigmaNN)
    ! the probability of no hardonic interaction at impact b=(bx,by)
    ! for AB collisions
    ! bx, by, RR should be in unit of fm
    ! sigmaNN should be in unit of fm^2
    IMPLICIT NONE
    REAL(KIND(1d0))::PNOHAD_AB_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A,Z
    REAL(KIND(1d0)),INTENT(IN)::sigmaNN ! inelastic NN xs (in unit of fm^2)
    REAL(KIND(1d0))::TAB,ABAB
    ABAB=A(1)*A(2)
    IF(.NOT.use_MC_Glauber)THEN
       TAB=TABhat_WoodsSaxon_grid(bx,by,RR,w,aa,A)*ABAB
    ELSE
       TAB=TAB_MC_Glauber(bx,by,A,Z)
    ENDIF
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
    IF(.NOT.use_MC_Glauber)THEN
       TA=TAhat_WoodsSaxon(bx,by,RR,w,aa,A,1,.FALSE.)*A
    ELSE
       TA=TA_MC_Glauber(bx,by,A)
    ENDIF
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

  FUNCTION PhotonNumberDensity_ChargeFormFactor_proton(b,Ega,gamma,CORRECTIONQ,MINUSDIPOLESQUAREQ)
    ! It gives us the photon number density with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonNumberDensity_ChargeFormFactor_proton
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma
    LOGICAL,INTENT(IN),OPTIONAL::CORRECTIONQ ! whether we need to include the correction beyond the dipole
                                             ! the default is no (.FALSE.)
    LOGICAL,INTENT(IN),OPTIONAL::MINUSDIPOLESQUAREQ ! whether we want to minus the dipole square
                                                    ! the default is no (.FALSE.)
    REAL(KIND(1d0)),PARAMETER::aa=1.1867816581938533d0 ! in unit of GeV-1 = 1/DSQRT(0.71 GeV^2)
    REAL(KIND(1d0))::btilde,atilde
    REAL(KIND(1d0))::Egaoga
    REAL(KIND(1d0)),EXTERNAL::BESSK1,BESSK0
    REAL(KIND(1d0))::integral,dipole,corr,sqrtterm
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
    dipole=0d0
    IF(PRESENT(MINUSDIPOLESQUAREQ))THEN
       IF(MINUSDIPOLESQUAREQ)THEN
          dipole=integral
       ENDIF
    ENDIF
    corr=0d0
    IF(PRESENT(CORRECTIONQ))THEN
       IF(CORRECTIONQ)THEN
          ! include the correction beyond the dipole
          corr=PhotonNumberDensity_ChargeFormFactor_proton_correction(b,Ega,gamma,1)
          integral=integral+corr
       ENDIF
    ENDIF
    IF(PRESENT(MINUSDIPOLESQUAREQ))THEN
       IF(MINUSDIPOLESQUAREQ)THEN
          integral=2d0*dipole*corr+corr**2
       ELSE
          integral=integral**2
       ENDIF
    ELSE
       integral=integral**2
    ENDIF
    PhotonNumberDensity_ChargeFormFactor_proton=one/pi2*integral
    RETURN
  END FUNCTION PhotonNumberDensity_ChargeFormFactor_proton

  FUNCTION PhotonNumberDensity_ChargeFormFactor_proton_correction(b,Ega,gamma,integ_method)
    ! It gives us the photon number density with Z=1 and alpha=1 (only the correction part beyond the dipole FF)
    ! b should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonNumberDensity_ChargeFormFactor_proton_correction
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
                                              ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::Egaoga,integral,xga,xga2,Egaoga2,Ega2
    REAL(KIND(1d0))::Egaoga_common,b_common
    COMMON/PND_CFF_proton_corr/Egaoga_common,b_common
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0))::btilde,atilde,db
    INTEGER::init=0
    INTEGER::NXA,NYA,i,j,n,k,l
    SAVE init,NXA,NYA
    INTEGER::NYA_save=0
    SAVE NYA_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA,YA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZA,ZA_save
    SAVE XA,YA,ZA,ZA_save
    REAL(KIND(1d0)),PARAMETER::bmaxoR=10d0
    INTEGER,PARAMETER::NBMAX=2
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=12
    ! NXSEG for x_gamma from 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::NXSEG=8
    INTEGER::log10xmin,log10xmin_before,ilog10x
    REAL(KIND(1d0))::XMIN=1D-8
    SAVE XMIN
    REAL(KIND(1d0))::log10x1,kTmax
    REAL(KIND(1d0))::rescaling_bmax_save
    SAVE rescaling_bmax_save
    INTEGER,PARAMETER::n_interp=6
    REAL(KIND(1d0)),DIMENSION(n_interp)::XA2,YA2
    REAL(KIND(1d0)),PARAMETER::PIPI=3.14159265358979323846264338328d0
    INTEGER::iter,npoints
    REAL(KIND(1d0))::integ1,integ2,integ3,integ4
    INTEGER,PARAMETER::itermax=8
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(nxseg+1)::YD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp,nxseg+1)::ZD2
    Egaoga=Ega/gamma ! gamma=Ebeam/mproton, xga=Ega/Ebeam, Egaoga=xga*mproton
    xga=Egaoga/mproton
    IF(xga.LE.1d-99)THEN
       PhotonNumberDensity_ChargeFormFactor_proton_correction=0d0
       RETURN
    ENDIF
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF
    IF(init.EQ.0.OR.xga.LT.xmin.OR.(integ_method6.LT.0))THEN
       ! first do a rescaling
       Egaoga_common=Egaoga
       b_common=b
       IF(integ_method6.LT.0)THEN
          IF(b.EQ.0d0)THEN
             PhotonNumberDensity_ChargeFormFactor_proton_correction=0d0
             RETURN
          ELSE
             npoints=50000
             CALL trapezoid_integration(npoints,PND_ChargeFormFactor_proton_corr_fxn,&
                  one,integral)
             PhotonNumberDensity_ChargeFormFactor_proton_correction=integral
             RETURN
          ENDIF
       ENDIF
       rescaling_bmax_save=MAX(b,bmaxoR*Rproton/GeVm12fm)
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF
       log10xmin_before=INT(DLOG10(xmin))
       IF(xga.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(xga))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(xga.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
          IF(ALLOCATED(YA))THEN
             WRITE(*,*)" "
          ENDIF
          WRITE(*,*)"INFO: in ChFF photon number density of proton, the xmin of the grid has been updated to ",xmin
          IF(ALLOCATED(YA))THEN
             DEALLOCATE(YA)
          ENDIF
          IF(ALLOCATED(ZA))THEN
             IF(init.EQ.1)THEN
                IF(ALLOCATED(ZA_save))THEN
                   DEALLOCATE(ZA_save)
                ENDIF
                ALLOCATE(ZA_save(NXA,NYA))
                ! save the values that have been calculated before
                NYA_save=NYA
                DO I=1,NXA
                   DO J=1,NYA
                      ZA_save(I,J)=ZA(I,J)
                   ENDDO
                ENDDO
             ENDIF
             ! then deallocate it
             DEALLOCATE(ZA)
          ENDIF
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       NYA=NXSEG*(-log10xmin)+1
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NYA))
       ENDIF
       IF(.NOT.ALLOCATED(ZA))THEN
          ALLOCATE(ZA(NXA,NYA))
       ENDIF
       db=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of photon number density correction from charge form factor"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1 (not multiplied Egaoga yet !)
             XA(NXA-k+1)=(10d0**(-n-1))*(1d0+DBLE(NBSEG+1-i)*db)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in PhotonNumberDensity_ChargeFormFactor_proton_correction"
          STOP
       ENDIF
       XA(1)=0d0
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nxseg
             log10x1=-1d0/DBLE(nxseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             YA(K)=log10x1
          ENDDO
       ENDDO
       IF(K.NE.NYA-1)THEN
          WRITE(*,*)"ERROR: K != NYA-1"
          STOP
       ENDIF
       kTmax=DSQRT(3d0) ! match with Q2max in PND_ChargeFormFactor_proton_corr_fxn
       YA(NYA)=DBLE(log10xmin)
       DO I=1,NXA
          IF(NYA_save.EQ.0)THEN
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             ZA(I,J)=0d0
             CYCLE
          ENDIF
          DO J=1,NYA
             ! use the ones that have been calculated before
             IF(NYA_save.GT.0.AND.J.LE.NYA_save)THEN
                ZA(I,J)=ZA_save(I,J)
                CYCLE
             ENDIF
             xga2=10d0**(YA(J)) ! x_gamma
             Egaoga2=xga2*mproton
             Ega2=Egaoga2*gamma
             b_common=XA(I) ! = b
             Egaoga_common=Egaoga2
             npoints=10000
             CALL trapezoid_integration(npoints,PND_ChargeFormFactor_proton_corr_fxn,&
                  kTmax,integral)
             ! we try to do some numerical improvement
             integ4=0d0
             iter=1
             DO WHILE((iter.EQ.1.OR.&
                  DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)&
                  .AND.iter.LE.itermax)
                integ4=integral
                ! increase the points by a factor of 2
                npoints=npoints*2
                CALL trapezoid_integration(npoints,PND_ChargeFormFactor_proton_corr_fxn,&
                     kTmax,integral)
                iter=iter+1
             END DO
             IF(DABS(integ4/integral).GT.1.5d0.OR.DABS(integ4/integral).LT.0.67d0)THEN
                WRITE(*,*)"WARNING: the integral is not stable (b,Ega,gamma,npoints)=",&
                     b_common,Ega2,gamma,npoints
                WRITE(*,*)"WARNING: integral in two iterations #1:",integ4,integral
             ENDIF
             IF(ISNAN(integral))THEN
                WRITE(*,*)"ERROR: the integral is not stable (b,Ega,gamma,RA,wA,aA)=",&
                     b_common,Ega2,gamma
                STOP
             ENDIF
             ZA(I,J)=integral
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(b.GT.rescaling_bmax_save.OR.b.LE.0d0)THEN
       PhotonNumberDensity_ChargeFormFactor_proton_correction=0d0
    ELSE
       XI(1)=b
       YI(1)=DLOG10(xga)
       
       db=MIN(b/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(db))-1 ! b is from 10**(-n-1)*bmax to 10**(-n)*bmax
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.b)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.b)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.b)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF
       IF(YI(1).GE.0d0)THEN
          ilog10x=-1
       ELSE
          ilog10x=FLOOR(YI(1))
       ENDIF
       L=NXSEG*(-ilog10x-1)

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
       ENDDO
       DO I=1,NXSEG+1
          YD2_1D(I)=YA(L+I)
       ENDDO
       DO I=1,n_interp
          DO J=1,NXSEG+1
             ZD2(I,J)=ZA(K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(n_interp-1,NXSEG,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
       integral=ZI(1)
       PhotonNumberDensity_ChargeFormFactor_proton_correction=integral
    ENDIF
    RETURN
  END FUNCTION PhotonNumberDensity_ChargeFormFactor_proton_correction

  FUNCTION PND_ChargeFormFactor_proton_corr_fxn(kT)
    ! x = ArcTan[kT*gamma/Ega]*2/Pi
    IMPLICIT NONE
    REAL(KIND(1d0))::PND_ChargeFormFactor_proton_corr_fxn
    REAL(KIND(1d0)),INTENT(IN)::kT ! in unit of GeV
    REAL(KIND(1d0))::Q,CFF,pref,bkT,dipole,ratio
    REAL(KIND(1d0)),PARAMETER::PIo2=1.57079632679489661923132169164d0
    REAL(KIND(1d0)),EXTERNAL::BESSJ1
    REAL(KIND(1d0))::Egaoga_common,b_common
    COMMON/PND_CFF_proton_corr/Egaoga_common,b_common
    REAL(KIND(1d0)),PARAMETER::Q2max=3d0 ! in GeV**2 (negative means no cuts)
    REAL(KIND(1d0)),PARAMETER::Q2lim=1.12d0 ! in GeV**2 limit between the two fit
    REAL(KIND(1d0))::Q2 ! ,expo
    REAL(KIND(1d0)),PARAMETER::one=1d0
    ! the parameter in the dipole form factor
    REAL(KIND(1d0)),PARAMETER::aa=1.1867816581938533d0 ! in unit of GeV-1 = 1/DSQRT(0.71 GeV^2)
    ! fitted parameters
    !REAL(KIND(1d0)),PARAMETER::a0=0.999781d0
     REAL(KIND(1d0)),PARAMETER::a0=0.999871d0
    !REAL(KIND(1d0)),PARAMETER::a1=-0.347088d0
    REAL(KIND(1d0)),PARAMETER::a1=-0.215829d0
    !REAL(KIND(1d0)),PARAMETER::a2=1.96245d0
    REAL(KIND(1d0)),PARAMETER::a2=0.509109d0
    !REAL(KIND(1d0)),PARAMETER::a3=-5.3141d0
    REAL(KIND(1d0)),PARAMETER::a3=-0.621597d0
    !REAL(KIND(1d0)),PARAMETER::a4=4.91061d0
    REAL(KIND(1d0)),PARAMETER::a4=0.246705d0
    !REAL(KIND(1d0)),PARAMETER::b0=1.07751d0
    REAL(KIND(1d0)),PARAMETER::b0=1.01809d0
    !REAL(KIND(1d0)),PARAMETER::b1=-2.04371d0
    REAL(KIND(1d0)),PARAMETER::b1=-0.778974d-1
    !REAL(KIND(1d0)),PARAMETER::b2=5.35245d0
    REAL(KIND(1d0)),PARAMETER::b2=-0.184511d-1
    REAL(KIND(1d0)),PARAMETER::b3=0.289159d-2
    REAL(KIND(1d0)),PARAMETER::b4=-0.121585d-3
    !REAL(KIND(1d0)),PARAMETER::c1=-0.071461765d0
    !REAL(KIND(1d0)),PARAMETER::c2=2.881405896d0
    !IF(x.GE.1d0.OR.x.LE.0d0)THEN
    IF(kT.LE.0d0)THEN
       PND_ChargeFormFactor_proton_corr_fxn=0d0
       RETURN
    ENDIF
    !pref=DTAN(PIo2*x)
    !kT=pref*Egaoga_common
    bkT=kT*b_common
    pref=kT**2/(kT**2+Egaoga_common**2)
    !pref=pref**2*PIo2*Egaoga_common
    Q2=kT**2+Egaoga_common**2
    IF(Q2max.GT.0d0.AND.Q2.GT.Q2max)THEN
       PND_ChargeFormFactor_proton_corr_fxn=0d0
       RETURN
    ENDIF
    dipole=one/(one+Q2*aa**2)**2
    Q=DSQRT(Q2)
    !IF(Q2.LT.0.5d0.AND.Q2.GT.0.06d0)THEN
    IF(Q2.LT.Q2lim)THEN
       ! use the polynomial fit
       ratio=a0+a1*Q2+a2*Q2**2+a3*Q2**3+a4*Q2**4
       CFF=dipole*(ratio-one)
    !ELSEIF(Q2.LT.0.06d0)THEN
    ELSEIF(Q2.LT.Q2max.AND.Q2.GT.Q2lim)THEN
       ! rational fit
       !CFF=(one+c1*Q2)/(one+c2*Q2)-dipole
       ! use the polynomial fit
       ratio=b0+b1*Q2+b2*Q2**2+b3*Q2**3+b4*Q2**4
       CFF=dipole*(ratio-one)
    ENDIF
    !ELSE
    !   ! use the Gaussian fit
    !   expo=(Q2-b1)**2/b2**2*0.5d0
    !   ratio=b0*DEXP(-expo)
    !   CFF=dipole*(ratio-one)
    !ENDIF
    IF(ISNAN(CFF))THEN
       PRINT *, "ChargeFormFactor is NaN with ",Q
       STOP
       CFF=0d0
    ENDIF
    PND_ChargeFormFactor_proton_corr_fxn=pref*BESSJ1(bkT)*CFF
    RETURN
  END FUNCTION PND_ChargeFormFactor_proton_corr_fxn

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
    INTEGER,DIMENSION(2)::NYA_save=(/0,0/)
    SAVE NYA_save
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA,YA,ZA1_save,ZA2_save
    REAL(KIND(1d0)),DIMENSION(:,:,:),ALLOCATABLE::ZA
    SAVE XA,YA,ZA,ZA1_save,ZA2_save
    REAL(KIND(1d0)),PARAMETER::bmaxoR=10d0
    INTEGER,PARAMETER::NBMAX=2
    ! From Nicolas Crepet:
    ! For x>0.1, it turns out that we only use NBSEG=28 and NXSEG=16 to be
    ! smooth enough for the curve
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=12
    ! NXSEG for x_gamma from 10**(-n-1) to 10**(-n) 
    INTEGER,PARAMETER::NXSEG=8
    INTEGER::log10xmin,log10xmin_before,ilog10x
    REAL(KIND(1d0))::XMIN=1D-8
    SAVE XMIN
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
    IF(xga.LE.1d-99)THEN
       PhotonNumberDensity_ChargeFormFactor_WS=0d0
       RETURN
    ENDIF
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF
    IF(init(ibeam).EQ.0.OR.xga.LT.xmin.OR.(integ_method6.LT.0))THEN
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
       log10xmin_before=INT(DLOG10(xmin))
       IF(xga.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(xga))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(xga.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
          IF(ALLOCATED(YA))THEN
             WRITE(*,*)" "
          ENDIF
          WRITE(*,*)"INFO: in ChFF photon number density of nucleus, the xmin of the grid has been updated to ",xmin
          IF(ALLOCATED(YA))THEN
             DEALLOCATE(YA)
          ENDIF
          IF(ALLOCATED(ZA))THEN
             ! save the values that have been calculated before
             IF(init(1).EQ.1)THEN
                NYA_save(1)=NYA
                IF(ALLOCATED(ZA1_save))THEN
                   DEALLOCATE(ZA1_save)
                ENDIF
                ALLOCATE(ZA1_save(NXA,NYA))
                DO I=1,NXA
                   DO J=1,NYA
                      ZA1_save(I,J)=ZA(1,I,J)
                   ENDDO
                ENDDO
             ENDIF
             IF(init(2).EQ.1)THEN
                NYA_save(2)=NYA
                IF(ALLOCATED(ZA2_save))THEN
                   DEALLOCATE(ZA2_save)
                ENDIF
                ALLOCATE(ZA2_save(NXA,NYA))
                DO I=1,NXA
                   DO J=1,NYA
                      ZA2_save(I,J)=ZA(2,I,J)
                   ENDDO
                ENDDO
             ENDIF
             ! then deallocate it
             DEALLOCATE(ZA)
          ENDIF
          init(1:2)=0
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
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam,I).EQ.0d0)THEN
                ZA(ibeam,I,J)=0d0
                CYCLE
             ENDIF
             DO J=1,NYA
                ! use the ones that have been calculated before
                IF(NYA_save(ibeam).GT.0.AND.J.LE.NYA_save(ibeam))THEN
                   IF(ibeam.EQ.1)THEN
                      ZA(ibeam,I,J)=ZA1_save(I,J)
                   ELSEIF(ibeam.EQ.2)THEN
                      ZA(ibeam,I,J)=ZA2_save(I,J)
                   ELSE
                      WRITE(*,*)"ERROR: do not know ibeam=",ibeam
                   ENDIF
                   CYCLE
                ENDIF
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
    REAL(KIND(1d0))::Q,CFF,pref,bkT,kT
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

  ! alpha=1
  ! see my notes OpticalGlauber.pdf
  FUNCTION NGAMMA_proton_dipole(aptilde)
    IMPLICIT NONE
    REAL(KIND(1d0))::NGAMMA_proton_dipole
    REAL(KIND(1d0)),INTENT(IN)::aptilde ! aptilde=ap*x*mproton, ap=1/sqrt(0.71) GeV-1
    REAL(KIND(1d0)),PARAMETER::PIPI=3.14159265358979323846264338328d0
    NGAMMA_proton_dipole=1d0/PIPI*((1d0+4d0*aptilde**2)*DLOG(1d0+aptilde**(-2))&
         -(24d0*aptilde**4+42d0*aptilde**2+17d0)/(6d0*(1d0+aptilde**2)**2))
    RETURN
  END FUNCTION NGAMMA_PROTON_DIPOLE

  FUNCTION PhotonFlux_proton_EDFF_nob(x,gamma)
    ! set PNOHARD=1
    ! for proton with EDFF
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_EDFF_nob
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
    PhotonFlux_proton_EDFF_nob=alpha/x*Z**2*NGAMMA(xi,gamma)
    RETURN
  END FUNCTION PhotonFlux_proton_EDFF_nob

  FUNCTION PhotonFlux_proton_dipole_nob(x)
    ! set PNOHARD=1 but with dipole ChFF
    ! for proton
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_dipole_nob
    REAL(KIND(1d0)),INTENT(IN)::x
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::aa=1.1867816581938533d0 ! in unit of GeV-1 = 1/DSQRT(0.71 GeV^2)
    REAL(KIND(1d0))::alpha
    SAVE alpha
    REAL(KIND(1d0))::aptilde
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
       ! for proton
       init=1
    ENDIF
    aptilde=mproton*x*aa
    PhotonFlux_proton_dipole_nob=alpha/x*NGAMMA_proton_dipole(aptilde)
    RETURN
  END FUNCTION PhotonFlux_proton_dipole_nob

  FUNCTION PhotonFlux_proton_ChFF_nob(x,gamma,CORRECTIONQ)
    ! set PNOHARD=1
    ! for proton with ChFF
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_ChFF_nob
    REAL(KIND(1d0)),INTENT(IN)::x,gamma
    LOGICAL,INTENT(IN),OPTIONAL::CORRECTIONQ ! whether we need to include the correction beyond the dipole
                                             ! the default is no (.FALSE.)
    LOGICAL::corrQ
    INTEGER::init=0
    SAVE init
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::alpha
    SAVE alpha
    REAL(KIND(1d0))::aptilde,Egaoga
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
    IF(PRESENT(CORRECTIONQ))THEN
       corrQ=CORRECTIONQ
    ELSE
       corrQ=.FALSE.
    ENDIF
    IF(corrQ)THEN
       ! include the correction beyond the dipole
       PhotonFlux_proton_ChFF_nob=PhotonFlux_proton_ChFF_corr_nob_eval(x,gamma)
       PhotonFlux_proton_ChFF_nob=alpha/x*PhotonFlux_proton_ChFF_nob
       PhotonFlux_proton_ChFF_nob=PhotonFlux_proton_ChFF_nob+PhotonFlux_proton_dipole_nob(x)
    ELSE
       PhotonFlux_proton_ChFF_nob=PhotonFlux_proton_dipole_nob(x)
    ENDIF
    RETURN
  END FUNCTION PhotonFlux_proton_ChFF_nob

  FUNCTION PhotonFlux_proton_ChFF_corr_nob_eval(x,gamma)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_ChFF_corr_nob_eval
    REAL(KIND(1d0)),INTENT(IN)::x,gamma
    REAL(KIND(1d0))::Ega
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::npoints
    REAL(KIND(1d0))::bupper,Egaoga,xx1
    REAL(KIND(1d0))::Ega_common,gamma_common
    COMMON /proton_ChFF_corr_nob_fxn/ Ega_common,gamma_common
    INTEGER::init=0
    SAVE init
    INTEGER::MX_save=0
    SAVE MX_save
    REAL(KIND(1d0))::xmin=1D-8
    SAVE xmin
    INTEGER::log10xmin,log10xmin_before
    REAL(KIND(1d0))::log10x
    INTEGER::ilog10x
    ! nseg for 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::nseg=10
    INTEGER::MX,I,J,K,L
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,XD_1D,ZD,ZD_save
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1)::ZD2
    REAL(KIND(1d0)),DIMENSION(1)::XI,ZI
    REAL(KIND(1d0)),PARAMETER::bmaxoR=10d0
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    IF(x.LE.1d-99.OR.gamma.LE.0d0)THEN
       PhotonFlux_proton_ChFF_corr_nob_eval=0d0
       RETURN
    ENDIF
    IF(init.EQ.0.OR.x.LT.xmin)THEN
       log10xmin_before=INT(DLOG10(xmin))
       IF(x.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a grid first
       MX=nseg*(-log10xmin)
       ! try to deallocate first
       IF(ALLOCATED(XD_1D))THEN
          DEALLOCATE(XD_1D)
       ENDIF
       ALLOCATE(XD_1D(MX+1))
       IF(ALLOCATED(ZD))THEN
          DEALLOCATE(ZD)
       ENDIF
       ALLOCATE(ZD(MX+1))
       K=0
       DO I=0,log10xmin+1,-1
          DO J=1,nseg
             log10x=-1d0/DBLE(nseg)*DBLE(J-1)+DBLE(I)
             K=K+1
             XD_1D(K)=log10x
          ENDDO
       ENDDO
       IF(K.NE.MX)THEN
          WRITE(*,*)"ERROR: K != MX"
          STOP
       ENDIF
       XD_1D(MX+1)=DBLE(log10xmin)
       bupper=bmaxoR*Rproton/GeVm12fm ! the hard cut in the grid of PhotonNumberDensity_ChargeFormFactor_proton_correction 
       DO I=1,MX+1
          IF(MX_save.GT.0.AND.I.LE.MX_save+1)THEN
             ZD(I)=ZD_save(I)
             CYCLE
          ENDIF
          xx1=10d0**(XD_1D(I))
          Ega=xx1*gamma*mproton
          npoints=5000
          Ega_common=Ega
          gamma_common=gamma
          !IF(Ega/gamma.LT.1d-6.AND.MOD(I,3).NE.0)THEN
          !IF(Ega/gamma.LT.1d-8)THEN
          !   ZD(I)=ZD(I-1)
          !ELSE
          CALL trapezoid_integration(npoints,PhotonFlux_proton_ChFF_corr_nob_fxn,&
               bupper,ZD(I))
          IF(MX_save.EQ.0)THEN
             IF(I.EQ.1)THEN
                WRITE(*,*)"INFO: generate grid of proton photon flux beyond dipole correction (will take a few seconds)"
                CALL progress(INT(I*50d0/(MX+1)),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/(MX+1)),50)
             ENDIF
          ENDIF
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1))
       ! save the values that have been calculated before
       MX_save=MX
       DO I=1,MX+1
          ZD_save(I)=ZD(I)
       ENDDO
       init=1
    ENDIF
    XI(1)=DLOG10(x)
    IF(XI(1).GE.0d0)THEN
       ilog10x=-1
    ELSE
       ilog10x=FLOOR(XI(1))
    ENDIF
    K=nseg*(-ilog10x-1)
    ! it turns out to be important with spline
    ! to reverse the order for a fast dropping curve
    ! x is now always from smaller to larger values
    DO I=1,nseg+1
       XD2_1D(nseg+2-I)=XD_1D(K+I)
       ZD2(nseg+2-I)=ZD(K+I)
    ENDDO
    CALL SPLINE_INTERPOLATE(XD2_1D,ZD2,nseg+1,XI(1),PhotonFlux_proton_ChFF_corr_nob_eval)
    RETURN
  END FUNCTION PhotonFlux_proton_ChFF_corr_nob_eval

  FUNCTION PhotonFlux_proton_ChFF_corr_nob_fxn(b)
    ! b is in unit of GeV-1
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_proton_ChFF_corr_nob_fxn
    REAL(KIND(1d0)),INTENT(IN)::b
    REAL(KIND(1d0))::Ega, gamma
    COMMON /proton_ChFF_corr_nob_fxn/ Ega, gamma
    REAL(KIND(1d0)),PARAMETER::twopi=6.2831853071795864769252867d0
    REAL(KIND(1d0))::Egaoga
    IF(b.LE.0d0)THEN
       PhotonFlux_proton_ChFF_corr_nob_fxn=0d0
       RETURN
    ENDIF
    PhotonFlux_proton_ChFF_corr_nob_fxn=PhotonNumberDensity_ChargeFormFactor_proton(b,Ega,gamma,.TRUE.,.TRUE.)
    PhotonFlux_proton_ChFF_corr_nob_fxn=PhotonFlux_proton_ChFF_corr_nob_fxn*twopi*b
    RETURN
  END FUNCTION PhotonFlux_proton_ChFF_corr_nob_fxn

  FUNCTION PhotonFlux_nucleus_EDFF_nob(x,gamma,Z,RA)
    ! set PNOHARD=1
    ! for nucleus with EDFF
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_nucleus_EDFF_nob
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
    PhotonFlux_nucleus_EDFF_nob=alpha/x*Z**2*NGAMMA(xi,gamma)
    RETURN
  END FUNCTION PhotonFlux_nucleus_EDFF_nob

  FUNCTION PhotonFlux_nucleus_ChFF_nob(ibeam,x,gamma,Z,RA,wA,aA)
    ! set PNOHARD=1
    ! for nucleus with ChFF
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_nucleus_ChFF_nob
    INTEGER,INTENT(IN)::ibeam
    REAL(KIND(1d0)),INTENT(IN)::x,gamma,Z
    REAL(KIND(1d0)),INTENT(IN)::RA,wA,aA ! RA is the radius of nucleus (in unit of fm)
                                         ! aA is the diffusivity of nucleus (in unit of fm)
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA
    SAVE init,NXA
    INTEGER,DIMENSION(2)::NXA_save=(/0,0/)
    SAVE NXA_save
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    ! NXSEG for x_gamma from 10**(-n-1) to 10**(-n)
    INTEGER,PARAMETER::NXSEG=20
    INTEGER::log10xmin,log10xmin_before,ilog10x,i,j,k,l,n
    REAL(KIND(1d0))::XMIN=1D-8
    SAVE XMIN
    REAL(KIND(1d0))::log10x1
    REAL(KIND(1d0)),DIMENSION(2)::R_save,w_save,aa_save,gamma_save
    SAVE R_save,w_save,aa_save,gamma_save
    REAL(KIND(1d0)),DIMENSION(1)::XI,ZI
    INTEGER,PARAMETER::n_interp=10
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::ZD2
    REAL(KIND(1d0))::alpha
    SAVE alpha
    REAL(KIND(1d0))::Ega,xga2,Ega2,Egaoga2
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::ZA1_save,ZA2_save
    SAVE XA,ZA,ZA1_save,ZA2_save
    IF(ibeam.GT.2.OR.ibeam.LT.1)THEN
       WRITE(*,*)"Error: ibeam=/=1,2 in PhotonFlux_nucleus_ChFF_nob"
       STOP
    ENDIF
    IF(x.LT.1d-99)THEN
       PhotonFlux_nucleus_ChFF_nob=0d0
       RETURN
    ENDIF
    IF(init(1).EQ.0.AND.init(2).EQ.0)THEN
       IF(alphaem_elasticphoton.LT.0d0)THEN
          IF(aqedup.GT.0d0)THEN
             alpha=aqedup
          ELSE
             alpha = 0.0072992701d0
          ENDIF
       ELSE
          alpha=alphaem_elasticphoton
       ENDIF
    ENDIF
    IF(init(ibeam).EQ.0.OR.x.LT.xmin)THEN
       R_save(ibeam)=RA
       w_save(ibeam)=wA
       aa_save(ibeam)=aA
       gamma_save(ibeam)=gamma
       log10xmin_before=INT(DLOG10(xmin))
       IF(x.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
          !IF(ALLOCATED(XA))THEN
          !   WRITE(*,*)" "
          !ENDIF
          WRITE(*,*)"INFO: in PhotonFlux_nucleus_ChFF_nob, the xmin of the grid has been updated to ",xmin
          IF(ALLOCATED(XA))THEN
             DEALLOCATE(XA)
          ENDIF
          IF(ALLOCATED(ZA))THEN
             ! save the values that have been calculated before
             IF(init(1).EQ.1)THEN
                NXA_save(1)=NXA
                IF(ALLOCATED(ZA1_save))THEN
                   DEALLOCATE(ZA1_save)
                ENDIF
                ALLOCATE(ZA1_save(NXA))
                DO I=1,NXA
                   ZA1_save(I)=ZA(1,I)
                ENDDO
             ENDIF
             IF(init(2).EQ.1)THEN
                NXA_save(2)=NXA
                IF(ALLOCATED(ZA2_save))THEN
                   DEALLOCATE(ZA2_save)
                ENDIF
                ALLOCATE(ZA2_save(NXA))
                DO I=1,NXA
                   ZA2_save(I)=ZA(2,I)
                ENDDO
             ENDIF
             ! then deallocate it
             DEALLOCATE(ZA)
          ENDIF
          init(1:2)=0
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       NXA=NXSEG*(-log10xmin)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       IF(.NOT.ALLOCATED(ZA))THEN
          ALLOCATE(ZA(2,NXA))
       ENDIF
       IF(init(3-ibeam).EQ.1.AND.&
            R_save(ibeam).EQ.R_save(3-ibeam).AND.&
            w_save(ibeam).EQ.w_save(3-ibeam).AND.&
            aa_save(ibeam).EQ.aa_save(3-ibeam).AND.&
            gamma_save(ibeam).EQ.gamma_save(3-ibeam))THEN
          DO k=1,NXA
             XA(ibeam,k)=XA(3-ibeam,k)
             ZA(ibeam,k)=ZA(3-ibeam,k)
          ENDDO
       ELSE
          K=0
          DO I=0,log10xmin+1,-1
             DO J=1,nxseg
                log10x1=-1d0/DBLE(nxseg)*DBLE(J-1)+DBLE(I)
                K=K+1
                XA(ibeam,K)=log10x1
             ENDDO
          ENDDO
          IF(K.NE.NXA-1)THEN
             WRITE(*,*)"ERROR: K != NXA-1"
             STOP
          ENDIF
          XA(ibeam,NXA)=DBLE(log10xmin)
          DO I=1,NXA
             ! use the ones that have been calculated before
             IF(NXA_save(ibeam).GT.0.AND.I.LE.NXA_save(ibeam))THEN
                IF(ibeam.EQ.1)THEN
                   ZA(ibeam,I)=ZA1_save(I)
                ELSEIF(ibeam.EQ.2)THEN
                   ZA(ibeam,I)=ZA2_save(I)
                ELSE
                   WRITE(*,*)"ERROR: do not known ibeam=",ibeam
                ENDIF
                CYCLE
             ENDIF
             xga2=10d0**(XA(ibeam,I)) ! x_gamma
             Egaoga2=xga2*mN
             Ega2=Egaoga2*gamma
             ZA(ibeam,I)=alpha*Z**2*&
                  PhotonFlux_nucleus_ChFF_nob_eval(ibeam,Ega2,gamma,RA,wA,aA)
             IF(NXA_save(ibeam).EQ.0)THEN
                IF(I.EQ.1)THEN
                   WRITE(*,*)"INFO: generate grid of b-integrated photon flux from charge form factor of beam=",ibeam
                   WRITE(*,*)"INFO: it will take tens of seconds !"
                   CALL progress(INT(I*50d0/NXA),50,.TRUE.)
                ELSE
                   CALL progress(INT(I*50d0/NXA),50)
                ENDIF
             ENDIF
          ENDDO
       ENDIF
       init(ibeam)=1
    ENDIF
    IF(R_save(ibeam).NE.RA.OR.w_save(ibeam).NE.wA.OR.aa_save(ibeam).NE.aA.OR.gamma_save(ibeam).NE.gamma)THEN
       WRITE(*,*)"ERROR: gamma,RA,wA,aA are not consistent in PhotonFlux_nucleus_ChFF_nob"
       WRITE(*,*)"INFO: ibeam=",ibeam
       WRITE(*,*)"INFO: Saved ones (gamma,RA,wA,aA)=",gamma_save(ibeam),R_save(ibeam),w_save(ibeam),aa_save(ibeam)
       WRITE(*,*)"INFO: New ones (gamma,RA,wA,aA)=",gamma,RA,wA,aA
       STOP
    ENDIF
    XI(1)=DLOG10(x)
    N=-FLOOR(XI(1))-1 ! x is from 10**(-n-1) to 10**(-n)
    IF(N.LT.0)THEN
       K=0
    ELSE
       IF(N.GE.-INT(DLOG10(xmin)))THEN
          k=NXA-NXSEG
       ELSE
          k=NXSEG*n+1
       ENDIF
       IF(XA(ibeam,k+NXSEG).GT.XI(1))THEN
          WRITE(*,*)"Error: k is not proper #1"
          STOP
       ENDIF
       IF(XA(ibeam,k).LT.XI(1))THEN
          WRITE(*,*)"Error: k is not proper #2"
          STOP
       ENDIF
       DO i=k,k+NXSEG
          IF(XA(ibeam,i).LE.XI(1))EXIT
       ENDDO
       IF(i-n_interp/2.GE.1.AND.i-n_interp/2-1+n_interp.LE.NXA)THEN
          K=i-n_interp/2-1
       ELSEIF(i-n_interp/2.LT.1)THEN
          K=0
       ELSEIF(i-n_interp/2-1+n_interp.GT.NXA)THEN
          K=NXA-n_interp
       ELSE
          WRITE(*,*)"Error: you cannot reach here !"
          STOP
       ENDIF
    ENDIF
    ! it turns out to be important with spline
    ! to reverse the order for a fast dropping curve
    ! x is now always from smaller to larger values
    DO I=1,n_interp
       XD2_1D(n_interp+1-I)=XA(ibeam,K+I)
       ZD2(n_interp+1-I)=ZA(ibeam,K+I)
    ENDDO
    CALL SPLINE_INTERPOLATE(XD2_1D,ZD2,n_interp,XI(1),PhotonFlux_nucleus_ChFF_nob)
    PhotonFlux_nucleus_ChFF_nob=PhotonFlux_nucleus_ChFF_nob/x
    RETURN
  END FUNCTION PhotonFlux_nucleus_ChFF_nob

  FUNCTION PhotonFlux_nucleus_ChFF_nob_eval(ibeam,Ega,gamma,Rval,wVal,aaVal)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_nucleus_ChFF_nob_eval
    INTEGER,INTENT(IN)::ibeam
    ! Ega in unit of GeV
    ! Rval and aaVal in unit of fm
    ! gamma and wVal are dimensionless
    REAL(KIND(1d0)),INTENT(IN)::Ega,gamma,Rval,wVal,aaVal
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::npoints
    REAL(KIND(1d0))::bupper,blower,Egaoga
    INTEGER::ibeam_common
    COMMON /nucleus_ChFF_nob_fxn_beam/ ibeam_common
    REAL(KIND(1d0))::Ega_common,gamma_common,Rval_common,wVal_common,aaVal_common
    COMMON /nucleus_ChFF_nob_fxn/ Ega_common, gamma_common, &
         Rval_common, wVal_common, aaVal_common
    npoints=50000
    ibeam_common=ibeam
    Ega_common=Ega ! in unit of GeV
    gamma_common=gamma
    ! converting from fm to GeV-1
    Rval_common=Rval/GeVm12fm
    wVal_common=wVal
    ! converting from fm to GeV-1
    aaVal_common=aaVal/GeVm12fm
    ! Exp(bA)=x*mN*b
    bupper=MAX(1.5d0,DLOG(0.8d0*Rval_common*DSQRT(Ega/gamma)))
    blower=DLOG(LOWER_BFactor_Limit*Rval_common*Ega/gamma)
    CALL simpson(PhotonFlux_nucleus_ChFF_nob_fxn,blower,bupper,&
         PhotonFlux_nucleus_ChFF_nob_eval,npoints)
    RETURN
  END FUNCTION PhotonFlux_nucleus_ChFF_nob_eval

  FUNCTION PhotonFlux_nucleus_ChFF_nob_fxn(bA)
    ! b is in unit of GeV-1
    ! Exp(bA)=x*mN*b
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonFlux_nucleus_ChFF_nob_fxn
    REAL(KIND(1d0)),INTENT(IN)::bA
    REAL(KIND(1d0))::b
    REAL(KIND(1d0))::btilde
    INTEGER::ibeam
    COMMON /nucleus_ChFF_nob_fxn_beam/ ibeam
    REAL(KIND(1d0))::Ega, gamma, Rval, wVal, aaVal
    COMMON /nucleus_ChFF_nob_fxn/ Ega, gamma, Rval, wVal, aaVal
    REAL(KIND(1d0)),PARAMETER::twopi=6.2831853071795864769252867d0
    b=DEXP(bA)/(Ega/gamma)
    IF(b.LE.0d0)THEN
       PhotonFlux_nucleus_ChFF_nob_fxn=0d0
       RETURN
    ENDIF
    PhotonFlux_nucleus_ChFF_nob_fxn=PhotonNumberDensity_ChargeFormFactor_WS(b,Ega,gamma,&
               Rval,wVal,aaVal,3d0,0.7d0,ibeam)
!    PhotonFlux_nucleus_ChFF_nob_fxn=PhotonNumberDensity(b,Ega,gamma)
    PhotonFlux_nucleus_ChFF_nob_fxn=PhotonFlux_nucleus_ChFF_nob_fxn*twopi*b**2
    RETURN
  END FUNCTION PhotonFlux_nucleus_ChFF_nob_fxn

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
    INTEGER::MX_save=0,MY_save=0
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD_save
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pp=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1&
         .AND.GENERATE_PhotonPhotonFlux_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
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
          WRITE(*,*)"INFO: in PhotonPhotonFlux_pp, the xmin of the grid has been updated to",xmin
          ! then deallocate it
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
             ! use the ones that have been calculated before
             IF(MX_save.GT.0.AND.MY_save.GT.0.AND.I.LE.MX_save+1.AND.J.LE.MY_save+1)THEN
                ZD(I,J)=ZD_save(I,J)
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pp_eval(xx1,xx2)
             IF(MX_save.EQ.0.AND.MY_save.EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   WRITE(*,*)"INFO: generate grid of photon-photon flux in pp (will take tens of seconds)"
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1,MY+1))
       ! save the values that have been calculated before
       MX_save=MX
       MY_save=MY
       DO I=1,MX+1
          DO J=1,MY+1
             ZD_save(I,J)=ZD(I,J)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       IF(GENERATE_PhotonPhotonFlux_GRID)THEN
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
       ELSE
          ZI(1)=PhotonPhotonFlux_pp_eval(x1,x2)
       ENDIF
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_pp_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pp=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.DABS(ZI(1)/pnohadval).GT.1D2)THEN
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
    CALC_DELTABQ=.FALSE.
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
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          PhotonPhotonFlux_pp_eval=PhotonFlux_proton_ChFF_nob(x1_common,gamma1_common,proton_FF_correction)
          PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
               PhotonFlux_proton_ChFF_nob(x2_common,gamma2_common,proton_FF_correction)
       ELSE
          PhotonPhotonFlux_pp_eval=PhotonFlux_proton_EDFF_nob(x1_common,gamma1_common)
          PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
               PhotonFlux_proton_EDFF_nob(x2_common,gamma2_common)
       ENDIF
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
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for approximation, we just use dipole
          PhotonPhotonFlux_pp_eval=PhotonFlux_proton_ChFF_nob(x1_common,gamma1_common)
          PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
               PhotonFlux_proton_ChFF_nob(x2_common,gamma2_common)
       ELSE
          PhotonPhotonFlux_pp_eval=PhotonFlux_proton_EDFF_nob(x1_common,gamma1_common)
          PhotonPhotonFlux_pp_eval=PhotonPhotonFlux_pp_eval*&
               PhotonFlux_proton_EDFF_nob(x2_common,gamma2_common)
       ENDIF
       PhotonPhotonFlux_pp_eval=MAX(PhotonPhotonFlux_pp_eval,0d0)
    ELSE
       PhotonPhotonFlux_pp_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pp_eval

  FUNCTION DeltaB_pp_at_x1x2(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::DeltaB_pp_at_x1x2
    REAL(KIND(1d0))::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    LOGICAL::force_pnohad1
    REAL(KIND(1d0))::num,den
    LOGICAL::use_grid_bu
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    ! numerator
    ! PhotonPhotonFlux*Delta b in unit of fm
    num=PhotonPhotonDeltaB_pp(x1,x2,force_pnohad1)
    ! denominator
    ! PhotonPhotonFlux
    use_grid_bu=GENERATE_PhotonPhotonFlux_GRID
    ! Let us make it to be coherent with DELTAB grid case
    GENERATE_PhotonPhotonFlux_GRID=GENERATE_DELTAB_GRID
    den=PhotonPhotonFlux_pp(x1,x2,force_pnohad1)
    GENERATE_PhotonPhotonFlux_GRID=use_grid_bu ! recover the original one
    IF(den.LE.0d0.OR.num.EQ.0d0)THEN
       DeltaB_pp_at_x1x2=0d0
    ELSE
       DeltaB_pp_at_x1x2=num/den
    ENDIF
    RETURN
  END FUNCTION DeltaB_pp_at_x1x2

  FUNCTION PhotonPhotonDeltaB_pp(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonDeltaB_pp
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
    INTEGER,DIMENSION(2)::MX_save=(/0,0/),MY_save=(/0,0/)
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:,:),ALLOCATABLE::ZD
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD1_save,ZD2_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD1_save,ZD2_save
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    LOGICAL,DIMENSION(2)::gridready=(/.FALSE.,.FALSE./)
    SAVE gridready
    INTEGER::igrid
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonDeltaB_pp=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       igrid=2
    ELSE
       igrid=1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin.OR..NOT.gridready(igrid)).AND.GENERATE_DELTAB_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       !IF(log10xmin.NE.log10xmin_before.or.init.EQ.0)THEN
       ! try to deallocate first
       IF(init.EQ.0.OR.log10xmin_before.GT.log10xmin)THEN
          IF(ALLOCATED(XD_1D))THEN
             DEALLOCATE(XD_1D)
          ENDIF
          ALLOCATE(XD_1D(MX+1))
          IF(ALLOCATED(YD_1D))THEN
             DEALLOCATE(YD_1D)
          ENDIF
          ALLOCATE(YD_1D(MY+1))
          IF(ALLOCATED(ZD))THEN
             WRITE(*,*)"INFO: in PhotonPhotonDeltaB_pp, the xmin of the grid has been updated to ",xmin
             ! save the values that have been calculated before
             IF(gridready(1))THEN
                MX_save(1)=MX
                MY_save(1)=MY
                IF(ALLOCATED(ZD1_save))THEN
                   DEALLOCATE(ZD1_save)
                ENDIF
                ALLOCATE(ZD1_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD1_save(I,J)=ZD(1,I,J)
                   ENDDO
                ENDDO
             ENDIF
             IF(gridready(2))THEN
                MX_save(2)=MX
                MY_save(2)=MY
                IF(ALLOCATED(ZD2_save))THEN
                   DEALLOCATE(ZD2_save)
                ENDIF
                ALLOCATE(ZD2_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD2_save(I,J)=ZD(2,I,J)
                   ENDDO
                ENDDO
             ENDIF
             DEALLOCATE(ZD)
          ENDIF
          ALLOCATE(ZD(2,MX+1,MY+1))
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
          gridready(1:2)=.FALSE.
       ENDIF
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             ! save the values that have been calculated before
             IF(MX_save(igrid).GT.0.AND.MY_save(igrid).GT.0.AND.I.LE.MX_save(igrid)+1&
                  .AND.J.LE.MY_save(igrid)+1)THEN
                IF(igrid.EQ.1)THEN
                   ZD(igrid,I,J)=ZD1_save(I,J)
                ELSEIF(igrid.EQ.2)THEN
                   ZD(igrid,I,J)=ZD2_save(I,J)
                ELSE
                   WRITE(*,*)"ERROR: do not known igrid=",igrid
                ENDIF
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(igrid,I,J)=PhotonPhotonDeltaB_pp_eval(xx1,xx2,force_pnohad1)
             IF(MX_save(igrid).EQ.0.AND.MY_save(igrid).EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   IF(igrid.EQ.1)THEN
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in pp (will take tens of seconds)"
                   ELSE
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in pp with PNOHAD=1 (will take tens of seconds)"
                   ENDIF
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       gridready(igrid)=.TRUE.
       init=1
    ENDIF
    IF(GENERATE_DELTAB_GRID)THEN
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
             ZD2(I,J)=ZD(igrid,K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
       IF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0)THEN
          PhotonPhotonDeltaB_pp=0d0
       ELSE
          PhotonPhotonDeltaB_pp=ZI(1)
       ENDIF
    ELSE
       PhotonPhotonDeltaB_pp=PhotonPhotonDeltaB_pp_eval(x1,x2,force_pnohad1)
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_pp

  FUNCTION PhotonPhotonDeltaB_pp_eval(x1,x2,FORCEPNOHAD1)
    ! this is PhotonPhotonFlux*Delta b, where Delta b is in unit of fm
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonDeltaB_pp_eval
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
       PhotonPhotonDeltaB_pp_eval=0d0
       RETURN
    ENDIF
    CALC_DELTABQ=.TRUE.
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
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact*x1)
    ! b2*x2*mproton = Exp(bA(2))
    aax(2)=DLOG(bfact*x2)
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
            integral,ind,eval_num)
    ELSE
       CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax,bbx,3,sub_num,1,1d-5,&
            integral,ind,eval_num)
    ENDIF
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bupper*2d0**(iter)
          bbx(2)=bupper*2d0**(iter)
          IF(force_pnohad1)THEN
             ! we only use PNOHAD=1
             CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
                  integral,ind,eval_num)
          ELSE
             CALL ROMBERG_ND(PhotonPhotonFlux_pp_fxn,aax,bbx,3,sub_num,1,1d-5,&
                  integral,ind,eval_num)
          ENDIF
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       bbx(2)=bupper
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon <delta b> at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: most probably need to increase bupper"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonDeltaB_pp_eval=0d0
    ELSE
       PhotonPhotonDeltaB_pp_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_pp_eval

  FUNCTION PhotonPhotonFlux_pp_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_pp_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3 or 2
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
    REAL(KIND(1d0)),EXTERNAL::elliptic_em
    REAL(KIND(1d0))::arg1,arg2
    IF(dim_num.NE.3.AND.dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_pp_fxn is not a three or two dimensional function"
       STOP
    ENDIF
    ! in unit of GeV-1
    ! x1*b1*mproton=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mproton
    ! x2*b2*mproton=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mproton
    IF(dim_num.EQ.3)THEN
       costh=DCOS(bA(3))
       b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
       pnohad=PNOHAD_pp(b12,0d0,b0_common)
       IF(CALC_DELTABQ)THEN
          pnohad=pnohad*b12*GeVm12fm ! convert from GeV-1 to fm
       ENDIF
    ELSE
       ! dim_num = 2
       ! it must be a case with pnohad=1
       ! and CALC_DELTABQ=.TRUE.
       IF(.NOT.CALC_DELTABQ)THEN
          WRITE(*,*)"ERROR: cannot reach here with CALC_DELTABQ=.FALSE."
       ENDIF
       ! pnohad is Integrate[Sqrt[b1^2 + b2^2 - 2 b1*b2*Cos[phi]], {phi, 0, 2 Pi}]
       arg2=4d0*b1*b2/(b1+b2)**2
       IF(b1.NE.b2.AND.arg2.LT.1d0)THEN
          arg1=-4d0*b1*b2/(b1-b2)**2
          ! elliptic_em(m) is EllipticE[m] with m<1
          pnohad=DABS(b1-b2)*elliptic_em(arg1)
          pnohad=pnohad+(b1+b2)*elliptic_em(arg2)
       ELSE
          pnohad=4d0*b1
       ENDIF
       pnohad=pnohad*2d0*GeVm12fm ! convert from GeV-1 to fm
    ENDIF
    IF(pnohad.LE.0d0)THEN
       PhotonPhotonFlux_pp_fxn=0d0
       RETURN
    ENDIF
    IF(.NOT.USE_CHARGEFORMFACTOR4PHOTON)THEN
       ! EDFF
       Ngamma1=PhotonNumberDensity(b1,E1_common,gamma1_common)
       Ngamma2=PhotonNumberDensity(b2,E2_common,gamma2_common)
    ELSE
       ! ChFF
       Ngamma1=PhotonNumberDensity_ChargeFormFactor_proton(b1,E1_common,gamma1_common,&
            proton_FF_correction)
       Ngamma2=PhotonNumberDensity_ChargeFormFactor_proton(b2,E2_common,gamma2_common,&
            proton_FF_correction)
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
    INTEGER::MX_save=0,MY_save=0
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD_save
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pA_hardsphere=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1&
         .AND.GENERATE_PhotonPhotonFlux_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
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
          ! then deallocate it
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
             IF(MX_save.GT.0.AND.MY_save.GT.0.AND.I.LE.MX_save+1&
                  .AND.J.LE.MY_save+1)THEN
                ZD(I,J)=ZD_save(I,J)
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pA_hardsphere_eval(xx1,xx2)
             IF(MX_save.EQ.0.AND.MY_save.EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   WRITE(*,*)"INFO: generate grid of photon-photon flux in pA or Ap (will take tens of seconds)"
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1,MY+1))
       ! save the values that have been calculated before
       MX_save=MX
       MY_save=MY
       DO I=1,MX+1
          DO J=1,MY+1
             ZD_save(I,J)=ZD(I,J)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       IF(GENERATE_PhotonPhotonFlux_GRID)THEN
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
       ELSE
          ZI(1)=PhotonPhotonFlux_pA_hardsphere_eval(x1,x2)
       ENDIF
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
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonFlux_proton_EDFF_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonPhotonFlux_pA_hardsphere_eval*&
            PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common)
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
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonFlux_proton_EDFF_nob(x1_common,&
            gamma1_common)
       PhotonPhotonFlux_pA_hardsphere_eval=PhotonPhotonFlux_pA_hardsphere_eval*&
            PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common)
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
    INTEGER::MX_save=0,MY_save=0
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD_save
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_pA_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1&
         .AND.GENERATE_PhotonPhotonFlux_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
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
             IF(MX_save.GT.0.AND.MY_save.GT.0.AND.I.LE.MX_save+1.AND.J.LE.MY_save+1)THEN
                ZD(I,J)=ZD_save(I,J)
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_pA_WoodsSaxon_eval(xx1,xx2)
             IF(MX_save.EQ.0.AND.MY_save.EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   WRITE(*,*)"INFO: generate grid of photon-photon flux in pA or Ap (will take a few minutes)"
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1,MY+1))
       MX_save=MX
       MY_save=MY
       DO I=1,MX+1
          DO J=1,MY+1
             ZD_save(I,J)=ZD(I,J)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       IF(GENERATE_PhotonPhotonFlux_GRID)THEN
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
       ELSE
          ZI(1)=PhotonPhotonFlux_pA_WoodsSaxon_eval(x1,x2)
       ENDIF
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_pA_WoodsSaxon_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_pA_WoodsSaxon=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.DABS(ZI(1)/pnohadval).GT.1D2)THEN
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
    REAL(KIND(1d0)),PARAMETER::bupper=2d0
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
    CALC_DELTABQ=.FALSE.
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
       IF(IEPSILON_EDFF.EQ.0)THEN
          bfact2=RA_common/GeVm12fm*mN
       ELSE
          bfact2=(RA_common+aaVal_common)/GeVm12fm*mN
       ENDIF
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
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_ChFF_nob(x1_common,&
               gamma1_common,proton_FF_correction)
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
            PhotonFlux_nucleus_ChFF_nob(2,x2_common,gamma2_common,Z2,RA_common,wVal_common,aaVal_common)
       ELSE
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_EDFF_nob(x1_common,&
               gamma1_common)
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
               PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common)
       ENDIF
       PhotonPhotonFlux_pA_WoodsSaxon_eval=MAX(PhotonPhotonFlux_pA_WoodsSaxon_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2))
    aax(2)=DLOG(bfact2*x2_common)
    ! we should also choose the upper limit dynamically for the ion beam
    bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common/GeVm12fm*DSQRT(x2_common*mN)))
    CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          !bbx(1)=bupper*2d0**(iter)
          !bbx(2)=bupper*2d0**(iter)
          bbx(1)=bbx(1)*2d0
          bbx(2)=bbx(2)*2d0
          CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       !bbx(2)=bupper
       bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common/GeVm12fm*DSQRT(x2_common*mN)))
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for approximation
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_ChFF_nob(x1_common,&
               gamma1_common,proton_FF_correction)
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
            PhotonFlux_nucleus_ChFF_nob(2,x2_common,gamma2_common,Z2,RA_common,wVal_common,aaVal_common)
       ELSE
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonFlux_proton_EDFF_nob(x1_common,&
               gamma1_common)
          PhotonPhotonFlux_pA_WoodsSaxon_eval=PhotonPhotonFlux_pA_WoodsSaxon_eval*&
               PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common)
       ENDIF
       PhotonPhotonFlux_pA_WoodsSaxon_eval=MAX(PhotonPhotonFlux_pA_WoodsSaxon_eval,0d0)
    ELSE
       PhotonPhotonFlux_pA_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_eval

  FUNCTION DeltaB_pA_at_x1x2(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::DeltaB_pA_at_x1x2
    REAL(KIND(1d0))::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    LOGICAL::force_pnohad1
    REAL(KIND(1d0))::num,den
    LOGICAL::use_grid_bu
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    ! numerator
    ! PhotonPhotonFlux*Delta b in unit of fm
    num=PhotonPhotonDeltaB_pA_WoodsSaxon(x1,x2,force_pnohad1)
    ! denominator
    ! PhotonPhotonFlux
    use_grid_bu=GENERATE_PhotonPhotonFlux_GRID
    ! Let us make it to be coherent with DELTAB grid case
    GENERATE_PhotonPhotonFlux_GRID=GENERATE_DELTAB_GRID
    den=PhotonPhotonFlux_pA_WoodsSaxon(x1,x2,force_pnohad1)
    GENERATE_PhotonPhotonFlux_GRID=use_grid_bu ! recover the original one
    IF(den.LE.0d0.OR.num.EQ.0d0)THEN
       DeltaB_pA_at_x1x2=0d0
    ELSE
       DeltaB_pA_at_x1x2=num/den
    ENDIF
    RETURN
  END FUNCTION DeltaB_pA_at_x1x2

  FUNCTION PhotonPhotonDeltaB_pA_WoodsSaxon(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonDeltaB_pA_WoodsSaxon
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
    INTEGER,DIMENSION(2)::MX_save=(/0,0/),MY_save=(/0,0/)
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:,:),ALLOCATABLE::ZD
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD1_save,ZD2_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD1_save,ZD2_save
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    LOGICAL,DIMENSION(2)::gridready=(/.FALSE.,.FALSE./)
    SAVE gridready
    INTEGER::igrid
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonDeltaB_pA_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       igrid=2
    ELSE
       igrid=1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin.OR..NOT.gridready(igrid)).AND.GENERATE_DELTAB_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(init.EQ.0.OR.log10xmin_before.GT.log10xmin)THEN
          IF(ALLOCATED(XD_1D))THEN
             DEALLOCATE(XD_1D)
          ENDIF
          ALLOCATE(XD_1D(MX+1))
          IF(ALLOCATED(YD_1D))THEN
             DEALLOCATE(YD_1D)
          ENDIF
          ALLOCATE(YD_1D(MY+1))
          IF(ALLOCATED(ZD))THEN
             WRITE(*,*)"INFO: in PhotonPhotonDeltaB_pA_WoodsSaxon, the xmin of the grid has been updated to ",xmin
             ! save the values that have been calculated before
             IF(gridready(1))THEN
                MX_save(1)=MX
                MY_save(1)=MY
                IF(ALLOCATED(ZD1_save))THEN
                   DEALLOCATE(ZD1_save)
                ENDIF
                ALLOCATE(ZD1_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD1_save(I,J)=ZD(1,I,J)
                   ENDDO
                ENDDO
             ENDIF
             IF(gridready(2))THEN
                MX_save(2)=MX
                MY_save(2)=MY
                IF(ALLOCATED(ZD2_save))THEN
                   DEALLOCATE(ZD2_save)
                ENDIF
                ALLOCATE(ZD2_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD2_save(I,J)=ZD(2,I,J)
                   ENDDO
                ENDDO
             ENDIF
             DEALLOCATE(ZD)
          ENDIF
          ALLOCATE(ZD(2,MX+1,MY+1))
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
          gridready(1:2)=.FALSE.
       ENDIF
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             ! save the values that have been calculated before
             IF(MX_save(igrid).GT.0.AND.MY_save(igrid).GT.0.AND.I.LE.MX_save(igrid)+1&
                  .AND.J.LE.MY_save(igrid)+1)THEN
                IF(igrid.EQ.1)THEN
                   ZD(igrid,I,J)=ZD1_save(I,J)
                ELSEIF(igrid.EQ.2)THEN
                   ZD(igrid,I,J)=ZD2_save(I,J)
                ELSE
                   WRITE(*,*)"ERROR: do not know igrid=",igrid
                ENDIF
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(igrid,I,J)=PhotonPhotonDeltaB_pA_WoodsSaxon_eval(xx1,xx2,force_pnohad1)
             IF(MX_save(igrid).EQ.0.AND.MY_save(igrid).EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   IF(igrid.EQ.1)THEN
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in pA or Ap (will take a few minutes)"
                   ELSE
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in pA or Ap with PNOHAD=1 (will take a few minutes)"
                   ENDIF
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       gridready(igrid)=.TRUE.
       init=1
    ENDIF
    IF(GENERATE_DELTAB_GRID)THEN
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
             ZD2(I,J)=ZD(igrid,K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
       IF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0)THEN
          PhotonPhotonDeltaB_pA_WoodsSaxon=0d0
       ELSE
          PhotonPhotonDeltaB_pA_WoodsSaxon=ZI(1)
       ENDIF
    ELSE
       PhotonPhotonDeltaB_pA_WoodsSaxon=PhotonPhotonDeltaB_pA_WoodsSaxon_eval(x1,x2,force_pnohad1)
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_pA_WoodsSaxon

  FUNCTION PhotonPhotonDeltaB_pA_WoodsSaxon_eval(x1,x2,FORCEPNOHAD1)
    ! this is PhotonPhotonFlux*Delta b, where Delta b is in unit of fm
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonDeltaB_pA_WoodsSaxon_eval
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
    REAL(KIND(1d0)),PARAMETER::bupper=2d0
    CHARACTER(len=7)::Aname
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonDeltaB_pA_WoodsSaxon_eval=0d0
       RETURN
    ENDIF
    CALC_DELTABQ=.TRUE.
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
       IF(IEPSILON_EDFF.EQ.0)THEN
          bfact2=RA_common/GeVm12fm*mN
       ELSE
          bfact2=(RA_common+aaVal_common)/GeVm12fm*mN
       ENDIF
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
    ! we should choose the lower limit dynamically
    ! b1*x1*mproton = Exp(bA(1))
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2))
    aax(2)=DLOG(bfact2*x2_common)
    ! we should also choose the upper limit dynamically for the ion beam
    bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common/GeVm12fm*DSQRT(x2_common*mN)))
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
            integral,ind,eval_num)
    ELSE
       CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
            integral,ind,eval_num)
    ENDIF
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          !bbx(1)=bupper*2d0**(iter)
          !bbx(2)=bupper*2d0**(iter)
          bbx(1)=bbx(1)*2d0
          bbx(2)=bbx(2)*2d0
          IF(force_pnohad1)THEN
             ! we only use PNOHAD=1
             CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
                  integral,ind,eval_num)
          ELSE
             CALL ROMBERG_ND(PhotonPhotonFlux_pA_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
                  integral,ind,eval_num)
          ENDIF
          iter=iter+1
       ENDDO
       bbx(1)=bupper
       !bbx(2)=bupper
       bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common/GeVm12fm*DSQRT(x2_common*mN)))
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon <delta b> at (x1,x2)=",x1_common,x2_common
          WRITE(*,*)"WARNING: most probably need to increase bupper"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonDeltaB_pA_WoodsSaxon_eval=0d0
    ELSE
       PhotonPhotonDeltaB_pA_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_pA_WoodsSaxon_eval

  FUNCTION PhotonPhotonFlux_pA_WoodsSaxon_fxn(dim_num,bA)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonFlux_pA_WoodsSaxon_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3 or 2
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mproton=Exp(bA(1))
    ! x2*b2*mproton=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::b12GeVm1,pbreak
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
    REAL(KIND(1d0)),EXTERNAL::elliptic_em
    REAL(KIND(1d0))::arg1,arg2
    IF(dim_num.NE.3.AND.dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_pA_WoodsSaxon_fxn is not a three or two dimensional function"
       STOP
    ENDIF
    ! in unit of GeV-1
    ! x1*b1*mproton=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mproton
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    IF(dim_num.EQ.3)THEN
       costh=DCOS(bA(3))
       b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
       b12GeVm1=b12 ! in unit of GeV-1
       b12=b12*GeVm12fm ! from GeV-1 to fm
       pnohad=PNOHAD_pA_WoodsSaxon(b12,0d0,RA_common,wVal_common,aaVal_common,&
            A_common,sigNN_inel_common)
       IF(nuclearA_beam1.NE.0)THEN
          ! A+p
          IF(neutron_tagging(1).NE.-2)THEN
             ! multiply the forward neutron tagging probability
             pbreak=Pbreak_pA_WoodsSaxon(neutron_tagging(1),b12GeVm1)
             pnohad=pnohad*pbreak
          ENDIF
       ELSE
          ! p+A
          IF(neutron_tagging(2).NE.-2)THEN
             ! multiply the forward neutron tagging probability
             pbreak=Pbreak_pA_WoodsSaxon(neutron_tagging(2),b12GeVm1)
             pnohad=pnohad*pbreak
          ENDIF
       ENDIF
       IF(CALC_DELTABQ)THEN
          pnohad=pnohad*b12
       ENDIF
    ELSE
       ! dim_num = 2
       ! it must be a case with pnohad=1
       ! and CALC_DELTABQ=.TRUE
       IF(.NOT.CALC_DELTABQ)THEN
          WRITE(*,*)"ERROR: cannot reach here with CALC_DELTABQ=.FALSE."
       ENDIF
       ! pnohad is Integrate[Sqrt[b1^2 + b2^2 - 2 b1*b2*Cos[phi]], {phi, 0, 2 Pi}]
       arg2=4d0*b1*b2/(b1+b2)**2
       IF(b1.NE.b2.AND.arg2.LT.1d0)THEN
          arg1=-4d0*b1*b2/(b1-b2)**2
          ! elliptic_em(m) is EllipticE[m] with m<1
          pnohad=DABS(b1-b2)*elliptic_em(arg1)
          pnohad=pnohad+(b1+b2)*elliptic_em(arg2)
       ELSE
          pnohad=4d0*b1
       ENDIF
       pnohad=pnohad*2d0*GeVm12fm ! convert from GeV-1 to fm
    ENDIF
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
          Ngamma2=PhotonNumberDensity_ChargeFormFactor_proton(b2,E2_common,gamma2_common,&
               proton_FF_correction)
       ELSE
          Ngamma1=PhotonNumberDensity_ChargeFormFactor_proton(b1,E1_common,gamma1_common,&
               proton_FF_correction)
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
    INTEGER::MX_save=0,MY_save=0
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD_save
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_AB_hardsphere=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1&
         .AND.GENERATE_PhotonPhotonFlux_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
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
          WRITE(*,*)"INFO: in PhotonPhotonFlux_AB_hardsphere, the xmin of the grid has been updated to ",xmin
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
             ! save the values that have been calculated before
             IF(MX_save.GT.0.AND.MY_save.GT.0.AND.I.LE.MX_save+1.AND.J.LE.MY_save+1)THEN
                ZD(I,J)=ZD_SAVE(I,J)
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_AB_hardsphere_eval(xx1,xx2)
             IF(MX_SAVE.EQ.0.AND.MY_SAVE.EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   WRITE(*,*)"INFO: generate grid of photon-photon flux in AB (will take tens of seconds)"
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1,MY+1))
       ! save the values that have been calculated before
       MX_save=MX
       MY_save=MY
       DO I=1,MX+1
          DO J=1,MY+1
             ZD_save(I,J)=ZD(I,J)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       IF(GENERATE_PhotonPhotonFlux_GRID)THEN
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
       ELSE
          ZI(1)=PhotonPhotonFlux_AB_hardsphere_eval(x1,x2)
       ENDIF
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
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonFlux_nucleus_EDFF_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonPhotonFlux_AB_hardsphere_eval*&
            PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common(2))
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
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonFlux_nucleus_EDFF_nob(x1_common,&
            gamma1_common,Z1,RA_common(1))
       PhotonPhotonFlux_AB_hardsphere_eval=PhotonPhotonFlux_AB_hardsphere_eval*&
            PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common(2))
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
    INTEGER::MX_save=0,MY_save=0
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD,ZD_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD_save
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonFlux_AB_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin).AND..NOT.force_pnohad1&
         .AND.GENERATE_PhotonPhotonFlux_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
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
          WRITE(*,*)"INFO: in PhotonPhotonFlux_AB_WoodsSaxon, the xmin of the grid has been updated to ",xmin
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
             ! save the values that have been calculated before
             IF(MX_save.GT.0.AND.MY_save.GT.0.AND.I.LE.MX_save+1&
                  .AND.J.LE.MY_save+1)THEN
                ZD(I,J)=ZD_save(I,J)
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(I,J)=PhotonPhotonFlux_AB_WoodsSaxon_eval(xx1,xx2)
             IF(MX_save.EQ.0.AND.MY_save.EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   WRITE(*,*)"INFO: generate grid of photon-photon flux in AB (will take a few minutes)"
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
          ENDDO
       ENDDO
       IF(ALLOCATED(ZD_save))THEN
          DEALLOCATE(ZD_save)
       ENDIF
       ALLOCATE(ZD_save(MX+1,MY+1))
       ! save the values that have been calculated before
       MX_save=MX
       MY_save=MY
       DO I=1,MX+1
          DO J=1,MY+1
             ZD_save(I,J)=ZD(I,J)
          ENDDO
       ENDDO
       init=1
    ENDIF
    IF(.NOT.force_pnohad1)THEN
       IF(GENERATE_PhotonPhotonFlux_GRID)THEN
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
       ELSE
          ZI(1)=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2)
       ENDIF
    ENDIF
    ! Let us always evaluate PNOHAD=1 as a reference to compare
    pnohadval=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2,.TRUE.)
    IF(.NOT.force_pnohad1)THEN
       IF(ISNAN(pnohadval).OR.pnohadval.EQ.0d0)THEN
          PhotonPhotonFlux_AB_WoodsSaxon=0d0
       ELSEIF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0.OR.DABS(ZI(1)/pnohadval).GT.1D2)THEN
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
    REAL(KIND(1d0)),DIMENSION(2)::Z_common
    REAL(KIND(1d0)),DIMENSION(2)::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_AB_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,Z_common,aaVal_common,wVal_common
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
    REAL(KIND(1d0)),PARAMETER::bupper=2d0
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
    CALC_DELTABQ=.FALSE.
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
       CALL GetNuclearInfo(Aname1,A_common(1),Z1,RA_common(1),aaVal_common(1),wVal_common(1))
       Z_common(1)=Z1
       IF(nuclearA_beam2.NE.nuclearA_beam1.OR.nuclearZ_beam1.NE.nuclearZ_beam2)THEN
          Aname2=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
          CALL GetNuclearInfo(Aname2,A_common(2),Z2,RA_common(2),aaVal_common(2),wVal_common(2))
          Z_common(2)=Z2
       ELSE
          Aname2=Aname1
          A_common(2)=A_common(1)
          Z_common(2)=Z_common(1)
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
       IF(IEPSILON_EDFF.EQ.0)THEN
          bfact1=RA_common(1)/GeVm12fm*mN
          bfact2=RA_common(2)/GeVm12fm*mN
       ELSE
          bfact1=(RA_common(1)+aaVal_common(1))/GeVm12fm*mN
          bfact2=(RA_common(2)+aaVal_common(2))/GeVm12fm*mN
       ENDIF
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we can integrate b down to zero
          bfact1=bfact1*LOWER_BFactor_Limit
          bfact2=bfact2*LOWER_BFactor_Limit
       ENDIF
       !bbx(1)=bupper
       !bbx(2)=bupper
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
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_ChFF_nob(1,x1_common,&
               gamma1_common,Z1,RA_common(1),wVal_common(1),aaVal_common(1))
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
               PhotonFlux_nucleus_ChFF_nob(2,x2_common,gamma2_common,Z2,RA_common(2),&
               wVal_common(2),aaVal_common(2))
       ELSE
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_EDFF_nob(x1_common,&
               gamma1_common,Z1,RA_common(1))
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
               PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common(2))
       ENDIF
       PhotonPhotonFlux_AB_WoodsSaxon_eval=MAX(PhotonPhotonFlux_AB_WoodsSaxon_eval,0d0)
       RETURN
    ENDIF
    ! we should choose the lower limit dynamically
    ! b1*x1*mN = Exp(bA(1)) = b1*E_gamma1/gamma1 = b1tilde
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2)) = b2*E_gamma2/gamma2 = b2tilde
    aax(2)=DLOG(bfact2*x2_common)
    ! we should also choose the upper limit dynamically
    bbx(1)=MAX(1.5d0,DLOG(0.8d0*RA_common(1)/GeVm12fm*DSQRT(x1_common*mN)))
    bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common(2)/GeVm12fm*DSQRT(x2_common*mN)))
    CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
         integral,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bbx(1)*2d0
          bbx(2)=bbx(2)*2d0
          CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
               integral,ind,eval_num)
          iter=iter+1
       ENDDO
       !bbx(1)=bupper
       !bbx(2)=bupper
       bbx(1)=MAX(1.5d0,DLOG(0.8d0*RA_common(1)/GeVm12fm*DSQRT(x1_common*mN)))
       bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common(2)/GeVm12fm*DSQRT(x2_common*mN)))
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon flux at (x1,x2)=",x1,x2
          WRITE(*,*)"WARNING: use PNOHAD=1 approx. instead (most probably need to increase bupper)"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_ChFF_nob(1,x1_common,&
               gamma1_common,Z1,RA_common(1),wVal_common(1),aaVal_common(1))
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
               PhotonFlux_nucleus_ChFF_nob(2,x2_common,gamma2_common,Z2,RA_common(2),&
               wVal_common(2),aaVal_common(2))
       ELSE
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonFlux_nucleus_EDFF_nob(x1_common,&
               gamma1_common,Z1,RA_common(1))
          PhotonPhotonFlux_AB_WoodsSaxon_eval=PhotonPhotonFlux_AB_WoodsSaxon_eval*&
               PhotonFlux_nucleus_EDFF_nob(x2_common,gamma2_common,Z2,RA_common(2))
       ENDIF
       PhotonPhotonFlux_AB_WoodsSaxon_eval=MAX(PhotonPhotonFlux_AB_WoodsSaxon_eval,0d0)
    ELSE
       PhotonPhotonFlux_AB_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_eval

  FUNCTION DeltaB_AB_at_x1x2(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::DeltaB_AB_at_x1x2
    REAL(KIND(1d0))::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1 ! If true, it only evaluates with PNOHAD=1
    LOGICAL::force_pnohad1
    REAL(KIND(1d0))::num,den
    LOGICAL::use_grid_bu
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    ! numerator
    ! PhotonPhotonFlux*Delta b in unit of fm
    num=PhotonPhotonDeltaB_AB_WoodsSaxon(x1,x2,force_pnohad1)
    ! denominator
    ! PhotonPhotonFlux
    use_grid_bu=GENERATE_PhotonPhotonFlux_GRID
    ! Let us make it to be coherent with DELTAB grid case
    GENERATE_PhotonPhotonFlux_GRID=GENERATE_DELTAB_GRID
    den=PhotonPhotonFlux_AB_WoodsSaxon(x1,x2,force_pnohad1)
    GENERATE_PhotonPhotonFlux_GRID=use_grid_bu ! recover the original one
    IF(den.LE.0d0.OR.num.EQ.0d0)THEN
       DeltaB_AB_at_x1x2=0d0
    ELSE
       DeltaB_AB_at_x1x2=num/den
    ENDIF
    RETURN
  END FUNCTION DeltaB_AB_at_x1x2

  FUNCTION PhotonPhotonDeltaB_AB_WoodsSaxon(x1,x2,FORCEPNOHAD1)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonDeltaB_AB_WoodsSaxon
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
    INTEGER,DIMENSION(2)::MX_save=(/0,0/),MY_save=(/0,0/)
    SAVE MX_save,MY_save
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XD_1D,YD_1D
    REAL(KIND(1d0)),DIMENSION(:,:,:),ALLOCATABLE::ZD
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::ZD1_save,ZD2_save
    SAVE MX,MY,XD_1D,YD_1D,ZD,ZD1_save,ZD2_save
    REAL(KIND(1d0)),DIMENSION(nseg+1)::XD2_1D,YD2_1D
    REAL(KIND(1d0)),DIMENSION(nseg+1,nseg+1)::ZD2
    REAL(KIND(1d0))::xx1,xx2
    REAL(KIND(1d0)),DIMENSION(1)::XI,YI,ZI
    REAL(KIND(1d0))::pnohadval
    LOGICAL::force_pnohad1
    LOGICAL,DIMENSION(2)::gridready=(/.FALSE.,.FALSE./)
    SAVE gridready
    INTEGER::igrid
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    IF(x1.LE.1d-99.OR.x2.LE.1d-99.OR.x1.GT.1d0.OR.x2.GT.1d0)THEN
       PhotonPhotonDeltaB_AB_WoodsSaxon=0d0
       RETURN
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(.NOT.PRESENT(FORCEPNOHAD1))THEN
       force_pnohad1=.FALSE.
    ELSE
       force_pnohad1=FORCEPNOHAD1
    ENDIF
    IF(force_pnohad1)THEN
       igrid=2
    ELSE
       igrid=1
    ENDIF
    IF((init.EQ.0.OR.x1.LT.xmin.OR.x2.LT.xmin.OR..NOT.gridready(igrid)).AND.GENERATE_DELTAB_GRID)THEN
       ! initialisation
       log10xmin_before=INT(DLOG10(xmin))
       IF(x1.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x1))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x1.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       IF(x2.LT.xmin)THEN
          log10xmin=FLOOR(DLOG10(x2))
          xmin=10d0**(log10xmin)
          ! to avoid some numerical artificial
          DO WHILE(x2.LT.xmin)
             log10xmin=log10xmin-1
             xmin=10d0**(log10xmin)
          ENDDO
       ENDIF
       log10xmin=INT(DLOG10(xmin))
       ! let us generate a 2-dim grid [xmin,1]x[xmin,1] first
       MX=nseg*(-log10xmin)
       MY=MX
       ! try to deallocate first
       IF(init.EQ.0.OR.log10xmin_before.GT.log10xmin)THEN
          IF(ALLOCATED(XD_1D))THEN
             DEALLOCATE(XD_1D)
          ENDIF
          ALLOCATE(XD_1D(MX+1))
          IF(ALLOCATED(YD_1D))THEN
             DEALLOCATE(YD_1D)
          ENDIF
          ALLOCATE(YD_1D(MY+1))
          IF(ALLOCATED(ZD))THEN
             WRITE(*,*)"INFO: in PhotonPhotonDeltaB_AB_WoodsSaxon, the xmin of the grid has been updated to ",xmin
             ! save the values that have been calculated before
             IF(gridready(1))THEN
                MX_save(1)=MX
                MY_save(1)=MY
                IF(ALLOCATED(ZD1_save))THEN
                   DEALLOCATE(ZD1_save)
                ENDIF
                ALLOCATE(ZD1_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD1_save(I,J)=ZD(1,I,J)
                   ENDDO
                ENDDO
             ENDIF
             IF(gridready(2))THEN
                MX_save(2)=MX
                MY_save(2)=MY
                IF(ALLOCATED(ZD2_save))THEN
                   DEALLOCATE(ZD2_save)
                ENDIF
                ALLOCATE(ZD2_save(MX+1,MY+1))
                DO I=1,MX+1
                   DO J=1,MY+1
                      ZD2_save(I,J)=ZD(2,I,J)
                   ENDDO
                ENDDO
             ENDIF
             DEALLOCATE(ZD)
          ENDIF
          ALLOCATE(ZD(2,MX+1,MY+1))
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
          gridready(1:2)=.FALSE.
       ENDIF
       DO I=1,MX+1
          xx1=10d0**(XD_1D(I))
          DO J=1,MY+1
             ! save the values that have been calculated before
             IF(MX_save(igrid).GT.0.AND.MY_save(igrid).GT.0.AND.I.LE.MX_save(igrid)+1&
                  .AND.J.LE.MY_save(igrid)+1)THEN
                IF(igrid.EQ.1)THEN
                   ZD(igrid,I,J)=ZD1_save(I,J)
                ELSEIF(igrid.EQ.2)THEN
                   ZD(igrid,I,J)=ZD2_save(I,J)
                ELSE
                   WRITE(*,*)"ERROR: do not know igrid=",igrid
                ENDIF
                CYCLE
             ENDIF
             xx2=10d0**(YD_1D(J))
             ZD(igrid,I,J)=PhotonPhotonDeltaB_AB_WoodsSaxon_eval(xx1,xx2,force_pnohad1)
             IF(MX_save(igrid).EQ.0.AND.MY_save(igrid).EQ.0)THEN
                IF(I.EQ.2.AND.J.EQ.2)THEN
                   IF(igrid.EQ.1)THEN
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in AB (will take a few minutes)"
                   ELSE
                      WRITE(*,*)"INFO: generate grid of photon-photon <delta b> in AB with PNOHAD=1 (will take a few minutes)"
                   ENDIF
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50,.TRUE.)
                ELSEIF(.NOT.(I.EQ.1.OR.J.EQ.1))THEN
                   ! we skip I=1 or J=1 which just returns 0
                   CALL progress(INT(((I-1)*(MY+1)+J)*50d0/((MX+1)*(MY+1))),50)
                ENDIF
             ENDIF
           ENDDO
       ENDDO
       gridready(igrid)=.TRUE.
       init=1
    ENDIF
    IF(GENERATE_DELTAB_GRID)THEN
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
             ZD2(I,J)=ZD(igrid,K+I,L+J)
          ENDDO
       ENDDO
       CALL lagrange_interp_2d(nseg,nseg,XD2_1D,YD2_1D,ZD2,1,XI,YI,ZI)
       IF(ISNAN(ZI(1)).OR.ZI(1).LT.0d0)THEN
          PhotonPhotonDeltaB_AB_WoodsSaxon=0d0
       ELSE
          PhotonPhotonDeltaB_AB_WoodsSaxon=pnohadval
       ENDIF
    ELSE
       PhotonPhotonDeltaB_AB_WoodsSaxon=PhotonPhotonDeltaB_AB_WoodsSaxon_eval(x1,x2,force_pnohad1)
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_AB_WoodsSaxon

  FUNCTION PhotonPhotonDeltaB_AB_WoodsSaxon_eval(x1,x2,FORCEPNOHAD1)
    ! this is PhotonPhotonFlux*Delta b, where Delta b is in unit of fm
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::PhotonPhotonDeltaB_AB_WoodsSaxon_eval
    REAL(KIND(1d0)),INTENT(IN)::x1,x2
    LOGICAL,INTENT(IN),OPTIONAL::FORCEPNOHAD1
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0)),DIMENSION(2)::Z_common
    REAL(KIND(1d0)),DIMENSION(2)::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_AB_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,Z_common,aaVal_common,wVal_common
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
    REAL(KIND(1d0)),PARAMETER::bupper=2d0
    CHARACTER(len=7)::Aname1,Aname2
    REAL(KIND(1d0))::cmenergy
    INTEGER,PARAMETER::itermax=5
    INTEGER::printnum=0,iter
    SAVE printnum
    LOGICAL::force_pnohad1
    IF(x1.LE.0d0.OR.x1.GE.1d0.OR.x2.LE.0d0.OR.x2.GE.1d0)THEN
       PhotonPhotonDeltaB_AB_WoodsSaxon_eval=0d0
       RETURN
    ENDIF
    CALC_DELTABQ=.TRUE.
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
       CALL GetNuclearInfo(Aname1,A_common(1),Z1,RA_common(1),aaVal_common(1),wVal_common(1))
       Z_common(1)=Z1
       IF(nuclearA_beam2.NE.nuclearA_beam1.OR.nuclearZ_beam1.NE.nuclearZ_beam2)THEN
          Aname2=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
          CALL GetNuclearInfo(Aname2,A_common(2),Z2,RA_common(2),aaVal_common(2),wVal_common(2))
          Z_common(2)=Z2
       ELSE
          Aname2=Aname1
          A_common(2)=A_common(1)
          Z_common(2)=Z_common(1)
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
       IF(IEPSILON_EDFF.EQ.0)THEN
          bfact1=RA_common(1)/GeVm12fm*mN
          bfact2=RA_common(2)/GeVm12fm*mN
       ELSE
          bfact1=(RA_common(1)+aaVal_common(1))/GeVm12fm*mN
          bfact2=(RA_common(2)+aaVal_common(2))/GeVm12fm*mN
       ENDIF
       IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
          ! for the charge form factor
          ! we can integrate b down to zero
          bfact1=bfact1*LOWER_BFactor_Limit
          bfact2=bfact2*LOWER_BFactor_Limit
       ENDIF
       !bbx(1)=bupper
       !bbx(2)=bupper
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
    ! we should choose the lower limit dynamically
    ! b1*x1*mN = Exp(bA(1)) = b1*E_gamma1/gamma1 = b1tilde
    aax(1)=DLOG(bfact1*x1_common)
    ! b2*x2*mN = Exp(bA(2)) = b2*E_gamma2/gamma2 = b2tilde
    aax(2)=DLOG(bfact2*x2_common)
    ! we should also choose the upper limit dynamically
    bbx(1)=MAX(1.5d0,DLOG(0.8d0*RA_common(1)/GeVm12fm*DSQRT(x1_common*mN)))
    bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common(2)/GeVm12fm*DSQRT(x2_common*mN)))
    IF(force_pnohad1)THEN
       ! we only use PNOHAD=1
       CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
            integral,ind,eval_num)
    ELSE
       CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
            integral,ind,eval_num)
    ENDIF
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    IF(integral.LT.0d0)THEN
       ! try to rescue it by increasing bupper
       iter=1
       DO WHILE(integral.LT.0d0.AND.iter.LE.itermax)
          bbx(1)=bbx(1)*2d0
          bbx(2)=bbx(2)*2d0
          IF(force_pnohad1)THEN
             ! we only use PNOHAD=1
             CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax(1:2),bbx(1:2),2,sub_num(1:2),1,1d-5,&
                  integral,ind,eval_num)
          ELSE
             CALL ROMBERG_ND(PhotonPhotonFlux_AB_WoodsSaxon_fxn,aax,bbx,3,sub_num,1,1d-5,&
                  integral,ind,eval_num)
          ENDIF
          iter=iter+1
       ENDDO
       !bbx(1)=bupper
       !bbx(2)=bupper
       bbx(1)=MAX(1.5d0,DLOG(0.8d0*RA_common(1)/GeVm12fm*DSQRT(x1_common*mN)))
       bbx(2)=MAX(1.5d0,DLOG(0.8d0*RA_common(2)/GeVm12fm*DSQRT(x2_common*mN)))
    ENDIF
    IF(integral.LT.0d0)THEN
       printnum=printnum+1
       IF(printnum.LE.5)THEN
          WRITE(*,*)"WARNING: negative photon <delta b> at (x1,x2)=",x1,x2
          WRITE(*,*)"WARNING: most probably need to increase bupper"
          IF(printnum.EQ.5)WRITE(*,*)"WARNING: Further warning will be suppressed"
       ENDIF
       PhotonPhotonDeltaB_AB_WoodsSaxon_eval=0d0
    ELSE
       PhotonPhotonDeltaB_AB_WoodsSaxon_eval=TWOPI/(x1*x2)*alpha**2*Z1**2*Z2**2*integral
    ENDIF
    RETURN
  END FUNCTION PhotonPhotonDeltaB_AB_WoodsSaxon_eval

  FUNCTION PhotonPhotonFlux_AB_WoodsSaxon_fxn(dim_num,bA)
    IMPLICIT NONE
    REAL(KIND(1d0))::PhotonPhotonFlux_AB_WoodsSaxon_fxn
    INTEGER,INTENT(IN)::dim_num ! should be 3 or 2
    ! 1/0.1973d0 from fm to GeV-1 for b
    ! x1*b1*mN=Exp(bA(1))
    ! x2*b2*mN=Exp(bA(2))
    ! bA(3) = theta_{12}
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::bA
    REAL(KIND(1d0))::b1,b2,b12,costh,pnohad
    REAL(KIND(1d0))::b12GeVm1,pbreak
    REAL(KIND(1d0))::Ngamma1,Ngamma2
    REAL(KIND(1d0))::gamma1_common,gamma2_common ! Lorentz boost factors
    REAL(KIND(1d0))::E1_common,E2_common      ! energies of two photons
    REAL(KIND(1d0))::x1_common,x2_common      ! x1 and x2 of two photons
    REAL(KIND(1d0))::sigNN_inel_common        ! NN inelastic cross section
    REAL(KIND(1d0)),DIMENSION(2)::RA_common, A_common      ! radius of nuclei and atom number of nuclei
    REAL(KIND(1d0)),DIMENSION(2)::Z_common
    REAL(KIND(1d0)),DIMENSION(2)::aaVal_common,wVal_common ! parameters in Woods-Saxon potential
    COMMON/PhotonPhoton_AB_WS/gamma1_common,gamma2_common,E1_common,E2_common,x1_common,x2_common,&
         sigNN_inel_common,RA_common,A_common,Z_common,aaVal_common,wVal_common
    REAL(KIND(1d0)),DIMENSION(2)::RR,aaa
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),EXTERNAL::elliptic_em
    REAL(KIND(1d0))::arg1,arg2
    IF(dim_num.NE.3.AND.dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: PhotonPhotonFlux_AB_WoodsSaxon_fxn is not a three or two dimensional function"
       STOP
    ENDIF
    ! in unit of GeV-1
    ! x1*b1*mN=Exp(bA(1))
    b1=DEXP(bA(1))/x1_common/mN
    ! x2*b2*mN=Exp(bA(2))
    b2=DEXP(bA(2))/x2_common/mN
    IF(dim_num.EQ.3)THEN
       costh=DCOS(bA(3))
       b12=DSQRT(b1**2+b2**2-2d0*b1*b2*costh)
       b12GeVm1=b12 ! in unit of GeV-1
       b12=b12*GeVm12fm ! from GeV-1 to fm

       pnohad=PNOHAD_AB_WoodsSaxon(b12,0d0,RA_common,wVal_common,aaVal_common,&
            A_common,Z_common,sigNN_inel_common)
       IF(neutron_tagging(1).NE.-2.OR.neutron_tagging(2).NE.-2)THEN
          ! multiply the forward neutron tagging probability
          pbreak=Pbreak_AB_WoodsSaxon(neutron_tagging(1),neutron_tagging(2),&
               b12GeVm1,neutron_conj_sum)
          pnohad=pnohad*pbreak
       ENDIF
       IF(CALC_DELTABQ)THEN
          pnohad=pnohad*b12
       ENDIF
    ELSE
       ! dim_num = 2
       ! it must be a case with pnohad=1
       ! and CALC_DELTABQ=.TRUE
       IF(.NOT.CALC_DELTABQ)THEN
          WRITE(*,*)"ERROR: cannot reach here with CALC_DELTABQ=.FALSE."
       ENDIF
       ! pnohad is Integrate[Sqrt[b1^2 + b2^2 - 2 b1*b2*Cos[phi]], {phi, 0, 2 Pi}]
       arg2=4d0*b1*b2/(b1+b2)**2
       IF(b1.NE.b2.AND.arg2.LT.1d0)THEN
          arg1=-4d0*b1*b2/(b1-b2)**2
          ! elliptic_em(m) is EllipticE[m] with m<1
          pnohad=DABS(b1-b2)*elliptic_em(arg1)
          pnohad=pnohad+(b1+b2)*elliptic_em(arg2)
       ELSE
          pnohad=4d0*b1
       ENDIF
       pnohad=pnohad*2d0*GeVm12fm ! convert from GeV-1 to fm
    ENDIF
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
    INTEGER::npoints
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
    IF(GENERATE_PhotonPhotonFlux_GRID)THEN
       npoints=1000
    ELSE
       npoints=100
    ENDIF
    CALL trapezoid_integration(npoints,Lgammagamma_UPC_fxn,&
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
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

  ! <delta b> at W=scale
  FUNCTION DeltaB_UPC_at_W(scale,icoll,iprofile)
    ! icoll: 1 - pp; 2 - pA; 3 - AB
    ! iprofile: 0; P_{NOHAD}=1; 1 - Woods-Saxon; 2 - hard-sphere (does not supported)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::DeltaB_UPC_at_W
    INTEGER,INTENT(IN)::icoll,iprofile
    REAL(KIND(1d0)),INTENT(IN)::scale ! scale=W
    REAL(KIND(1d0))::num,den
    LOGICAL::use_grid_bu
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
       WRITE(*,*)"|    ultraperipheral proton and nuclear collisions (v1.7)     |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    By Hua-Sheng Shao (LPTHE) and David d'Enterria (CERN)    |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"|    Please cite arXiv:2207.03012 [JHEP 09 (2022) 248]        |"
       WRITE(*,*)"|                                                             |"
       WRITE(*,*)"==============================================================="
       print_banner=.TRUE.
    ENDIF
    ! numerator
    ! PhotonPhotonFlux*Delta b in unit of fm
    num=LgammagammaDeltaB_UPC(scale,icoll,iprofile)
    ! denominator
    ! PhotonPhotonFlux
    use_grid_bu=GENERATE_PhotonPhotonFlux_GRID
    ! Let us make it to be coherent with DELTAB grid case
    GENERATE_PhotonPhotonFlux_GRID=GENERATE_DELTAB_GRID
    den=Lgammagamma_UPC(scale,icoll,iprofile)
    GENERATE_PhotonPhotonFlux_GRID=use_grid_bu ! recover the original one
    IF(den.LE.0d0.OR.num.EQ.0d0)THEN
       DeltaB_UPC_at_W=0d0
    ELSE
       DeltaB_UPC_at_W=num/den
    ENDIF
    RETURN
  END FUNCTION DeltaB_UPC_at_W

  ! photon-photon Luminosity*Delta B
  FUNCTION LgammagammaDeltaB_UPC(scale,icoll,iprofile)
    ! icoll: 1 - pp; 2 - pA; 3 - AB
    ! iprofile: 0: P_{NOHAD}=1; 1 - Woods-Saxon; 2 - hard-sphere (does not supported)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::LgammagammaDeltaB_UPC
    INTEGER,INTENT(IN)::icoll,iprofile
    REAL(KIND(1d0)),INTENT(IN)::scale ! shat=scale**2
    INTEGER::npoints
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
    IF(GENERATE_DELTAB_GRID)THEN
       npoints=1000
    ELSE
       npoints=100
    ENDIF
    CALL trapezoid_integration(npoints,LgammagammaDeltaB_UPC_fxn,&
         log1oxmax,LgammagammaDeltaB_UPC)
    RETURN
  END FUNCTION LgammagammaDeltaB_UPC

  FUNCTION LgammagammaDeltaB_UPC_fxn(log1ox)
    IMPLICIT NONE
    REAL(KIND(1d0))::LgammagammaDeltaB_UPC_fxn
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
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_pp(x1,x2,.TRUE.)
       ELSE
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_pp(x1,x2)
       ENDIF
    ELSEIF(collision_type_common.EQ.2)THEN
       ! pA or Ap
       IF(profile_type_common.EQ.1)THEN
          ! Woods-Saxon
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_pA_WoodsSaxon(x1,x2)
       ELSEIF(profile_type_common.EQ.2)THEN
          ! Hard-Sphere
          WRITE(*,*)"ERROR: LgammagammaDeltaB_UPC does not support hard-sphere"
          STOP
       ELSEIF(profile_type_common.EQ.0)THEN
          ! P_{NOHAD}=1
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_pA_WoodsSaxon(x1,x2,.TRUE.)
       ELSE
          WRITE(*,*)"ERROR: do not know the profile type = ",profile_type_common
          STOP
       ENDIF
    ELSEIF(collision_type_common.EQ.3)THEN
       ! AB
       IF(profile_type_common.EQ.1)THEN
          ! Woods-Saxon
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_AB_WoodsSaxon(x1,x2)
       ELSEIF(profile_type_common.EQ.2)THEN
          ! Hard-Sphere
          WRITE(*,*)"ERROR: LgammagammaDeltaB_UPC does not support hard-sphere"
          STOP
       ELSEIF(profile_type_common.EQ.0)THEN
          ! P_{NOHAD}=1
          LgammagammaDeltaB_UPC_fxn=PhotonPhotonDeltaB_AB_WoodsSaxon(x1,x2,.TRUE.)
       ELSE
          WRITE(*,*)"ERROR: do not know the profile type = ",profile_type_common
          STOP
       ENDIF
    ELSE
       WRITE(*,*)"ERROR: do not know the collision type = ",collision_type_common
       STOP
    ENDIF
    RETURN
  END FUNCTION LgammagammaDeltaB_UPC_fxn

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                        !
!   Tagging forward neutrons                                             !
!                                                                        !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  FUNCTION Pbreak_AB_WoodsSaxon(in1,in2,Db,CONJUGATESUM)
    ! It gives us the probability Pbreak for Coulomb breakup of nuclei in nuclear-nulcear collisions
    ! Db is in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    ! in1, in2 = -2(0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)
    ! CONJUGATESUM is an optional logical variable (default is .FALSE.)
    ! If it is true, for in1=!=in2, it will sum (in1,in2)+(in2,in1)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::Pbreak_AB_WoodsSaxon
    INTEGER,INTENT(IN)::in1,in2
    REAL(KIND(1d0)),INTENT(IN)::Db
    LOGICAL,INTENT(IN),OPTIONAL::CONJUGATESUM ! If it is true, for in1=!=in2, it will sum (in1,in2)+(in2,in1) [default .FALSE.]
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),SAVE::gamma1_save,gamma2_save ! Lorentz boost factors in the lab frame
    REAL(KIND(1d0)),DIMENSION(2),SAVE::RA_save ! radius of nuclei
    INTEGER,DIMENSION(2),SAVE::A_save,Z_save ! atomic mass and atomic number of nuclei
    REAL(KIND(1d0)),DIMENSION(2),SAVE::aaVal_save,wVal_save ! parameters in Woods-Saxon potential
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0))::Z1,Z2,A1,A2
    INTEGER::i
    CHARACTER(len=7)::Aname1,Aname2
    LOGICAL::conj_sum
    REAL(KIND(1d0))::Pbreak_conj
    REAL(KIND(1d0))::PXn_LO_proj1,P1n_LO_proj1,P2n_LO_proj1,P3n_LO_proj1,P4n_LO_proj1
    REAL(KIND(1d0))::PXn_LO_proj2,P1n_LO_proj2,P2n_LO_proj2,P3n_LO_proj2,P4n_LO_proj2
    REAL(KIND(1d0))::PXn_proj1,P0n_proj1,P1n_proj1,P2n_proj1,P3n_proj1,P4n_proj1
    REAL(KIND(1d0))::PXn_proj2,P0n_proj2,P1n_proj2,P2n_proj2,P3n_proj2,P4n_proj2
    
    IF(in1.LT.-2.OR.in1.GT.4)THEN
       WRITE(*,*)"ERROR: only the following options for in1: -2 (0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)"
       STOP
    ENDIF
    IF(in2.LT.-1.OR.in2.GT.4)THEN
       WRITE(*,*)"ERROR: only the following options for in2: -2 (0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)"
       STOP
    ENDIF

    IF(in1.EQ.-2.AND.in2.EQ.-2)THEN
       Pbreak_AB_WoodsSaxon=1d0
       RETURN
    ENDIF
    
    conj_sum=.FALSE.
    IF(PRESENT(CONJUGATESUM))THEN
       conj_sum=CONJUGATESUM
    ENDIF
    IF(in1.EQ.in2)conj_sum=.FALSE.
    
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
       gamma1_save=ebeam_PN(1)/mN
       gamma2_save=ebeam_PN(2)/mN
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
       CALL GetNuclearInfo(Aname1,A1,Z1,RA_save(1),aaVal_save(1),wVal_save(1))
       Z_save(1)=INT(Z1)
       A_save(1)=INT(A1)
       IF(nuclearA_beam2.NE.nuclearA_beam1.OR.nuclearZ_beam1.NE.nuclearZ_beam2)THEN
          Aname2=GetASymbol(nuclearA_beam2,nuclearZ_beam2)
          CALL GetNuclearInfo(Aname2,A2,Z2,RA_save(2),aaVal_save(2),wVal_save(2))
          Z_save(2)=INT(Z2)
          A_save(2)=INT(A2)
       ELSE
          Aname2=Aname1
          A_save(2)=A_save(1)
          Z_save(2)=Z_save(1)
          Z2=Z1
          RA_save(2)=RA_save(1)
          aaVal_save(2)=aaVal_save(1)
          wVal_save(2)=wVal_save(1)
       ENDIF
       ! 0.1973 is from fm to GeV-1
       DO i=1,2
          RA_save(i)=RA_save(i)/GeVm12fm
          aaVal_save(i)=aaVal_save(i)/GeVm12fm
       ENDDO
       
       init=1
    ENDIF

    Pbreak_AB_WoodsSaxon=1d0
    Pbreak_conj=1d0
    ! For in2, projectile is beam 1 and target is beam 2
    ! P^{Xn,(1)}
    IF(in2.NE.-2.OR.conj_sum)THEN
       PXn_LO_proj1=PbreakXn_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
            RA_save(1),wVal_save(1),aaVal_save(1),1d-3,0d0,1)
       PXn_LO_proj1=PXn_LO_proj1*Z_save(1)**2*alpha
       P0n_proj1=DEXP(-PXn_LO_proj1)
       IF(in2.EQ.-1.OR.(conj_sum.AND.in1.EQ.-1))THEN
          ! Xn
          PXn_proj1=MAX(1d0-P0n_proj1,0d0)
          IF(in2.EQ.-1)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*PXn_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.-1)THEN
             Pbreak_conj=Pbreak_conj*PXn_proj1
          ENDIF
       ENDIF
       IF(in2.EQ.0.OR.(conj_sum.AND.in1.EQ.0))THEN
          ! 0n
          IF(in2.EQ.0)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P0n_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.0)THEN
             Pbreak_conj=Pbreak_conj*P0n_proj1
          ENDIF
       ENDIF
       IF(in2.EQ.1.OR.(conj_sum.AND.in1.EQ.1))THEN
          ! 1n
          P1n_LO_proj1=Pbreak1n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1d-3,4d-2,1)
          P1n_LO_proj1=P1n_LO_proj1*Z_save(1)**2*alpha
          P1n_proj1=P1n_LO_proj1*P0n_proj1
          IF(in2.EQ.1)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P1n_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.1)THEN
             Pbreak_conj=Pbreak_conj*P1n_proj1
          ENDIF
       ENDIF
       IF(in2.EQ.2.OR.(conj_sum.AND.in1.EQ.2))THEN
          ! 2n
          P1n_LO_proj1=Pbreak1n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1d-3,4d-2,1)
          P1n_LO_proj1=P1n_LO_proj1*Z_save(1)**2*alpha
          P2n_LO_proj1=Pbreak2n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1.3d-2,1.4d-1,1)
          P2n_LO_proj1=P2n_LO_proj1*Z_save(1)**2*alpha
          P2n_proj1=(P1n_LO_proj1**2/2d0+P2n_LO_proj1)*P0n_proj1
          IF(in2.EQ.2)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P2n_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.2)THEN
             Pbreak_conj=Pbreak_conj*P2n_proj1
          ENDIF
       ENDIF
       IF(in2.EQ.3.OR.(conj_sum.AND.in1.EQ.3))THEN
          ! 3n
          P1n_LO_proj1=Pbreak1n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1d-3,4d-2,1)
          P1n_LO_proj1=P1n_LO_proj1*Z_save(1)**2*alpha
          P2n_LO_proj1=Pbreak2n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1.3d-2,1.4d-1,1)
          P2n_LO_proj1=P2n_LO_proj1*Z_save(1)**2*alpha
          P3n_LO_proj1=Pbreak3n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),2.2d-2,1.4d-1,1)
          P3n_LO_proj1=P3n_LO_proj1*Z_save(1)**2*alpha
          P3n_proj1=(P1n_LO_proj1**3/6d0+P1n_LO_proj1*P2n_LO_proj1+P3n_LO_proj1)*P0n_proj1
          IF(in2.EQ.3)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P3n_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.3)THEN
             Pbreak_conj=Pbreak_conj*P3n_proj1
          ENDIF
       ENDIF
       IF(in2.EQ.4.OR.(conj_sum.AND.in1.EQ.4))THEN
          ! 4n
          P1n_LO_proj1=Pbreak1n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1d-3,4d-2,1)
          P1n_LO_proj1=P1n_LO_proj1*Z_save(1)**2*alpha
          P2n_LO_proj1=Pbreak2n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),1.3d-2,1.4d-1,1)
          P2n_LO_proj1=P2n_LO_proj1*Z_save(1)**2*alpha
          P3n_LO_proj1=Pbreak3n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),2.2d-2,1.4d-1,1)
          P3n_LO_proj1=P3n_LO_proj1*Z_save(1)**2*alpha
          P4n_LO_proj1=Pbreak4n_LO_AB_WS(Db,A_save(2),Z_save(2),gamma2_save,gamma1_save,&
               RA_save(1),wVal_save(1),aaVal_save(1),3.2d-2,1.4d-1,1)
          P4n_LO_proj1=P4n_LO_proj1*Z_save(1)**2*alpha
          P4n_proj1=(P1n_LO_proj1**4/24d0+P1n_LO_proj1**2*P2n_LO_proj1/2d0&
               +P1n_LO_proj1*P3n_LO_proj1+P2n_LO_proj1**2/2d0+P4n_LO_proj1)*P0n_proj1
          IF(in2.EQ.4)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P4n_proj1
          ENDIF
          IF(conj_sum.AND.in1.EQ.4)THEN
             Pbreak_conj=Pbreak_conj*P4n_proj1
          ENDIF
       ENDIF
    ENDIF
    ! For in1, projectile is beam 2 and target is beam 1
    IF(in1.NE.-2.OR.conj_sum)THEN
       PXn_LO_proj2=PbreakXn_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
            RA_save(2),wVal_save(2),aaVal_save(2),1d-3,0d0,2)
       PXn_LO_proj2=PXn_LO_proj2*Z_save(2)**2*alpha
       P0n_proj2=DEXP(-PXn_LO_proj2)
       IF(in1.EQ.-1.OR.(conj_sum.AND.in2.EQ.-1))THEN
          ! Xn
          PXn_proj2=MAX(1d0-P0n_proj2,0d0)
          IF(in1.EQ.-1)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*PXn_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.-1)THEN
             Pbreak_conj=Pbreak_conj*PXn_proj2
          ENDIF
       ENDIF
       IF(in1.EQ.0.OR.(conj_sum.AND.in2.EQ.0))THEN
          ! 0n
          IF(in1.EQ.0)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P0n_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.0)THEN
             Pbreak_conj=Pbreak_conj*P0n_proj2
          ENDIF
       ENDIF
       IF(in1.EQ.1.OR.(conj_sum.AND.in2.EQ.1))THEN
          ! 1n
          P1n_LO_proj2=Pbreak1n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1d-3,4d-2,2)
          P1n_LO_proj2=P1n_LO_proj2*Z_save(2)**2*alpha
          P1n_proj2=P1n_LO_proj2*P0n_proj2
          IF(in1.EQ.1)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P1n_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.1)THEN
             Pbreak_conj=Pbreak_conj*P1n_proj2
          ENDIF
       ENDIF
       IF(in1.EQ.2.OR.(conj_sum.AND.in2.EQ.2))THEN
          ! 2n
          P1n_LO_proj2=Pbreak1n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1d-3,4d-2,2)
          P1n_LO_proj2=P1n_LO_proj2*Z_save(2)**2*alpha
          P2n_LO_proj2=Pbreak2n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1.3d-2,1.4d-1,2)
          P2n_LO_proj2=P2n_LO_proj2*Z_save(2)**2*alpha
          P2n_proj2=(P1n_LO_proj2**2/2d0+P2n_LO_proj2)*P0n_proj2
          IF(in1.EQ.2)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P2n_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.2)THEN
             Pbreak_conj=Pbreak_conj*P2n_proj2
          ENDIF
       ENDIF
       IF(in1.EQ.3.OR.(conj_sum.AND.in2.EQ.3))THEN
          ! 3n
          P1n_LO_proj2=Pbreak1n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1d-3,4d-2,2)
          P1n_LO_proj2=P1n_LO_proj2*Z_save(2)**2*alpha
          P2n_LO_proj2=Pbreak2n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1.3d-2,1.4d-1,2)
          P2n_LO_proj2=P2n_LO_proj2*Z_save(2)**2*alpha
          P3n_LO_proj2=Pbreak3n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),2.2d-2,1.4d-1,2)
          P3n_LO_proj2=P3n_LO_proj2*Z_save(2)**2*alpha
          P3n_proj2=(P1n_LO_proj2**3/6d0+P1n_LO_proj2*P2n_LO_proj2+P3n_LO_proj2)*P0n_proj2
          IF(in1.EQ.3)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P3n_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.3)THEN
             Pbreak_conj=Pbreak_conj*P3n_proj2
          ENDIF
       ENDIF
       IF(in1.EQ.4.OR.(conj_sum.AND.in2.EQ.4))THEN
          ! 4n
          P1n_LO_proj2=Pbreak1n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1d-3,4d-2,2)
          P1n_LO_proj2=P1n_LO_proj2*Z_save(2)**2*alpha
          P2n_LO_proj2=Pbreak2n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),1.3d-2,1.4d-1,2)
          P2n_LO_proj2=P2n_LO_proj2*Z_save(2)**2*alpha
          P3n_LO_proj2=Pbreak3n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),2.2d-2,1.4d-1,2)
          P3n_LO_proj2=P3n_LO_proj2*Z_save(2)**2*alpha
          P4n_LO_proj2=Pbreak4n_LO_AB_WS(Db,A_save(1),Z_save(1),gamma1_save,gamma2_save,&
               RA_save(2),wVal_save(2),aaVal_save(2),3.2d-2,1.4d-1,2)
          P4n_LO_proj2=P4n_LO_proj2*Z_save(2)**2*alpha
          P4n_proj2=(P1n_LO_proj2**4/24d0+P1n_LO_proj2**2*P2n_LO_proj2/2d0&
               +P1n_LO_proj2*P3n_LO_proj2+P2n_LO_proj2**2/2d0+P4n_LO_proj2)*P0n_proj2
          IF(in1.EQ.4)THEN
             Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon*P4n_proj2
          ENDIF
          IF(conj_sum.AND.in2.EQ.4)THEN
             Pbreak_conj=Pbreak_conj*P4n_proj2
          ENDIF
       ENDIF
    ENDIF
    ! to sum the conjugated one
    IF(conj_sum)THEN
       Pbreak_AB_WoodsSaxon=Pbreak_AB_WoodsSaxon+Pbreak_conj
    ENDIF
    
    RETURN
  END FUNCTION Pbreak_AB_WoodsSaxon

  FUNCTION Pbreak_pA_WoodsSaxon(in2,Db)
    ! It gives us the probability Pbreak for Coulomb breakup of nuclei in proton-ion collisions
    ! Db is in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    ! in2 = -2(0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)
    IMPLICIT NONE
    include 'run90.inc'
    REAL(KIND(1d0))::Pbreak_pA_WoodsSaxon
    INTEGER,INTENT(IN)::in2
    REAL(KIND(1d0)),INTENT(IN)::Db
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0)),PARAMETER::Rproton=0.877d0 ! the charge radius of proton (in fm)
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),SAVE::gamma1_save,gamma2_save ! Lorentz boost factors
    REAL(KIND(1d0)),SAVE::RA_save ! radius of nuclei
    INTEGER,SAVE::A_save,Z_save ! atomic mass and atomic number of nuclei
    REAL(KIND(1d0)),SAVE::aaVal_save,wVal_save ! parameters in Woods-Saxon potential
    REAL(KIND(1d0))::alpha
    INTEGER::init=0
    SAVE init,alpha
    REAL(KIND(1d0))::Z1,Z2,A1,A2
    CHARACTER(len=7)::Aname
    REAL(KIND(1d0))::PXn_LO_proj1,P1n_LO_proj1,P2n_LO_proj1,P3n_LO_proj1,P4n_LO_proj1
    REAL(KIND(1d0))::PXn_proj1,P0n_proj1,P1n_proj1,P2n_proj1,P3n_proj1,P4n_proj1

    IF(in2.LT.-1.OR.in2.GT.4)THEN
       WRITE(*,*)"ERROR: only the following options for in2: -2 (0n+Xn), -1 (Xn), 0 (0n), 1 (1n), 2 (2n), 3 (3n), 4 (4n)"
       STOP
    ENDIF

    IF(in2.EQ.-2)THEN
       Pbreak_pA_WoodsSaxon=1d0
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
          gamma1_save=ebeam_PN(2)/mproton
          gamma2_save=ebeam_PN(1)/mN
       ELSE
          gamma1_save=ebeam_PN(1)/mproton
          gamma2_save=ebeam_PN(2)/mN
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
       CALL GetNuclearInfo(Aname,A2,Z2,RA_save,aaVal_save,wVal_save)
       A_save=INT(A2)
       Z_save=INT(Z2)
       ! 0.1973 is from fm to GeV-1
       RA_save=RA_save/GeVm12fm
       aaVal_save=aaVal_save/GeVm12fm
       init=1
    ENDIF

    Pbreak_pA_WoodsSaxon=1d0
    PXn_LO_proj1=PbreakXn_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1d-3,0d0)
    PXn_LO_proj1=PXn_LO_proj1*alpha
    P0n_proj1=DEXP(-PXn_LO_proj1)
    IF(in2.EQ.-1)THEN
       ! Xn
       PXn_proj1=MAX(1d0-P0n_proj1,0d0)
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*PXn_proj1
    ELSEIF(in2.EQ.0)THEN
       ! 0n
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*P0n_proj1
    ELSEIF(in2.EQ.1)THEN
       ! 1n
       P1n_LO_proj1=Pbreak1n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1d-3,4d-2)
       P1n_LO_proj1=P1n_LO_proj1*alpha
       P1n_proj1=P1n_LO_proj1*P0n_proj1
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*P1n_proj1
    ELSEIF(in2.EQ.2)THEN
       ! 2n
       P1n_LO_proj1=Pbreak1n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1d-3,4d-2)
       P1n_LO_proj1=P1n_LO_proj1*alpha
       P2n_LO_proj1=Pbreak2n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1.3d-2,1.4d-1)
       P2n_LO_proj1=P2n_LO_proj1*alpha
       P2n_proj1=(P1n_LO_proj1**2/2d0+P2n_LO_proj1)*P0n_proj1
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*P2n_proj1
    ELSEIF(in2.EQ.3)THEN
       ! 3n
       P1n_LO_proj1=Pbreak1n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1d-3,4d-2)
       P1n_LO_proj1=P1n_LO_proj1*alpha
       P2n_LO_proj1=Pbreak2n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1.3d-2,1.4d-1)
       P2n_LO_proj1=P2n_LO_proj1*alpha
       P3n_LO_proj1=Pbreak3n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,2.2d-2,1.4d-1)
       P3n_LO_proj1=P3n_LO_proj1*alpha
       P3n_proj1=(P1n_LO_proj1**3/6d0+P1n_LO_proj1*P2n_LO_proj1+P3n_LO_proj1)*P0n_proj1
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*P3n_proj1
    ELSEIF(in2.EQ.4)THEN
       ! 4n
       P1n_LO_proj1=Pbreak1n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1d-3,4d-2)
       P1n_LO_proj1=P1n_LO_proj1*alpha
       P2n_LO_proj1=Pbreak2n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,1.3d-2,1.4d-1)
       P2n_LO_proj1=P2n_LO_proj1*alpha
       P3n_LO_proj1=Pbreak3n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,2.2d-2,1.4d-1)
       P3n_LO_proj1=P3n_LO_proj1*alpha
       P4n_LO_proj1=Pbreak4n_LO_pA(Db,A_save,Z_save,gamma2_save,gamma1_save,3.2d-2,1.4d-1)
       P4n_LO_proj1=P4n_LO_proj1*alpha
       P4n_proj1=(P1n_LO_proj1**4/24d0+P1n_LO_proj1**2*P2n_LO_proj1/2d0&
            +P1n_LO_proj1*P3n_LO_proj1+P2n_LO_proj1**2/2d0+P4n_LO_proj1)*P0n_proj1
       Pbreak_pA_WoodsSaxon=Pbreak_pA_WoodsSaxon*P4n_proj1
    ENDIF

    RETURN
  END FUNCTION Pbreak_pA_WoodsSaxon

  FUNCTION PbreakXn_LO_AB_WS(Db,A_target,Z_target,gamma_target,&
       gamma_proj,RR_proj,w_proj,aa_proj,Egamma_min_target,Egamma_max_target,&
       ibeam_proj,integ_method)
    ! It gives us the lowest-order probability P^{Xn,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ibeam_proj
    REAL(KIND(1d0))::PbreakXn_LO_AB_WS
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::RR_proj,w_proj,aa_proj ! nucleu parameters for projectile
    
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 1 MeV (0.001 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/PbreakXn_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/PbreakXn_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=4
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_proj_save,w_proj_save,aa_proj_save
    REAL(KIND(1d0)),DIMENSION(2)::gamma_proj_save,gamma_target_save
    INTEGER,DIMENSION(2)::A_target_save,Z_target_save
    SAVE R_proj_save,w_proj_save,aa_proj_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save
    
    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D
    
    IF(ibeam_proj.GT.2.OR.ibeam_proj.LT.1)THEN
       WRITE(*,*)"Error: ibeam_proj=/=1,2 in PbreakXn_LO_AB_WS"
       STOP
    ENDIF
    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mN)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)
    
    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    ibeam_proj_common=ibeam_proj
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj
    R_proj_common=RR_proj
    w_proj_common=w_proj
    aa_proj_common=aa_proj
  
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init(ibeam_proj).EQ.0.OR.(integ_method6.LT.0))THEN
       R_proj_save(ibeam_proj)=RR_proj
       w_proj_save(ibeam_proj)=w_proj
       aa_proj_save(ibeam_proj)=aa_proj
       gamma_proj_save(ibeam_proj)=gamma_proj
       gamma_target_save(ibeam_proj)=gamma_target
       A_target_save(ibeam_proj)=A_target
       Z_target_save(ibeam_proj)=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             PbreakXn_LO_AB_WS=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,PbreakXn_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(PbreakXn_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints) 
             ENDIF
             PbreakXn_LO_AB_WS=integral
             RETURN
          ENDIF
       ENDIF
       
       rescaling_bmax_save(ibeam_proj)=MAX(db,bmaxoR*RR_proj)
       
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NXA))
       ENDIF
       
       dbb=9d0/DBLE(NBSEG)
       IF(init(3-ibeam_proj).EQ.1.AND.&
            rescaling_bmax_save(ibeam_proj).EQ.rescaling_bmax_save(3-ibeam_proj).AND.&
            R_proj_save(ibeam_proj).EQ.R_proj_save(3-ibeam_proj).AND.&
            w_proj_save(ibeam_proj).EQ.w_proj_save(3-ibeam_proj).AND.&
            aa_proj_save(ibeam_proj).EQ.aa_proj_save(3-ibeam_proj).AND.&
            gamma_proj_save(ibeam_proj).EQ.gamma_proj_save(3-ibeam_proj).AND.&
            gamma_target_save(ibeam_proj).EQ.gamma_target_save(3-ibeam_proj).AND.&
            A_target_save(ibeam_proj).EQ.A_target_save(3-ibeam_proj).AND.&
            Z_target_save(ibeam_proj).EQ.Z_target_save(3-ibeam_proj))THEN
          DO k=1,NXA
             XA(ibeam_proj,k)=XA(3-ibeam_proj,k)
             YA(ibeam_proj,k)=YA(3-ibeam_proj,k)
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of LO Pbreak(Xn) from charge form factor of beam_proj=",ibeam_proj
          WRITE(*,*)"INFO: it will take a few seconds !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1
                XA(ibeam_proj,NXA-k+1)=(10d0**(-n-1))*(one&
                     +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save(ibeam_proj)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in PbreakXn_LO_AB_WS"
             STOP
          ENDIF
          XA(ibeam_proj,1)=0d0
          DO I=1,NXA
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam_proj,I).EQ.0d0)THEN
                YA(ibeam_proj,I)=0d0
                CYCLE
             ENDIF

             npoints=30000
             b_proj_common=XA(ibeam_proj,I)
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,PbreakXn_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(PbreakXn_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             YA(ibeam_proj,I)=integral
          ENDDO
       ENDIF
       init(ibeam_proj)=1
    ENDIF
  
    IF(R_proj_save(ibeam_proj).NE.RR_proj&
         .OR.w_proj_save(ibeam_proj).NE.w_proj&
         .OR.aa_proj_save(ibeam_proj).NE.aa_proj)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in PbreakXn_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_proj_save(ibeam_proj),&
            w_proj_save(ibeam_proj),aa_proj_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR_proj,w_proj,aa_proj
       STOP
    ENDIF
    IF(gamma_proj_save(ibeam_proj).NE.gamma_proj&
         .OR.gamma_target_save(ibeam_proj).NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in PbreakXn_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save(ibeam_proj),&
            gamma_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save(ibeam_proj).NE.A_target&
         .OR.Z_target_save(ibeam_proj).NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in PbreakXn_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save(ibeam_proj),&
            Z_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF
  
    IF(db.GT.rescaling_bmax_save(ibeam_proj).OR.db.LE.0d0)THEN
       PbreakXn_LO_AB_WS=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save(ibeam_proj),1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam_proj,k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam_proj,k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam_proj,i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam_proj,K+I)
          YD2_1D(I)=YA(ibeam_proj,K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       PbreakXn_LO_AB_WS=integral
       
    ENDIF
    RETURN
  END FUNCTION PbreakXn_LO_AB_WS

  FUNCTION PbreakXn_LO_AB_WS_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::PbreakXn_LO_AB_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/PbreakXn_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/PbreakXn_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    
    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       PbreakXn_LO_AB_WS_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mN*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_WS(b_proj_common,Ega_lab,&
            gamma_proj_common,R_proj_common,w_proj_common,aa_proj_common,&
            3d0,0.7d0,ibeam_proj_common)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    PbreakXn_LO_AB_WS_fxn=GAMMAABS(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    PbreakXn_LO_AB_WS_fxn=PbreakXn_LO_AB_WS_fxn*CFF
    
    RETURN
  END FUNCTION PbreakXn_LO_AB_WS_fxn

  FUNCTION PbreakXn_LO_pA(Db,A_target,Z_target,gamma_target,&
       gamma_proj,Egamma_min_target,Egamma_max_target,&
       integ_method)
    ! It gives us the lowest-order probability P^{Xn,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::PbreakXn_LO_pA
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 1 MeV (0.001 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    
    INTEGER::A_target_common,Z_target_common
    COMMON/PbreakXn_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/PbreakXn_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    INTEGER::init=0
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::RA=35d0 ! in unit of GeV-1
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=5
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0))::gamma_proj_save,gamma_target_save
    INTEGER::A_target_save,Z_target_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save

    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D

    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mproton)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)

    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj

    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init.EQ.0.OR.integ_method6.LT.0)THEN
       gamma_proj_save=gamma_proj
       gamma_target_save=gamma_target
       A_target_save=A_target
       Z_target_save=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             PbreakXn_LO_pA=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
		Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,PbreakXn_LO_pA_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(PbreakXn_LO_pA_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             PbreakXn_LO_pA=integral
             RETURN
          ENDIF
       ENDIF

       rescaling_bmax_save=MAX(db,bmaxoR*RA)

       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF

       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NXA))
       ENDIF

       dbb=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of LO Pbreak(Xn) from charge form factor of pA"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1
             XA(NXA-k+1)=(10d0**(-n-1))*(one&
                  +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in PbreakXn_LO_pA"
          STOP
       ENDIF
       XA(1)=0d0
       DO I=1,NXA
          IF(I.EQ.1)THEN
             CALL progress(INT(I*50d0/NXA),50,.TRUE.)
          ELSE
             CALL progress(INT(I*50d0/NXA),50)
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             YA(I)=0d0
             CYCLE
          ENDIF
          npoints=30000
          b_proj_common=XA(I)
          IF(Egamma_max_target.LT.0d0)THEN
             ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0       
             Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
             Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
             xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
             log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
          ENDIF
          IF(log1oxgamma_min.EQ.0d0)THEN
             CALL trapezoid_integration(npoints,PbreakXn_LO_pA_fxn,&
                  log1oxgamma_max,integral)
          ELSE
             CALL simpson(PbreakXn_LO_pA_fxn,log1oxgamma_min,&
                  log1oxgamma_max,integral,npoints)
          ENDIF
          YA(I)=integral
       ENDDO
       init=1
    ENDIF

    IF(gamma_proj_save.NE.gamma_proj&
         .OR.gamma_target_save.NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in PbreakXn_LO_pA"
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save,&
            gamma_target_save
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save.NE.A_target&
         .OR.Z_target_save.NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in PbreakXn_LO_pA"
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save,&
            Z_target_save
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF

    IF(db.GT.rescaling_bmax_save.OR.db.LE.0d0)THEN
       PbreakXn_LO_pA=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
          YD2_1D(I)=YA(K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       PbreakXn_LO_pA=integral

    ENDIF
    RETURN
  END FUNCTION PbreakXn_LO_pA

  FUNCTION PbreakXn_LO_pA_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)                                                            
    ! where xgamma is the momentum fraction in the lab frame                                 
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::PbreakXn_LO_pA_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::A_target_common,Z_target_common
    COMMON/PbreakXn_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/PbreakXn_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       PbreakXn_LO_pA_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mproton*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_proton(b_proj_common,Ega_lab,&
            gamma_proj_common,proton_FF_correction,.FALSE.)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    PbreakXn_LO_pA_fxn=GAMMAABS(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    PbreakXn_LO_pA_fxn=PbreakXn_LO_pA_fxn*CFF

    RETURN
  END FUNCTION PbreakXn_LO_pA_fxn

  FUNCTION Pbreak1n_LO_AB_WS(Db,A_target,Z_target,gamma_target,&
       gamma_proj,RR_proj,w_proj,aa_proj,Egamma_min_target,Egamma_max_target,&
       ibeam_proj,integ_method)
    ! It gives us the lowest-order probability P^{1n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ibeam_proj
    REAL(KIND(1d0))::Pbreak1n_LO_AB_WS
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::RR_proj,w_proj,aa_proj ! nucleu parameters for projectile
    
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 1 MeV (0.001 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 40 MeV (0.04 GeV).
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak1n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak1n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=4
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_proj_save,w_proj_save,aa_proj_save
    REAL(KIND(1d0)),DIMENSION(2)::gamma_proj_save,gamma_target_save
    INTEGER,DIMENSION(2)::A_target_save,Z_target_save
    SAVE R_proj_save,w_proj_save,aa_proj_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save
    
    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D
    
    IF(ibeam_proj.GT.2.OR.ibeam_proj.LT.1)THEN
       WRITE(*,*)"Error: ibeam_proj=/=1,2 in Pbreak1n_LO_AB_WS"
       STOP
    ENDIF
    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mN)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)
    
    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    ibeam_proj_common=ibeam_proj
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj
    R_proj_common=RR_proj
    w_proj_common=w_proj
    aa_proj_common=aa_proj
  
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init(ibeam_proj).EQ.0.OR.(integ_method6.LT.0))THEN
       R_proj_save(ibeam_proj)=RR_proj
       w_proj_save(ibeam_proj)=w_proj
       aa_proj_save(ibeam_proj)=aa_proj
       gamma_proj_save(ibeam_proj)=gamma_proj
       gamma_target_save(ibeam_proj)=gamma_target
       A_target_save(ibeam_proj)=A_target
       Z_target_save(ibeam_proj)=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak1n_LO_AB_WS=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak1n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak1n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints) 
             ENDIF
             Pbreak1n_LO_AB_WS=integral
             RETURN
          ENDIF
       ENDIF
       
       rescaling_bmax_save(ibeam_proj)=MAX(db,bmaxoR*RR_proj)
       
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NXA))
       ENDIF
       
       dbb=9d0/DBLE(NBSEG)
       IF(init(3-ibeam_proj).EQ.1.AND.&
            rescaling_bmax_save(ibeam_proj).EQ.rescaling_bmax_save(3-ibeam_proj).AND.&
            R_proj_save(ibeam_proj).EQ.R_proj_save(3-ibeam_proj).AND.&
            w_proj_save(ibeam_proj).EQ.w_proj_save(3-ibeam_proj).AND.&
            aa_proj_save(ibeam_proj).EQ.aa_proj_save(3-ibeam_proj).AND.&
            gamma_proj_save(ibeam_proj).EQ.gamma_proj_save(3-ibeam_proj).AND.&
            gamma_target_save(ibeam_proj).EQ.gamma_target_save(3-ibeam_proj).AND.&
            A_target_save(ibeam_proj).EQ.A_target_save(3-ibeam_proj).AND.&
            Z_target_save(ibeam_proj).EQ.Z_target_save(3-ibeam_proj))THEN
          DO k=1,NXA
             XA(ibeam_proj,k)=XA(3-ibeam_proj,k)
             YA(ibeam_proj,k)=YA(3-ibeam_proj,k)
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of LO Pbreak(1n) from charge form factor of beam_proj=",ibeam_proj
          WRITE(*,*)"INFO: it will take a few seconds !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1
                XA(ibeam_proj,NXA-k+1)=(10d0**(-n-1))*(one&
                     +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save(ibeam_proj)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak1n_LO_AB_WS"
             STOP
          ENDIF
          XA(ibeam_proj,1)=0d0
          DO I=1,NXA
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam_proj,I).EQ.0d0)THEN
                YA(ibeam_proj,I)=0d0
                CYCLE
             ENDIF

             npoints=30000
             b_proj_common=XA(ibeam_proj,I)
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak1n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak1n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             YA(ibeam_proj,I)=integral
          ENDDO
       ENDIF
       init(ibeam_proj)=1
    ENDIF
  
    IF(R_proj_save(ibeam_proj).NE.RR_proj&
         .OR.w_proj_save(ibeam_proj).NE.w_proj&
         .OR.aa_proj_save(ibeam_proj).NE.aa_proj)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in Pbreak1n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_proj_save(ibeam_proj),&
            w_proj_save(ibeam_proj),aa_proj_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR_proj,w_proj,aa_proj
       STOP
    ENDIF
    IF(gamma_proj_save(ibeam_proj).NE.gamma_proj&
         .OR.gamma_target_save(ibeam_proj).NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak1n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save(ibeam_proj),&
            gamma_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save(ibeam_proj).NE.A_target&
         .OR.Z_target_save(ibeam_proj).NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak1n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save(ibeam_proj),&
            Z_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF
  
    IF(db.GT.rescaling_bmax_save(ibeam_proj).OR.db.LE.0d0)THEN
       Pbreak1n_LO_AB_WS=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save(ibeam_proj),1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam_proj,k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam_proj,k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam_proj,i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam_proj,K+I)
          YD2_1D(I)=YA(ibeam_proj,K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak1n_LO_AB_WS=integral
       
    ENDIF
    RETURN
  END FUNCTION Pbreak1n_LO_AB_WS

  FUNCTION Pbreak1n_LO_AB_WS_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak1n_LO_AB_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak1n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak1n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    
    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak1n_LO_AB_WS_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mN*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_WS(b_proj_common,Ega_lab,&
            gamma_proj_common,R_proj_common,w_proj_common,aa_proj_common,&
            3d0,0.7d0,ibeam_proj_common)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak1n_LO_AB_WS_fxn=GAMMATO1N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak1n_LO_AB_WS_fxn=Pbreak1n_LO_AB_WS_fxn*CFF
    
    RETURN
  END FUNCTION Pbreak1n_LO_AB_WS_fxn

    FUNCTION Pbreak1n_LO_pA(Db,A_target,Z_target,gamma_target,&
       gamma_proj,Egamma_min_target,Egamma_max_target,&
       integ_method)
    ! It gives us the lowest-order probability P^{1n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak1n_LO_pA
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 1 MeV (0.001 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 40 MeV (0.04 GeV)
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak1n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak1n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    INTEGER::init=0
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::RA=35d0 ! in unit of GeV-1
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=5
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0))::gamma_proj_save,gamma_target_save
    INTEGER::A_target_save,Z_target_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save

    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D

    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mproton)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)

    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj

    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init.EQ.0.OR.integ_method6.LT.0)THEN
       gamma_proj_save=gamma_proj
       gamma_target_save=gamma_target
       A_target_save=A_target
       Z_target_save=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak1n_LO_pA=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
		Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak1n_LO_pA_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak1n_LO_pA_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             Pbreak1n_LO_pA=integral
             RETURN
          ENDIF
       ENDIF

       rescaling_bmax_save=MAX(db,bmaxoR*RA)

       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF

       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NXA))
       ENDIF

       dbb=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of LO Pbreak(1n) from charge form factor of pA"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1
             XA(NXA-k+1)=(10d0**(-n-1))*(one&
                  +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak1n_LO_pA"
          STOP
       ENDIF
       XA(1)=0d0
       DO I=1,NXA
          IF(I.EQ.1)THEN
             CALL progress(INT(I*50d0/NXA),50,.TRUE.)
          ELSE
             CALL progress(INT(I*50d0/NXA),50)
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             YA(I)=0d0
             CYCLE
          ENDIF
          npoints=30000
          b_proj_common=XA(I)
          IF(Egamma_max_target.LT.0d0)THEN
             ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0       
             Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
             Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
             xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
             log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
          ENDIF
          IF(log1oxgamma_min.EQ.0d0)THEN
             CALL trapezoid_integration(npoints,Pbreak1n_LO_pA_fxn,&
                  log1oxgamma_max,integral)
          ELSE
             CALL simpson(Pbreak1n_LO_pA_fxn,log1oxgamma_min,&
                  log1oxgamma_max,integral,npoints)
          ENDIF
          YA(I)=integral
       ENDDO
       init=1
    ENDIF

    IF(gamma_proj_save.NE.gamma_proj&
         .OR.gamma_target_save.NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak1n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save,&
            gamma_target_save
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save.NE.A_target&
         .OR.Z_target_save.NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak1n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save,&
            Z_target_save
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF

    IF(db.GT.rescaling_bmax_save.OR.db.LE.0d0)THEN
       Pbreak1n_LO_pA=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
          YD2_1D(I)=YA(K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak1n_LO_pA=integral

    ENDIF
    RETURN
  END FUNCTION Pbreak1n_LO_pA

  FUNCTION Pbreak1n_LO_pA_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak1n_LO_pA_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak1n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak1n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak1n_LO_pA_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mproton*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_proton(b_proj_common,Ega_lab,&
            gamma_proj_common,proton_FF_correction,.FALSE.)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak1n_LO_pA_fxn=GAMMATO1N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak1n_LO_pA_fxn=Pbreak1n_LO_pA_fxn*CFF

    RETURN
  END FUNCTION Pbreak1n_LO_pA_fxn

  FUNCTION Pbreak2n_LO_AB_WS(Db,A_target,Z_target,gamma_target,&
       gamma_proj,RR_proj,w_proj,aa_proj,Egamma_min_target,Egamma_max_target,&
       ibeam_proj,integ_method)
    ! It gives us the lowest-order probability P^{2n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ibeam_proj
    REAL(KIND(1d0))::Pbreak2n_LO_AB_WS
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::RR_proj,w_proj,aa_proj ! nucleu parameters for projectile
    
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 13 MeV (0.013 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV).
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak2n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak2n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=4
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_proj_save,w_proj_save,aa_proj_save
    REAL(KIND(1d0)),DIMENSION(2)::gamma_proj_save,gamma_target_save
    INTEGER,DIMENSION(2)::A_target_save,Z_target_save
    SAVE R_proj_save,w_proj_save,aa_proj_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save
    
    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D
    
    IF(ibeam_proj.GT.2.OR.ibeam_proj.LT.1)THEN
       WRITE(*,*)"Error: ibeam_proj=/=1,2 in Pbreak2n_LO_AB_WS"
       STOP
    ENDIF
    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mN)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)
    
    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    ibeam_proj_common=ibeam_proj
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj
    R_proj_common=RR_proj
    w_proj_common=w_proj
    aa_proj_common=aa_proj
  
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init(ibeam_proj).EQ.0.OR.(integ_method6.LT.0))THEN
       R_proj_save(ibeam_proj)=RR_proj
       w_proj_save(ibeam_proj)=w_proj
       aa_proj_save(ibeam_proj)=aa_proj
       gamma_proj_save(ibeam_proj)=gamma_proj
       gamma_target_save(ibeam_proj)=gamma_target
       A_target_save(ibeam_proj)=A_target
       Z_target_save(ibeam_proj)=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak2n_LO_AB_WS=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak2n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak2n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints) 
             ENDIF
             Pbreak2n_LO_AB_WS=integral
             RETURN
          ENDIF
       ENDIF
       
       rescaling_bmax_save(ibeam_proj)=MAX(db,bmaxoR*RR_proj)
       
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NXA))
       ENDIF
       
       dbb=9d0/DBLE(NBSEG)
       IF(init(3-ibeam_proj).EQ.1.AND.&
            rescaling_bmax_save(ibeam_proj).EQ.rescaling_bmax_save(3-ibeam_proj).AND.&
            R_proj_save(ibeam_proj).EQ.R_proj_save(3-ibeam_proj).AND.&
            w_proj_save(ibeam_proj).EQ.w_proj_save(3-ibeam_proj).AND.&
            aa_proj_save(ibeam_proj).EQ.aa_proj_save(3-ibeam_proj).AND.&
            gamma_proj_save(ibeam_proj).EQ.gamma_proj_save(3-ibeam_proj).AND.&
            gamma_target_save(ibeam_proj).EQ.gamma_target_save(3-ibeam_proj).AND.&
            A_target_save(ibeam_proj).EQ.A_target_save(3-ibeam_proj).AND.&
            Z_target_save(ibeam_proj).EQ.Z_target_save(3-ibeam_proj))THEN
          DO k=1,NXA
             XA(ibeam_proj,k)=XA(3-ibeam_proj,k)
             YA(ibeam_proj,k)=YA(3-ibeam_proj,k)
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of LO Pbreak(2n) from charge form factor of beam_proj=",ibeam_proj
          WRITE(*,*)"INFO: it will take a few seconds !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1
                XA(ibeam_proj,NXA-k+1)=(10d0**(-n-1))*(one&
                     +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save(ibeam_proj)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak2n_LO_AB_WS"
             STOP
          ENDIF
          XA(ibeam_proj,1)=0d0
          DO I=1,NXA
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam_proj,I).EQ.0d0)THEN
                YA(ibeam_proj,I)=0d0
                CYCLE
             ENDIF

             npoints=30000
             b_proj_common=XA(ibeam_proj,I)
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak2n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak2n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             YA(ibeam_proj,I)=integral
          ENDDO
       ENDIF
       init(ibeam_proj)=1
    ENDIF
  
    IF(R_proj_save(ibeam_proj).NE.RR_proj&
         .OR.w_proj_save(ibeam_proj).NE.w_proj&
         .OR.aa_proj_save(ibeam_proj).NE.aa_proj)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in Pbreak2n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_proj_save(ibeam_proj),&
            w_proj_save(ibeam_proj),aa_proj_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR_proj,w_proj,aa_proj
       STOP
    ENDIF
    IF(gamma_proj_save(ibeam_proj).NE.gamma_proj&
         .OR.gamma_target_save(ibeam_proj).NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak2n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save(ibeam_proj),&
            gamma_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save(ibeam_proj).NE.A_target&
         .OR.Z_target_save(ibeam_proj).NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak2n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save(ibeam_proj),&
            Z_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF
  
    IF(db.GT.rescaling_bmax_save(ibeam_proj).OR.db.LE.0d0)THEN
       Pbreak2n_LO_AB_WS=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save(ibeam_proj),1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam_proj,k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam_proj,k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam_proj,i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam_proj,K+I)
          YD2_1D(I)=YA(ibeam_proj,K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak2n_LO_AB_WS=integral
       
    ENDIF
    RETURN
  END FUNCTION Pbreak2n_LO_AB_WS

  FUNCTION Pbreak2n_LO_AB_WS_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak2n_LO_AB_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak2n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak2n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    
    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak2n_LO_AB_WS_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mN*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_WS(b_proj_common,Ega_lab,&
            gamma_proj_common,R_proj_common,w_proj_common,aa_proj_common,&
            3d0,0.7d0,ibeam_proj_common)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak2n_LO_AB_WS_fxn=GAMMATO2N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak2n_LO_AB_WS_fxn=Pbreak2n_LO_AB_WS_fxn*CFF
    
    RETURN
  END FUNCTION Pbreak2n_LO_AB_WS_fxn

  FUNCTION Pbreak2n_LO_pA(Db,A_target,Z_target,gamma_target,&
       gamma_proj,Egamma_min_target,Egamma_max_target,&
       integ_method)
    ! It gives us the lowest-order probability P^{2n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak2n_LO_pA
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 14 MeV (0.014 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV)
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak2n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak2n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    INTEGER::init=0
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::RA=35d0 ! in unit of GeV-1
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=5
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0))::gamma_proj_save,gamma_target_save
    INTEGER::A_target_save,Z_target_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save

    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D

    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mproton)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)

    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj

    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init.EQ.0.OR.integ_method6.LT.0)THEN
       gamma_proj_save=gamma_proj
       gamma_target_save=gamma_target
       A_target_save=A_target
       Z_target_save=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak2n_LO_pA=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
		Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak2n_LO_pA_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak2n_LO_pA_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             Pbreak2n_LO_pA=integral
             RETURN
          ENDIF
       ENDIF

       rescaling_bmax_save=MAX(db,bmaxoR*RA)

       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF

       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NXA))
       ENDIF

       dbb=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of LO Pbreak(2n) from charge form factor of pA"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1
             XA(NXA-k+1)=(10d0**(-n-1))*(one&
                  +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak2n_LO_pA"
          STOP
       ENDIF
       XA(1)=0d0
       DO I=1,NXA
          IF(I.EQ.1)THEN
             CALL progress(INT(I*50d0/NXA),50,.TRUE.)
          ELSE
             CALL progress(INT(I*50d0/NXA),50)
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             YA(I)=0d0
             CYCLE
          ENDIF
          npoints=30000
          b_proj_common=XA(I)
          IF(Egamma_max_target.LT.0d0)THEN
             ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0       
             Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
             Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
             xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
             log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
          ENDIF
          IF(log1oxgamma_min.EQ.0d0)THEN
             CALL trapezoid_integration(npoints,Pbreak2n_LO_pA_fxn,&
                  log1oxgamma_max,integral)
          ELSE
             CALL simpson(Pbreak2n_LO_pA_fxn,log1oxgamma_min,&
                  log1oxgamma_max,integral,npoints)
          ENDIF
          YA(I)=integral
       ENDDO
       init=1
    ENDIF

    IF(gamma_proj_save.NE.gamma_proj&
         .OR.gamma_target_save.NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak2n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save,&
            gamma_target_save
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save.NE.A_target&
         .OR.Z_target_save.NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak2n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save,&
            Z_target_save
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF

    IF(db.GT.rescaling_bmax_save.OR.db.LE.0d0)THEN
       Pbreak2n_LO_pA=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
          YD2_1D(I)=YA(K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak2n_LO_pA=integral

    ENDIF
    RETURN
  END FUNCTION Pbreak2n_LO_pA

  FUNCTION Pbreak2n_LO_pA_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak2n_LO_pA_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak2n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak2n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak2n_LO_pA_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mproton*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_proton(b_proj_common,Ega_lab,&
            gamma_proj_common,proton_FF_correction,.FALSE.)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak2n_LO_pA_fxn=GAMMATO2N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak2n_LO_pA_fxn=Pbreak2n_LO_pA_fxn*CFF

    RETURN
  END FUNCTION Pbreak2n_LO_pA_fxn

  FUNCTION Pbreak3n_LO_AB_WS(Db,A_target,Z_target,gamma_target,&
       gamma_proj,RR_proj,w_proj,aa_proj,Egamma_min_target,Egamma_max_target,&
       ibeam_proj,integ_method)
    ! It gives us the lowest-order probability P^{3n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ibeam_proj
    REAL(KIND(1d0))::Pbreak3n_LO_AB_WS
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::RR_proj,w_proj,aa_proj ! nucleu parameters for projectile
    
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 22 MeV (0.022 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV).
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak3n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak3n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=4
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_proj_save,w_proj_save,aa_proj_save
    REAL(KIND(1d0)),DIMENSION(2)::gamma_proj_save,gamma_target_save
    INTEGER,DIMENSION(2)::A_target_save,Z_target_save
    SAVE R_proj_save,w_proj_save,aa_proj_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save
    
    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D
    
    IF(ibeam_proj.GT.2.OR.ibeam_proj.LT.1)THEN
       WRITE(*,*)"Error: ibeam_proj=/=1,2 in Pbreak3n_LO_AB_WS"
       STOP
    ENDIF
    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mN)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)
    
    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    ibeam_proj_common=ibeam_proj
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj
    R_proj_common=RR_proj
    w_proj_common=w_proj
    aa_proj_common=aa_proj
  
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init(ibeam_proj).EQ.0.OR.(integ_method6.LT.0))THEN
       R_proj_save(ibeam_proj)=RR_proj
       w_proj_save(ibeam_proj)=w_proj
       aa_proj_save(ibeam_proj)=aa_proj
       gamma_proj_save(ibeam_proj)=gamma_proj
       gamma_target_save(ibeam_proj)=gamma_target
       A_target_save(ibeam_proj)=A_target
       Z_target_save(ibeam_proj)=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak3n_LO_AB_WS=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak3n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak3n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints) 
             ENDIF
             Pbreak3n_LO_AB_WS=integral
             RETURN
          ENDIF
       ENDIF
       
       rescaling_bmax_save(ibeam_proj)=MAX(db,bmaxoR*RR_proj)
       
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NXA))
       ENDIF
       
       dbb=9d0/DBLE(NBSEG)
       IF(init(3-ibeam_proj).EQ.1.AND.&
            rescaling_bmax_save(ibeam_proj).EQ.rescaling_bmax_save(3-ibeam_proj).AND.&
            R_proj_save(ibeam_proj).EQ.R_proj_save(3-ibeam_proj).AND.&
            w_proj_save(ibeam_proj).EQ.w_proj_save(3-ibeam_proj).AND.&
            aa_proj_save(ibeam_proj).EQ.aa_proj_save(3-ibeam_proj).AND.&
            gamma_proj_save(ibeam_proj).EQ.gamma_proj_save(3-ibeam_proj).AND.&
            gamma_target_save(ibeam_proj).EQ.gamma_target_save(3-ibeam_proj).AND.&
            A_target_save(ibeam_proj).EQ.A_target_save(3-ibeam_proj).AND.&
            Z_target_save(ibeam_proj).EQ.Z_target_save(3-ibeam_proj))THEN
          DO k=1,NXA
             XA(ibeam_proj,k)=XA(3-ibeam_proj,k)
             YA(ibeam_proj,k)=YA(3-ibeam_proj,k)
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of LO Pbreak(3n) from charge form factor of beam_proj=",ibeam_proj
          WRITE(*,*)"INFO: it will take a few seconds !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1
                XA(ibeam_proj,NXA-k+1)=(10d0**(-n-1))*(one&
                     +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save(ibeam_proj)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak3n_LO_AB_WS"
             STOP
          ENDIF
          XA(ibeam_proj,1)=0d0
          DO I=1,NXA
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam_proj,I).EQ.0d0)THEN
                YA(ibeam_proj,I)=0d0
                CYCLE
             ENDIF

             npoints=30000
             b_proj_common=XA(ibeam_proj,I)
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak3n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak3n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             YA(ibeam_proj,I)=integral
          ENDDO
       ENDIF
       init(ibeam_proj)=1
    ENDIF
  
    IF(R_proj_save(ibeam_proj).NE.RR_proj&
         .OR.w_proj_save(ibeam_proj).NE.w_proj&
         .OR.aa_proj_save(ibeam_proj).NE.aa_proj)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in Pbreak3n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_proj_save(ibeam_proj),&
            w_proj_save(ibeam_proj),aa_proj_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR_proj,w_proj,aa_proj
       STOP
    ENDIF
    IF(gamma_proj_save(ibeam_proj).NE.gamma_proj&
         .OR.gamma_target_save(ibeam_proj).NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak3n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save(ibeam_proj),&
            gamma_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save(ibeam_proj).NE.A_target&
         .OR.Z_target_save(ibeam_proj).NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak3n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save(ibeam_proj),&
            Z_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF
  
    IF(db.GT.rescaling_bmax_save(ibeam_proj).OR.db.LE.0d0)THEN
       Pbreak3n_LO_AB_WS=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save(ibeam_proj),1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam_proj,k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam_proj,k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam_proj,i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam_proj,K+I)
          YD2_1D(I)=YA(ibeam_proj,K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak3n_LO_AB_WS=integral
       
    ENDIF
    RETURN
  END FUNCTION Pbreak3n_LO_AB_WS

  FUNCTION Pbreak3n_LO_AB_WS_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak3n_LO_AB_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak3n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak3n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    
    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak3n_LO_AB_WS_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mN*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_WS(b_proj_common,Ega_lab,&
            gamma_proj_common,R_proj_common,w_proj_common,aa_proj_common,&
            3d0,0.7d0,ibeam_proj_common)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak3n_LO_AB_WS_fxn=GAMMATO3N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak3n_LO_AB_WS_fxn=Pbreak3n_LO_AB_WS_fxn*CFF
    
    RETURN
  END FUNCTION Pbreak3n_LO_AB_WS_fxn

  FUNCTION Pbreak3n_LO_pA(Db,A_target,Z_target,gamma_target,&
       gamma_proj,Egamma_min_target,Egamma_max_target,&
       integ_method)
    ! It gives us the lowest-order probability P^{3n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak3n_LO_pA
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 22 MeV (0.022 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV)
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak3n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak3n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    INTEGER::init=0
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::RA=35d0 ! in unit of GeV-1
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=5
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0))::gamma_proj_save,gamma_target_save
    INTEGER::A_target_save,Z_target_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save

    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D

    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mproton)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)

    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj

    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init.EQ.0.OR.integ_method6.LT.0)THEN
       gamma_proj_save=gamma_proj
       gamma_target_save=gamma_target
       A_target_save=A_target
       Z_target_save=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak3n_LO_pA=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
		Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak3n_LO_pA_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak3n_LO_pA_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             Pbreak3n_LO_pA=integral
             RETURN
          ENDIF
       ENDIF

       rescaling_bmax_save=MAX(db,bmaxoR*RA)

       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF

       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NXA))
       ENDIF

       dbb=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of LO Pbreak(3n) from charge form factor of pA"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1
             XA(NXA-k+1)=(10d0**(-n-1))*(one&
                  +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak3n_LO_pA"
          STOP
       ENDIF
       XA(1)=0d0
       DO I=1,NXA
          IF(I.EQ.1)THEN
             CALL progress(INT(I*50d0/NXA),50,.TRUE.)
          ELSE
             CALL progress(INT(I*50d0/NXA),50)
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             YA(I)=0d0
             CYCLE
          ENDIF
          npoints=30000
          b_proj_common=XA(I)
          IF(Egamma_max_target.LT.0d0)THEN
             ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0       
             Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
             Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
             xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
             log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
          ENDIF
          IF(log1oxgamma_min.EQ.0d0)THEN
             CALL trapezoid_integration(npoints,Pbreak3n_LO_pA_fxn,&
                  log1oxgamma_max,integral)
          ELSE
             CALL simpson(Pbreak3n_LO_pA_fxn,log1oxgamma_min,&
                  log1oxgamma_max,integral,npoints)
          ENDIF
          YA(I)=integral
       ENDDO
       init=1
    ENDIF

    IF(gamma_proj_save.NE.gamma_proj&
         .OR.gamma_target_save.NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak3n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save,&
            gamma_target_save
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save.NE.A_target&
         .OR.Z_target_save.NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak3n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save,&
            Z_target_save
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF

    IF(db.GT.rescaling_bmax_save.OR.db.LE.0d0)THEN
       Pbreak3n_LO_pA=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
          YD2_1D(I)=YA(K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak3n_LO_pA=integral

    ENDIF
    RETURN
  END FUNCTION Pbreak3n_LO_pA

  FUNCTION Pbreak3n_LO_pA_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak3n_LO_pA_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak3n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak3n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak3n_LO_pA_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mproton*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_proton(b_proj_common,Ega_lab,&
            gamma_proj_common,proton_FF_correction,.FALSE.)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak3n_LO_pA_fxn=GAMMATO3N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak3n_LO_pA_fxn=Pbreak3n_LO_pA_fxn*CFF

    RETURN
  END FUNCTION Pbreak3n_LO_pA_fxn

  FUNCTION Pbreak4n_LO_AB_WS(Db,A_target,Z_target,gamma_target,&
       gamma_proj,RR_proj,w_proj,aa_proj,Egamma_min_target,Egamma_max_target,&
       ibeam_proj,integ_method)
    ! It gives us the lowest-order probability P^{4n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ibeam_proj
    REAL(KIND(1d0))::Pbreak4n_LO_AB_WS
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::RR_proj,w_proj,aa_proj ! nucleu parameters for projectile
    
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 32 MeV (0.032 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV).
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak4n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak4n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    INTEGER,DIMENSION(2)::init=(/0,0/)
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=4
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    REAL(KIND(1d0)),DIMENSION(2)::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0)),DIMENSION(2)::R_proj_save,w_proj_save,aa_proj_save
    REAL(KIND(1d0)),DIMENSION(2)::gamma_proj_save,gamma_target_save
    INTEGER,DIMENSION(2)::A_target_save,Z_target_save
    SAVE R_proj_save,w_proj_save,aa_proj_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save
    
    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D
    
    IF(ibeam_proj.GT.2.OR.ibeam_proj.LT.1)THEN
       WRITE(*,*)"Error: ibeam_proj=/=1,2 in Pbreak4n_LO_AB_WS"
       STOP
    ENDIF
    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mN)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)
    
    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    ibeam_proj_common=ibeam_proj
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj
    R_proj_common=RR_proj
    w_proj_common=w_proj
    aa_proj_common=aa_proj
  
    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init(ibeam_proj).EQ.0.OR.(integ_method6.LT.0))THEN
       R_proj_save(ibeam_proj)=RR_proj
       w_proj_save(ibeam_proj)=w_proj
       aa_proj_save(ibeam_proj)=aa_proj
       gamma_proj_save(ibeam_proj)=gamma_proj
       gamma_target_save(ibeam_proj)=gamma_target
       A_target_save(ibeam_proj)=A_target
       Z_target_save(ibeam_proj)=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak4n_LO_AB_WS=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak4n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak4n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints) 
             ENDIF
             Pbreak4n_LO_AB_WS=integral
             RETURN
          ENDIF
       ENDIF
       
       rescaling_bmax_save(ibeam_proj)=MAX(db,bmaxoR*RR_proj)
       
       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(2,NXA))
       ENDIF
       
       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(2,NXA))
       ENDIF
       
       dbb=9d0/DBLE(NBSEG)
       IF(init(3-ibeam_proj).EQ.1.AND.&
            rescaling_bmax_save(ibeam_proj).EQ.rescaling_bmax_save(3-ibeam_proj).AND.&
            R_proj_save(ibeam_proj).EQ.R_proj_save(3-ibeam_proj).AND.&
            w_proj_save(ibeam_proj).EQ.w_proj_save(3-ibeam_proj).AND.&
            aa_proj_save(ibeam_proj).EQ.aa_proj_save(3-ibeam_proj).AND.&
            gamma_proj_save(ibeam_proj).EQ.gamma_proj_save(3-ibeam_proj).AND.&
            gamma_target_save(ibeam_proj).EQ.gamma_target_save(3-ibeam_proj).AND.&
            A_target_save(ibeam_proj).EQ.A_target_save(3-ibeam_proj).AND.&
            Z_target_save(ibeam_proj).EQ.Z_target_save(3-ibeam_proj))THEN
          DO k=1,NXA
             XA(ibeam_proj,k)=XA(3-ibeam_proj,k)
             YA(ibeam_proj,k)=YA(3-ibeam_proj,k)
          ENDDO
       ELSE
          WRITE(*,*)"INFO: generate grid of LO Pbreak(4n) from charge form factor of beam_proj=",ibeam_proj
          WRITE(*,*)"INFO: it will take a few seconds !"
          k=0
          DO n=0,nbmax
             ! from 10**(-n-1)*bmax to 10**(-n)*bmax
             DO i=1,NBSEG
                k=NBSEG*n+i
                ! these are b in unit GeV-1
                XA(ibeam_proj,NXA-k+1)=(10d0**(-n-1))*(one&
                     +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save(ibeam_proj)
             ENDDO
          ENDDO
          IF(k+1.NE.NXA)THEN
             WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak4n_LO_AB_WS"
             STOP
          ENDIF
          XA(ibeam_proj,1)=0d0
          DO I=1,NXA
             IF(I.EQ.1)THEN
                CALL progress(INT(I*50d0/NXA),50,.TRUE.)
             ELSE
                CALL progress(INT(I*50d0/NXA),50)
             ENDIF
             IF(XA(ibeam_proj,I).EQ.0d0)THEN
                YA(ibeam_proj,I)=0d0
                CYCLE
             ENDIF

             npoints=30000
             b_proj_common=XA(ibeam_proj,I)
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
                Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mN),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak4n_LO_AB_WS_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak4n_LO_AB_WS_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             YA(ibeam_proj,I)=integral
          ENDDO
       ENDIF
       init(ibeam_proj)=1
    ENDIF
  
    IF(R_proj_save(ibeam_proj).NE.RR_proj&
         .OR.w_proj_save(ibeam_proj).NE.w_proj&
         .OR.aa_proj_save(ibeam_proj).NE.aa_proj)THEN
       WRITE(*,*)"ERROR: RA,wA,aA are not consistent in Pbreak4n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (RA,wA,aA)=",R_proj_save(ibeam_proj),&
            w_proj_save(ibeam_proj),aa_proj_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (RA,wA,aA)=",RR_proj,w_proj,aa_proj
       STOP
    ENDIF
    IF(gamma_proj_save(ibeam_proj).NE.gamma_proj&
         .OR.gamma_target_save(ibeam_proj).NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak4n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save(ibeam_proj),&
            gamma_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save(ibeam_proj).NE.A_target&
         .OR.Z_target_save(ibeam_proj).NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak4n_LO_AB_WS"
       WRITE(*,*)"INFO: ibeam_proj=",ibeam_proj
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save(ibeam_proj),&
            Z_target_save(ibeam_proj)
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF
  
    IF(db.GT.rescaling_bmax_save(ibeam_proj).OR.db.LE.0d0)THEN
       Pbreak4n_LO_AB_WS=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save(ibeam_proj),1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(ibeam_proj,k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(ibeam_proj,k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(ibeam_proj,i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(ibeam_proj,K+I)
          YD2_1D(I)=YA(ibeam_proj,K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak4n_LO_AB_WS=integral
       
    ENDIF
    RETURN
  END FUNCTION Pbreak4n_LO_AB_WS

  FUNCTION Pbreak4n_LO_AB_WS_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak4n_LO_AB_WS_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
    INTEGER::A_target_common,Z_target_common,ibeam_proj_common
    COMMON/Pbreak4n_AB_CFF_WS_INT/A_target_common,Z_target_common,ibeam_proj_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common,R_proj_common,w_proj_common,aa_proj_common
    COMMON/Pbreak4n_AB_CFF_WS_REAL/gamma_target_common,gamma_proj_common,b_proj_common,&
         R_proj_common,w_proj_common,aa_proj_common
    
    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak4n_LO_AB_WS_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mN*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_WS(b_proj_common,Ega_lab,&
            gamma_proj_common,R_proj_common,w_proj_common,aa_proj_common,&
            3d0,0.7d0,ibeam_proj_common)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak4n_LO_AB_WS_fxn=GAMMATO4N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak4n_LO_AB_WS_fxn=Pbreak4n_LO_AB_WS_fxn*CFF
    
    RETURN
  END FUNCTION Pbreak4n_LO_AB_WS_fxn

  FUNCTION Pbreak4n_LO_pA(Db,A_target,Z_target,gamma_target,&
       gamma_proj,Egamma_min_target,Egamma_max_target,&
       integ_method)
    ! It gives us the lowest-order probability P^{4n,(1)}
    ! for Coulomb breakup of a nucleus with Z=1 and alpha=1
    ! b should be written in unit of GeV-1
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak4n_LO_pA
    REAL(KIND(1d0)),INTENT(IN)::Db ! Delta b
    INTEGER,INTENT(IN)::A_target,Z_target ! atomic mass and atomic numbers for the target
    ! note Lorentz gamma should be >= 1
    REAL(KIND(1d0)),INTENT(IN)::gamma_proj ! Lorentz gamma for projectile in lab frame
    REAL(KIND(1d0)),INTENT(IN)::gamma_target ! Lorentz gamma for target in lab frame
    REAL(KIND(1d0)),INTENT(IN)::Egamma_min_target ! Minimal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 32 MeV (0.032 GeV).
    REAL(KIND(1d0)),INTENT(IN)::Egamma_max_target ! Maximal Egamma in the target frame in unit of GeV
                                                  ! A nominal value is 140 MeV (0.14 GeV)
    ! If it is negative, we take Abs(Egamma_max_target)*(gamma_proj_target/Db),
    ! where gamma_proj_target=gamma_proj*gamma_target+sqrt((gamma_proj**2-1)*(gamma_target**2-1))
    ! which is the Lorentz gamma of projectile in the target frame
    ! If it is zero, xgamma upper limit is 1
    INTEGER,INTENT(IN),OPTIONAL::integ_method ! 1: direct trapezoid rule
    ! do not generate the grid but direct integration when integ_method < 0
    INTEGER::integ_method6
    REAL(KIND(1d0)),PARAMETER::one=1d0
    REAL(KIND(1d0))::gamma_proj_target
    REAL(KIND(1d0))::Egamma_min_lab,Egamma_max_lab
    REAL(KIND(1d0))::xgamma_min_lab,xgamma_max_lab
    REAL(KIND(1d0))::log1oxgamma_max, log1oxgamma_min
    REAL(KIND(1d0))::integral
    
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak4n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak4n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    INTEGER::init=0
    INTEGER::NXA,i,j,n,k,l
    SAVE init,NXA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::YA
    SAVE XA,YA
    REAL(KIND(1d0)),PARAMETER::RA=35d0 ! in unit of GeV-1
    REAL(KIND(1d0)),PARAMETER::bmaxoR=1d3
    INTEGER,PARAMETER::NBMAX=5
    ! 0 to 10**(-nbmax)*bmax
    ! 10**(-n-1)*bmax to 10**(-n)*bmax
    INTEGER,PARAMETER::NBSEG=50
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    REAL(KIND(1d0))::rescaling_bmax_save
    REAL(KIND(1d0))::dbb,bb
    SAVE rescaling_bmax_save
    REAL(KIND(1d0))::gamma_proj_save,gamma_target_save
    INTEGER::A_target_save,Z_target_save
    SAVE gamma_proj_save,gamma_target_save
    SAVE A_target_save, Z_target_save

    INTEGER,PARAMETER::n_interp=10
    INTEGER::npoints
    REAL(KIND(1d0)),DIMENSION(n_interp)::XD2_1D
    REAL(KIND(1d0)),DIMENSION(n_interp)::YD2_1D

    IF(gamma_target.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_target < 1"
       STOP
    ENDIF
    IF(gamma_proj.LT.1d0)THEN
       WRITE(*,*)"Error: gamma_proj < 1"
       STOP
    ENDIF
    gamma_proj_target=gamma_proj*gamma_target&
         +dsqrt((gamma_proj**2-one)*(gamma_target**2-one))

    ! lowest gamma energy [GeV] in the lab frame
    Egamma_min_lab=MAX(Egamma_min_target,0d0)/(gamma_target+dsqrt(gamma_target**2-one))
    xgamma_min_lab=Egamma_min_lab/(gamma_proj*mproton)
    log1oxgamma_max=DLOG(1d0/xgamma_min_lab)

    IF(Egamma_max_target.GT.0d0)THEN
       IF(Egamma_max_target.LE.Egamma_min_target)THEN
          WRITE(*,*)"Error: Egamma_max_target <= Egamma_min_target"
          STOP
       ENDIF
       ! highest gamma energy [GeV] in the lab frame if Egamma_max_target > 0
       Egamma_max_lab=Egamma_max_target/(gamma_target+dsqrt(gamma_target**2-one))
       xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
       log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
    ELSE
       log1oxgamma_min=0d0
    ENDIF

    A_target_common=A_target
    Z_target_common=Z_target
    gamma_target_common=gamma_target
    gamma_proj_common=gamma_proj

    integ_method6=1
    IF(PRESENT(integ_method))THEN
       integ_method6=integ_method
    ENDIF

    IF(init.EQ.0.OR.integ_method6.LT.0)THEN
       gamma_proj_save=gamma_proj
       gamma_target_save=gamma_target
       A_target_save=A_target
       Z_target_save=Z_target

       IF(integ_method6.LT.0)THEN
          IF(db.EQ.0d0)THEN
             Pbreak4n_LO_pA=0d0
             RETURN
          ELSE
             npoints=30000
             b_proj_common=db
             IF(Egamma_max_target.LT.0d0)THEN
                ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0
                Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/db
		Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
                xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
                log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
             ENDIF
             IF(log1oxgamma_min.EQ.0d0)THEN
                CALL trapezoid_integration(npoints,Pbreak4n_LO_pA_fxn,&
                     log1oxgamma_max,integral)
             ELSE
                CALL simpson(Pbreak4n_LO_pA_fxn,log1oxgamma_min,&
                     log1oxgamma_max,integral,npoints)
             ENDIF
             Pbreak4n_LO_pA=integral
             RETURN
          ENDIF
       ENDIF

       rescaling_bmax_save=MAX(db,bmaxoR*RA)

       NXA=NBSEG*(nbmax+1)+1
       IF(.NOT.ALLOCATED(XA))THEN
          ALLOCATE(XA(NXA))
       ENDIF

       IF(.NOT.ALLOCATED(YA))THEN
          ALLOCATE(YA(NXA))
       ENDIF

       dbb=9d0/DBLE(NBSEG)
       WRITE(*,*)"INFO: generate grid of LO Pbreak(4n) from charge form factor of pA"
       WRITE(*,*)"INFO: it will take a few seconds !"
       k=0
       DO n=0,nbmax
          ! from 10**(-n-1)*bmax to 10**(-n)*bmax
          DO i=1,NBSEG
             k=NBSEG*n+i
             ! these are b in unit GeV-1
             XA(NXA-k+1)=(10d0**(-n-1))*(one&
                  +DBLE(NBSEG+1-i)*dbb)*rescaling_bmax_save
          ENDDO
       ENDDO
       IF(k+1.NE.NXA)THEN
          WRITE(*,*)"ERROR: mismatching k+1 and NXA in Pbreak4n_LO_pA"
          STOP
       ENDIF
       XA(1)=0d0
       DO I=1,NXA
          IF(I.EQ.1)THEN
             CALL progress(INT(I*50d0/NXA),50,.TRUE.)
          ELSE
             CALL progress(INT(I*50d0/NXA),50)
          ENDIF
          IF(XA(I).EQ.0d0)THEN
             YA(I)=0d0
             CYCLE
          ENDIF
          npoints=30000
          b_proj_common=XA(I)
          IF(Egamma_max_target.LT.0d0)THEN
             ! highest gamma energy [GeV] in the lab frame if Egamma_max_target < 0       
             Egamma_max_lab=DABS(Egamma_max_target)*gamma_proj_target/b_proj_common
             Egamma_max_lab=Egamma_max_lab/(gamma_target+dsqrt(gamma_target**2-one))
             xgamma_max_lab=MIN(Egamma_max_lab/(gamma_proj*mproton),1d0)
             log1oxgamma_min=DLOG(1d0/xgamma_max_lab)
          ENDIF
          IF(log1oxgamma_min.EQ.0d0)THEN
             CALL trapezoid_integration(npoints,Pbreak4n_LO_pA_fxn,&
                  log1oxgamma_max,integral)
          ELSE
             CALL simpson(Pbreak4n_LO_pA_fxn,log1oxgamma_min,&
                  log1oxgamma_max,integral,npoints)
          ENDIF
          YA(I)=integral
       ENDDO
       init=1
    ENDIF

    IF(gamma_proj_save.NE.gamma_proj&
         .OR.gamma_target_save.NE.gamma_target)THEN
       WRITE(*,*)"ERROR: gamma_proj,gamma_target are not consistent in Pbreak4n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (gamma_proj,gamma_target)=",gamma_proj_save,&
            gamma_target_save
       WRITE(*,*)"INFO: New ones (gamma_proj,gamma_target)=",gamma_proj,gamma_target
       STOP
    ENDIF
    IF(A_target_save.NE.A_target&
         .OR.Z_target_save.NE.Z_target)THEN
       WRITE(*,*)"ERROR: A_target,Z_target are not consistent in Pbreak4n_LO_pA"
       WRITE(*,*)"INFO: Saved ones (A_target,Z_target)=",A_target_save,&
            Z_target_save
       WRITE(*,*)"INFO: New ones (A_target,Z_target)=",A_target,Z_target
       STOP
    ENDIF

    IF(db.GT.rescaling_bmax_save.OR.db.LE.0d0)THEN
       Pbreak4n_LO_pA=0d0
    ELSE
       
       dbb=MIN(db/rescaling_bmax_save,1d0)
       N=-FLOOR(DLOG10(dbb))-1 ! db is in 10**(-n-1) to 10**(-n)
       IF(N.LT.0)THEN
          K=NXA-n_interp
       ELSE
          IF(N.LT.NBMAX)THEN
             k=NXA-NBSEG*n
          ELSE
             k=NBSEG+1
          ENDIF
          IF(XA(k-NBSEG).GT.db)THEN
             WRITE(*,*)"Error: k is not proper #1"
             STOP
          ENDIF
          IF(XA(k).LT.db)THEN
             WRITE(*,*)"Error: k is not proper #2"
             STOP
          ENDIF
          DO i=k-NBSEG,k
             IF(XA(i).GE.db)EXIT
          ENDDO
          IF(i-n_interp/2+2.GE.1.AND.i-n_interp/2+1+n_interp.LE.NXA)THEN
             K=i-n_interp/2+1
          ELSEIF(i-n_interp/2+2.LT.1)THEN
             K=0
          ELSEIF(i-n_interp/2+1+n_interp.GT.NXA)THEN
             K=NXA-n_interp
          ELSE
             WRITE(*,*)"Error: you cannot reach here !"
             STOP
          ENDIF
       ENDIF

       DO I=1,n_interp
          XD2_1D(I)=XA(K+I)
          YD2_1D(I)=YA(K+I)
       ENDDO

       CALL SPLINE_INTERPOLATE(XD2_1D,YD2_1D,n_interp,db,integral)
       Pbreak4n_LO_pA=integral

    ENDIF
    RETURN
  END FUNCTION Pbreak4n_LO_pA

  FUNCTION Pbreak4n_LO_pA_fxn(log1oxgamma)
    ! log1oxgamma = log(1/xgamma)
    ! where xgamma is the momentum fraction in the lab frame
    USE photoabsorption
    IMPLICIT NONE
    REAL(KIND(1d0))::Pbreak4n_LO_pA_fxn
    REAL(KIND(1d0)),INTENT(IN)::log1oxgamma
    REAL(KIND(1d0))::xgamma,Ega_lab,Ega_target,CFF
    REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV)
    INTEGER::A_target_common,Z_target_common
    COMMON/Pbreak4n_pA_CFF_INT/A_target_common,Z_target_common
    REAL(KIND(1d0))::gamma_proj_common,gamma_target_common
    REAL(KIND(1d0))::b_proj_common
    COMMON/Pbreak4n_pA_CFF_REAL/gamma_target_common,gamma_proj_common,b_proj_common

    xgamma=DEXP(-log1oxgamma)
    IF(xgamma.GE.1d0.OR.xgamma.LE.0d0)THEN
       Pbreak4n_LO_pA_fxn=0d0
       RETURN
    ENDIF

    ! gamma energy in lab [in unit of GeV]
    Ega_lab=xgamma*mproton*gamma_proj_common

    IF(USE_CHARGEFORMFACTOR4PHOTON)THEN
       CFF=PhotonNumberDensity_ChargeFormFactor_proton(b_proj_common,Ega_lab,&
            gamma_proj_common,proton_FF_correction,.FALSE.)
    ELSE
       CFF=PhotonNumberDensity(b_proj_common,Ega_lab,gamma_proj_common)
    ENDIF

    ! gamma energy in target frame [in unit of MeV]
    Ega_target=Ega_lab*(gamma_target_common+DSQRT(gamma_target_common**2-1d0))
    Ega_target=Ega_target*1d3 ! from GeV to MeV

    ! in unit of GeV**(-2)
    Pbreak4n_LO_pA_fxn=GAMMATO4N(Ega_target,&
         A_target_common,Z_target_common)*0.1d0/GeVm12fm**2

    Pbreak4n_LO_pA_fxn=Pbreak4n_LO_pA_fxn*CFF

    RETURN
  END FUNCTION Pbreak4n_LO_pA_fxn
  
  subroutine progress(j,nmax,forceinit)
    implicit none
    integer,intent(in)::j,nmax
    logical,intent(in),optional::forceinit
    integer::k
    character(:), allocatable :: bar, bar0
    character(5)::nmax_str
    !character(len=)::bar="???% |                                     |"
    integer::init=0
    save init,bar,bar0,nmax_str
    IF(present(forceinit))THEN
       IF(forceinit)THEN
          init=0
          IF(ALLOCATED(bar))THEN
             DEALLOCATE(bar)
          ENDIF
          IF(ALLOCATED(bar0))THEN
             DEALLOCATE(bar0)
          ENDIF
       ENDIF
    ENDIF
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

  ! erf^-1(x)
  ! from http://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
  ! there is a built-in Fortran function of erf (derf)
  function erfinv(x)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::x
    REAL(KIND(1d0))::erfinv
    REAL(KIND(1d0))::w,p
    IF(DABS(x).GE.1d0)THEN
       WRITE(*,*)"ERROR: |x|>= 1 in erfinv"
       STOP
    ENDIF
    w=-DLOG(1d0-x**2)
    IF(w.LT.5d0)THEN
       w=w-2.5d0
       p=2.81022636d-8
       p=3.43273939d-7+p*w
       p=-3.5233877d-6+p*w
       p=-4.39150654d-6+p*w
       p=0.00021858087d0+p*w
       p=-0.00125372503d0+p*w
       p=-0.00417768164d0+p*w
       p=0.246640727d0+p*w
       p=1.50140941d0+p*w
    ELSE
       w=dsqrt(w)-3d0
       p=-0.000200214257d0
       p=0.000100950558d0+p*w
       p=0.00134934322d0+p*w
       p=-0.00367342844d0+p*w
       p=0.00573950773d0+p*w
       p=-0.0076224613d0+p*w
       p=0.00943887047d0+p*w
       p=1.00167406d0+p*w
       p=2.83297682d0+p*w
    ENDIF
    erfinv=p*x
    RETURN
  END function erfinv

  SUBROUTINE generate_Q2_epa_proton_iWW(x,Q2max,Q2)
    use interpolation
    implicit none
    real(kind(1d0)),intent(in)::x,Q2max
    real(kind(1d0)),intent(out)::Q2
    real(kind(1d0)),parameter::mp=0.938272081d0 ! proton mass in unit of GeV
    real(kind(1d0)),parameter::mupomuN=2.793d0
    real(kind(1d0)),parameter::Q02=0.71d0 ! in unit of GeV**2
    real(kind(1d0))::mp2,Q2min,xx
    real(kind(1d0))::xmax,xmaxQ2max
    real(kind(1d0))::logQ2oQ02max,logQ2oQ02min
    real(kind(1d0))::log10xmaxQ2maxm1,tlog10xm1
    real(kind(1d0))::log10xm1
    real(kind(1d0))::max_Q2,max_fun,logQ2oQ02now
    real(kind(1d0))::Q2max_save
    data Q2max_save /-1d0/
    save Q2max_save
    real(kind(1d0)),dimension(100)::p_x_array
    real(kind(1d0)),dimension(100)::p_xmax_array
    real(kind(1d0)),dimension(100)::p_fmax_array
    save p_x_array,p_xmax_array,p_fmax_array
    real(kind(1d0))::r1,r2,w
    integer::i,j
    if(x.ge.1d0.or.x.le.0d0)then
       write(*,*)"ERROR: x>=1 or x<=0"
       stop
    endif
    mp2=mp**2
    Q2min=mp2*x**2/(1d0-x)

    xmax=xmaxvalue(mp2,Q2max)

    if(Q2max.le.Q2min.or.x.ge.xmax)then
       Q2=Q2max
       return
    endif

    logQ2oQ02max=dlog(Q2max/Q02)
    logQ2oQ02min=dlog(Q2min/Q02)
    
    if(Q2max_save.ne.Q2max)then
       ! we need to generate the grid first
       Q2max_save=Q2max
       xmaxQ2max=xmaxvalue(mp2,Q2max)
       log10xmaxQ2maxm1=dlog10(1d0/xmaxQ2max)
       do i=0,9
          do j=0,9
             tlog10xm1=log10xmaxQ2maxm1+0.1D0*dble(j)+dble(i)
             p_x_array(1+j+i*10)=tlog10xm1
             xx=10d0**(-tlog10xm1)
             if(i.eq.0.and.j.eq.0)then
                max_Q2=logQ2oQ02max
                max_fun=proton_iWW_distfun(xx,max_Q2)
                p_xmax_array(1+j+i*10)=max_Q2
                p_fmax_array(1+j+i*10)=max_fun
             else
                call calc_proton_iWW_distfun_max(xx,logQ2oQ02min,&
                     logQ2oQ02max,max_Q2)
                max_fun=proton_iWW_distfun(xx,max_Q2)
                p_xmax_array(1+j+i*10)=max_Q2
                p_fmax_array(1+j+i*10)=max_fun
             endif
          enddo
       enddo
    endif
    log10xm1=DLOG10(1d0/x)
    call spline_interpolate(p_x_array,p_fmax_array,100,&
         log10xm1,max_fun)
    logQ2oQ02now=logQ2oQ02min
    DO WHILE(.TRUE.)
       r1=rand() ! a random float number between 0 and 1
       logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
       w=proton_iWW_distfun(x,logQ2oQ02now)/max_fun
       r2=rand() ! a random float number between 0 and 1
       if(r2.le.w)exit
    ENDDO
    Q2=dexp(logQ2oQ02now)*Q02
    return
  END SUBROUTINE generate_Q2_epa_proton_iWW

  SUBROUTINE generate_Q2_epa_ion_ChFF(ion_Form,ibeam,x,Q2max,RA,aA,wA,Q2)
    use interpolation
    implicit none
    integer,intent(in)::ion_Form ! Form1: Q**2=kT**2+(mn*x)**2, Qmin**2=(mn*x)**2;
                                 ! Form2: Q**2=kT**2/(1-x)+(mn*x)**2/(1-x), Qmin**2=(mn*x)**2/(1-x) [more exact]
    integer,intent(in)::ibeam ! first beam: ibeam=1; second beam: ibeam=2
    real(kind(1d0)),intent(in)::x,Q2max
    real(kind(1d0)),intent(in)::RA,aA,wA ! RA and aA are in fm
    real(kind(1d0)),intent(out)::Q2
    real(kind(1d0)),parameter::mn=0.9315d0 ! averaged nucleon mass in unit of GeV
    real(kind(1d0)),parameter::Q02=0.71d0 ! in unit of GeV**2
    real(kind(1d0))::mn2,Q2min,xx,RAA,aAA
    real(kind(1d0))::xmax,xmaxQ2max
    real(kind(1d0))::logQ2oQ02max,logQ2oQ02min
    real(kind(1d0))::log10xmaxQ2maxm1,tlog10xm1
    real(kind(1d0))::log10xm1
    real(kind(1d0))::max_Q2,max_fun,logQ2oQ02now
    real(kind(1d0)),dimension(2)::Q2max_save
    data Q2max_save /-1d0,-1d0/
    save Q2max_save
    real(kind(1d0)),dimension(2,100)::A_x_array
    real(kind(1d0)),dimension(2,100)::A_xmax_array
    real(kind(1d0)),dimension(2,100)::A_fmax_array
    save A_x_array,A_xmax_array,A_fmax_array
    real(kind(1d0))::r1,r2,w
    integer::i,j
    if(ion_Form.ne.1.and.ion_Form.ne.2)then
       write(*,*)"ERROR: ion_Form =/= 1 or 2"
       stop
    endif
    if(ibeam.ne.1.and.ibeam.ne.2)then
       write(*,*)"ERROR: ibeam =/= 1 or 2"
       stop
    endif
    if(x.ge.1d0.or.x.le.0d0)then
       write(*,*)"ERROR: x>=1 or x<=0"
       stop
    endif
    mn2=mn**2
    if(ion_Form.eq.2)then
       Q2min=mn2*x**2/(1d0-x)
    else
       Q2min=mn2*x**2
    endif

    RAA=RA/GeVm12fm ! from fm to GeV-1
    aAA=aA/GeVm12fm ! from fm to GeV-1
      

    xmax=xmaxvalue(mn2,Q2max)

    if(Q2max.le.Q2min.or.x.ge.xmax)then
       Q2=Q2max
       return
    endif

    logQ2oQ02max=dlog(Q2max/Q02)
    logQ2oQ02min=dlog(Q2min/Q02)

    if(Q2max_save(ibeam).ne.Q2max)then
       ! we need to generate the grid first
       Q2max_save(ibeam)=Q2max
       xmaxQ2max=xmaxvalue(mn2,Q2max)
       log10xmaxQ2maxm1=dlog10(1d0/xmaxQ2max)
       do i=0,9
          do j=0,9
             tlog10xm1=log10xmaxQ2maxm1+0.1D0*dble(j)+dble(i)
             A_x_array(ibeam,1+j+i*10)=tlog10xm1
             xx=10d0**(-tlog10xm1)
             if(i.eq.0.and.j.eq.0)then
                max_Q2=logQ2oQ02max
                max_fun=ion_ChFF_distfun(ion_Form,aAA,RAA,wA,&
                     Q2min,xx,max_Q2)
                A_xmax_array(ibeam,1+j+i*10)=max_Q2
                A_fmax_array(ibeam,1+j+i*10)=max_fun
             else
                call calc_ion_ChFF_distfun_max(ion_Form,aAA,RAA,wA,&
                     Q2min,xx,logQ2oQ02min,logQ2oQ02max,max_Q2)
                max_fun=ion_ChFF_distfun(ion_Form,aAA,RAA,wA,&
                     Q2min,xx,max_Q2)
                A_xmax_array(ibeam,1+j+i*10)=max_Q2
                A_fmax_array(ibeam,1+j+i*10)=max_fun
             endif
          enddo
       enddo
    endif
    log10xm1=DLOG10(1d0/x)
    call spline_interpolate(A_x_array(ibeam,1:100),&
         A_fmax_array(ibeam,1:100),100,log10xm1,max_fun)
    logQ2oQ02now=logQ2oQ02min
    DO WHILE(.TRUE.)
       r1=rand() ! a random float number between 0 and 1
       logQ2oQ02now=(logQ2oQ02max-logQ2oQ02min)*r1+logQ2oQ02min
       w=ion_ChFF_distfun(ion_Form,aAA,RAA,wA,Q2min,x,logQ2oQ02now)/max_fun
       r2=rand() ! a random float number between 0 and 1
       if(r2.le.w)exit
    ENDDO
    Q2=dexp(logQ2oQ02now)*Q02
    return
  END SUBROUTINE generate_Q2_epa_ion_ChFF

  function xmaxvalue(mp2,q2max)
    implicit none
    real(kind(1d0))::xmaxvalue
    real(kind(1d0)),intent(in)::mp2,q2max
    xmaxvalue=(DSQRT(Q2max*(4d0*mp2+Q2max))-Q2max)/(2d0*mp2)
    return
  end function xmaxvalue

  ! (1 - x)*FE + x^2/2*FM - (1 - x)*Q2min/Q2*FE
  function proton_iWW_distfun(xx,logQ2oQ02)
    implicit none
    real(kind(1d0))::proton_iWW_distfun
    real(kind(1d0)),intent(in)::xx,logQ2oQ02
    real(kind(1d0)),parameter::mp=0.938272081d0 ! proton mass in unit of GeV
    real(kind(1d0)),parameter::mupomuN=2.793d0
    real(kind(1d0)),parameter::Q02=0.71d0 ! in unit of GeV**2
    real(kind(1d0))::expp,mp2
    mp2=mp**2
    expp=DEXP(logQ2oQ02)
    proton_iWW_distfun=(-8D0*mp2**2*xx**2+expp**2*mupomuN**2*Q02**2*&
         (2D0-2D0*xx+xx**2)+2D0*expp*mp2*Q02*(4D0-4D0*xx&
         +mupomuN**2*xx**2))/(2D0*expp*(1D0+expp)**4*Q02&
         *(4D0*mp2+expp*Q02))
    return
  end function proton_iWW_distfun

  function proton_iWW_mdistfun_min(logQ2oQ02)
    implicit none
    real(kind(1d0))::proton_iWW_mdistfun_min
    real(kind(1d0)),intent(in)::logQ2oQ02
    real(kind(1d0))::x
    common/proton_iWW_dist/x
      
    proton_iWW_mdistfun_min=proton_iWW_distfun(x,logQ2oQ02)
    proton_iWW_mdistfun_min=-proton_iWW_mdistfun_min
    return
  end function proton_iWW_mdistfun_min

  subroutine calc_proton_iWW_distfun_max(xx,logQ2oQ02min,&
       logQ2oQ02max,logQ2oQ02)
    use fmin_module
    implicit none
    real(kind(1d0)),intent(in)::xx
    real(kind(1d0)),intent(out)::logQ2oQ02
    real(kind(1d0)),intent(in)::logQ2oQ02min,logQ2oQ02max
    real(kind(1d0)),parameter::tol=1d-8
    real(kind(1d0))::x
    common/proton_iWW_dist/x
    x=xx
    logQ2oQ02=fmin(proton_iWW_mdistfun_min,logQ2oQ02min,&
         logQ2oQ02max,tol)
    return
  end subroutine calc_proton_iWW_distfun_max

  function ion_ChFF_distfun(ion_Form,aAA,RAA,wA,Q2min,xx,logQ2oQ02)
    implicit none
    integer,intent(in)::ion_Form ! Form1: Q**2=kT**2+(mn*x)**2, Qmin**2=(mn*x)**2;
                                 ! Form2: Q**2=kT**2/(1-x)+(mn*x)**2/(1-x), Qmin**2=(mn*x)**2/(1-x) [more exact]
    real(kind(1d0)),intent(in)::Q2min,RAA,aAA,wA
    real(kind(1d0)),parameter::Q02=0.71d0 ! in unit of GeV**2
    real(kind(1d0))::ion_ChFF_distfun
    real(kind(1d0)),intent(in)::xx,logQ2oQ02
    real(kind(1d0))::qq,FchA,expp
    expp=DEXP(logQ2oQ02)*Q02
    if(ion_Form.eq.2)then
       qq=dsqrt((1d0-xx)*expp)
    else
       qq=dsqrt(expp)
    endif
    ! keep the 10 terms
    FchA=ChargeFormFactor_WoodsSaxon(qq,RAA,wA,aAA,10)
    ion_ChFF_distfun=(1d0-Q2min/expp)*FchA**2
    return
  end function ion_ChFF_distfun

  function ion_ChFF_mdistfun_min(logQ2oQ02)
    implicit none
    real(kind(1d0)),intent(in)::logQ2oQ02
    real(kind(1d0))::ion_ChFF_mdistfun_min
    integer::ion_Form
    common/ion_form_par/ion_Form
    real(kind(1d0))::aAA,RAA,wA,x,Q2min
    common/ion_ChFF_dist/aAA,RAA,wA,x,Q2min

    ion_ChFF_mdistfun_min=ion_ChFF_distfun(ion_Form,aAA,RAA,wA,&
         Q2min,x,logQ2oQ02)
    ion_ChFF_mdistfun_min=-ion_ChFF_mdistfun_min
    return
  end function ion_ChFF_mdistfun_min

  subroutine calc_ion_ChFF_distfun_max(ion_Form,aAA,RAA,wA,&
       Q2min,xx,logQ2oQ02min,logQ2oQ02max,logQ2oQ02)
    use fmin_module
    implicit none
    integer,intent(in)::ion_Form
    real(kind(1d0)),intent(in)::aAA,RAA,wA,Q2min
    real(kind(1d0)),intent(in)::xx
    real(kind(1d0)),intent(out)::logQ2oQ02
    real(kind(1d0)),intent(in)::logQ2oQ02min,logQ2oQ02max
    real(kind(1d0)),parameter::tol=1d-8
    integer::ion_Form_com
    common/ion_form_par/ion_Form_com
    real(kind(1d0))::aAA_com,RAA_com,wA_com,x,Q2min_com
    common/ion_ChFF_dist/aAA_com,RAA_com,wA_com,x,Q2min_com
    ion_Form_com=ion_Form
    Q2min_com=Q2min
    x=xx
    aAA_com=aAA
    RAA_com=RAA
    wA_com=wA
    logQ2oQ02=fmin(ion_ChFF_mdistfun_min,logQ2oQ02min,&
         logQ2oQ02max,tol)
    return
  end subroutine calc_ion_ChFF_distfun_max

  SUBROUTINE InitialMomentumReshuffle_PhotonPhoton(NP,PA,Ecm,x1,x2,Q1,Q2,&
       PO,reshuffled)
    ! work in the rest frame of the two colliding nucleons
    IMPLICIT NONE
    integer,intent(in)::NP
    ! Ecm**2=s_{NN}
    real(kind(1d0)),intent(in)::Ecm,x1,x2,Q1,Q2
    real(kind(1d0)),dimension(NP,0:3),intent(in)::PA
    real(kind(1d0)),dimension(NP,0:3),intent(out)::PO
    LOGICAL,INTENT(out)::reshuffled
    real(kind(1d0))::r1,r2,ph1,ph2,x1bar,x2bar
    real(kind(1d0)),parameter::pipi=3.14159265358979323846264338328d0
    real(kind(1d0))::kperp2,kperp2max,Q
    real(kind(1d0)),dimension(0:3)::PBOO1,PBOO2
    integer::j
    reshuffled=.FALSE. ! if reshuffled is .False., we need to regenerate Q1 and Q2
    r1=rand() ! a random float number between 0 and 1
    r2=rand() ! a random float number between 0 and 1
    ph1=2d0*pipi*r1
    ph2=2d0*pipi*r2
    kperp2=Q1**2+Q2**2+2d0*Q1*Q2*DCOS(ph1-ph2)
    kperp2max=Ecm**2*(MIN(1d0,x1/x2,x2/x1)-x1*x2)
    IF(kperp2.GE.kperp2max)RETURN
    x1bar=DSQRT(x1/x2*kperp2/Ecm**2+x1**2)
    x2bar=DSQRT(x2/x1*kperp2/Ecm**2+x2**2)
    IF(x1bar.GE.1d0.or.x2bar.GE.1d0)RETURN
    ! new initial state
    PO(1,0)=Ecm/2d0*x1bar
    PO(1,1)=Q1*DCOS(ph1)
    PO(1,2)=Q1*DSIN(ph1)
    PO(1,3)=Ecm/2d0*x1bar
    PO(2,0)=Ecm/2d0*x2bar
    PO(2,1)=Q2*DCOS(ph2)
    PO(2,2)=Q2*DSIN(ph2)
    PO(2,3)=-Ecm/2d0*x2bar
    ! new final state
    DO j=0,3
       PBOO1(j)=PA(1,j)+PA(2,j)
       PBOO2(j)=PO(1,j)+PO(2,j)
    ENDDO
    Q=DSQRT(x1*x2)*Ecm
    DO j=3,NP
       PO(j,0:3)=PA(j,0:3)
       CALL BOOSTL52(Q,PBOO1,PBOO2,PO(j,0:3))
    ENDDO
    reshuffled=.TRUE.
    RETURN
  END SUBROUTINE InitialMomentumReshuffle_PhotonPhoton

  ! When PBOO(0)=P(0) and PBOO(1:3)=-P(1:3), it boosts P to its rest frame (0,0,0,Q)
  SUBROUTINE BOOSTL5(Q,PBOO,P)
    ! momentums are in normal representation with fourth comp is the zero comp
    ! Boost P via PBOO(PBOO^2=Q^2) to PLB,and set to P
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::Q
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::PBOO
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(INOUT)::P
    REAL(KIND(1d0)),DIMENSION(0:3)::PCM,PLB
    REAL(KIND(1d0))::FACT
    INTEGER::J
    PCM(0:3)=P(0:3)
    PLB(0)=(PBOO(0)*PCM(0)&
         +PBOO(3)*PCM(3)+PBOO(2)*PCM(2)+PBOO(1)*PCM(1))/Q
    FACT=(PLB(0)+PCM(0))/(Q+PBOO(0))
    DO J=1,3
       PLB(J)=PCM(J)+FACT*PBOO(J)
    ENDDO
    P(0:3)=PLB(0:3)
    RETURN
  END SUBROUTINE BOOSTL5

  SUBROUTINE BOOSTL52(Q,PBOO1,PBOO2,P)
    ! Boost P from PBOO1 (PBOO1^2=Q^2) to PBOO2 (PBOO2^2=Q^2) frame
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::Q
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::PBOO1,PBOO2
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(INOUT)::P
    REAL(KIND(1d0)),DIMENSION(0:3)::PBOO10,PRES
    PBOO10(0)=PBOO1(0)
    PBOO10(1:3)=-PBOO1(1:3)
    PRES(0:3)=P(0:3)
    CALL BOOSTL5(Q,PBOO10,PRES) ! PRES is in (Q,0,0,0) frame
    CALL BOOSTL5(Q,PBOO2,PRES)  ! PRES is in PBOO2 frame
    P(0:3)=PRES(0:3)
    RETURN
  END SUBROUTINE BOOSTL52

  SUBROUTINE BOOSTTOECM(NP,E1,E2,PA)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NP
    REAL(KIND(1d0)),INTENT(IN)::E1,E2
    REAL(KIND(1d0)),DIMENSION(NP,0:3),INTENT(INOUT)::PA
    REAL(KIND(1d0))::Ecm
    REAL(KIND(1d0)),DIMENSION(0:3)::PBOO
    INTEGER::J
    Ecm=2d0*DSQRT(E1*E2)
    PBOO(0)=E1+E2
    PBOO(1:2)=0d0
    PBOO(3)=E2-E1
    DO J=1,NP
       CALL BOOSTL5(Ecm,PBOO,PA(J,0:3))
    ENDDO
    RETURN
  END SUBROUTINE BOOSTTOECM

  SUBROUTINE BOOSTFROMECM(NP,E1,E2,PA)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NP
    REAL(KIND(1d0)),INTENT(IN)::E1,E2
    REAL(KIND(1d0)),DIMENSION(NP,0:3),INTENT(INOUT)::PA
    REAL(KIND(1d0))::Ecm
    REAL(KIND(1d0)),DIMENSION(0:3)::PBOO
    INTEGER::J
    Ecm=2d0*DSQRT(E1*E2)
    PBOO(0)=E1+E2
    PBOO(1:2)=0d0
    PBOO(3)=E1-E2
    DO J=1,NP
       CALL BOOSTL5(Ecm,PBOO,PA(J,0:3))
    ENDDO
    RETURN
  END SUBROUTINE BOOSTFROMECM

END MODULE ElasticPhotonPhotonFlux
