PROGRAM test
  USE OpticalGlauber_Geometry
  USE ElasticPhotonPhotonFlux
  IMPLICIT NONE
  include 'run90.inc'
  INTEGER::I
  REAL(KIND(1d0))::x1,x2
  REAL(KIND(1d0))::flux1,flux2,flux3,flux4,flux5,flux6
  REAL(KIND(1d0))::gamma1,gamma2
  REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV) 
  REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
  REAL(KIND(1d0)),PARAMETER::mchic=3.55d0
  REAL(KIND(1d0))::cmsenergy,tau,dy,ychic,ymin,ymax,gagam
  INTEGER::nybin
  REAL(KIND(1d0))::mass1,mass2,mass3,mass4
  REAL(KIND(1d0))::dM,db,RR_proj,w_proj,aa_proj
  REAL(KIND(1d0))::width1,width2,width3,width4
  INTEGER::J1,J2,J3,J4
  REAL(KIND(1d0))::br1,br2,br3,br4
  REAL(KIND(1d0)),PARAMETER::convfac=3.8938573d5 ! from GeV-2 to nb
  REAL(KIND(1d0)),PARAMETER::FOURPI2=39.4784176043574344753379639995d0
  REAL(KIND(1d0)),PARAMETER::pipipi=3.14159265358979323846264338328d0
  REAL(KIND(1d0)),DIMENSION(5)::array,mass,flux,flux_inel,delta_B,delta_B_inel

  USE_CHARGEFORMFACTOR4PHOTON=.TRUE.
  !  proton_FF_correction=.FALSE.
  ebeam(1)=2510d0
  ebeam(2)=2510d0
  nuclearA_beam1=208
  nuclearA_beam2=208
  nuclearZ_beam1=82
  nuclearZ_beam2=82
  x1=0.00099601593625498d0
  x2=x1
  !flux1=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2)
  !flux1=Lgammagamma_UPC(5d0,3,1)
  !PRINT *, "default:",flux1
  !RETURN
  neutron_tagging(1)=-1
  neutron_tagging(2)=-1
  !flux1=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2)
  !flux1=Lgammagamma_UPC(5d0,3,1)
  !PRINT *, "XnXn:",flux1
  !RETURN
  neutron_tagging(1)=0
  neutron_tagging(2)=0
  !flux1=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2)
  !flux1=Lgammagamma_UPC(5d0,3,1) 
  !PRINT *, "0n0n:",flux1
  !RETURN
  neutron_tagging(1)=-1
  neutron_tagging(2)=0
  neutron_conj_sum=.TRUE.
  !flux1=PhotonPhotonFlux_AB_WoodsSaxon_eval(x1,x2)
  flux1=Lgammagamma_UPC(5d0,3,1)
  PRINT *, "0nXn+Xn0n:",flux1
  RETURN
!  gamma1=5020d0/2d0/mN
!  !gamma1=6500d0/mproton
!  gamma2=5020d0/2d0/mN
!  ! 15 fm; 1  GeV^-1 = 0.1973  fm
!  db=15d0/0.1973d0 ! in unit of GeV-1
!  RR_proj=6.624d0/0.1973d0 ! in unit of GeV-1
!  aa_proj=0.549d0/0.1973d0 ! in unit of GeV-1
!  w_proj=0d0
!  flux1=Pbreak4n_LO_AB_WS(db,208,82,gamma2,gamma1,&
!       RR_proj,w_proj,aa_proj,3.2d-2,1.4d-1,1,-1)
!  PRINT *, flux1
!  RETURN

!  OPEN(UNIT=2033,FILE="/Users/hua-shengshao/Physics/gammaUPC/ExcitingIons/gammaUPC/Xn1n_PbPb_EDFF.dat")
!  !DO I=-40,73
!  DO I=1,73
!     db=1.1d0**I/0.1973d0 ! in unit of GeV-1
!     ! Pb+Pb
!     flux1=PbreakXn_LO_AB_WS(db,208,82,gamma2,gamma1,&
!          RR_proj,w_proj,aa_proj,1d-3,0d0,1)
!     flux1=flux1*82d0**2/137.036d0 ! P^{Xn,(1)}
!     flux2=Pbreak1n_LO_AB_WS(db,208,82,gamma2,gamma1,&
!          RR_proj,w_proj,aa_proj,1d-3,4d-2,1)
!     flux2=flux2*82d0**2/137.036d0 ! P^{1n,(1)}
!     flux3=DEXP(-flux1) ! 0n
!     flux1=1d0-flux3 ! Xn
!     flux2=flux2*flux3 ! 1n
!     flux4=flux2**2 ! 1n1n
!     flux5=2d0*flux2*flux1 ! Xn1n
!     flux6=2d0*flux3*flux2 ! 0n1n
!     ! p+Pb
!     !flux1=PbreakXn_LO_pA(db,208,82,gamma2,gamma1,1d-3,0d0,1)
!     !flux1=flux1/137.036d0
!     !flux2=DEXP(-flux1) ! 0n
!     !flux1=1d0-flux2    ! Xn
!     !flux3=flux1*flux2*2d0 ! Xn0n+0nXn
!     !flux2=flux2**2 ! 0n0n
!     !flux1=flux1**2 ! XnXn
!  
!     !WRITE(2033,*)0.1973d0*db,flux1,flux3,flux2
!     !WRITE(2033,*)0.1973d0*db,flux1,flux2
!     WRITE(2033,*)0.1973d0*db,flux4,flux5,flux6
!  ENDDO
!  CLOSE(UNIT=2033)
  
!  RETURN
  USE_CHARGEFORMFACTOR4PHOTON=.True.
  GENERATE_PhotonPhotonFlux_GRID=.FALSE.
  nuclearA_beam1=208
  nuclearZ_beam1=82
  nuclearA_beam2=208
  nuclearZ_beam2=82
  ebeam(1)=2.75d3
  ebeam(2)=2.75d3
  
  
  do i=1,5
  array(i) = 0 + 2.18 * real(i-1)/(100-1)
  mass(i) = 10**array(i)
  flux(i) = dLgammagammadW_UPC(mass(i),3,1)
  flux_inel(i) = dLgammagammadW_UPC(mass(i),3,0)
  delta_B(i) = DeltaB_UPC_at_W(mass(i),3,1)
  delta_B_inel(i) = DeltaB_UPC_at_W(mass(i),3,0)
  END do
  
  PRINT *, mass
  PRINT *, flux
  PRINT *, flux_inel
  PRINT *, delta_B
  PRINT *, delta_B_inel
  RETURN

  IEPSILON_EDFF=1
  USE_MC_GLAUBER=.FALSE.
  USE_CHARGEFORMFACTOR4PHOTON=.FALSE.
  alphaem_elasticphoton=0.0072992700729927005d0
  nuclearA_beam1=208
  nuclearZ_beam1=82
  nuclearA_beam2=208
  nuclearZ_beam2=82
  nuclearA_beam1=0
  nuclearZ_beam1=0
  nuclearA_beam2=0
  nuclearZ_beam2=0
  ebeam(1)=2760d0
  ebeam(2)=2760d0
  mass1=3.41475d0 ! chi_c0
  mass2=3.5562d0 ! chi_c2
  mass3=3.6389d0 ! etac(2S)
  mass4=3.55d0   ! tau onium
  flux1=dLgammagammadW_UPC(mass1,3,1)
  flux2=dLgammagammadW_UPC(mass2,3,1)
  flux3=dLgammagammadW_UPC(mass3,3,1)
  flux4=dLgammagammadW_UPC(mass4,3,1)

  PRINT *, "chic0     chic2     etac(2S)      tauonium"
  PRINT *, "dL/dW [GeV-1]"
  PRINT *, flux1,flux2,flux3,flux4

  ! total width [GeV]
  width1=0.0108d0
  width2=0.00197d0
  width3=0.0113d0
  width4=1.84d-11
  ! branching fraction to diphoton
  br1=2.04d-4
  br2=2.85d-4
  br3=1.9d-4
  br4=1d0
  ! spin
  J1=0
  J2=2
  J3=0
  J4=0

  ! cross sections [nb]
  flux1=FOURPI2*DBLE(2*J1+1)*br1**2*width1/mass1**2*convfac*flux1
  flux2=FOURPI2*DBLE(2*J2+1)*br2**2*width2/mass2**2*convfac*flux2
  flux3=FOURPI2*DBLE(2*J3+1)*br3**2*width3/mass3**2*convfac*flux3
  flux4=FOURPI2*DBLE(2*J4+1)*br4**2*width4/mass4**2*convfac*flux4

  PRINT *, "cross section*Br [nb]"
  PRINT *, flux1,flux2,flux3,flux4

  ! for axion mass scan
  OPEN(UNIT=20344,FILE="AxionXS_M.dat")
  J1=0
  br1=1d0
  gagam=1d-4 ! in unit of GeV-1
  ! for pp
  dM=1d0
  DO I=1, 10000
  ! for others
  !dM=0.2d0
  !DO I=5,1000
     mass1=dM*DBLE(I)
     width1=gagam**2*mass1**3/(64d0*pipipi)
  !   ! PbPb, XeXe, KrKr, ArAr, CaCa, OO
  !   !flux1=dLgammagammadW_UPC(mass1,3,1)
  !   ! pPb
  !   !flux1=dLgammagammadW_UPC(mass1,2,1)
     ! pp
     flux1=dLgammagammadW_UPC(mass1,1,1)
     flux1=FOURPI2*DBLE(2*J1+1)*br1**2*width1/mass1**2*convfac*flux1
     WRITE(20344,*)mass1,flux1
  ENDDO
  CLOSE(UNIT=20344)

  RETURN

!  OPEN(UNIT=20344,FILE="/Users/erdissshaw/Works/Plots/Test_Centrality/data_UPC_PhotonFlux/PbPb5.5TeV_dLdW_M.dat")
!  dM=0.2d0
!  DO I=5,1000
!     mass1=dM*DBLE(I)
!     flux1=dLgammagammadW_UPC(mass1,3,1)
!     WRITE(20344,*)mass1,flux1
!  ENDDO
!  CLOSE(UNIT=20344)

!  RETURN
END PROGRAM test
