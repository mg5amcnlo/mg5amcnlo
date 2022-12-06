PROGRAM test
  USE ElasticPhotonPhotonFlux
  IMPLICIT NONE
  include 'run90.inc'
  INTEGER::I
  REAL(KIND(1d0))::x1,x2
  REAL(KIND(1d0))::flux1,flux2,flux3,flux4,flux5
  REAL(KIND(1d0))::gamma1,gamma2
  REAL(KIND(1d0)),PARAMETER::mproton=0.938272081d0 ! the mass of proton (GeV) 
  REAL(KIND(1d0)),PARAMETER::mN=0.9315d0 ! average nucleaon mass in nuclei (GeV)
  REAL(KIND(1d0)),PARAMETER::mchic=3.55d0
  REAL(KIND(1d0))::cmsenergy,tau,dy,ychic,ymin,ymax
  INTEGER::nybin
  REAL(KIND(1d0))::mass1,mass2,mass3,mass4
  REAL(KIND(1d0))::dM
  REAL(KIND(1d0))::width1,width2,width3,width4
  INTEGER::J1,J2,J3,J4
  REAL(KIND(1d0))::br1,br2,br3,br4
  REAL(KIND(1d0)),PARAMETER::convfac=3.8938573d5 ! from GeV-2 to nb
  REAL(KIND(1d0)),PARAMETER::FOURPI2=39.4784176043574344753379639995d0
  USE_CHARGEFORMFACTOR4PHOTON=.FALSE.
  alphaem_elasticphoton=0.0072992700729927005d0
  nuclearA_beam1=208
  nuclearZ_beam1=82
  nuclearA_beam2=208
  nuclearZ_beam2=82
  ebeam_PN(1)=2760d0
  ebeam_PN(2)=2760d0
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
