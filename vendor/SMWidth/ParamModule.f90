MODULE ParamModule
  IMPLICIT NONE
  REAL(KIND(1d0))::Decay_MU_R
  REAL(KIND(1d0))::Decay_aEWM1
  REAL(KIND(1d0))::Decay_Gf
  REAL(KIND(1d0))::Decay_aS
  REAL(KIND(1d0))::Decay_MT
  REAL(KIND(1d0))::Decay_MZ
  REAL(KIND(1d0))::Decay_MH
  REAL(KIND(1d0))::Decay_MW
  INTEGER::Decay_scheme
  REAL(KIND(1d0))::Decay_gs,Decay_e
  COMPLEX(KIND(1d0))::Decay_C_SW2,Decay_C_CW2,Decay_C_SW,Decay_C_CW
  REAL(KIND(1d0))::Decay_R_SW2,Decay_R_CW2,Decay_SW,Decay_CW
  REAL(KIND(1d0))::Decay_WW,Decay_WZ,Decay_WT
  REAL(KIND(1d0)),PARAMETER::pi=3.1415926535897932384626433832795d0
  REAL(KIND(1d0)),PARAMETER::EPS=1d-10
  ! Running for ALPHAS
  !     alphaQCD2 -- value of alpha_s at the mass of the Z-boson
  !     nloop -- the number of loops (1,2, or 3) at which beta
  INTEGER::NLOOP
  REAL(KIND(1d0))::alphaQCD2
  LOGICAL::print_banner=.FALSE.
  LOGICAL::Skip_scheme=.FALSE.
  ! For the finite mass effect
  ! On shell masses
  ! in the PDG, the quark masses are MSbar masses
  REAL(KIND(1d0)),PARAMETER::Finite_MB=4.49D0
  REAL(KIND(1d0)),PARAMETER::Finite_MC=1.42D0
  REAL(KIND(1d0)),PARAMETER::Finite_MTAU=1.77684D0
  REAL(KIND(1d0)),PARAMETER::Finite_MMU=0.105658367D0
  CHARACTER(len=100)::identcard=""
END MODULE ParamModule
