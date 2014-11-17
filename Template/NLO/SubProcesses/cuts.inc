C
C     DEFAULT CUTS FROM RUN_CARD>DAt
C
      REAL*8 JETALGO,JETRADIUS,PTJ,ETAJ       ! jet cuts
      REAL*8 PTL,ETAL,DRLL,DRLL_SF,MLL,MLL_SF ! lepton cuts
      COMMON /TO_NEW_AUTO_CUTS/
     &     JETALGO,JETRADIUS,PTJ,ETAJ,
     &     PTL,ETAL,DRLL,DRLL_SF,MLL,MLL_SF
      REAL*8 PTGMIN,R0GAMMA,XN,EPSGAMMA,ETAGAMMA  ! photon cuts
      LOGICAL ISOEM                           ! photon cuts
      COMMON /TO_ISOGAMMA_CUTS/PTGMIN,R0GAMMA,XN,EPSGAMMA,ETAGAMMA
      COMMON /TO_ISOGAMMA_EM/ISOEM
      integer maxjetflavor
      COMMON /to_min_max_cuts/maxjetflavor
C
C     SPECIAL CUTS (FOR MERGING)
C
      real*8 xmtc,xqcut
      common /to_specxpt/xmtc,xqcut
