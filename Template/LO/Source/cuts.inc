C
C     AUTOMATIC CUTS
C
      REAL*8 PTJ,PTB,PTA,PTL,PTONIUM  ! ptmin
      REAL*8 EJ,EB,EA,EL,ETAONIUM      ! min energy
      REAL*8 ETAJ,ETAB,ETAA,ETAL ! min rapidity
      REAL*8 DRJJ,DRBB,DRLL,DRAA,DRBJ,DRAJ,DRJL,DRAB,DRBL,DRAL ! min distance
      REAL*8 MMJJ,MMLL,MMAA,MMBB !min inv mass
      COMMON /TO_AUTO_CUTS/
     &     PTJ ,PTB ,PTA ,PTL ,PTONIUM, ! pt
     &     EJ  ,EB  ,EA  ,EL  , ! energy
     &     ETAJ,ETAB,ETAA,ETAL,ETAONIUM, ! rapidity 
     &     DRJJ,DRAJ,DRBJ,DRJL, ! delta R
     &     DRBB,DRAB,DRBL,
     &     DRAA,DRAL,
     &     DRLL,
     &     MMJJ,MMLL,MMAA,MMBB ! invariant masses of opposite sign

C     
C     MIN/MAX cuts
C
      REAL*8 PTJmax ,PTBmax ,PTAmax ,PTLmax    ! max pt
      REAL*8 EJmax  ,EBmax  ,EAmax  ,ELmax     ! max rap
      REAL*8 ETAJmin,ETABmin,ETAAmin,ETALmin   ! min rap
      REAL*8 DRJJmax,DRBBmax,DRLLmax,DRAAmax
      REAL*8 DRBJmax,DRAJmax,DRJLmax,DRABmax,DRBLmax,DRALmax
      REAL*8 MMJJmax,MMLLmax,MMAAmax,MMBBmax !max inv mass 
      REAL*8 MMNL,MMNLMAX ! invariant mass of all leptons
      REAL*8 cutuse
      REAL*8 ptj1min,ptj2min,ptj3min,ptj4min
      REAL*8 ptj1max,ptj2max,ptj3max,ptj4max
      REAL*8 ptl1min,ptl2min,ptl3min,ptl4min
      REAL*8 ptl1max,ptl2max,ptl3max,ptl4max
      REAL*8 ht2min,ht3min,ht4min
      REAL*8 ht2max,ht3max,ht4max
      REAL*8 htjmin,htjmax,ihtmin,ihtmax
      REAL*8 misset,missetmax,ptheavy
      REAL*8 ptllmin,ptllmax
      integer maxjetflavor
	
      COMMON /to_min_max_cuts/
     &     PTJmax,PTBmax,PTAmax,PTLmax,
     &     EJmax,EBmax,EAmax,ELmax,
     &     ETAJmin,ETABmin,ETAAmin,ETALmin,
     &     DRJJmax,DRBBmax,DRLLmax,DRAAmax,
     &     DRBJmax,DRAJmax,DRJLmax,DRABmax,DRBLmax,DRALmax,
     &     MMJJmax,MMLLmax,MMAAmax,MMBBmax,
     &     MMNL,MMNLmax,
     &     ptj1min,ptj1max,
     &     ptj2min,ptj2max,
     &     ptj3min,ptj3max,
     &     ptj4min,ptj4max,
     &     ptl1min,ptl1max,
     &     ptl2min,ptl2max,
     &     ptl3min,ptl3max,
     &     ptl4min,ptl4max,
     &     cutuse,
     &     ht2min,ht3min,ht4min,
     &     ht2max,ht3max,ht4max,
     &     htjmin,htjmax,ihtmin,ihtmax,
     &     misset,missetmax,ptheavy,
     &     ptllmin,ptllmax,
     &     maxjetflavor

C
C     SPECIAL CUTS
C
C      Frixione Isolation cut
       REAL*8 PTGMIN,R0GAMMA,XN,EPSGAMMA ! photon cuts
       LOGICAL ISOEM ! photon cuts
       COMMON /TO_ISOGAMMA_CUTS/PTGMIN,R0GAMMA,XN,EPSGAMMA
       COMMON /TO_ISOGAMMA_EM/ISOEM

C-- this are somewhat old kind of cuts that where used for specific
c   purposes
c
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta

C
C    FLAG FOR CUTTING DECAY PRODUCTS AND AUTOMATICALLY SETTING PTJ/MJJ
C
      logical cut_decays, auto_ptj_mjj
      common /to_cut_decays/cut_decays,auto_ptj_mjj

C COMMON BLOCK FOR JET MEASURE CUTS: DURHAM KT
      REAL*8 KT_DURHAM, D_PARAMETER
      LOGICAL DO_KT_DURHAM
      COMMON /JET_MEASURE_CUTS/ KT_DURHAM, D_PARAMETER