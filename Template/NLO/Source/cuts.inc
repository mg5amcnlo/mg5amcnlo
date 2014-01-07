C
C     NEW CUTS
C
      REAL*8 JETALGO,JETRADIUS,PTJ,ETAJ  ! jet cuts
      REAL*8 PTL,ETAL,DRLL,MLL           ! lepton cuts
      COMMON /TO_NEW_AUTO_CUTS/
     &     JETALGO,JETRADIUS,PTJ,ETAJ,
     &     PTL,ETAL,DRLL,MLL
      REAL*8 PTGMIN,R0GAMMA,XN,EPSGAMMA,ETAGAMMA  ! photon cuts
      LOGICAL ISOEM                           ! photon cuts
      COMMON /TO_ISOGAMMA_CUTS/PTGMIN,R0GAMMA,XN,EPSGAMMA,ETAGAMMA
      COMMON /TO_ISOGAMMA_EM/ISOEM

C
C     AUTOMATIC CUTS
C
c      REAL*8 PTJ,PTB,PTA,PTL,PTONIUM  ! ptmin
      REAL*8 PTB,PTA,PTONIUM  ! ptmin
      REAL*8 EJ,EB,EA,EL,ETAONIUM      ! min energy
c      REAL*8 ETAJ,ETAB,ETAA,ETAL ! min rapidity
      REAL*8 ETAB,ETAA ! min rapidity
c      REAL*8 DRJJ,DRBB,DRLL,DRAA,DRBJ,DRAJ,DRJL,DRAB,DRBL,DRAL ! min distance
      REAL*8 DRJJ,DRBB,DRAA,DRBJ,DRAJ,DRJL,DRAB,DRBL,DRAL ! min distance
      REAL*8 MMJJ,MMLL,MMAA,MMBB !min inv mass
      COMMON /TO_AUTO_CUTS/
c     &     PTJ ,PTB ,PTA ,PTL ,PTONIUM, ! pt
     &     PTB ,PTA ,PTONIUM, ! pt
     &     EJ  ,EB  ,EA  ,EL  , ! energy
c     &     ETAJ,ETAB,ETAA,ETAL,ETAONIUM, ! rapidity 
     &     ETAB,ETAA,ETAONIUM, ! rapidity 
     &     DRJJ,DRAJ,DRBJ,DRJL, ! delta R
     &     DRBB,DRAB,DRBL,
     &     DRAA,DRAL,
c     &     DRLL,
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
      REAL*8 ht2min,ht3min,ht4min
      REAL*8 ht2max,ht3max,ht4max
      REAL*8 htjmin,htjmax
      REAL*8 misset,missetmax,ptheavy
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
     &     cutuse,
     &     ht2min,ht3min,ht4min,
     &     ht2max,ht3max,ht4max,
     &     htjmin,htjmax,
     &     misset,missetmax,ptheavy,
     &     maxjetflavor

C
C     SPECIAL CUTS
C
C-- this are somewhat old kind of cuts that where used for specific
c   purposes
c
      real*8 xptj,xptb,xpta,xptl,xmtc
      real*8 xetamin,xqcut,deltaeta
      common /to_specxpt/xptj,xptb,xpta,xptl,xmtc,xetamin,xqcut,deltaeta
