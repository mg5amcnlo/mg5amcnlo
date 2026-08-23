MODULE OpticalGlauber_Geometry
  ! This is module to provide the functions for the Optical Glauber model
  ! via the profiles. 
  ! Besides the Optical Glauber model, another possiblity is to
  !  use the Monte Carlo Glauber model, in which the nucleons are
  ! populated stochastically according to the given nuclear density profile
  ! There are a few public Monte Carlo Glauber tools (e.g. PHOBOS in
  ! https://arxiv.org/pdf/0805.4411.pdf (or 1408.2549), which requires the ROOT pre-installation).
  ! Other references for Glauber modelling in high-energy nuclear collisions are
  ! http://www.physi.uni-heidelberg.de/~reygers/lectures/2014/qgp_journal_club/talks/2014-08-18-glauber-model.pdf
  ! The geometrical dependent shadowng can be found (e.g. Eq.6) in 
  ! https://arxiv.org/pdf/0809.4684.pdf, which is equivalent to 
  ! https://arxiv.org/pdf/nucl-th/0305046.pdf
  ! it is important to check my derived formula in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  USE NINTLIB ! for multiple dimensional integrations
  USE interpolation
  IMPLICIT NONE
  CHARACTER(len=20)::nuclear_dir='./nuclear/'
  ! the parameter for evaluating sigma_inelastic
  ! 1: from a DdE parameterisation (2011.14909)
  ! 2: data from nuclear/input/sigmapp_inel.inp and use spline to interpolate
  INTEGER::sigmaNN_inelastic_eval=1
CONTAINS
  ! The parameters of R, A, w, a (the Woods-Saxon distribution) are
  ! given in Ramona Vogt's lecture or H. DeVries, C.W. De Jager, C. DeVries, 1987 etc
  ! They are determined via e-=nucleus scattering (and difference between protons and neutrons negligible)
  
  FUNCTION SigmaInelAB_hardsphere(RR,A,sigma_inel)
    ! in unit of fm^2, 1 fm^2 = 10 mb
    ! calculate the total inelastic cross section of A+B collision
    ! via the integration of Eq.(2.7) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::SigmaInelAB_hardsphere
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb; at 7-60 GeV, it is averaged as 31.5 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
                                           ! (see Table in http://www.phys.ufl.edu/~korytov/phz4390/note_01_NaturalUnits_SMsummary.pdf) 
                                           ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
                                           ! 1 GeV^-2 = 0.38938573 mb = 0.38938573e-31 m^2 = 0.038938573 fm^2 
                                           ! 1 mb = 1e-31 m^2 = 0.1 fm^2
    REAL(KIND(1d0)),DIMENSION(2)::R_common,A_common
    REAL(KIND(1d0))::sigmainel_common
    COMMON/SigmaInel_hardsphere/R_common,A_common,sigmainel_common
    R_common(1:2)=RR(1:2)
    A_common(1:2)=A(1:2)
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    CALL trapezoid_integration(1000,SigmaInelAB_fxn_hardsphere,&
         RR(1)+RR(2),SigmaInelAB_hardsphere)
    RETURN
  END FUNCTION SigmaInelAB_hardsphere

  FUNCTION SigmaInelAB_fxn_hardsphere(b)
    ! in unit of fm, 1 fm = 10 mb/fm
    IMPLICIT NONE
    REAL(KIND(1d0))::SigmaInelAB_fxn_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::b
    REAL(KIND(1d0)),DIMENSION(2)::R_common,A_common
    REAL(KIND(1d0))::sigmainel_common
    COMMON/SigmaInel_hardsphere/R_common,A_common,sigmainel_common
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    REAL(KIND(1d0))::TAB
    INTEGER::ABAB
    ABAB=INT(A_common(1))*INT(A_common(2))
    TAB=TABhat_hardsphere(b,0d0,R_common)
    SigmaInelAB_fxn_hardsphere=(1d0-GOOD_POWER(1d0-TAB*sigmainel_common,ABAB))*2d0*pi*b
    RETURN
  END FUNCTION SigmaInelAB_fxn_hardsphere

  FUNCTION Npart_avg_hardsphere(bmin,bmax,RR,A,sigma_inel)
    ! integration of bmin to bmax and divide by the bin size, i.e. <Npart>=Int[Npart[bx,by]dbx dby]/Int[dbx dby]
    ! where Npart is Eq.(2.9) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_avg_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb; at 7-60 GeV, it is averaged as 31.5 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
                                           ! (see Table in http://www.phys.ufl.edu/~korytov/phz4390/note_01_NaturalUnits_SMsummary.pdf)
                                           ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
                                           ! 1 GeV^-2 = 0.38938573 mb = 0.38938573e-31 m^2 = 0.038938573 fm^2
                                           ! 1 mb = 1e-31 m^2 = 0.1 fm^2
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::RA_common,RB_common,AA_common,AB_common,sigmainel_common
    LOGICAL::bjac_common
    COMMON/Npart_avg_hardsphere/RA_common,RB_common,AA_common,AB_common,sigmainel_common
    IF(bmin.GE.bmax)THEN
       Npart_avg_hardsphere=0d0
       RETURN
    ENDIF
    RA_common=RR(1)
    RB_common=RR(2)
    AA_common=A(1)
    AB_common=A(2)
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    aax(1)=-RR(1)
    bbx(1)=RR(1)
    aax(2)=-RR(1)
    bbx(2)=RR(1)
    aax(3)=bmin
    bbx(3)=bmax
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=100
    CALL ROMBERG_ND(Npart_avg_fxn_hardsphere,aax,bbx,3,sub_num,1,1d-5,&
         Npart_avg_hardsphere,ind,eval_num)
    Npart_avg_hardsphere=Npart_avg_hardsphere/(0.5d0*(bmax**2-bmin**2))
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
  END FUNCTION NPART_AVG_HARDSPHERE

  FUNCTION Npart_avg_fxn_hardsphere(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_avg_fxn_hardsphere
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::RA_common,RB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_avg_hardsphere/RA_common,RB_common,AA_common,AB_common,sigmainel_common
    REAL(KIND(1d0))::s1,s2
    REAL(KIND(1d0))::TTA,TTB
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: Npart_avg_fxn_hardsphere is not a three dimensional function"
       STOP
    ENDIF
    s1=DSQRT(sA(1)**2+sA(2)**2)
    IF(s1.GT.RA_common)THEN
       Npart_avg_fxn_hardsphere=0d0
       RETURN
    ENDIF
    s2=DSQRT((sA(1)-sA(3))**2+(sA(2))**2)
    IF(s2.GT.RB_common)THEN
       Npart_avg_fxn_hardsphere=0d0
       RETURN
    ENDIF
    TTA=3d0/4d0/pi/RA_common**3*2d0*DSQRT(RA_common**2-s1**2)
    TTB=3d0/4d0/pi/RB_common**3*2d0*DSQRT(RB_common**2-s2**2)
    ! first term
    Npart_avg_fxn_hardsphere=AA_common*TTA*(1D0-(1D0-TTB*sigmainel_common)**INT(AB_common))
    Npart_avg_fxn_hardsphere=Npart_avg_fxn_hardsphere+&
         AB_common*TTB*(1D0-(1D0-TTA*sigmainel_common)**INT(AA_common))
    ! jaccobi d^2b -> 2*pi*b*db (drop 2*pi)
    Npart_avg_fxn_hardsphere=Npart_avg_fxn_hardsphere*sA(3)
    RETURN
  END FUNCTION Npart_avg_fxn_hardsphere

  FUNCTION Npart_hardsphere(bx,by,RR,A,sigma_inel)
    ! Eq.(2.9) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb; at 7-60 GeV, it is averaged as 31.5 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
                                           ! (see Table in http://www.phys.ufl.edu/~korytov/phz4390/note_01_NaturalUnits_SMsummary.pdf)
                                           ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
                                           ! 1 GeV^-2 = 0.38938573 mb = 0.38938573e-31 m^2 = 0.038938573 fm^2
                                           ! 1 mb = 1e-31 m^2 = 0.1 fm^2 
    REAL(KIND(1d0)),DIMENSION(2)::aax,bbx
    INTEGER,DIMENSION(2)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_hardsphere/b_common,RA_common,RB_common,AA_common,AB_common,sigmainel_common
    b_common(1)=bx
    b_common(2)=by
    RA_common=RR(1)
    RB_common=RR(2)
    AA_common=A(1)
    AB_common=A(2)
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    aax(1)=-RR(1)
    bbx(1)=RR(1)
    aax(2)=-RR(1)
    bbx(2)=RR(1)
    sub_num(1)=100
    sub_num(2)=100
    CALL ROMBERG_ND(Npart_fxn_hardsphere,aax,bbx,2,sub_num,1,1d-5,&
         Npart_hardsphere,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
  END FUNCTION NPART_HARDSPHERE

  FUNCTION Npart_fxn_hardsphere(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_fxn_hardsphere
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_hardsphere/b_common,RA_common,RB_common,AA_common,AB_common,sigmainel_common
    REAL(KIND(1d0))::s1,s2
    REAL(KIND(1d0))::TTA,TTB
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: Npart_fxn_hardsphere is not a two dimensional function"
       STOP
    ENDIF
    Npart_fxn_hardsphere=0d0
    s1=DSQRT(sA(1)**2+sA(2)**2)
    IF(s1.LE.RA_common)THEN
       Npart_fxn_hardsphere=0d0
       RETURN
    ENDIF
    s2=DSQRT((sA(1)-b_common(1))**2+(sA(2)-b_common(2))**2)
    IF(s2.GT.RB_common)THEN
       Npart_fxn_hardsphere=0d0
       RETURN
    ENDIF
    TTA=3d0/4d0/pi/RA_common**3*2d0*DSQRT(RA_common**2-s1**2)
    TTB=3d0/4d0/pi/RB_common**3*2d0*DSQRT(RB_common**2-s2**2)
    ! first term
    Npart_fxn_hardsphere=AA_common*TTA*(1D0-(1D0-TTB*sigmainel_common)**INT(AB_common))
    Npart_fxn_hardsphere=Npart_fxn_hardsphere+&
         AB_common*TTB*(1D0-(1D0-TTA*sigmainel_common)**INT(AA_common))
    RETURN
  END FUNCTION Npart_fxn_hardsphere

  FUNCTION Ncoll_avg_hardsphere(bmin,bmax,RR,A,sigma_inel)
    ! integration of bmin to bmax and divide by the bin size, i.e. <Ncoll>=Int[Ncoll[bx,by]dbx dby]/Int[dbx dby]
    ! where Ncoll is Eq.(2.8) in (in unit of 1)
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Ncoll_avg_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax,sigma_inel
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,A
    ! 1D-1 is a factor from mb (the unit of sigma_inel) to fm^2
    Ncoll_avg_hardsphere=TABhat_avg_hardsphere(bmin,bmax,RR)*A(1)*A(2)*sigma_inel*1D-1
    RETURN
  END FUNCTION Ncoll_avg_hardsphere

  FUNCTION TABhat_avg_hardsphere(bmin,bmax,RR)
    ! integration of bmin to bmax and divide by the bin size, i.e. <TABhat>=Int[TABhat[bx,by]dbx dby]/Int[dbx dby]
    ! Thickness function defined in Eq.(2.4) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_avg_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::RA_common,RB_common
    COMMON/TAB_avg_hardsphere/RA_common,RB_common
    ! normalized to 1
    IF(bmin.GE.bmax.OR.bmin.GT.RR(1)+RR(2))THEN
       TABhat_avg_hardsphere=0d0
       RETURN
    ENDIF
    RA_common=RR(1)
    RB_common=RR(2)
    aax(1)=-RR(1)
    bbx(1)=RR(1)
    aax(2)=-RR(1)
    bbx(2)=RR(1)
    aax(3)=bmin
    bbx(3)=bmax
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=100
    CALL ROMBERG_ND(TABhat_avg_fxn_hardsphere,aax,bbx,3,sub_num,1,1d-5,&
         TABhat_avg_hardsphere,ind,eval_num)
    TABhat_avg_hardsphere=TABhat_avg_hardsphere/(0.5d0*(bmax**2-bmin**2))
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    RETURN
  END FUNCTION TABhat_avg_hardsphere

  FUNCTION TABhat_avg_fxn_hardsphere(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_avg_fxn_hardsphere
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::RA_common,RB_common
    COMMON/TAB_avg_hardsphere/RA_common,RB_common
    REAL(KIND(1d0))::s1,s2
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: TABhat_avg_fxn_hardsphere is not a three dimensional function"
       STOP
    ENDIF
    s1=DSQRT(sA(1)**2+sA(2)**2)
    IF(s1.GT.RA_common)THEN
       TABhat_avg_fxn_hardsphere=0d0
       RETURN
    ENDIF
    s2=DSQRT((sA(1)-sA(3))**2+(sA(2))**2)
    IF(s2.GT.RB_common)THEN
       TABhat_avg_fxn_hardsphere=0d0
       RETURN
    ENDIF
    TABhat_avg_fxn_hardsphere=3d0/4d0/pi/RA_common**3*2d0*DSQRT(RA_common**2-s1**2)
    TABhat_avg_fxn_hardsphere=TABhat_avg_fxn_hardsphere*&
         3d0/4d0/pi/RB_common**3*2d0*DSQRT(RB_common**2-s2**2)
    ! jaccobi d^2b -> 2*pi*b*db (drop 2*pi)
    TABhat_avg_fxn_hardsphere=TABhat_avg_fxn_hardsphere*sA(3)
    RETURN
  END FUNCTION TABhat_avg_fxn_hardsphere

  FUNCTION Ncoll_hardsphere(bx,by,RR,A,sigma_inel)
    ! Eq.(2.8) in (in unit of 1) 
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Ncoll_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bx,by,sigma_inel
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,A
    ! 1D-1 is a factor from mb (the unit of sigma_inel) to fm^2
    Ncoll_hardsphere=TABhat_hardsphere(bx,by,RR)*A(1)*A(2)*sigma_inel*1D-1
    RETURN
  END FUNCTION Ncoll_hardsphere

  FUNCTION TABhat_hardsphere_grid(bx,by,RR)
    ! this function will generate a grid first
    ! and store it in the memory
    ! then use interpolations for the next runs
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_hardsphere_grid
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR
    INTEGER::init=0,NA
    SAVE init,NA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA,YA
    SAVE XA,YA
    INTEGER,PARAMETER::NSTEPS=199
    INTEGER::i
    REAL(KIND(1d0))::db,bb
    IF(init.EQ.0)THEN
       WRITE(*,*)"INFO: generate a grid for TABhat in hard sphere (may take a few seconds)"
       NA=NSTEPS+1
       ALLOCATE(XA(NA))
       ALLOCATE(YA(NA))
       db=(RR(1)+RR(2))/DBLE(NSTEPS)
       DO i=1,NA
          XA(i)=db*DBLE(i-1)
          YA(i)=TABhat_hardsphere(XA(i),0d0,RR)
       ENDDO
       init=1
    ENDIF
    bb=DSQRT(bx**2+by**2)
    IF(bb.GT.RR(1)+RR(2))THEN
       TABhat_hardsphere_grid=0d0
    ELSE
       CALL SPLINE_INTERPOLATE(XA,YA,NA,bb,TABhat_hardsphere_grid)
    ENDIF
    RETURN
  END FUNCTION TABhat_hardsphere_grid

  FUNCTION TABhat_hardsphere(bx,by,RR)
    ! Thickness function defined in Eq.(2.4) in 
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR
    REAL(KIND(1d0)),DIMENSION(2)::aax,bbx
    INTEGER,DIMENSION(2)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common
    COMMON/TAB_hardsphere/b_common,RA_common,RB_common
    ! normalized to 1
    IF(DSQRT(bx**2+by**2).GT.RR(1)+RR(2))THEN
       TABhat_hardsphere=0d0
       RETURN
    ENDIF
    b_common(1)=bx
    b_common(2)=by
    RA_common=RR(1)
    RB_common=RR(2)
    aax(1)=-RR(1)
    bbx(1)=RR(1)
    aax(2)=-RR(1)
    bbx(2)=RR(1)
    sub_num(1)=100
    sub_num(2)=100
    CALL ROMBERG_ND(TABhat_fxn_hardsphere,aax,bbx,2,sub_num,1,1d-5,&
         TABhat_hardsphere,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    RETURN
  END FUNCTION TABhat_hardsphere

  FUNCTION TABhat_fxn_hardsphere(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_fxn_hardsphere
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common
    COMMON/TAB_hardsphere/b_common,RA_common,RB_common
    REAL(KIND(1d0))::s1,s2
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: TABhat_fxn_hardsphere is not a two dimensional function"
       STOP
    ENDIF
    s1=DSQRT(sA(1)**2+sA(2)**2)
    IF(s1.GT.RA_common)THEN
       TABhat_fxn_hardsphere=0d0
       RETURN
    ENDIF
    s2=DSQRT((sA(1)-b_common(1))**2+(sA(2)-b_common(2))**2)
    IF(s2.GT.RB_common)THEN
       TABhat_fxn_hardsphere=0d0
       RETURN
    ENDIF
    TABhat_fxn_hardsphere=3d0/4d0/pi/RA_common**3*2d0*DSQRT(RA_common**2-s1**2)
    TABhat_fxn_hardsphere=TABhat_fxn_hardsphere*&
         3d0/4d0/pi/RB_common**3*2d0*DSQRT(RB_common**2-s2**2)
    RETURN
  END FUNCTION TABhat_fxn_hardsphere

  FUNCTION TAhat_hardsphere(ssx,ssy,RR)
    ! Eq.(2.1) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::TAhat_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::ssx,ssy,RR
    REAL(KIND(1d0))::ss
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    ! it is normalized to 1
    ss=DSQRT(ssx**2+ssy**2)
    IF(ss.GT.RR)THEN
       TAhat_hardsphere=0d0
       RETURN
    ENDIF
    TAhat_hardsphere=3d0/4d0/pi/RR**3*2d0*DSQRT(RR**2-ss**2)
    RETURN
  END FUNCTION TAhat_hardsphere
  
  FUNCTION rho_hardsphere(r,RR,A)
    IMPLICIT NONE
    REAL(KIND(1d0))::rho_hardsphere
    REAL(KIND(1d0)),INTENT(IN)::r,RR,A
    REAL(KIND(1d0))::rho
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(r.GE.RR.OR.r.LE.0d0)THEN
       rho_hardsphere=0d0
       RETURN
    ENDIF
    ! rho is try to normalized to A
    ! via d^3r*rho_hardspere(r,R,A)=A
    rho=3d0/4d0/pi/RR**3*A
    rho_hardsphere=rho
    RETURN
  END FUNCTION rho_hardsphere

  FUNCTION SigmaInelAB_WoodsSaxon(RR,w,aa,A,sigma_inel)
    ! in unit of fm^2, 1 fm^2 = 10 mb
    ! calculate the total inelastic cross section of A+B collision
    ! via the integration of Eq.(2.7) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::SigmaInelAB_WoodsSaxon
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb; at 7-60 GeV, it is averaged as 31.5 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
                                           ! (see Table in http://www.phys.ufl.edu/~korytov/phz4390/note_01_NaturalUnits_SMsummary.pdf)
                                           ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
                                           ! 1 GeV^-2 = 0.38938573 mb = 0.38938573e-31 m^2 = 0.038938573 fm^2
                                           ! 1 mb = 1e-31 m^2 = 0.1 fm^2
    REAL(KIND(1d0)),DIMENSION(2)::R_common,w_common,aa_common,A_common
    REAL(KIND(1d0))::sigmainel_common
    COMMON/SigmaInel_WoodsSaxon/R_common,w_common,aa_common,A_common,sigmainel_common
    R_common(1:2)=RR(1:2)
    w_common(1:2)=w(1:2)
    aa_common(1:2)=aa(1:2)
    A_common(1:2)=A(1:2)    
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    CALL trapezoid_integration(1000,SigmaInelAB_fxn_WoodsSaxon,&
         10d0*RR(1)+10d0*RR(2),SigmaInelAB_WoodsSaxon)
    RETURN
  END FUNCTION SigmaInelAB_WoodsSaxon

  FUNCTION SigmaInelAB_fxn_WoodsSaxon(b)
    ! in unit of fm, 1 fm = 10 mb/fm
    IMPLICIT NONE
    REAL(KIND(1d0))::SigmaInelAB_fxn_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::b
    REAL(KIND(1d0)),DIMENSION(2)::R_common,w_common,aa_common,A_common
    REAL(KIND(1d0))::sigmainel_common
    COMMON/SigmaInel_WoodsSaxon/R_common,w_common,aa_common,A_common,sigmainel_common
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    REAL(KIND(1d0))::TAB
    INTEGER::ABAB
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    storegrid=.TRUE.
    ABAB=INT(A_common(1))*INT(A_common(2))
    TAB=TABhat_WoodsSaxon(b,0d0,R_common,w_common,aa_common,A_common)
    SigmaInelAB_fxn_WoodsSaxon=(1d0-GOOD_POWER(1d0-TAB*sigmainel_common,ABAB))*2d0*pi*b
    RETURN
  END FUNCTION SigmaInelAB_fxn_WoodsSaxon

  FUNCTION Npart_avg_WoodsSaxon(bmin,bmax,RR,w,aa,A,sigma_inel)
    ! integration of bmin to bmax and divide by the bin size, i.e. <Npart>=Int[Npart[bx,by]dbx dby]/Int[dbx dby]
    ! where Npart is Eq.(2.9) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_avg_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_avg_WoodsSaxon/RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    IF(bmin.GE.bmax)THEN
       Npart_avg_WoodsSaxon=0d0
       RETURN
    ENDIF
    RA_common=RR(1)
    RB_common=RR(2)
    wA_common=w(1)
    wB_common=w(2)
    aaA_common=aa(1)
    aaB_common=aa(2)
    AA_common=A(1)
    AB_common=A(2)
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    aax(1)=-10d0*RR(1)
    bbx(1)=10d0*RR(1)
    aax(2)=-10d0*RR(1)
    bbx(2)=10d0*RR(1)
    aax(3)=bmin
    bbx(3)=bmax
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=100
    CALL ROMBERG_ND(Npart_avg_fxn_WoodsSaxon,aax,bbx,3,sub_num,1,1d-5,&
         Npart_avg_WoodsSaxon,ind,eval_num)
    Npart_avg_WoodsSaxon=Npart_avg_WoodsSaxon/(0.5d0*(bmax**2-bmin**2))
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
  END FUNCTION Npart_avg_WoodsSaxon

  FUNCTION Npart_avg_fxn_WoodsSaxon(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_avg_fxn_WoodsSaxon
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_avg_WoodsSaxon/RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    REAL(KIND(1d0))::ssx,ssy
    REAL(KIND(1d0))::TTA,TTB
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: Npart_avg_fxn_WoodsSaxon is not a three dimensional function"
       STOP
    ENDIF
    ssx=sA(1)-sA(3)
    ssy=sA(2)
    TTA=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1)
    TTB=TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2)
    ! first term
    Npart_avg_fxn_WoodsSaxon=AA_common*TTA*(1D0-(1D0-TTB*sigmainel_common)**INT(AB_common))
    Npart_avg_fxn_WoodsSaxon=Npart_avg_fxn_WoodsSaxon+&
         AB_common*TTB*(1D0-(1D0-TTA*sigmainel_common)**INT(AA_common))
    ! jaccobi d^2b -> 2*pi*b*db (drop 2*pi)
    Npart_avg_fxn_WoodsSaxon=Npart_avg_fxn_WoodsSaxon*sA(3)
    RETURN
  END FUNCTION Npart_avg_fxn_WoodsSaxon

  FUNCTION Npart_WoodsSaxon(bx,by,RR,w,aa,A,sigma_inel)
    ! Eq.(2.9) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),INTENT(IN)::sigma_inel ! at RHIC 200 GeV, it is 42 mb;
                                           ! at LHC, it is 72 mb (large uncertainty from elastic cross section)
    REAL(KIND(1d0)),DIMENSION(2)::aax,bbx
    INTEGER,DIMENSION(2)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_WoodsSaxon/b_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    b_common(1)=bx
    b_common(2)=by
    RA_common=RR(1)
    RB_common=RR(2)
    wA_common=w(1)
    wB_common=w(2)
    aaA_common=aa(1)
    aaB_common=aa(2)
    AA_common=A(1)
    AB_common=A(2)
    sigmainel_common=sigma_inel*1D-1 ! from mb to fm^2
    aax(1)=-10d0*RR(1)
    bbx(1)=10d0*RR(1)
    aax(2)=-10d0*RR(1)
    bbx(2)=10d0*RR(1)
    sub_num(1)=100
    sub_num(2)=100
    CALL ROMBERG_ND(Npart_fxn_WoodsSaxon,aax,bbx,2,sub_num,1,1d-5,&
         Npart_WoodsSaxon,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
  END FUNCTION NPART_WoodsSaxon

  FUNCTION Npart_fxn_WoodsSaxon(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::Npart_fxn_WoodsSaxon
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    COMMON/Npart_WoodsSaxon/b_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common,sigmainel_common
    REAL(KIND(1d0))::ssx,ssy
    REAL(KIND(1d0))::TTA,TTB
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    IF(dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: Npart_fxn_WoodsSaxon is not a two dimensional function"
       STOP
    ENDIF
    ssx=sA(1)-b_common(1)
    ssy=sA(2)-b_common(2)
    TTA=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1)
    TTB=TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2)
    ! first term
    Npart_fxn_WoodsSaxon=AA_common*TTA*(1D0-(1D0-TTB*sigmainel_common)**INT(AB_common))
    Npart_fxn_WoodsSaxon=Npart_fxn_WoodsSaxon+&
         AB_common*TTB*(1D0-(1D0-TTA*sigmainel_common)**INT(AA_common))
    RETURN
  END FUNCTION Npart_fxn_WoodsSaxon

  FUNCTION Ncoll_avg_WoodsSaxon(bmin,bmax,RR,w,aa,A,sigma_inel)
    ! integration of bmin to bmax and divide by the bin size, i.e. <Ncoll>=Int[Ncoll[bx,by]dbx dby]/Int[dbx dby]
    ! where Ncoll is Eq.(2.8) in (in unit of 1)
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Ncoll_avg_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax,sigma_inel
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    ! 1D-1 is a factor from mb (the unit of sigma_inel) to fm^2
    Ncoll_avg_WoodsSaxon=TABhat_avg_WoodsSaxon(bmin,bmax,RR,w,aa,A)*A(1)*A(2)*sigma_inel*1D-1
    RETURN
  END FUNCTION Ncoll_avg_WoodsSaxon

  FUNCTION TABhat_avg_WoodsSaxon(bmin,bmax,RR,w,aa,A)
    ! integration of bmin to bmax and divide by the bin size, i.e. <TABhat>=Int[TABhat[bx,by]dbx dby]/Int[dbx dby]
    ! Thickness function defined in Eq.(2.4) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_avg_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bmin,bmax
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::RA_common,RB_common
    REAL(KIND(1d0))::wA_common,wB_common
    REAL(KIND(1d0))::aaA_common,aaB_common
    REAL(KIND(1d0))::AA_common,AB_common
    COMMON/TAB_avg_WoodsSaxon/RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    ! normalized to 1
    RA_common=RR(1)
    RB_common=RR(2)
    wA_common=w(1)
    wB_common=w(2)
    aaA_common=aa(1)
    aaB_common=aa(2)
    AA_common=A(1)
    AB_common=A(2)
    aax(1)=-10d0*RR(1)
    bbx(1)=10d0*RR(1)
    aax(2)=-10d0*RR(1)
    bbx(2)=10d0*RR(1)
    aax(3)=bmin
    bbx(3)=bmax
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=100
    CALL ROMBERG_ND(TABhat_avg_fxn_WoodsSaxon,aax,bbx,3,sub_num,1,1d-5,&
         TABhat_avg_WoodsSaxon,ind,eval_num)
    TABhat_avg_WoodsSaxon=TABhat_avg_WoodsSaxon/(0.5d0*(bmax**2-bmin**2))
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    RETURN
  END FUNCTION TABhat_avg_WoodsSaxon

  FUNCTION TABhat_avg_fxn_WoodsSaxon(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_avg_fxn_WoodsSaxon
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::RA_common,RB_common
    REAL(KIND(1d0))::wA_common,wB_common
    REAL(KIND(1d0))::aaA_common,aaB_common
    REAL(KIND(1d0))::AA_common,AB_common
    COMMON/TAB_avg_WoodsSaxon/RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    REAL(KIND(1d0))::ssx,ssy
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: TABhat_avg_fxn_WoodsSaxon is not a three dimensional function"
       STOP
    ENDIF
    ssx=sA(1)-sA(3)
    ssy=sA(2)
    TABhat_avg_fxn_WoodsSaxon=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1)
    TABhat_avg_fxn_WoodsSaxon=TABhat_avg_fxn_WoodsSaxon*&
         TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2)
    ! jaccobi d^2b -> 2*pi*b*db (drop 2*pi)
    TABhat_avg_fxn_WoodsSaxon=TABhat_avg_fxn_WoodsSaxon*sA(3)
    RETURN
  END FUNCTION TABhat_avg_fxn_WoodsSaxon

  FUNCTION Ncoll_WoodsSaxon(bx,by,RR,w,aa,A,sigma_inel)
    ! Eq.(2.8) in (in unit of 1)
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::Ncoll_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by,sigma_inel
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    storegrid=.TRUE.
    ! 1D-1 is a factor from mb (the unit of sigma_inel) to fm^2
    Ncoll_WoodsSaxon=TABhat_WoodsSaxon(bx,by,RR,w,aa,A)*A(1)*A(2)*sigma_inel*1D-1
    RETURN
  END FUNCTION Ncoll_WoodsSaxon

  FUNCTION TAB_MC_Glauber(bx,by,A,Z)
    ! this gives us TAB from Glauber MC
    ! in unit of fm-2 for TAB
    ! 1 fm^2 = 10 mb
    ! 1 mb-1 = 10 fm-2
    IMPLICIT NONE
    REAL(KIND(1d0))::TAB_MC_Glauber ! in unit of fm-2
    REAL(KIND(1d0)),INTENT(IN)::bx,by ! in unit of fm
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::A,Z
    REAL(KIND(1d0)),DIMENSION(0:9)::pp
    REAL(KIND(1d0))::bb,pden
    INTEGER::i
    IF(A(1).EQ.208d0.AND.A(2).EQ.208d0.AND.Z(1).EQ.82d0.AND.Z(2).EQ.82d0)THEN
       ! fitted by David d'Enterria
       ! Pb208+Pb208
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.17d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=68.3d0
       pp(0)=2.24771D+09/pden
       pp(1)=1.10956D+06/pden
       pp(2)=-5.43303D+07/pden
       pp(3)=3.57902D+06/pden
       pp(4)=93705.9D0/pden
       pp(5)=-13485D0/pden
       pp(6)=297.642D0/pden
       pp(7)=1.10214D+06
       pp(8)=-11.53D0
       pp(9)=1.81509D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSEIF(A(1).EQ.129d0.AND.A(2).EQ.129d0.AND.Z(1).EQ.54d0.AND.Z(2).EQ.54d0)THEN
       ! fitted by David d'Enterria
       ! Xe129+Xe129
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.15d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=68.1d0
       pp(0)=1.99693D+07/pden
       pp(1)=222138D0/pden
       pp(2)=-729708D0/pden
       pp(3)=59685.2D0/pden
       pp(4)=1061.42D0/pden
       pp(5)=-251.573D0/pden
       pp(6)=6.75182D0/pden
       pp(7)=18440.5D0
       pp(8)=-9.13656D0
       pp(9)=2.13687D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSEIF(A(1).EQ.78d0.AND.A(2).EQ.78d0.AND.Z(1).EQ.36d0.AND.Z(2).EQ.36d0)THEN
       ! fitted by David d'Enterria
       ! Kr78+Kr78
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.13d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=70d0
       pp(0)=2.89965D+07/pden
       pp(1)=391154D0/pden
       pp(2)=-1.40566D+06/pden
       pp(3)=117449D0/pden
       pp(4)=6235.26D0/pden
       pp(5)=-1019.49D0/pden
       pp(6)=30.2962D0/pden
       pp(7)=50524.8D0
       pp(8)=-13.4926D0
       pp(9)=2.17738D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSEIF(A(1).EQ.40d0.AND.A(2).EQ.40d0.AND.Z(1).EQ.20d0.AND.Z(2).EQ.20d0)THEN
       ! fitted by David d'Enterria
       ! Ca40+Ca40
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.11d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=70.8d0
       pp(0)=2.30564D+07/pden
       pp(1)=573687D0/pden
       pp(2)=-1.05777D+06/pden
       pp(3)=-39228D0/pden
       pp(4)=36164.8D0/pden
       pp(5)=-3456.49D0/pden
       pp(6)=100.796D0/pden
       pp(7)=105117D0
       pp(8)=-30.3963D0
       pp(9)=3.28039D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSEIF(A(1).EQ.40d0.AND.A(2).EQ.40d0.AND.Z(1).EQ.18d0.AND.Z(2).EQ.18d0)THEN
       ! fitted by David d'Enterria
       ! Ar40+Ar40
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.12d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=69.7d0
       pp(0)=2.71179D+07/pden
       pp(1)=1.47978D+06/pden
       pp(2)=-1.44297D+06/pden
       pp(3)=91226.3D0/pden
       pp(4)=4224.01D0/pden
       pp(5)=-85.6026D0/pden
       pp(6)=-20.5078D0/pden
       pp(7)=116171D0
       pp(8)=-23.2043D0
       pp(9)=2.40569D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSEIF(A(1).EQ.16d0.AND.A(2).EQ.16d0.AND.Z(1).EQ.8d0.AND.Z(2).EQ.8d0)THEN
       ! fitted by David d'Enterria
       ! O16+O16
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.11d0)THEN
          TAB_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=70.8d0
       pp(0)=-881.419D0/pden
       pp(1)=176115D0/pden
       pp(2)=33767.2D0/pden
       pp(3)=-21042.1D0/pden
       pp(4)=154.837D0/pden
       pp(5)=521.076D0/pden
       pp(6)=-39.6948D0/pden
       pp(7)=-10584.2D0
       pp(8)=-30.5333D0
       pp(9)=3.29537D0
       ! the fit gives in unit of mb-1
       TAB_MC_Glauber=0d0
       DO i=0,6
          TAB_MC_Glauber=TAB_MC_Glauber+pp(i)*bb**i
       ENDDO
       TAB_MC_Glauber=TAB_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TAB_MC_Glauber=TAB_MC_Glauber*10d0
    ELSE
       WRITE(*,*)"Error: it is not implemented yet for A1,A2,Z1,Z2=",A(1:2),Z(1:2)
       STOP
    ENDIF
    RETURN
  END FUNCTION TAB_MC_Glauber

  FUNCTION TABhat_WoodsSaxon_grid(bx,by,RR,w,aa,A)
    ! this function will generate a grid first
    ! and store it in the memory
    ! then use interpolations for the next runs
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_WoodsSaxon_grid
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    INTEGER::init=0,NA
    SAVE init,NA
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA,YA
    SAVE XA,YA
    INTEGER,PARAMETER::NSTEPS=199
    INTEGER::i
    REAL(KIND(1d0))::db,bb
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    IF(init.EQ.0)THEN
       storegrid=.FALSE.
       WRITE(*,*)"INFO: generate a grid for TABhat in Woods Saxon (may take a few seconds)"
       NA=NSTEPS+1
       ALLOCATE(XA(NA))
       ALLOCATE(YA(NA))
       db=10d0*(RR(1)+RR(2))/DBLE(NSTEPS)
       DO i=1,NA
          XA(i)=db*DBLE(i-1)
          YA(i)=TABhat_WoodsSaxon(XA(i),0d0,RR,w,aa,A)
       ENDDO
       init=1
    ENDIF
    bb=DSQRT(bx**2+by**2)
    IF(bb.GT.10d0*(RR(1)+RR(2)))THEN
       TABhat_WoodsSaxon_grid=0d0
    ELSE
       CALL SPLINE_INTERPOLATE(XA,YA,NA,bb,TABhat_WoodsSaxon_grid)
    ENDIF
    RETURN
  END FUNCTION TABhat_WoodsSaxon_grid

  FUNCTION TABhat_WoodsSaxon(bx,by,RR,w,aa,A)
    ! Thickness function defined in Eq.(2.4) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::bx,by
    REAL(KIND(1d0)),DIMENSION(2),INTENT(IN)::RR,w,aa,A
    REAL(KIND(1d0)),DIMENSION(2)::aax,bbx
    INTEGER,DIMENSION(2)::sub_num
    INTEGER::ind,eval_num
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common
    REAL(KIND(1d0))::wA_common,wB_common
    REAL(KIND(1d0))::aaA_common,aaB_common
    REAL(KIND(1d0))::AA_common,AB_common
    COMMON/TAB_WoodsSaxon/b_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    ! normalized to 1
    b_common(1)=bx
    b_common(2)=by
    RA_common=RR(1)
    RB_common=RR(2)
    wA_common=w(1)
    wB_common=w(2)
    aaA_common=aa(1)
    aaB_common=aa(2)
    AA_common=A(1)
    AB_common=A(2)
    aax(1)=-10d0*RR(1)
    bbx(1)=10d0*RR(1)
    aax(2)=-10d0*RR(1)
    bbx(2)=10d0*RR(1)
    sub_num(1)=100
    sub_num(2)=100
    CALL ROMBERG_ND(TABhat_fxn_WoodsSaxon,aax,bbx,2,sub_num,1,1d-5,&
         TABhat_WoodsSaxon,ind,eval_num)
    IF(ind.EQ.-1)THEN
       WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
    ENDIF
    RETURN
  END FUNCTION TABhat_WoodsSaxon

  FUNCTION TABhat_fxn_WoodsSaxon(dim_num,sA)
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat_fxn_WoodsSaxon
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0)),DIMENSION(2)::b_common
    REAL(KIND(1d0))::RA_common,RB_common
    REAL(KIND(1d0))::wA_common,wB_common
    REAL(KIND(1d0))::aaA_common,aaB_common
    REAL(KIND(1d0))::AA_common,AB_common
    COMMON/TAB_WoodsSaxon/b_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    REAL(KIND(1d0))::ssx,ssy
    IF(dim_num.NE.2)THEN
       WRITE(*,*)"ERROR: TABhat_fxn_WoodsSaxon is not a two dimensional function"
       STOP
    ENDIF
    ssx=sA(1)-b_common(1)
    ssy=sA(2)-b_common(2)
    TABhat_fxn_WoodsSaxon=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1,storegrid)
    TABhat_fxn_WoodsSaxon=TABhat_fxn_WoodsSaxon*&
         TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2,storegrid)
    RETURN
  END FUNCTION TABhat_fxn_WoodsSaxon

  FUNCTION TA_MC_Glauber(bx,by,A)
    ! this gives us TA from Glauber MC
    ! in unit of fm-2 for TA
    ! 1 fm^2 = 10 mb
    ! 1 mb-1 = 10 fm-2
    IMPLICIT NONE
    REAL(KIND(1d0))::TA_MC_Glauber ! in unit of fm-2                                                                                    
    REAL(KIND(1d0)),INTENT(IN)::bx,by ! in unit of fm                                                                                   
    REAL(KIND(1d0)),INTENT(IN)::A
    REAL(KIND(1d0)),DIMENSION(0:9)::pp
    REAL(KIND(1d0))::bb,pden
    INTEGER::i
    IF(A.EQ.208d0)THEN
       ! fitted by David d'Enterria
       ! Pb208
       bb=DSQRT(bx**2+by**2)
       IF(bb.GT.10d0)THEN
          TA_MC_Glauber=0d0
          RETURN
       ENDIF
       pden=73.2d0
       pp(0)=50108.2D0/pden
       pp(1)=-376.504D0/pden
       pp(2)=405.334D0/pden
       pp(3)=-728.447D0/pden
       pp(4)=247.782D0/pden
       pp(5)=-36.4531D0/pden
       pp(6)=1.94578D0/pden
       pp(7)=3236.32D0
       pp(8)=-2.02486D0
       pp(9)=1.02466D0
       ! the fit gives in unit of mb-1
       TA_MC_Glauber=0d0
       DO i=0,6
          TA_MC_Glauber=TA_MC_Glauber+pp(i)*bb**i
       ENDDO
       TA_MC_Glauber=TA_MC_Glauber/(pp(7)+DEXP((bb-pp(8))/pp(9)))
       ! from mb-1 to 1 fm-2
       TA_MC_Glauber=TA_MC_Glauber*10d0
    ELSE
       WRITE(*,*)"Error: it is not implemented yet for A=",A
       STOP
    ENDIF
    RETURN
  END FUNCTION TA_MC_Glauber

  FUNCTION TAhat_WoodsSaxon(ssx,ssy,RR,w,aa,A,IMETH,STOREGRID)
    ! Eq.(2.1) in
    ! http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    ! IMETH: 0, always get the TAhat from direct calculation
    ! IMETH: 1, generate the A grid and then use interpolation
    ! IMETH: 2, generate the B grid and then use interpolation
    IMPLICIT NONE
    REAL(KIND(1d0))::TAhat_WoodsSaxon
    INTEGER,INTENT(IN)::IMETH
    REAL(KIND(1d0)),INTENT(IN)::ssx,ssy,RR,w,aa,A
    LOGICAL,INTENT(IN),OPTIONAL::STOREGRID ! if true, store the grid on disk. Otherwise, store it in memory
    REAL(KIND(1d0))::ss
    REAL(KIND(1d0))::ss_common,R_common,w_common,aa_common,A_common
    COMMON/TA_WoodsSaxon/ss_common,R_common,w_common,aa_common,A_common
    INTEGER::init1=0,init2=0,NA,NB
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::XA,YA,XB,YB
    SAVE init1,init2,NA,XA,YA,NB,XB,YB
    INTEGER::imethod,i
    REAL(KIND(1d0))::error
    LOGICAL::lexist,store_grid
    INTEGER,PARAMETER::NSTEPS=199
    REAL(KIND(1d0))::ds
    imethod=IMETH
    IF(init1.EQ.0.AND.imethod.EQ.1)THEN
       IF(.NOT.PRESENT(STOREGRID))THEN
          store_grid=.TRUE.
       ELSE
          store_grid=STOREGRID
       ENDIF
       ! first to check the grid
       IF(store_grid)THEN
          INQUIRE(FILE=TRIM(nuclear_dir)//"grid/TAhat_WoodsSaxon.grid",EXIST=lexist)
       ELSE
          lexist=.FALSE.
       ENDIF
       IF(lexist)THEN
          OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TAhat_WoodsSaxon.grid')
          READ(30307,*)NA,R_common,w_common,aa_common
          IF(R_common.NE.RR.OR.w_common.NE.w.OR.aa_common.NE.aa)THEN
             ! regenerate the grid
             CLOSE(UNIT=30307)
             WRITE(*,*)"INFO: generate A grid for TAhat in Woods-Saxon (may take a few seconds)"
             IF(store_grid)OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TAhat_WoodsSaxon.grid')
             R_common=RR
             w_common=w
             aa_common=aa
             A_common=A
             ds=10d0*RR/NSTEPS
             IF(store_grid)THEN
                WRITE(30307,*)NSTEPS+1,R_common,w_common,aa_common
             ELSE
                NA=NSTEPS+1
                ALLOCATE(XA(NA))
                ALLOCATE(YA(NA))
             ENDIF
             DO i=0,NSTEPS
                ss=i*ds
                ss_common=ss
                CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
                TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
                IF(store_grid)THEN
                   WRITE(30307,*)ss,TAhat_WoodsSaxon
                ELSE
                   XA(i+1)=ss
                   YA(i+1)=TAhat_WoodsSaxon
                ENDIF
             ENDDO
             IF(store_grid)CLOSE(UNIT=30307)
          ELSE
             CLOSE(UNIT=30307)
          ENDIF
       ELSE
          WRITE(*,*)"INFO: generate A grid for TAhat in Woods-Saxon (may take a few seconds)"
          IF(store_grid)OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TAhat_WoodsSaxon.grid')
          R_common=RR
          w_common=w
          aa_common=aa
          A_common=A
          ds=10d0*RR/NSTEPS
          IF(store_grid)THEN
             WRITE(30307,*)NSTEPS+1,R_common,w_common,aa_common
          ELSE
             NA=NSTEPS+1
             ALLOCATE(XA(NA))
             ALLOCATE(YA(NA))
          ENDIF
          DO i=0,NSTEPS
             ss=i*ds
             ss_common=ss
             CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
             TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
             IF(store_grid)THEN
                WRITE(30307,*)ss,TAhat_WoodsSaxon
             ELSE
                XA(i+1)=ss
                YA(i+1)=TAhat_WoodsSaxon
             ENDIF
          ENDDO
          IF(store_grid)CLOSE(UNIT=30307)
       ENDIF
       IF(store_grid)THEN
          OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TAhat_WoodsSaxon.grid')
          READ(30307,*)NA,R_common,w_common,aa_common
          ALLOCATE(XA(NA))
          ALLOCATE(YA(NA))
          DO i=1,NA
             READ(30307,*)XA(i),YA(i)
          ENDDO
          CLOSE(UNIT=30307)
       ENDIF
       init1=1
    ENDIF
    IF(init2.EQ.0.AND.imethod.EQ.2)THEN
       IF(.NOT.PRESENT(STOREGRID))THEN
          store_grid=.TRUE.
       ELSE
          store_grid=STOREGRID
       ENDIF
       ! first to check the grid
       IF(store_grid)THEN
          INQUIRE(FILE=TRIM(nuclear_dir)//"/grid/TBhat_WoodsSaxon.grid",EXIST=lexist)
       ELSE
          lexist=.FALSE.
       ENDIF
       IF(lexist)THEN
          OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'/grid/TBhat_WoodsSaxon.grid')
          READ(30307,*)NB,R_common,w_common,aa_common
          IF(R_common.NE.RR.OR.w_common.NE.w.OR.aa_common.NE.aa)THEN
             ! regenerate the grid
             CLOSE(UNIT=30307)
             WRITE(*,*)"INFO: generate B grid for TAhat in Woods-Saxon (may take a few seconds)"
             IF(store_grid)OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TBhat_WoodsSaxon.grid')
             R_common=RR
             w_common=w
             aa_common=aa
             A_common=A
             ds=10d0*RR/NSTEPS
             IF(store_grid)THEN
                WRITE(30307,*)NSTEPS+1,R_common,w_common,aa_common
             ELSE
                NB=NSTEPS+1
                ALLOCATE(XB(NB))
                ALLOCATE(YB(NB))
             ENDIF
             DO i=0,NSTEPS
                ss=i*ds
                ss_common=ss
                CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
                TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
                IF(store_grid)THEN
                   WRITE(30307,*)ss,TAhat_WoodsSaxon
                ELSE
                   XB(i+1)=ss
                   YB(i+1)=TAhat_WoodsSaxon
                ENDIF
             ENDDO
             IF(store_grid)CLOSE(UNIT=30307)
          ELSE
             CLOSE(UNIT=30307)
          ENDIF
       ELSE
          WRITE(*,*)"INFO: generate B grid for TAhat in Woods-Saxon (may take a few seconds)"
          IF(store_grid)OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TBhat_WoodsSaxon.grid')
          R_common=RR
          w_common=w
          aa_common=aa
          A_common=A
          ds=10d0*RR/NSTEPS
          IF(store_grid)THEN
             WRITE(30307,*)NSTEPS+1,R_common,w_common,aa_common
          ELSE
             NB=NSTEPS+1
             ALLOCATE(XB(NB))
             ALLOCATE(YB(NB))
          ENDIF
          DO i=0,NSTEPS
             ss=i*ds
             ss_common=ss
             CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
             TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
             IF(store_grid)THEN
                WRITE(30307,*)ss,TAhat_WoodsSaxon
             ELSE
                XB(i+1)=ss
                YB(i+1)=TAhat_WoodsSaxon
             ENDIF
          ENDDO
          IF(store_grid)CLOSE(UNIT=30307)
       ENDIF
       IF(store_grid)THEN
          OPEN(UNIT=30307,FILE=TRIM(nuclear_dir)//'grid/TBhat_WoodsSaxon.grid')
          READ(30307,*)NB,R_common,w_common,aa_common
          ALLOCATE(XB(NB))
          ALLOCATE(YB(NB))
          DO i=1,NB
             READ(30307,*)XB(i),YB(i)
          ENDDO
       ENDIF
       init2=1
    ENDIF
    IF(imethod.EQ.0)THEN
       ss=DSQRT(ssx**2+ssy**2)
       ss_common=ss
       R_common=RR
       w_common=w
       aa_common=aa
       A_common=A
       CALL trapezoid_integration(10000,TAhat_fxn_WoodsSaxon,50d0*RR,TAhat_WoodsSaxon)
       TAhat_WoodsSaxon=TAhat_WoodsSaxon/A
    ELSEIF(imethod.EQ.1)THEN
       ss=DSQRT(ssx**2+ssy**2)
       IF(ss.GT.10d0*RR)THEN
          TAhat_WoodsSaxon=0d0
       ELSE
          CALL SPLINE_INTERPOLATE(XA,YA,NA,ss,TAhat_WoodsSaxon)
          !CALL POLYNOMINAL_INTERPOLATE(XA,YA,NA,ss,TAhat_WoodsSaxon,error)
       ENDIF
    ELSE
       ss=DSQRT(ssx**2+ssy**2)
       IF(ss.GT.10d0*RR)THEN
          TAhat_WoodsSaxon=0d0
       ELSE
          CALL SPLINE_INTERPOLATE(XA,YA,NA,ss,TAhat_WoodsSaxon)
          !CALL POLYNOMINAL_INTERPOLATE(XB,YB,NB,ss,TAhat_WoodsSaxon,error)
       ENDIF
    ENDIF
    RETURN
  END FUNCTION TAhat_WoodsSaxon

  FUNCTION TAhat_fxn_WoodsSaxon(zA)
    IMPLICIT NONE
    REAL(KIND(1d0))::TAhat_fxn_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::zA
    REAL(KIND(1d0))::ss_common,R_common,w_common,aa_common,A_common
    COMMON/TA_WoodsSaxon/ss_common,R_common,w_common,aa_common,A_common
    REAL(KIND(1d0))::r
    r=DSQRT(ss_common**2+zA**2)
    ! the prefactor 2 is coming from the symmetric of zA
    TAhat_fxn_WoodsSaxon=2d0*rho_WoodsSaxon(r,R_common,w_common,aa_common,A_common)
    RETURN
  END FUNCTION TAhat_fxn_WoodsSaxon

  FUNCTION rho_WoodsSaxon(r,RR,w,aa,A,NumericIntQ)
    USE nielsen_generalized_polylog
    IMPLICIT NONE
    LOGICAL,INTENT(IN),OPTIONAL::NumericIntQ
    LOGICAL::numericintqq
    REAL(KIND(1d0))::rho_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::r,RR,w,aa,A
    INTEGER::init=0
    REAL(KIND(1d0))::rho0_save,R_save,aa_save,A_save,w_save
    SAVE init,rho0_save,R_save,aa_save,A_save,w_save
    REAL(KIND(1d0))::R_common,w_common,aa_common ! used by norho0_WoodsSaxon
    COMMON/WoodsSaxon/R_common,w_common,aa_common
    REAL(KIND(1d0))::rho0
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    COMPLEX(KIND(1d0))::Li3val,Li5val
    REAL(KIND(1d0))::RoA
    IF(init.EQ.0)THEN
       ! integrate the function via d^3r in order to normalize it
       ! to A
       A_save=A
       w_save=w
       w_common=w
       aa_save=aa
       aa_common=aa
       R_save=RR
       R_common=RR
       numericintqq=.FALSE.
       IF(PRESENT(NumericIntQ))THEN
          numericintqq=NumericIntQ
       ENDIF
       IF(numericintqq)THEN
          CALL trapezoid_integration(10000,norho0_WoodsSaxon,50d0*RR,rho0)                                            
          rho0=A/4d0/rho0/pi
       ELSE
          IF(w.NE.0d0)THEN
             ! for w=!=0, we also know it analytically
             ! cf. eq.(7.12) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
             RoA=RR/aa
             Li3val=Nielsen_PolyLog(2,1,-DEXP(RoA))
             Li5val=Nielsen_PolyLog(4,1,-DEXP(RoA))
             rho0=A/(-8d0*pi*aa**3*(DREAL(Li3val)+12d0*w/RoA**2*DREAL(Li5val)))
          ELSE
             ! for w=0, we know it analytically
             ! cf. eq.(21) in Maximon and Schrack, J. Res. Natt. Bur. Stand B70 (1966)
             ! or eq.(7.9) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
             Li3val=Nielsen_PolyLog(2,1,-DEXP(-RR/aa))
             rho0=A/(4d0*pi/3d0*RR*(RR**2+(pi*aa)**2)-8d0*pi*aa**3*DREAL(Li3val))
          ENDIF
       ENDIF
       rho0_save=rho0
       init=1
    ELSE
       IF(A_save.NE.A.OR.w_save.NE.w.OR.aa_save.NE.aa.OR.R_save.NE.RR)THEN
          ! update the rho0_save
          WRITE(*,*)"WARNING:Will update the saved parameters in rho_WoodsSaxon !"
          A_save=A
          w_save=w
          w_common=w
          aa_save=aa
          aa_common=aa
          R_save=RR
          R_common=RR
          numericintqq=.FALSE.
          IF(PRESENT(NumericIntQ))THEN
             numericintqq=NumericIntQ
          ENDIF
          IF(numericintqq)THEN
             CALL trapezoid_integration(10000,norho0_WoodsSaxon,50d0*RR,rho0)
             rho0=A/4d0/rho0/pi
          ELSE
             IF(w.NE.0d0)THEN
                ! for w=!=0, we also know it analytically
                ! cf. eq.(7.12) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
                RoA=RR/aa
                Li3val=Nielsen_PolyLog(2,1,-DEXP(RoA))
                Li5val=Nielsen_PolyLog(4,1,-DEXP(RoA))
                rho0=A/(-8d0*pi*aa**3*(DREAL(Li3val)+12d0*w/RoA**2*DREAL(Li5val)))
             ELSE
                ! for w=0, we know it analytically
                ! cf. eq.(21) in Maximon and Schrack, J. Res. Natt. Bur. Stand B70 (1966)
                ! or eq.(7.9) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
                Li3val=Nielsen_PolyLog(2,1,-DEXP(-RR/aa))
                rho0=A/(4d0*pi/3d0*RR*(RR**2+(pi*aa)**2)-8d0*pi*aa**3*DREAL(Li3val))
             ENDIF
          ENDIF
          rho0_save=rho0
       ENDIF
    ENDIF
    rho_WoodsSaxon=rho0_save*(1d0+w*(r/RR)**2)/(1d0+DEXP((r-RR)/aa))
    RETURN
  END FUNCTION rho_WoodsSaxon

  FUNCTION norho0_WoodsSaxon(r)
    ! Eq.(1.1) in http://cds.cern.ch/record/1595014/files/CERN%20report.pdf
    ! with rho0=1
    ! times r**2 (the measure)
    IMPLICIT NONE
    REAL(KIND(1d0))::norho0_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::r
    REAL(KIND(1d0))::RR,w,aa
    COMMON/WoodsSaxon/RR,w,aa
    norho0_WoodsSaxon=r**2*(1d0+w*(r/RR)**2)/(1d0+DEXP((r-RR)/aa))
    RETURN
  END FUNCTION norho0_WoodsSaxon

  ! it used the one-dimensional integration via 
  ! a to b
  SUBROUTINE simpson(f,a,b,integral,n)
    !==========================================================
    ! Integration of f(x) on [a,b]
    ! Method: Simpson rule for n intervals  
    ! written by: Alex Godunov (October 2009)
    !----------------------------------------------------------
    ! IN:
    ! f   - Function to integrate (supplied by a user)
    ! a  - Lower limit of integration
    ! b  - Upper limit of integration
    ! n   - number of intervals
    ! OUT:
    ! integral - Result of integration
    !==========================================================
    IMPLICIT NONE
    REAL(KIND(1d0)),EXTERNAL::f
    REAL(KIND(1d0)),INTENT(IN)::a, b
    REAL(KIND(1d0)),INTENT(OUT)::integral
    REAL(KIND(1d0))::s
    REAL(KIND(1d0))::h, x
    INTEGER::ninit,i
    INTEGER,INTENT(INOUT)::n
    ! if n is odd we add +1 to make it even
    IF((n/2)*2.ne.n) n=n+1
    ! loop over n (number of intervals)
    s = 0.0D0
    h = (b-a)/DBLE(n)
    DO i=2, n-2, 2
       x   = a+DBLE(i)*h
       s = s + 2.0*f(x) + 4.0*f(x+h)
    ENDDO
    integral = (s + f(a) + f(b) + 4.0*f(a+h))*h/3.0
    RETURN
  END SUBROUTINE simpson

  ! it used the one-dimensional integration via
  ! 0 to end_val
  SUBROUTINE trapezoid_integration(n,fxn,end_val,res)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(OUT)::res
    REAL(KIND(1d0)),EXTERNAL::fxn
    INTEGER,INTENT(IN)::n ! it is total number of intervals in the x
    REAL(KIND(1d0)),INTENT(IN)::end_val ! the upper value of integration
    REAL(KIND(1d0))::u,h
    INTEGER::i
    res=0d0
    DO i=0,n
       u=(end_val*i)/n
       IF(i.EQ.0.OR.i.EQ.n)THEN
          res=res+fxn(u)
       ELSE
          res=res+2d0*fxn(u)
       ENDIF
    ENDDO
    h=end_val/n
    res=(h/2d0)*res
    RETURN
  END SUBROUTINE trapezoid_integration

  FUNCTION GOOD_POWER(x,n)
    IMPLICIT NONE
    REAL(KIND(1d0))::GOOD_POWER
    REAL(KIND(1d0)),INTENT(IN)::x
    INTEGER,INTENT(IN)::n
    REAL(KIND(1d0))::expx
    REAL(KIND(1d0)),PARAMETER::threshold=-20d0
    expx=DLOG(x)*n
    IF(expx.LT.threshold)THEN
       GOOD_POWER=0d0
    ELSE
       GOOD_POWER=DEXP(expx)
    ENDIF
    RETURN
  END FUNCTION GOOD_POWER

  FUNCTION sigma_inelastic(energy)
    ! in unit of mb, 1 mb = 0.1 fm^2
    ! use the interpolation to get the sigma inelastic in unit of mb
    ! most of the input data are from Figure 4 in 1712.06153
    IMPLICIT NONE
    REAL(KIND(1d0))::sigma_inelastic
    REAL(KIND(1d0)),INTENT(IN)::energy ! in unit of GeV
    INTEGER,PARAMETER::NMAXD=100
    INTEGER::NDATA
    REAL(KIND(1d0)),DIMENSION(NMAXD,2)::sigma_grid
    INTEGER::init=0,i
    CHARACTER(len=100)::COMMENT
    SAVE init,NDATA,sigma_grid
    IF(init.EQ.0.AND.sigmaNN_inelastic_eval.EQ.2)THEN
       NDATA=0
       OPEN(UNIT=230555,FILE=TRIM(nuclear_dir)//"input/sigmapp_inel.inp")
       ! three comment lines
       READ(230555,*)COMMENT
       READ(230555,*)COMMENT
       READ(230555,*)COMMENT
       DO WHILE(.TRUE.)
          NDATA=NDATA+1
          READ(230555,*,ERR=230,END=230)sigma_grid(NDATA,1),sigma_grid(NDATA,2)
       ENDDO
230    CONTINUE
       CLOSE(UNIT=230555)
       NDATA=NDATA-1
       init=1
    ENDIF
    IF(sigmaNN_inelastic_eval.EQ.2)THEN
       IF(NDATA.LE.0)THEN
          WRITE(*,*)"WARNING: failed to get sigma inelastic scattering in sigma_inelastic!"
          sigma_inelastic=0d0
          RETURN
       ENDIF
       CALL SPLINE_INTERPOLATE(sigma_grid(1:NDATA,1),sigma_grid(1:NDATA,2),&
            NDATA,energy,sigma_inelastic)
    ELSEIF(sigmaNN_inelastic_eval.EQ.1)THEN
       ! could also try the fitted parameterisation from DdE (2011.14909)
       ! a+b*log^n(s), with a=28.84 mb, b=0.0456 mb, n=2.374, s in GeV^2
       sigma_inelastic=28.84d0+0.0456d0*DLOG(energy**2)**(2.374d0)
    ELSE
       WRITE(*,*)"ERROR: do not know sigmaNN_inelastic_eval=",sigmaNN_inelastic_eval
       STOP
    ENDIF
    !PRINT *, sigma_inelastic
    RETURN
  END FUNCTION sigma_inelastic

  SUBROUTINE GetNuclearInfo(name,A,Z,R,aa,w)
    IMPLICIT NONE
    CHARACTER(len=7),INTENT(IN)::name
    REAL(KIND(1d0)),INTENT(OUT)::A,Z,R,aa,w
    CHARACTER(len=100)::COMMENT
    LOGICAL::found
    CHARACTER(len=7)::temp
    INTEGER,PARAMETER::data_len=42
    CHARACTER(len=5),DIMENSION(data_len)::ion_names=(/'H2   ','Li7  ','Be9  ','B10  ','B11  ','C13  ',&
         'C14  ','N14  ','N15  ','O16  ','Ne20 ','Mg24 ','Mg25 ','Al27 ','Si28 ',&
         'Si29 ','Si30 ','P31  ','Cl35 ','Cl37 ','Ar40 ','K39  ','Ca40 ','Ca48 ','Fe56 ',&
         'Ni58 ','Ni60 ','Ni61 ','Ni62 ','Ni64 ','Cu63 ','Kr78 ','Ag110','Sb122','Xe129',&
         'Xe132','Nd142','Er166','W186 ','Au197','Pb207','Pb208'/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_A=(/2d0,7d0,9d0,10d0,11d0,13d0,&
         14d0,14d0,15d0,16d0,20d0,24d0,25d0,27d0,28d0,&
         29d0,30d0,31d0,35d0,37d0,40d0,39d0,40d0,48d0,56d0,&
         58d0,60d0,61d0,62d0,64d0,63d0,78d0,110d0,122d0,129d0,&
         132d0,142d0,166d0,186d0,197d0,207d0,208d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_Z=(/1d0,3d0,4d0,5d0,5d0,6d0,&
         6d0,7d0,7d0,8d0,10d0,12d0,12d0,13d0,14d0,&
         14d0,14d0,15d0,17d0,17d0,18d0,19d0,20d0,20d0,26d0,&
         28d0,28d0,28d0,28d0,28d0,29d0,36d0,47d0,51d0,54d0,&
         54d0,60d0,68d0,74d0,79d0,82d0,82d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_R=(/0.01d0,1.77d0,1.791d0,1.71d0,1.69d0,1.635d0,&
         1.73d0,2.570d0,2.334d0,2.608d0,2.791d0,3.108d0,3.22d0,3.07d0,3.340d0,&
         3.338d0,3.338d0,3.369d0,3.476d0,3.554d0,3.766d0,3.743d0,3.766d0,3.7369d0,4.074d0,&
         4.3092d0,4.4891d0,4.4024d0,4.4425d0,4.5211d0,4.214d0,4.5d0,5.33d0,5.32d0,5.36d0,&
         5.4d0,5.6135d0,5.98d0,6.58d0,6.38d0,6.62d0,6.624d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_aa=(/0.5882d0,0.327d0,0.611d0,0.837d0,0.811d0,1.403d0,&
         1.38d0,0.5052d0,0.498d0,0.513d0,0.698d0,0.607d0,0.58d0,0.519d0,0.580d0,&
         0.547d0,0.547d0,0.582d0,0.599d0,0.588d0,0.586d0,0.595d0,0.586d0,0.5245d0,0.536d0,&
         0.5169d0,0.5369d0,0.5401d0,0.5386d0,0.5278d0,0.586d0,0.5d0,0.535d0,0.57d0,0.59d0,&
         0.61d0,0.5868d0,0.446d0,0.480d0,0.535d0,0.546d0,0.549d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_w=(/0d0,0d0,0d0,0d0,0d0,0d0,&
         0d0,-0.180d0,0.139d0,-0.051d0,-0.168d0,-0.163d0,-0.236d0,0d0,-0.233d0,&
         -0.203d0,-0.203d0,-0.173d0,-0.10d0,-0.13d0,-0.161d0,-0.201d0,-0.161d0,-0.030d0,0d0,&
         -0.1308d0,-0.2668d0,-0.1983d0,-0.2090d0,-0.2284d0,0d0,0d0,0d0,0d0,0d0,&
         0d0,0.096d0,0.19d0,0d0,0d0,0d0,0d0/)
    INTEGER::I
    !OPEN(UNIT=20565,FILE=TRIM(nuclear_dir)//'input/nuclear_info.inp')
    ! three comment lines
    !READ(20565,*)COMMENT
    !READ(20565,*)COMMENT
    !READ(20565,*)COMMENT
    found=.FALSE.
    !DO WHILE(.not.found)
    DO I=1,data_len
       temp=ion_names(I)
       A=ion_A(I)
       Z=ion_Z(I)
       R=ion_R(I)
       aa=ion_aa(I)
       w=ion_w(I)
       !READ(20565,*,ERR=240,END=240)temp,A,Z,R,aa,w
       IF(TRIM(temp).EQ.TRIM(name))THEN
          found=.TRUE.
          EXIT
       ENDIF
    ENDDO
!240 CONTINUE
!    CLOSE(UNIT=20565)
    IF(.NOT.found)THEN
       WRITE(*,*)"ERROR: do not find the '"//TRIM(name)//"'. Check input/nuclear_info.inp."
       STOP
    ENDIF
    RETURN
  END SUBROUTINE GetNuclearInfo

  SUBROUTINE GetCentralityImpactB(NC,NB,cbins,bbins,wmatrix)
    IMPLICIT NONE
    INTEGER,INTENT(OUT)::NC,NB
    INTEGER,PARAMETER::NMAX=20
    CHARACTER(len=100)::COMMENT
    REAL(KIND(1d0)),DIMENSION(NMAX,2),INTENT(OUT)::cbins,bbins
    REAL(KIND(1d0)),DIMENSION(NMAX,NMAX),INTENT(OUT)::wmatrix
    INTEGER::i,j
    OPEN(UNIT=230556,FILE=TRIM(nuclear_dir)//"input/centrality_brange.inp")
    NC=0
    NB=0
    DO WHILE(.TRUE.)
       READ(230556,*)COMMENT
       IF(COMMENT(1:12).EQ.'<centrality>')THEN
          NC=0
          DO WHILE(.TRUE.)
             NC=NC+1
             IF(NC.GT.NMAX+1)THEN
                WRITE(*,*)"ERROR: too many centrality bins (>20) to enlarge NMAX"
                CLOSE(UNIT=230556)
                STOP
             ENDIF
             READ(230556,*,ERR=250,END=250)cbins(NC,1),cbins(NC,2)
          ENDDO
       ELSEIF(COMMENT(1:3).EQ.'<b>')THEN
          NB=0
          DO WHILE(.TRUE.)
             NB=NB+1
             IF(NB.GT.NMAX+1)THEN
                WRITE(*,*)"ERROR: too many b bins (>20) to enlarge NMAX"
                CLOSE(UNIT=230556)
                STOP
             ENDIF
             READ(230556,*,ERR=251,END=251)bbins(NB,1),bbins(NB,2)
          ENDDO
       ELSEIF(COMMENT(1:8).EQ.'<weight>')THEN
          IF(NC.LE.0)THEN
             WRITE(*,*)"ERROR: there is no centrality bin"
             CLOSE(UNIT=230556)
             STOP
          ENDIF
          IF(NB.LE.0)THEN
             WRITE(*,*)"ERROR: there is no b bin"
             CLOSE(UNIT=230556)
             STOP
          ENDIF
          DO i=1,NC
             READ(230556,*,ERR=252)(wmatrix(i,j),j=1,NB)
          ENDDO
          EXIT
       ELSE
          CYCLE
       ENDIF
       CYCLE
250    CONTINUE
       NC=NC-1
       CYCLE
251    CONTINUE
       NB=NB-1
       CYCLE
252    CONTINUE
       WRITE(*,*)"ERROR: unable to read the weight matrix"
       CLOSE(UNIT=230556)
       STOP
    ENDDO
    CLOSE(UNIT=230556)
    RETURN
  END SUBROUTINE GetCentralityImpactB

  ! the following is useful for the factorised form, e.g. Eq.(4.9) in
  ! /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  SUBROUTINE CalculateTAhatTBhat_WoodsSaxon_centrality(nameA,nameB,nbbins,bbins,res,nA,nB)
    ! calculate int_{bmin}^{bmax}{int_{0}^{+inf}{TAhat(\vec{s})**nA*TBhat(\vec{s}-\vec{b})**nB*d^2\vec{s}}*2pi*bdb} for bin by bin
    ! nbbins,bbins are same as GetCentralityImpactB arguments
    IMPLICIT NONE
    CHARACTER(len=7),INTENT(IN)::nameA,nameB
    REAL(KIND(1d0)),INTENT(IN),OPTIONAL::nA,nB ! the power nA, nB                                  
    INTEGER,INTENT(IN)::NBBINS
    REAL(KIND(1d0)),DIMENSION(NBBINS,2),INTENT(IN)::bbins
    REAL(KIND(1d0)),DIMENSION(0:NBBINS),INTENT(OUT)::res
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    REAL(KIND(1d0))::ZvalA,ZvalB,integral,sum
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::nA_common,nB_common
    REAL(KIND(1d0))::RA_common,wA_common,aaA_common,AA_common
    REAL(KIND(1d0))::RB_common,wB_common,aaB_common,AB_common
    COMMON/CalTATB_WoodsSaxon/nA_common,nB_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    INTEGER::i
    IF(PRESENT(nA))THEN
       nA_common=nA
    ELSE
       nA_common=1D0
    ENDIF
    IF(PRESENT(nB))THEN
       nB_common=nB
    ELSE
       nB_common=1D0
    ENDIF
    CALL GetNuclearInfo(nameA,AA_common,ZvalA,RA_common,aaA_common,wA_common)
    CALL GetNuclearInfo(nameB,AB_common,ZvalB,RB_common,aaB_common,wB_common)
    sum=0D0
    aax(1)=-10d0*RA_common
    bbx(1)=10d0*RA_common
    aax(2)=-10d0*RA_common
    bbx(2)=10d0*RA_common
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=MAX(100/NBBINS,20)
    DO i=1,nbbins
       aax(3)=bbins(i,1)
       bbx(3)=bbins(i,2)
       CALL ROMBERG_ND(CalculateTAhatTBhat_WoodsSaxon_cfxn,aax,bbx,3,sub_num,1,1d-5,&
            integral,ind,eval_num)
       IF(ind.EQ.-1)THEN
          WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
       ENDIF
       sum=sum+integral
       res(i)=integral
    ENDDO
    res(0)=sum
    IF(ABS(sum-1d0).GT.1d-3.AND.nA_common.EQ.1d0.AND.nB_common.EQ.1d0)THEN
       WRITE(*,*)"ERROR: the sum of centrality integration over TABhat is not 1 in CalculateTAhatTBhat_WoodsSaxon_centrality"
       WRITE(*,*)"sum=",sum
       STOP
    ENDIF
    RETURN
  END SUBROUTINE CalculateTAhatTBhat_WoodsSaxon_centrality

  FUNCTION CalculateTAhatTBhat_WoodsSaxon_cfxn(dim_num,sA)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::CalculateTAhatTBhat_WoodsSaxon_cfxn,temp
    REAL(KIND(1d0))::nA_common,nB_common
    REAL(KIND(1d0))::RA_common,wA_common,aaA_common,AA_common
    REAL(KIND(1d0))::RB_common,wB_common,aaB_common,AB_common
    COMMON/CalTATB_WoodsSaxon/nA_common,nB_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    REAL(KIND(1d0))::ssx,ssy
    REAL(KIND(1d0)),PARAMETER::twopi=6.28318530717958647692528676656d0
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: CalculateTAhatTBhat_WoodsSaxon_cfxn is not a three dimensional function"
       STOP
    ENDIF
    IF(nA_common.EQ.0D0.AND.nB_common.EQ.0d0)THEN
       CalculateTAhatTBhat_WoodsSaxon_cfxn=1d0
    ELSE
       ssx=sA(1)-sA(3)
       ssy=sA(2)
       IF(nA_common.NE.0d0)THEN
          CalculateTAhatTBhat_WoodsSaxon_cfxn=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1)
          IF(nA_common.NE.1d0)THEN
             CalculateTAhatTBhat_WoodsSaxon_cfxn=CalculateTAhatTBhat_WoodsSaxon_cfxn**nA_common
          ENDIF
       ELSE
          CalculateTAhatTBhat_WoodsSaxon_cfxn=1d0
       ENDIF
       IF(nB_common.NE.0d0)THEN
          temp=TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2)
          IF(nB_common.NE.1d0)THEN
             temp=temp**nB_common
          ENDIF
       ELSE
          temp=1d0
       ENDIF
       CalculateTAhatTBhat_WoodsSaxon_cfxn=CalculateTAhatTBhat_WoodsSaxon_cfxn*temp
    ENDIF
    ! jacobi
    CalculateTAhatTBhat_WoodsSaxon_cfxn=CalculateTAhatTBhat_WoodsSaxon_cfxn*sA(3)*twopi
    RETURN
  END FUNCTION CalculateTAhatTBhat_WoodsSaxon_cfxn

  ! the following is useful for the nonfactorised form, e.g. Eq.(4.8) in 
  ! /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  SUBROUTINE CalculateTABhat_WoodsSaxon_centrality(nameA,nameB,nbbins,bbins,res,n)
    ! calculate int_{bmin}^{bmax}{TABhat(b)**n*2pi*bdb} for bin by bin
    ! nbbins,bbins are same as GetCentralityImpactB arguments
    IMPLICIT NONE
    CHARACTER(len=7),INTENT(IN)::nameA,nameB
    REAL(KIND(1d0)),INTENT(IN),OPTIONAL::n ! the power n
    INTEGER,INTENT(IN)::NBBINS
    REAL(KIND(1d0)),DIMENSION(NBBINS,2),INTENT(IN)::BBINS
    REAL(KIND(1d0)),DIMENSION(0:NBBINS),INTENT(OUT)::res
    REAL(KIND(1d0)),DIMENSION(3)::aax,bbx
    INTEGER,DIMENSION(3)::sub_num
    REAL(KIND(1d0))::ZvalA,ZvalB,integral,sum
    INTEGER::ind,eval_num
    REAL(KIND(1d0))::n_common
    REAL(KIND(1d0))::RA_common,wA_common,aaA_common,AA_common
    REAL(KIND(1d0))::RB_common,wB_common,aaB_common,AB_common
    COMMON/CalTAB_WoodsSaxon/n_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    INTEGER::i
    IF(PRESENT(n))THEN
       n_common=n
    ELSE
       n_common=1D0
    ENDIF
    CALL GetNuclearInfo(nameA,AA_common,ZvalA,RA_common,aaA_common,wA_common)
    CALL GetNuclearInfo(nameB,AB_common,ZvalB,RB_common,aaB_common,wB_common)
    sum=0D0
    aax(1)=-10d0*RA_common
    bbx(1)=10d0*RA_common
    aax(2)=-10d0*RA_common
    bbx(2)=10d0*RA_common
    sub_num(1)=100
    sub_num(2)=100
    sub_num(3)=MAX(100/NBBINS,20)
    DO i=1,nbbins
       aax(3)=bbins(i,1)
       bbx(3)=bbins(i,2)
       CALL ROMBERG_ND(CalculateTABhat_WoodsSaxon_cfxn,aax,bbx,3,sub_num,1,1d-5,&
            integral,ind,eval_num)
       IF(ind.EQ.-1)THEN
          WRITE(*,*)"WARNING: the precision 1e-5 is not achieved"
       ENDIF
       sum=sum+integral
       res(i)=integral
    ENDDO
    res(0)=sum
    IF(ABS(sum-1d0).GT.1d-3.AND.n_common.EQ.1d0)THEN
       WRITE(*,*)"ERROR: the sum of centrality integration over TABhat is not 1 in CalculateTABhat_WoodsSaxon_centrality"
       WRITE(*,*)"sum=",sum
       STOP
    ENDIF
    RETURN
  END SUBROUTINE CalculateTABhat_WoodsSaxon_centrality

  FUNCTION TABhat0_WoodsSaxon(nameA,nameB)
    ! get TABhat(0)
    IMPLICIT NONE
    REAL(KIND(1d0))::TABhat0_WoodsSaxon
    CHARACTER(len=7),INTENT(IN)::nameA,nameB
    REAL(KIND(1d0))::ZvalA,ZvalB
    REAL(KIND(1d0)),DIMENSION(2)::A,aa,w,RR
    LOGICAL::storegrid
    COMMON/TAB_WoodsSaxon_Grid/storegrid
    storegrid=.TRUE.
    CALL GetNuclearInfo(nameA,A(1),ZvalA,RR(1),aa(1),w(1))
    CALL GetNuclearInfo(nameB,A(2),ZvalB,RR(2),aa(2),w(2))
    TABhat0_WoodsSaxon=TABhat_WoodsSaxon(0d0,0d0,RR,w,aa,A)
    RETURN
  END FUNCTION TABhat0_WoodsSaxon

  FUNCTION CalculateTABhat_WoodsSaxon_cfxn(dim_num,sA)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::dim_num
    REAL(KIND(1d0)),DIMENSION(dim_num),INTENT(IN)::sA
    REAL(KIND(1d0))::CalculateTABhat_WoodsSaxon_cfxn
    REAL(KIND(1d0))::n_common
    REAL(KIND(1d0))::RA_common,wA_common,aaA_common,AA_common
    REAL(KIND(1d0))::RB_common,wB_common,aaB_common,AB_common
    COMMON/CalTAB_WoodsSaxon/n_common,RA_common,RB_common,wA_common,wB_common,&
         aaA_common,aaB_common,AA_common,AB_common
    REAL(KIND(1d0))::ssx,ssy
    REAL(KIND(1d0)),PARAMETER::twopi=6.28318530717958647692528676656d0
    IF(dim_num.NE.3)THEN
       WRITE(*,*)"ERROR: CalculateTABhat_WoodsSaxon_cfxn is not a three dimensional function"
       STOP
    ENDIF
    IF(n_common.EQ.0D0)THEN
       CalculateTABhat_WoodsSaxon_cfxn=1d0
    ELSE
       ssx=sA(1)-sA(3)
       ssy=sA(2)
       CalculateTABhat_WoodsSaxon_cfxn=TAhat_WoodsSaxon(sA(1),sA(2),RA_common,wA_common,aaA_common,AA_common,1)
       CalculateTABhat_WoodsSaxon_cfxn=CalculateTABhat_WoodsSaxon_cfxn*&
            TAhat_WoodsSaxon(ssx,ssy,RB_common,wB_common,aaB_common,AB_common,2)
       IF(n_common.NE.1d0)THEN
          CalculateTABhat_WoodsSaxon_cfxn=CalculateTABhat_WoodsSaxon_cfxn**n_common
       ENDIF
    ENDIF
    ! jacobi
    CalculateTABhat_WoodsSaxon_cfxn=CalculateTABhat_WoodsSaxon_cfxn*sA(3)*twopi
    RETURN
  END FUNCTION CalculateTABhat_WoodsSaxon_cfxn

  SUBROUTINE CalculateTAhat_WoodsSaxon_centrality(name,nbbins,bbins,res,n)
    ! calculate int_{bmin}^{bmax}{TAhat(b)**n*2pi*bdb} for bin by bin
    ! nbbins,bbins are same as GetCentralityImpactB arguments
    IMPLICIT NONE
    CHARACTER(len=7),INTENT(IN)::name
    INTEGER,INTENT(IN)::NBBINS
    REAL(KIND(1d0)),INTENT(IN),OPTIONAL::n ! the power n
    REAL(KIND(1d0)),DIMENSION(NBBINS,2),INTENT(IN)::BBINS
    REAL(KIND(1d0)),DIMENSION(0:NBBINS),INTENT(OUT)::res
    REAL(KIND(1d0))::Zval,integral,bmin,bmax,sum
    REAL(KIND(1d0))::n_common
    REAL(KIND(1d0))::R_common,w_common,aa_common,A_common
    COMMON/CalTA_WoodsSaxon/n_common,R_common,w_common,aa_common,A_common
    INTEGER::i,ninteg
    IF(PRESENT(n))THEN
       n_common=n
    ELSE
       n_common=1D0
    ENDIF
    CALL GetNuclearInfo(name,A_common,Zval,R_common,aa_common,w_common)
    sum=0D0
    ninteg=10000
    DO i=1,nbbins
       bmin=bbins(i,1)
       bmax=bbins(i,2)
       CALL simpson(CalculateTAhat_WoodsSaxon_cfxn,bmin,bmax,integral,ninteg)
       sum=sum+integral
       res(i)=integral
    ENDDO
    res(0)=sum
    IF(ABS(sum-1d0).GT.1d-3.AND.n_common.EQ.1D0)THEN
       WRITE(*,*)"ERROR: the sum of centrality integration over TAhat is not 1 in CalculateTAhat_WoodsSaxon_centrality"
       WRITE(*,*)"sum=",sum
       STOP
    ENDIF
    RETURN
  END SUBROUTINE CalculateTAhat_WoodsSaxon_centrality

  FUNCTION CalculateTAhat_WoodsSaxon_cfxn(sA)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::sA
    REAL(KIND(1d0))::CalculateTAhat_WoodsSaxon_cfxn
    REAL(KIND(1d0))::n_common
    REAL(KIND(1d0))::R_common,w_common,aa_common,A_common
    COMMON/CalTA_WoodsSaxon/n_common,R_common,w_common,aa_common,A_common
    REAL(KIND(1d0)),PARAMETER::twopi=6.28318530717958647692528676656d0
    IF(n_common.EQ.0D0)THEN
       CalculateTAhat_WoodsSaxon_cfxn=1d0
    ELSE
       CalculateTAhat_WoodsSaxon_cfxn=TAhat_WoodsSaxon(sA,0d0,R_common,w_common,aa_common,A_common,1)
       IF(n_common.NE.1d0)THEN
          CalculateTAhat_WoodsSaxon_cfxn=CalculateTAhat_WoodsSaxon_cfxn**n_common
       ENDIF
    ENDIF
    ! jacobi
    CalculateTAhat_WoodsSaxon_cfxn=CalculateTAhat_WoodsSaxon_cfxn*sA*twopi
    RETURN
  END FUNCTION CalculateTAhat_WoodsSaxon_cfxn

  FUNCTION TAhat0_WoodsSaxon(name)
    ! get TAhat(0)
    IMPLICIT NONE
    REAL(KIND(1d0))::TAhat0_WoodsSaxon
    CHARACTER(len=7),INTENT(IN)::name
    REAL(KIND(1d0))::Zval,Aval,Rval,aaval,wval
    CALL GetNuclearInfo(name,Aval,Zval,Rval,aaval,wval)
    TAhat0_WoodsSaxon=TAhat_WoodsSaxon(0d0,0d0,Rval,wval,aaval,Aval,1)
    RETURN
  END FUNCTION TAhat0_WoodsSaxon

  ! Charge form factor of ions
  ! This is defined in eq.(7.3) of /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  ! This is the same as eq.(7.16) of /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  ! with a = 0 (a real hard sphere from Woods-Saxon and w=0)
  FUNCTION ChargeFormFactor_Hardsphere(Q,RR)
    ! Q and RR should be in unit of GeV and GeV-1
    ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::Q,RR
    REAL(KIND(1d0))::ChargeFormFactor_Hardsphere
    REAL(KIND(1d0))::QR
    QR=Q*RR
    ChargeFormFactor_Hardsphere=3d0*(DSIN(QR)-QR*DCOS(QR))/QR**3
    RETURN
  END FUNCTION ChargeFormFactor_Hardsphere

  ! This is eq.(7.17) of /Users/erdissshaw/Works/Manuscript/OpticalGlauber
  FUNCTION ChargeFormFactor_dipole_proton(Q)
    ! Q is in unit of GeV
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::Q
    REAL(KIND(1d0))::ChargeFormFactor_dipole_proton
    REAL(KIND(1d0)),PARAMETER::aa=1.1867816581938533d0 ! in unit of GeV-1 = 1/(sqrt(0.71) GeV)
    REAL(KIND(1d0))::QA
    QA=Q*aa
    ChargeFormFactor_dipole_proton=1d0/(1d0+QA**2)**2
    RETURN
  END FUNCTION ChargeFormFactor_dipole_proton

  FUNCTION ChargeFormFactor_WoodsSaxon(Q,RR,w,aa,NTERMS)
    ! Q and RR/aa should be in unit of GeV and GeV-1
    ! 1 GeV^-1 = 0.197e-15 m = 0.197 fm
    USE nielsen_generalized_polylog
    IMPLICIT NONE
    REAL(KIND(1d0))::ChargeFormFactor_WoodsSaxon
    REAL(KIND(1d0)),INTENT(IN)::Q,RR,w,aa
    INTEGER,INTENT(IN)::NTERMS
    REAL(KIND(1d0))::QR,QA,PIQA
    REAL(KIND(1d0))::rho0hat,expterms
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    REAL(KIND(1d0)),PARAMETER::PI2=9.86960440108935861883449099988d0
    COMPLEX(KIND(1d0))::Li3val,Li5val
    REAL(KIND(1d0))::RoA,sinhpiqa,coshpiqa
    INTEGER::ii
    RoA=RR/aa
    IF(w.NE.0d0)THEN
       ! for w=!=0, we also know it analytically
       ! cf. eq.(7.12) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
       Li3val=Nielsen_PolyLog(2,1,-DEXP(RoA))
       Li5val=Nielsen_PolyLog(4,1,-DEXP(RoA))
       rho0hat=1d0/(-8d0*pi*aa**3*(DREAL(Li3val)+12d0*w/RoA**2*DREAL(Li5val)))
    ELSE
       ! for w=0, we know it analytically
       ! cf. eq.(21) in Maximon and Schrack, J. Res. Natt. Bur. Stand B70 (1966)
       ! or eq.(7.9) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
       Li3val=Nielsen_PolyLog(2,1,-DEXP(-RR/aa))
       rho0hat=1d0/(4d0*pi/3d0*RR*(RR**2+(pi*aa)**2)-8d0*pi*aa**3*DREAL(Li3val))
    ENDIF
    ! for w=!=0, we also know it analtycially
    ! eq.(7.6) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
    ! for w=0, we know it analytically
    ! cf. eq.(20) in Maximon and Schrack, J. Res. Natt. Bur. Stand B70 (1966)
    ! eq.(7.3) of /Users/erdissshaw/Works/Manuscript/OpticalGlauber
    QR=Q*RR
    QA=Q*aa
    PIQA=pi*QA
    IF(PIQA.LE.250d0)THEN
       coshpiqa=DCOSH(PIQA)
       sinhpiqa=DSINH(PIQA)
       ChargeFormFactor_WoodsSaxon=rho0hat*4d0*pi**2*aa**3/(QA**2*sinhpiqa**2)*&
            (PIQA*coshpiqa*DSIN(QR)*(1d0-w/RoA**2*(6d0*PI2/sinhpiqa**2+PI2-3d0*RoA**2))&
            -QR*sinhpiqa*DCOS(QR)*(1d0-w/RoA**2*(6d0*PI2/sinhpiqa**2+3d0*PI2-RoA**2)))
    ELSE
       ! the above term must be suppressed by Exp(-Pi*Q*aa)
       ChargeFormFactor_WoodsSaxon=0d0
    ENDIF
    IF(NTERMS.GT.0)THEN
       expterms=0d0
       DO ii=1,NTERMS
          expterms=expterms+(-1D0)**(ii-1)*ii*DEXP(-ii*RR/aa)/(ii**2+QA**2)**2*&
               (1d0+12d0*w/RoA**2*(ii**2-QA**2)/(ii**2+QA**2)**2)
       ENDDO
       ChargeFormFactor_WoodsSaxon=ChargeFormFactor_WoodsSaxon+8d0*pi*rho0hat*aa**3*expterms
    ENDIF
    RETURN
  END FUNCTION ChargeFormFactor_WoodsSaxon

  ! eq.(7.19) in my notes OpticalGlauber.pdf
  FUNCTION PhotonNumberDensity_AnalyticInt4Series_WS(b,Ega,gamma,RR,w,aa,NMIN,NMAX)
    ! b,RR,aa should be written in unit of GeV-1
    ! Ega should be in unit of GeV
    ! 1 GeV^-1 = 0.1973e-15 m = 0.1973 fm
    ! If NMIN < 0, we do not perform any infinite sum
    ! If NMIN > 0, we already perform infinite sum for K1(btil)
    USE nielsen_generalized_polylog
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::b,Ega,gamma,RR,w,aa
    INTEGER,INTENT(IN)::NMIN,NMAX
    REAL(KIND(1d0))::PhotonNumberDensity_AnalyticInt4Series_WS
    COMPLEX(KIND(1d0))::Li3val,Li5val
    REAL(KIND(1d0))::RoA,Egaoga,pref,rho0hat
    REAL(KIND(1d0)),PARAMETER::pi=3.14159265358979323846264338328d0
    INTEGER::ii
    REAL(KIND(1d0))::btil,atil,Rtil,Bntil
    REAL(KIND(1d0)),EXTERNAL::BESSK1,BESSK0
    REAL(KIND(1d0))::K1btil,K0Bntil,K1Bntil,wpref,exppref,expterm,sqrtterm
    INTEGER::nmin2,nmax2
    REAL(KIND(1d0))::logb
    REAL(KIND(1d0)),PARAMETER::eulergamma=0.577215664901532860606512090082d0
    REAL(KIND(1d0)),PARAMETER::logtwo=0.693147180559945309417232121458d0
    nmin2=MAX(1,ABS(NMIN))
    nmax2=ABS(NMAX)
    IF(nmin2.GT.nmax2)THEN
       PhotonNumberDensity_AnalyticInt4Series_WS=0d0
       RETURN
    ENDIF
    RoA=RR/aa
    ! for w=!=0, we also know it analytically
    ! cf. eq.(7.12) in /Users/erdissshaw/Works/Manuscript/OpticalGlauber
    Li3val=Nielsen_PolyLog(2,1,-DEXP(RoA))
    IF(w.NE.0d0)THEN
       Li5val=Nielsen_PolyLog(4,1,-DEXP(RoA))
       ! this is 8*pi*rho0hat*aa**3
       rho0hat=1d0/(-(DREAL(Li3val)+12d0*w/RoA**2*DREAL(Li5val)))
    ELSE
       rho0hat=1d0/(-(DREAL(Li3val)))
    ENDIF
    Egaoga=Ega/gamma
    ! rescaled variables
    btil=b*Egaoga
    atil=aa*Egaoga
    Rtil=RR*Egaoga
    pref=Egaoga*rho0hat
    PhotonNumberDensity_AnalyticInt4Series_WS=0d0
    K1btil=BESSK1(btil)
    wpref=12d0*w/RoA**2
    IF(NMIN.GT.0)THEN
       ! we will sum all K1(btil) term
       Li3val=Nielsen_PolyLog(2,1,-DEXP(-RoA))
       IF(w.NE.0d0)THEN
          Li5val=Nielsen_PolyLog(4,1,-DEXP(-RoA))
          PhotonNumberDensity_AnalyticInt4Series_WS=-K1btil*(DREAL(Li3val)+wpref*DREAL(Li5val))
       ELSE
          PhotonNumberDensity_AnalyticInt4Series_WS=-K1btil*DREAL(Li3val)
       ENDIF
    ENDIF
    IF(btil.GT.1D-2.OR.NMIN.GT.0.OR.btil/atil.GT.1D-2)THEN
       DO ii=nmin2,nmax2
          sqrtterm=DSQRT(1D0+DBLE(ii)**2/atil**2)
          Bntil=btil*sqrtterm
          K0Bntil=BESSK0(Bntil)
          K1Bntil=BESSK1(Bntil)
          exppref=(-1D0)**(ii-1)*DBLE(ii)*DEXP(-ii*RoA)
          expterm=-sqrtterm/DBLE(ii)**4*K1Bntil-btil/(2d0*DBLE(ii)**2*atil**2)*K0Bntil
          IF(NMIN.LE.0)expterm=expterm+K1btil/DBLE(ii)**4
          IF(w.NE.0d0)THEN
             expterm=expterm-wpref*(1d0/DBLE(ii)**6+btil**2*(5*DBLE(ii)**2+3*atil**2)/&
                  (24d0*DBLE(ii)**2*(DBLE(ii)**2+atil**2)**2*atil**2))*sqrtterm*K1Bntil&
                  -wpref*(btil/(2d0*DBLE(ii)**4*atil**2)+btil**3/(24d0*(atil**2+DBLE(ii)**2)*atil**4))*K0Bntil
             IF(NMIN.LE.0)expterm=expterm+wpref*K1btil/DBLE(ii)**6
          ENDIF
          expterm=expterm*exppref
          PhotonNumberDensity_AnalyticInt4Series_WS=PhotonNumberDensity_AnalyticInt4Series_WS+expterm
       ENDDO
    ELSE
       logb=DLOG(btil)
       ! there are large numerical cancellations between different terms
       ! we use the Taylor expansion terms
       DO ii=nmin2,nmax2
          exppref=(-1D0)**(ii-1)*DBLE(ii)*DEXP(-ii*RoA)
          ! for log(btil) terms we sum up to higher order
          expterm=(btil**3/(16d0*atil**4)+btil**5*(3*atil**2+2*DBLE(ii)**2)/(384d0*atil**6)&
               +btil**7*(6d0*atil**4+8d0*DBLE(ii)**2*atil**2+3d0*DBLE(ii)**4)/(18432d0*atil**8))*logb
          expterm=expterm+btil*(DBLE(ii)**2-atil**2*DLOG(DBLE(ii)**2/atil**2+1d0))/(4d0*atil**2*DBLE(ii)**4)
          expterm=expterm-btil**3/(64d0*atil**4*DBLE(ii)**4)*(2*(atil**4-DBLE(ii)**4)*DLOG(1d0+DBLE(ii)**2/atil**2)&
               -2*atil**2*DBLE(ii)**2+DBLE(ii)**4*(3d0-4d0*eulergamma+4d0*logtwo))
          IF(w.NE.0)THEN
             expterm=expterm+wpref*logb*(btil**5/(384d0*atil**6)&
                  +btil**7*(4d0*atil**2+5d0*DBLE(ii)**2)/(18432d0*atil**8))
             expterm=expterm+wpref*btil*((6d0*atil**4+9d0*atil**2*DBLE(ii)**2+DBLE(ii)**4)&
                  /(24d0*atil**2*(atil**2+DBLE(ii)**2)**2*DBLE(ii)**4)-DLOG(1d0+DBLE(ii)**2/atil**2)/(4d0*DBLE(ii)**6))
             expterm=expterm+wpref*btil**3*((6d0*atil**4+3d0*DBLE(ii)**2*atil**2+DBLE(ii)**4)&
                  /(192d0*DBLE(ii)**4*atil**4*(atil**2+DBLE(ii)**2))-DLOG(1d0+DBLE(ii)**2/atil**2)/(32d0*DBLE(ii)**6))
          ENDIF
          expterm=expterm*exppref
          PhotonNumberDensity_AnalyticInt4Series_WS=PhotonNumberDensity_AnalyticInt4Series_WS+expterm
       ENDDO
    ENDIF
    PhotonNumberDensity_AnalyticInt4Series_WS=PhotonNumberDensity_AnalyticInt4Series_WS*pref
    RETURN
  END FUNCTION PhotonNumberDensity_AnalyticInt4Series_WS

  FUNCTION AtomicFormFactor(Z,q)
    ! charge-neutral atomic form factors
    ! see table 6.1.1.1 in /Users/huasheng/Physics/Books/AtomicPhysics/Intensity_of _Diffracted _Intensities.pdf for
    ! (Z<=98 and q/(4*PI)<=6 angstrom^{-1})
    ! see table 1 in /Users/huasheng/Physics/Books/AtomicPhysics/Atomicformfactors_incoherentscatteringfunctions_photonscatteringcrosssections.pdf for
    ! (either Z=99,100 or q(4*PI)>6 angstrom^{-1})
    IMPLICIT NONE
    INTEGER,INTENT(IN)::Z  ! atomic number
    REAL(KIND(1d0)),INTENT(IN)::q ! in unit of GeV
    REAL(KIND(1d0))::AtomicFormFactor ! it is dimensionless
    REAL(KIND(1d0)),PARAMETER::FOURPI=12.5663706143591729538505735331d0
    INTEGER,PARAMETER::ZMAX=100,NQ=70
    INTEGER::I,J
    ! mean atomic scattering factors for free atoms
    ! this uses Hartree-Fock or Dirac-Slater wave functions
    ! the following Fortran output can be found in <<Atomic form factors/Output into Fortran for gamma-UPC>>
    ! in "/Users/huasheng/Physics/FLibatM/jpsi_resummation/CSS/quarkonium.nb"
    REAL(KIND(1d0)),DIMENSION(NQ)::QO4PI ! in unit of angstrom^{-1}
    DATA QO4PI /0d0,0.01d0,0.02d0,0.03d0,0.04d0,0.05d0,0.06d0,0.07d0,0.08d0,0.09d0,&
         0.1d0,0.11d0,0.12d0,0.13d0,0.14d0,0.15d0,0.16d0,0.17d0,0.18d0,0.19d0,&
         0.2d0,0.22d0,0.24d0,0.25d0,0.26d0,0.28d0,0.3d0,0.32d0,0.34d0,0.35d0,&
         0.36d0,0.38d0,0.4d0,0.42d0,0.44d0,0.45d0,0.46d0,0.48d0,0.5d0,0.55d0,&
         0.6d0,0.65d0,0.7d0,0.8d0,0.9d0,1.d0,1.1d0,1.2d0,1.3d0,1.4d0,&
         1.5d0,1.6d0,1.7d0,1.8d0,1.9d0,2.d0,2.5d0,3.d0,3.5d0,4.d0,&
         5.d0,6.d0,7.d0,8.d0,10.d0,15.d0,20.d0,50.d0,80.d0,100.d0/
    REAL(KIND(1d0)),DIMENSION(ZMAX,NQ)::ScatteringFactors
    DATA (ScatteringFactors(1,I),I=1,NQ) /&
    1d0,0.998d0,0.991d0,0.98d0,0.966d0,0.947d0,0.925d0,0.9d0,0.872d0,0.842d0,&
    0.811d0,0.778d0,0.744d0,0.71d0,0.676d0,0.641d0,0.608d0,0.574d0,0.542d0,0.511d0,&
    0.481d0,0.424d0,0.373d0,0.35d0,0.328d0,0.287d0,0.251d0,0.22d0,0.193d0,0.18d0,&
    0.169d0,0.148d0,0.13d0,0.115d0,0.101d0,0.095d0,0.09d0,0.079d0,0.071d0,0.053d0,&
    0.04d0,0.031d0,0.024d0,0.015d0,0.01d0,0.007d0,0.005d0,0.003d0,0.003d0,0.002d0,&
    0.001d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,&
    0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0/
    DATA (ScatteringFactors(2,I),I=1,NQ) /&
    2d0,1.998d0,1.993d0,1.984d0,1.972d0,1.957d0,1.939d0,1.917d0,1.893d0,1.866d0,&
    1.837d0,1.806d0,1.772d0,1.737d0,1.701d0,1.663d0,1.624d0,1.584d0,1.543d0,1.502d0,&
    1.46d0,1.377d0,1.295d0,1.254d0,1.214d0,1.136d0,1.06d0,0.988d0,0.92d0,0.887d0,&
    0.856d0,0.795d0,0.738d0,0.686d0,0.636d0,0.613d0,0.591d0,0.548d0,0.509d0,0.423d0,&
    0.353d0,0.295d0,0.248d0,0.177d0,0.129d0,0.095d0,0.072d0,0.055d0,0.042d0,0.033d0,&
    0.026d0,0.021d0,0.017d0,0.014d0,0.011d0,0.01d0,0.004d0,0.002d0,0.001d0,0.001d0,&
    0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0,0d0/
    DATA (ScatteringFactors(3,I),I=1,NQ) /&
    3d0,2.986d0,2.947d0,2.884d0,2.802d0,2.708d0,2.606d0,2.502d0,2.4d0,2.304d0,&
    2.215d0,2.135d0,2.065d0,2.004d0,1.95d0,1.904d0,1.863d0,1.828d0,1.796d0,1.768d0,&
    1.742d0,1.693d0,1.648d0,1.626d0,1.604d0,1.559d0,1.513d0,1.465d0,1.417d0,1.393d0,&
    1.369d0,1.32d0,1.27d0,1.221d0,1.173d0,1.149d0,1.125d0,1.078d0,1.033d0,0.924d0,&
    0.823d0,0.732d0,0.65d0,0.512d0,0.404d0,0.32d0,0.255d0,0.205d0,0.165d0,0.134d0,&
    0.11d0,0.091d0,0.075d0,0.063d0,0.053d0,0.044d0,0.021d0,0.011d0,0.006d0,0.004d0,&
    0.002d0,0.001d0,0.0004266d0,0.0002521d0,0.0001043d0,0.0000208d0,6.628d-6,1.72d-7,0d0,0d0/
    DATA (ScatteringFactors(4,I),I=1,NQ) /&
    4d0,3.987d0,3.95d0,3.889d0,3.807d0,3.707d0,3.592d0,3.468d0,3.336d0,3.201d0,&
    3.065d0,2.932d0,2.804d0,2.683d0,2.569d0,2.463d0,2.365d0,2.277d0,2.197d0,2.125d0,&
    2.06d0,1.951d0,1.864d0,1.828d0,1.795d0,1.739d0,1.692d0,1.652d0,1.616d0,1.6d0,&
    1.583d0,1.551d0,1.52d0,1.489d0,1.458d0,1.443d0,1.427d0,1.395d0,1.362d0,1.279d0,&
    1.195d0,1.112d0,1.03d0,0.876d0,0.74d0,0.622d0,0.522d0,0.439d0,0.369d0,0.311d0,&
    0.263d0,0.223d0,0.19d0,0.163d0,0.139d0,0.12d0,0.06d0,0.033d0,0.019d0,0.012d0,&
    0.005d0,0.003d0,0.001418d0,0.0008431d0,0.0003514d0,0.0000707d0,0.0000226d0,5.92d-7,0d0,0d0/
    DATA (ScatteringFactors(5,I),I=1,NQ) /&
    5d0,4.988d0,4.954d0,4.897d0,4.82d0,4.724d0,4.613d0,4.488d0,4.352d0,4.209d0,&
    4.06d0,3.908d0,3.756d0,3.606d0,3.459d0,3.316d0,3.179d0,3.048d0,2.924d0,2.808d0,&
    2.699d0,2.503d0,2.336d0,2.263d0,2.195d0,2.077d0,1.979d0,1.897d0,1.829d0,1.799d0,&
    1.771d0,1.723d0,1.681d0,1.644d0,1.611d0,1.596d0,1.581d0,1.553d0,1.526d0,1.463d0,&
    1.402d0,1.339d0,1.276d0,1.147d0,1.02d0,0.9d0,0.79d0,0.69d0,0.602d0,0.524d0,&
    0.457d0,0.398d0,0.347d0,0.304d0,0.266d0,0.233d0,0.126d0,0.072d0,0.043d0,0.027d0,&
    0.012d0,0.006d0,0.0034963d0,0.0020943d0,0.0008808d0,0.0001791d0,0.0000574d0,1.515d-6,2.36d-7,0d0/
    DATA (ScatteringFactors(6,I),I=1,NQ) /&
    6d0,5.99d0,5.958d0,5.907d0,5.837d0,5.749d0,5.645d0,5.526d0,5.396d0,5.255d0,&
    5.107d0,4.952d0,4.794d0,4.633d0,4.472d0,4.311d0,4.153d0,3.998d0,3.847d0,3.701d0,&
    3.56d0,3.297d0,3.058d0,2.949d0,2.846d0,2.658d0,2.494d0,2.351d0,2.227d0,2.171d0,&
    2.12d0,2.028d0,1.948d0,1.88d0,1.821d0,1.794d0,1.77d0,1.725d0,1.685d0,1.603d0,&
    1.537d0,1.479d0,1.426d0,1.322d0,1.219d0,1.114d0,1.012d0,0.914d0,0.822d0,0.736d0,&
    0.659d0,0.588d0,0.525d0,0.468d0,0.418d0,0.373d0,0.216d0,0.13d0,0.081d0,0.053d0,&
    0.025d0,0.013d0,0.0071471d0,0.0043194d0,0.0018364d0,0.0003777d0,0.0001216d0,3.234d-6,5.07d-7,2.11d-7/
    DATA (ScatteringFactors(7,I),I=1,NQ) /&
    7d0,6.991d0,6.963d0,6.918d0,6.855d0,6.776d0,6.682d0,6.574d0,6.453d0,6.321d0,&
    6.18d0,6.03d0,5.875d0,5.714d0,5.551d0,5.385d0,5.218d0,5.051d0,4.886d0,4.723d0,&
    4.563d0,4.254d0,3.963d0,3.825d0,3.693d0,3.445d0,3.219d0,3.014d0,2.831d0,2.747d0,&
    2.667d0,2.522d0,2.393d0,2.278d0,2.178d0,2.132d0,2.089d0,2.011d0,1.942d0,1.802d0,&
    1.697d0,1.616d0,1.551d0,1.445d0,1.353d0,1.265d0,1.172d0,1.09d0,1.004d0,0.921d0,&
    0.843d0,0.769d0,0.7d0,0.636d0,0.578d0,0.525d0,0.324d0,0.204d0,0.132d0,0.088d0,&
    0.043d0,0.023d0,0.0131d0,0.0101d0,0.0041d0,0.0007611d0,0.0002463d0,6.624d-6,1.042d-6,4.35d-7/
    DATA (ScatteringFactors(8,I),I=1,NQ) /&
    8d0,7.992d0,7.967d0,7.926d0,7.869d0,7.798d0,7.712d0,7.612d0,7.501d0,7.378d0,&
    7.245d0,7.103d0,6.954d0,6.798d0,6.637d0,6.472d0,6.304d0,6.134d0,5.964d0,5.793d0,&
    5.623d0,5.289d0,4.965d0,4.808d0,4.655d0,4.363d0,4.089d0,3.834d0,3.599d0,3.489d0,&
    3.383d0,3.186d0,3.006d0,2.844d0,2.697d0,2.629d0,2.564d0,2.445d0,2.338d0,2.115d0,&
    1.946d0,1.816d0,1.714d0,1.568d0,1.463d0,1.377d0,1.298d0,1.221d0,1.145d0,1.07d0,&
    0.997d0,0.926d0,0.857d0,0.792d0,0.731d0,0.674d0,0.443d0,0.292d0,0.196d0,0.134d0,&
    0.067d0,0.037d0,0.0211d0,0.0158d0,0.0052d0,0.0012872d0,0.0004191d0,0.0000114d0,1.801d-6,7.54d-7/
    DATA (ScatteringFactors(9,I),I=1,NQ) /&
    9d0,8.993d0,8.97d0,8.933d0,8.881d0,8.815d0,8.736d0,8.645d0,8.541d0,8.427d0,&
    8.302d0,8.168d0,8.026d0,7.876d0,7.721d0,7.56d0,7.395d0,7.226d0,7.055d0,6.883d0,&
    6.709d0,6.362d0,6.02d0,5.851d0,5.685d0,5.363d0,5.054d0,4.761d0,4.484d0,4.353d0,&
    4.225d0,3.983d0,3.759d0,3.551d0,3.36d0,3.27d0,3.183d0,3.022d0,2.874d0,2.559d0,&
    2.309d0,2.112d0,1.956d0,1.735d0,1.588d0,1.482d0,1.398d0,1.324d0,1.254d0,1.186d0,&
    1.12d0,1.055d0,0.99d0,0.928d0,0.868d0,0.81d0,0.564d0,0.389d0,0.27d0,0.19d0,&
    0.099d0,0.055d0,0.0325d0,0.0247d0,0.0091d0,0.0020409d0,0.0006689d0,0.0000184d0,2.92d-6,1.226d-6/
    DATA (ScatteringFactors(10,I),I=1,NQ) /&
    10d0,9.993d0,9.973d0,9.938d0,9.891d0,9.83d0,9.757d0,9.672d0,9.576d0,9.469d0,&
    9.351d0,9.225d0,9.09d0,8.948d0,8.799d0,8.643d0,8.483d0,8.318d0,8.15d0,7.978d0,&
    7.805d0,7.454d0,7.102d0,6.928d0,6.754d0,6.412d0,6.079d0,5.758d0,5.451d0,5.302d0,&
    5.158d0,4.88d0,4.617d0,4.37d0,4.139d0,4.029d0,3.923d0,3.722d0,3.535d0,3.126d0,&
    2.786d0,2.517d0,2.296d0,1.971d0,1.757d0,1.609d0,1.502d0,1.418d0,1.346d0,1.28d0,&
    1.218d0,1.158d0,1.099d0,1.041d0,0.984d0,0.929d0,0.68d0,0.489d0,0.331d0,0.254d0,&
    0.137d0,0.079d0,0.0469d0,0.0358d0,0.0136d0,0.0030746d0,0.0010151d0,0.0000282d0,4.507d-6,1.897d-6/
    DATA (ScatteringFactors(11,I),I=1,NQ) /&
    11d0,10.98d0,10.922d0,10.83d0,10.709d0,10.568d0,10.412d0,10.249d0,10.084d0,9.92d0,&
    9.76d0,9.605d0,9.455d0,9.309d0,9.166d0,9.027d0,8.888d0,8.751d0,8.613d0,8.475d0,&
    8.335d0,8.052d0,7.764d0,7.618d0,7.471d0,7.176d0,6.881d0,6.588d0,6.298d0,6.156d0,&
    6.015d0,5.739d0,5.471d0,5.214d0,4.967d0,4.848d0,4.731d0,4.506d0,4.293d0,3.811d0,&
    3.398d0,3.048d0,2.754d0,2.305d0,1.997d0,1.784d0,1.634d0,1.524d0,1.438d0,1.367d0,&
    1.304d0,1.247d0,1.191d0,1.137d0,1.084d0,1.032d0,0.791d0,0.591d0,0.438d0,0.325d0,&
    0.183d0,0.107d0,0.0653d0,0.0499d0,0.019d0,0.0044431d0,0.0014786d0,0.0000416d0,6.681d-6,2.819d-6/
    DATA (ScatteringFactors(12,I),I=1,NQ) /&
    12d0,11.978d0,11.914d0,11.811d0,11.674d0,11.507d0,11.319d0,11.116d0,10.903d0,10.687d0,&
    10.472d0,10.262d0,10.059d0,9.864d0,9.678d0,9.502d0,9.334d0,9.175d0,9.023d0,8.876d0,&
    8.735d0,8.465d0,8.205d0,8.078d0,7.951d0,7.698d0,7.446d0,7.194d0,6.943d0,6.817d0,&
    6.691d0,6.442d0,6.194d0,5.951d0,5.712d0,5.595d0,5.48d0,5.253d0,5.034d0,4.52d0,&
    4.059d0,3.652d0,3.297d0,2.729d0,2.317d0,2.022d0,1.812d0,1.66d0,1.546d0,1.459d0,&
    1.387d0,1.326d0,1.27d0,1.219d0,1.169d0,1.12d0,0.892d0,0.691d0,0.527d0,0.401d0,&
    0.234d0,0.141d0,0.0871d0,0.067d0,0.0268d0,0.0062022d0,0.0020816d0,0.0000594d0,9.582d-6,4.054d-6/
    DATA (ScatteringFactors(13,I),I=1,NQ) /&
    13d0,12.976d0,12.903d0,12.786d0,12.629d0,12.439d0,12.222d0,11.987d0,11.739d0,11.485d0,&
    11.23d0,10.978d0,10.733d0,10.498d0,10.273d0,10.059d0,9.857d0,9.667d0,9.487d0,9.318d0,&
    9.158d0,8.862d0,8.592d0,8.465d0,8.341d0,8.103d0,7.873d0,7.648d0,7.426d0,7.316d0,&
    7.205d0,6.985d0,6.766d0,6.548d0,6.33d0,6.222d0,6.115d0,5.902d0,5.692d0,5.186d0,&
    4.713d0,4.277d0,3.883d0,3.221d0,2.712d0,2.33d0,2.049d0,1.841d0,1.687d0,1.571d0,&
    1.481d0,1.408d0,1.346d0,1.292d0,1.243d0,1.195d0,0.979d0,0.783d0,0.615d0,0.478d0,&
    0.29d0,0.179d0,0.113d0,0.087d0,0.0349d0,0.0084081d0,0.0028477d0,0.0000824d0,0.0000134d0,5.67d-6/
    DATA (ScatteringFactors(14,I),I=1,NQ) /&
    14d0,13.976d0,13.904d0,13.787d0,13.628d0,13.434d0,13.209d0,12.961d0,12.695d0,12.417d0,&
    12.134d0,11.849d0,11.567d0,11.292d0,11.025d0,10.769d0,10.525d0,10.293d0,10.074d0,9.868d0,&
    9.673d0,9.319d0,9.004d0,8.859d0,8.722d0,8.467d0,8.231d0,8.011d0,7.8d0,7.698d0,&
    7.597d0,7.398d0,7.202d0,7.008d0,6.815d0,6.719d0,6.622d0,6.431d0,6.24d0,5.769d0,&
    5.312d0,4.878d0,4.47d0,3.75d0,3.164d0,2.702d0,2.346d0,2.076d0,1.872d0,1.717d0,&
    1.598d0,1.505d0,1.43d0,1.367d0,1.313d0,1.264d0,1.056d0,0.867d0,0.699d0,0.566d0,&
    0.349d0,0.222d0,0.1416d0,0.1096d0,0.0456d0,0.011116d0,0.0038011d0,0.0001117d0,0.0000182d0,7.744d-6/
    DATA (ScatteringFactors(15,I),I=1,NQ) /&
    15d0,14.977d0,14.909d0,14.798d0,14.646d0,14.458d0,14.237d0,13.99d0,13.721d0,13.435d0,&
    13.138d0,12.834d0,12.527d0,12.223d0,11.922d0,11.629d0,11.345d0,11.072d0,10.811d0,10.563d0,&
    10.327d0,9.894d0,9.51d0,9.335d0,9.17d0,8.869d0,8.6d0,8.357d0,8.134d0,8.029d0,&
    7.928d0,7.733d0,7.547d0,7.367d0,7.19d0,7.103d0,7.017d0,6.845d0,6.674d0,6.25d0,&
    5.829d0,5.418d0,5.02d0,4.284d0,3.649d0,3.122d0,2.698d0,2.364d0,2.104d0,1.903d0,&
    1.747d0,1.626d0,1.53d0,1.453d0,1.389d0,1.333d0,1.122d0,0.942d0,0.777d0,0.632d0,&
    0.411d0,0.268d0,0.1741d0,0.1354d0,0.0579d0,0.01438d0,0.0049671d0,0.0001483d0,0.0000243d0,0.0000104d0/
    DATA (ScatteringFactors(16,I),I=1,NQ) /&
    16d0,15.979d0,15.915d0,15.809d0,15.665d0,15.484d0,15.271d0,15.03d0,14.764d0,14.478d0,&
    14.177d0,13.865d0,13.546d0,13.224d0,12.902d0,12.583d0,12.27d0,11.964d0,11.668d0,11.382d0,&
    11.109d0,10.598d0,10.138d0,9.927d0,9.727d0,9.363d0,9.039d0,8.752d0,8.494d0,8.376d0,&
    8.262d0,8.051d0,7.856d0,7.673d0,7.501d0,7.417d0,7.335d0,7.174d0,7.017d0,6.633d0,&
    6.254d0,5.877d0,5.505d0,4.79d0,4.138d0,3.57d0,3.092d0,2.699d0,2.384d0,2.133d0,&
    1.935d0,1.779d0,1.655d0,1.557d0,1.477d0,1.411d0,1.182d0,1.009d0,0.849d0,0.705d0,&
    0.474d0,0.316d0,0.2098d0,0.1641d0,0.0726d0,0.01825d0,0.006311d0,0.0001933d0,0.0000319d0,0.0000136d0/
    DATA (ScatteringFactors(17,I),I=1,NQ) /&
    17d0,16.98d0,16.919d0,16.82d0,16.683d0,16.511d0,16.306d0,16.073d0,15.814d0,15.533d0,&
    15.234d0,14.921d0,14.597d0,14.266d0,13.932d0,13.597d0,13.263d0,12.934d0,12.611d0,12.297d0,&
    11.991d0,11.413d0,10.881d0,10.633d0,10.398d0,9.964d0,9.576d0,9.231d0,8.923d0,8.782d0,&
    8.649d0,8.403d0,8.181d0,7.979d0,7.794d0,7.706d0,7.621d0,7.459d0,7.305d0,6.941d0,&
    6.595d0,6.254d0,5.915d0,5.245d0,4.607d0,4.023d0,3.509d0,3.07d0,2.704d0,2.405d0,&
    2.162d0,1.967d0,1.811d0,1.686d0,1.585d0,1.502d0,1.24d0,1.069d0,0.915d0,0.773d0,&
    0.536d0,0.367d0,0.2468d0,0.1936d0,0.0873d0,0.022774d0,0.0080384d0,0.0002481d0,0.0000411d0,0.0000176d0/
    DATA (ScatteringFactors(18,I),I=1,NQ) /&
    18d0,17.981d0,17.924d0,17.83d0,17.7d0,17.536d0,17.34d0,17.116d0,16.865d0,16.591d0,&
    16.298d0,15.988d0,15.665d0,15.331d0,14.991d0,14.647d0,14.301d0,13.957d0,13.615d0,13.279d0,&
    12.949d0,12.315d0,11.721d0,11.441d0,11.172d0,10.671d0,10.216d0,9.807d0,9.441d0,9.272d0,&
    9.113d0,8.82d0,8.558d0,8.322d0,8.11d0,8.011d0,7.917d0,7.739d0,7.575d0,7.207d0,&
    6.875d0,6.56d0,6.252d0,5.639d0,5.036d0,4.46d0,3.931d0,3.462d0,3.056d0,2.713d0,&
    2.427d0,2.192d0,2d0,1.844d0,1.717d0,1.614d0,1.301d0,1.123d0,0.974d0,0.836d0,&
    0.597d0,0.419d0,0.2873d0,0.2268d0,0.1057d0,0.027995d0,0.0099948d0,0.000314d0,0.0000523d0,0.0000225d0/
    DATA (ScatteringFactors(19,I),I=1,NQ) /&
    19d0,18.963d0,18.854d0,18.683d0,18.462d0,18.204d0,17.924d0,17.63d0,17.332d0,17.032d0,&
    16.733d0,16.436d0,16.138d0,15.841d0,15.543d0,15.243d0,14.941d0,14.638d0,14.334d0,14.031d0,&
    13.728d0,13.13d0,12.55d0,12.268d0,11.994d0,11.468d0,10.977d0,10.521d0,10.103d0,9.908d0,&
    9.722d0,9.375d0,9.061d0,8.778d0,8.522d0,8.403d0,8.29d0,8.08d0,7.889d0,7.474d0,&
    7.125d0,6.814d0,6.523d0,5.961d0,5.406d0,4.859d0,4.337d0,3.855d0,3.423d0,3.045d0,&
    2.722d0,2.45d0,2.221d0,2.033d0,1.876d0,1.748d0,1.367d0,1.174d0,1.028d0,0.895d0,&
    0.657d0,0.472d0,0.3296d0,0.2614d0,0.1251d0,0.033951d0,0.012265d0,0.0003926d0,0.0000658d0,0.0000284d0/
    DATA (ScatteringFactors(20,I),I=1,NQ) /&
    20d0,19.959d0,19.838d0,19.645d0,19.392d0,19.091d0,18.758d0,18.405d0,18.045d0,17.685d0,&
    17.331d0,16.987d0,16.655d0,16.334d0,16.024d0,15.723d0,15.43d0,15.142d0,14.859d0,14.58d0,&
    14.304d0,13.76d0,13.225d0,12.961d0,12.701d0,12.194d0,11.705d0,11.24d0,10.8d0,10.59d0,&
    10.388d0,10.004d0,9.65d0,9.324d0,9.025d0,8.885d0,8.752d0,8.502d0,8.275d0,7.788d0,&
    7.392d0,7.057d0,6.762d0,6.228d0,5.717d0,5.209d0,4.71d0,4.233d0,3.791d0,3.391d0,&
    3.039d0,2.733d0,2.47d0,2.25d0,2.063d0,1.908d0,1.444d0,1.225d0,1.078d0,0.949d0,&
    0.715d0,0.524d0,0.3722d0,0.2972d0,0.1472d0,0.040674d0,0.014814d0,0.0004853d0,0.0000818d0,0.0000353d0/
    DATA (ScatteringFactors(21,I),I=1,NQ) /&
    21d0,20.962d0,20.848d0,20.665d0,20.422d0,20.131d0,19.805d0,19.455d0,19.091d0,18.723d0,&
    18.356d0,17.995d0,17.643d0,17.301d0,16.968d0,16.645d0,16.33d0,16.023d0,15.722d0,15.426d0,&
    15.135d0,14.564d0,14.006d0,13.732d0,13.462d0,12.933d0,12.423d0,11.934d0,11.467d0,11.244d0,&
    11.027d0,10.613d0,10.226d0,9.866d0,9.534d0,9.377d0,9.227d0,8.946d0,8.687d0,8.132d0,&
    7.682d0,7.312d0,6.996d0,6.46d0,5.975d0,5.501d0,5.03d0,4.57d0,4.131d0,3.722d0,&
    3.352d0,3.023d0,2.733d0,2.485d0,2.271d0,2.09d0,1.533d0,1.279d0,1.125d0,0.998d0,&
    0.77d0,0.577d0,0.4156d0,0.3338d0,0.1703d0,0.048192d0,0.017845d0,0.0005938d0,0.0001007d0,0.0000437d0/
    DATA (ScatteringFactors(22,I),I=1,NQ) /&
    22d0,21.964d0,21.856d0,21.682d0,21.451d0,21.171d0,20.854d0,20.511d0,20.15d0,19.781d0,&
    19.41d0,19.041d0,18.678d0,18.322d0,17.974d0,17.635d0,17.304d0,16.98d0,16.663d0,16.351d0,&
    16.044d0,15.444d0,14.859d0,14.572d0,14.289d0,13.735d0,13.198d0,12.682d0,12.187d0,11.949d0,&
    11.717d0,11.271d0,10.852d0,10.459d0,10.093d0,9.92d0,9.753d0,9.438d0,9.148d0,8.518d0,&
    8.007d0,7.588d0,7.24d0,6.676d0,6.2d0,5.752d0,5.31d0,4.872d0,4.445d0,4.038d0,&
    3.66d0,3.316d0,3.006d0,2.734d0,2.496d0,2.29d0,1.637d0,1.338d0,1.171d0,1.044d0,&
    0.821d0,0.627d0,0.4603d0,0.3112d0,0.1931d0,0.056526d0,0.021201d0,0.00072d0,0.0001228d0,0.0000534d0/
    DATA (ScatteringFactors(23,I),I=1,NQ) /&
    23d0,22.966d0,22.864d0,22.698d0,22.477d0,22.208d0,21.902d0,21.567d0,21.212d0,20.846d0,&
    20.474d0,20.102d0,19.733d0,19.369d0,19.011d0,18.661d0,18.317d0,17.98d0,17.649d0,17.323d0,&
    17.003d0,16.376d0,15.765d0,15.465d0,15.169d0,14.589d0,14.026d0,13.482d0,12.959d0,12.705d0,&
    12.458d0,11.982d0,11.53d0,11.105d0,10.705d0,10.515d0,10.332d0,9.984d0,9.66d0,8.952d0,&
    8.373d0,7.898d0,7.506d0,6.892d0,6.406d0,5.972d0,5.553d0,5.139d0,4.73d0,4.333d0,&
    3.956d0,3.604d0,3.281d0,2.992d0,2.733d0,2.506d0,1.756d0,1.404d0,1.217d0,1.087d0,&
    0.869d0,0.677d0,0.5051d0,0.4099d0,0.2195d0,0.065692d0,0.024962d0,0.0008656d0,0.0001486d0,0.0000648d0/
    DATA (ScatteringFactors(24,I),I=1,NQ) /&
    24d0,23.971d0,23.885d0,23.746d0,23.558d0,23.329d0,23.065d0,22.772d0,22.459d0,22.129d0,&
    21.789d0,21.441d0,21.089d0,20.734d0,20.378d0,20.022d0,19.667d0,19.312d0,18.96d0,18.609d0,&
    18.26d0,17.57d0,16.893d0,16.561d0,16.232d0,15.588d0,14.965d0,14.365d0,13.79d0,13.513d0,&
    13.242d0,12.72d0,12.227d0,11.762d0,11.326d0,11.118d0,10.917d0,10.536d0,10.18d0,9.4d0,&
    8.756d0,8.227d0,7.791d0,7.118d0,6.606d0,6.172d0,5.768d0,5.372d0,4.982d0,4.597d0,&
    4.226d0,3.874d0,3.545d0,3.244d0,2.971d0,2.727d0,1.888d0,1.479d0,1.266d0,1.129d0,&
    0.914d0,0.724d0,0.5476d0,0.4473d0,0.2466d0,0.075698d0,0.02915d0,0.0010329d0,0.0001785d0,0.000078d0/
    DATA (ScatteringFactors(25,I),I=1,NQ) /&
    25d0,24.969d0,24.876d0,24.726d0,24.523d0,24.274d0,23.988d0,23.671d0,23.331d0,22.976d0,&
    22.611d0,22.24d0,21.868d0,21.497d0,21.128d0,20.764d0,20.404d0,20.049d0,19.699d0,19.354d0,&
    19.012d0,18.342d0,17.686d0,17.364d0,17.045d0,16.417d0,15.806d0,15.211d0,14.634d0,14.353d0,&
    14.078d0,13.543d0,13.031d0,12.543d0,12.08d0,11.858d0,11.642d0,11.228d0,10.84d0,9.973d0,&
    9.245d0,8.639d0,8.137d0,7.368d0,6.808d0,6.359d0,5.962d0,5.586d0,5.215d0,4.849d0,&
    4.49d0,4.144d0,3.814d0,3.506d0,3.221d0,2.963d0,2.037d0,1.563d0,1.319d0,1.171d0,&
    0.956d0,0.769d0,0.5903d0,0.4847d0,0.2736d0,0.086549d0,0.033781d0,0.0012237d0,0.0002128d0,0.0000933d0/
    DATA (ScatteringFactors(26,I),I=1,NQ) /&
    26d0,25.97d0,25.882d0,25.738d0,25.543d0,25.304d0,25.026d0,24.719d0,24.387d0,24.038d0,&
    23.678d0,23.31d0,22.939d0,22.568d0,22.197d0,21.829d0,21.465d0,21.104d0,20.748d0,20.395d0,&
    20.046d0,19.359d0,18.685d0,18.354d0,18.025d0,17.378d0,16.744d0,16.127d0,15.527d0,15.233d0,&
    14.945d0,14.384d0,13.845d0,13.328d0,12.835d0,12.598d0,12.367d0,11.922d0,11.502d0,10.557d0,&
    9.753d0,9.077d0,8.512d0,7.645d0,7.023d0,6.545d0,6.143d0,5.775d0,5.42d0,5.07d0,&
    4.725d0,4.388d0,4.062d0,3.753d0,3.463d0,3.195d0,2.197d0,1.658d0,1.377d0,1.213d0,&
    0.995d0,0.813d0,0.6311d0,0.5227d0,0.3048d0,0.098241d0,0.038814d0,0.0014403d0,0.0002521d0,0.0001108d0/
    DATA (ScatteringFactors(27,I),I=1,NQ) /&
    27d0,26.972d0,26.887d0,26.749d0,26.562d0,26.331d0,26.063d0,25.764d0,25.44d0,25.098d0,&
    24.744d0,24.38d0,24.011d0,23.641d0,23.27d0,22.9d0,22.533d0,22.168d0,21.806d0,21.448d0,&
    21.093d0,20.393d0,19.704d0,19.364d0,19.027d0,18.361d0,17.709d0,17.072d0,16.45d0,16.145d0,&
    15.845d0,15.26d0,14.695d0,14.151d0,13.63d0,13.379d0,13.133d0,12.659d0,12.209d0,11.188d0,&
    10.309d0,9.561d0,8.93d0,7.955d0,7.259d0,6.738d0,6.318d0,5.95d0,5.601d0,5.27d0,&
    4.939d0,4.611d0,4.295d0,3.989d0,3.697d0,3.424d0,2.366d0,1.763d0,1.441d0,1.258d0,&
    1.033d0,0.853d0,0.6731d0,0.5593d0,0.3317d0,0.11077d0,0.044441d0,0.0016851d0,0.0002969d0,0.000131d0/
    DATA (ScatteringFactors(28,I),I=1,NQ) /&
    28d0,27.973d0,27.892d0,27.759d0,27.579d0,27.356d0,27.096d0,26.806d0,26.49d0,26.156d0,&
    25.807d0,25.448d0,25.083d0,24.714d0,24.344d0,23.973d0,23.604d0,23.237d0,22.872d0,22.51d0,&
    22.15d0,21.438d0,20.737d0,20.39d0,20.046d0,19.365d0,18.696d0,18.04d0,17.398d0,17.084d0,&
    16.773d0,16.165d0,15.576d0,15.008d0,14.461d0,14.196d0,13.937d0,13.435d0,12.956d0,11.862d0,&
    10.909d0,10.09d0,9.392d0,8.301d0,7.519d0,6.944d0,6.495d0,6.118d0,5.776d0,5.451d0,&
    5.133d0,4.819d0,4.511d0,4.211d0,3.922d0,3.647d0,2.543d0,1.878d0,1.512d0,1.306d0,&
    1.069d0,0.892d0,0.7138d0,0.5972d0,0.3641d0,0.12411d0,0.050496d0,0.0019605d0,0.0003478d0,0.0001538d0/
    DATA (ScatteringFactors(29,I),I=1,NQ) /&
    29d0,28.977d0,28.908d0,28.794d0,28.64d0,28.448d0,28.223d0,27.971d0,27.694d0,27.397d0,&
    27.084d0,26.758d0,26.422d0,26.077d0,25.726d0,25.37d0,25.009d0,24.645d0,24.278d0,23.91d0,&
    23.54d0,22.798d0,22.057d0,21.687d0,21.319d0,20.589d0,19.869d0,19.162d0,18.472d0,18.133d0,&
    17.799d0,17.145d0,16.514d0,15.904d0,15.318d0,15.034d0,14.757d0,14.219d0,13.707d0,12.533d0,&
    11.507d0,10.621d0,9.861d0,8.663d0,7.799d0,7.166d0,6.681d0,6.285d0,5.939d0,5.617d0,&
    5.308d0,5.005d0,4.705d0,4.413d0,4.128d0,3.855d0,2.721d0,2.001d0,1.59d0,1.358d0,&
    1.105d0,0.929d0,0.7498d0,0.6308d0,0.3929d0,0.13827d0,0.057051d0,0.0022689d0,0.0004053d0,0.0001798d0/
    DATA (ScatteringFactors(30,I),I=1,NQ) /&
    30d0,29.975d0,29.9d0,29.777d0,29.609d0,29.401d0,29.157d0,28.883d0,28.583d0,28.263d0,&
    27.927d0,27.579d0,27.222d0,26.859d0,26.492d0,26.124d0,25.754d0,25.385d0,25.017d0,24.649d0,&
    24.283d0,23.556d0,22.836d0,22.478d0,22.122d0,21.417d0,20.72d0,20.034d0,19.359d0,19.027d0,&
    18.698d0,18.051d0,17.421d0,16.809d0,16.216d0,15.926d0,15.642d0,15.09d0,14.559d0,13.328d0,&
    12.235d0,11.276d0,10.442d0,9.108d0,8.132d0,7.417d0,6.879d0,6.453d0,6.096d0,5.775d0,&
    5.473d0,5.18d0,4.892d0,4.61d0,4.332d0,4.063d0,2.908d0,2.135d0,1.677d0,1.414d0,&
    1.14d0,0.964d0,0.7837d0,0.6639d0,0.4242d0,0.1532d0,0.064113d0,0.0026131d0,0.00047d0,0.0002091d0/
    DATA (ScatteringFactors(31,I),I=1,NQ) /&
    31d0,30.971d0,30.883d0,30.74d0,30.546d0,30.308d0,30.031d0,29.724d0,29.391d0,29.04d0,&
    28.675d0,28.302d0,27.924d0,27.543d0,27.162d0,26.783d0,26.406d0,26.033d0,25.663d0,25.297d0,&
    24.935d0,24.121d0,23.52d0,23.174d0,22.83d0,22.151d0,21.481d0,20.82d0,20.169d0,19.847d0,&
    19.527d0,18.897d0,18.278d0,17.673d0,17.083d0,16.794d0,16.508d0,15.95d0,15.41d0,14.142d0,&
    12.996d0,11.974d0,11.073d0,9.604d0,8.51d0,7.702d0,7.099d0,6.633d0,6.254d0,5.926d0,&
    5.627d0,5.342d0,5.065d0,4.792d0,4.523d0,4.26d0,3.097d0,2.277d0,1.772d0,1.477d0,&
    1.176d0,0.998d0,0.8182d0,0.6969d0,0.4542d0,0.16889d0,0.07169d0,0.0029958d0,0.0005427d0,0.0002422d0/
    DATA (ScatteringFactors(32,I),I=1,NQ) /&
    32d0,31.97d0,31.878d0,31.729d0,31.526d0,31.276d0,30.984d0,30.657d0,30.302d0,29.926d0,&
    29.534d0,29.133d0,28.725d0,28.316d0,27.908d0,27.504d0,27.104d0,26.709d0,26.322d0,25.941d0,&
    25.567d0,24.839d0,24.135d0,23.791d0,23.452d0,22.787d0,22.136d0,21.498d0,20.87d0,20.56d0,&
    20.253d0,19.645d0,19.047d0,18.459d0,17.882d0,17.598d0,17.317d0,16.765d0,16.227d0,14.947d0,&
    13.77d0,12.702d0,11.745d0,10.151d0,8.937d0,8.028d0,7.348d0,6.83d0,6.419d0,6.076d0,&
    5.774d0,5.493d0,5.224d0,4.961d0,4.702d0,4.447d0,3.287d0,2.428d0,1.876d0,1.545d0,&
    1.213d0,1.03d0,0.8526d0,0.7307d0,0.4868d0,0.1853d0,0.079186d0,0.0034198d0,0.0006239d0,0.0002793d0/
    DATA (ScatteringFactors(33,I),I=1,NQ) /&
    33d0,32.97d0,32.879d0,32.73d0,32.527d0,32.274d0,31.977d0,31.642d0,31.276d0,30.884d0,&
    30.473d0,30.049d0,29.616d0,29.179d0,28.742d0,28.307d0,27.877d0,27.454d0,27.039d0,26.633d0,&
    26.235d0,25.469d0,24.739d0,24.386d0,24.041d0,23.37d0,22.724d0,22.097d0,21.486d0,21.185d0,&
    20.888d0,20.301d0,19.725d0,19.159d0,18.602d0,18.326d0,18.054d0,17.516d0,16.989d0,15.721d0,&
    14.535d0,13.44d0,12.442d0,10.741d0,9.411d0,8.396d0,7.631d0,7.05d0,6.597d0,6.231d0,&
    5.917d0,5.636d0,5.372d0,5.117d0,4.867d0,4.621d0,3.475d0,2.584d0,1.988d0,1.621d0,&
    1.251d0,1.061d0,0.886d0,0.7629d0,0.5166d0,0.20242d0,0.088404d0,0.003888d0,0.0007145d0,0.0003208d0/
    DATA (ScatteringFactors(34,I),I=1,NQ) /&
    34d0,33.97d0,33.881d0,33.734d0,33.532d0,33.28d0,32.982d0,32.645d0,32.273d0,31.872d0,&
    31.449d0,31.009d0,30.557d0,30.099d0,29.637d0,29.175d0,28.718d0,28.266d0,27.822d0,27.387d0,&
    26.962d0,26.145d0,25.372d0,25.001d0,24.641d0,23.947d0,23.288d0,22.656d0,22.048d0,21.751d0,&
    21.459d0,20.887d0,20.328d0,19.78d0,19.242d0,18.977d0,18.713d0,18.193d0,17.682d0,16.444d0,&
    15.269d0,14.166d0,13.145d0,11.362d0,9.928d0,8.809d0,7.952d0,7.299d0,6.795d0,6.395d0,&
    6.063d0,5.775d0,5.511d0,5.262d0,5.02d0,4.782d0,3.658d0,2.745d0,2.108d0,1.703d0,&
    1.292d0,1.092d0,0.9161d0,0.793d0,0.5468d0,0.22019d0,0.097545d0,0.0044035d0,0.0008153d0,0.0003672d0/
    DATA (ScatteringFactors(35,I),I=1,NQ) /&
    35d0,34.971d0,34.883d0,34.739d0,34.54d0,34.291d0,33.995d0,33.658d0,33.284d0,32.88d0,&
    32.45d0,32d0,31.535d0,31.06d0,30.578d0,30.095d0,29.613d0,29.136d0,28.664d0,28.202d0,&
    27.749d0,26.876d0,26.052d0,25.658d0,25.276d0,24.545d0,23.857d0,23.206d0,22.587d0,22.288d0,&
    21.995d0,21.425d0,20.874d0,20.338d0,19.816d0,19.558d0,19.304d0,18.801d0,18.307d0,17.107d0,&
    15.958d0,14.865d0,13.837d0,12.001d0,10.48d0,9.262d0,8.312d0,7.58d0,7.016d0,6.574d0,&
    6.216d0,5.913d0,5.645d0,5.398d0,5.162d0,4.932d0,3.836d0,2.909d0,2.235d0,1.793d0,&
    1.337d0,1.123d0,0.9418d0,0.8202d0,0.5769d0,0.23858d0,0.10721d0,0.0049695d0,0.000927d0,0.0004188d0/
    DATA (ScatteringFactors(36,I),I=1,NQ) /&
    36d0,35.972d0,35.886d0,35.744d0,35.549d0,35.304d0,35.011d0,34.677d0,34.305d0,33.899d0,&
    33.467d0,33.011d0,32.537d0,32.051d0,31.555d0,31.055d0,30.553d0,30.053d0,29.558d0,29.07d0,&
    28.59d0,27.663d0,26.784d0,26.364d0,25.957d0,25.181d0,24.453d0,23.771d0,23.128d0,22.82d0,&
    22.52d0,21.941d0,21.388d0,20.855d0,20.339d0,20.087d0,19.838d0,19.349d0,18.87d0,17.709d0,&
    16.594d0,15.524d0,14.504d0,12.645d0,11.057d0,9.752d0,8.711d0,7.898d0,7.266d0,6.773d0,&
    6.38d0,6.056d0,5.778d0,5.528d0,5.295d0,5.071d0,4.007d0,3.074d0,2.369d0,1.89d0,&
    1.384d0,1.154d0,0.9662d0,0.846d0,0.6057d0,0.25757d0,0.11739d0,0.0055892d0,0.0010506d0,0.0004762d0/
    DATA (ScatteringFactors(37,I),I=1,NQ) /&
    37d0,36.952d0,36.809d0,36.583d0,36.291d0,35.948d0,35.571d0,35.171d0,34.758d0,34.336d0,&
    33.907d0,33.473d0,33.034d0,32.588d0,32.137d0,31.681d0,31.22d0,30.757d0,30.293d0,29.83d0,&
    29.368d0,28.459d0,27.576d0,27.148d0,26.729d0,25.922d0,25.158d0,24.437d0,23.758d0,23.432d0,&
    23.116d0,22.51d0,21.934d0,21.386d0,20.86d0,20.605d0,20.354d0,19.866d0,19.391d0,18.252d0,&
    17.167d0,16.125d0,15.126d0,13.272d0,11.645d0,10.27d0,9.147d0,8.252d0,7.548d0,6.996d0,&
    6.562d0,6.21d0,5.913d0,5.656d0,5.42d0,5.2d0,4.168d0,3.239d0,2.507d0,1.993d0,&
    1.436d0,1.186d0,0.9922d0,0.8733d0,0.6356d0,0.2771d0,0.12809d0,0.0062659d0,0.0011869d0,0.0005397d0/
    DATA (ScatteringFactors(38,I),I=1,NQ) /&
    38d0,37.946d0,37.786d0,37.532d0,37.197d0,36.802d0,36.363d0,35.897d0,35.418d0,34.937d0,&
    34.458d0,33.986d0,33.522d0,33.066d0,32.616d0,32.171d0,31.73d0,31.292d0,30.856d0,30.421d0,&
    29.988d0,29.128d0,28.28d0,27.863d0,27.452d0,26.648d0,25.875d0,25.135d0,24.43d0,24.09d0,&
    23.76d0,23.125d0,22.522d0,21.95d0,21.404d0,21.141d0,20.883d0,20.383d0,19.902d0,18.764d0,&
    17.696d0,16.678d0,15.702d0,13.872d0,12.23d0,10.806d0,9.612d0,8.64d0,7.863d0,7.249d0,&
    6.764d0,6.376d0,6.055d0,5.785d0,5.544d0,5.323d0,4.32d0,3.401d0,2.649d0,2.103d0,&
    1.493d0,1.219d0,1.0188d0,0.9008d0,0.6647d0,0.29713d0,0.13929d0,0.0070031d0,0.001337d0,0.0006099d0/
    DATA (ScatteringFactors(39,I),I=1,NQ) /&
    39d0,38.947d0,38.792d0,38.543d0,38.212d0,37.816d0,37.369d0,36.889d0,36.387d0,35.876d0,&
    35.364d0,34.855d0,34.354d0,33.861d0,33.378d0,32.904d0,32.437d0,31.977d0,31.523d0,31.075d0,&
    30.631d0,29.758d0,28.904d0,28.485d0,28.071d0,27.263d0,26.483d0,25.734d0,25.018d0,24.673d0,&
    24.336d0,23.687d0,23.071d0,22.485d0,21.928d0,21.66d0,21.398d0,20.89d0,20.404d0,19.263d0,&
    18.204d0,17.203d0,16.246d0,14.443d0,12.798d0,11.339d0,10.088d0,9.046d0,8.2d0,7.523d0,&
    6.985d0,6.554d0,6.205d0,5.914d0,5.662d0,5.44d0,4.46d0,3.56d0,2.78d0,2.215d0,&
    1.55d0,1.25d0,1.045d0,0.9271d0,0.6914d0,0.31763d0,0.15099d0,0.0078042d0,0.0015019d0,0.0006873d0/
    DATA (ScatteringFactors(40,I),I=1,NQ) /&
    40d0,39.949d0,39.8d0,39.559d0,39.237d0,38.847d0,38.403d0,37.921d0,37.412d0,36.887d0,&
    36.356d0,35.824d0,35.296d0,34.775d0,34.262d0,33.758d0,33.263d0,32.776d0,32.298d0,31.827d0,&
    31.363d0,30.454d0,29.572d0,29.141d0,28.716d0,27.889d0,27.092d0,26.327d0,25.596d0,25.243d0,&
    24.899d0,24.236d0,23.606d0,23.008d0,22.439d0,22.166d0,21.899d0,21.384d0,20.892d0,19.745d0,&
    18.693d0,17.706d0,16.767d0,14.996d0,13.361d0,11.883d0,10.588d0,9.486d0,8.574d0,7.833d0,&
    7.238d0,6.76d0,6.375d0,6.059d0,5.79d0,5.558d0,4.59d0,3.72d0,2.92d0,2.335d0,&
    1.62d0,1.285d0,1.0701d0,0.9526d0,0.7175d0,0.33856d0,0.16319d0,0.008673d0,0.0016826d0,0.0007725d0/
    DATA (ScatteringFactors(41,I),I=1,NQ) /&
    41d0,40.956d0,40.824d0,40.61d0,40.323d0,39.97d0,39.565d0,39.116d0,38.634d0,38.128d0,&
    37.606d0,37.073d0,36.535d0,35.994d0,35.454d0,34.916d0,34.382d0,33.854d0,33.331d0,32.814d0,&
    32.305d0,31.31d0,30.348d0,29.881d0,29.424d0,28.538d0,27.692d0,26.888d0,26.126d0,25.76d0,&
    25.404d0,24.721d0,24.077d0,23.468d0,22.892d0,22.615d0,22.346d0,21.829d0,21.336d0,20.195d0,&
    19.156d0,18.187d0,17.268d0,15.533d0,13.915d0,12.427d0,11.098d0,9.945d0,8.972d0,8.169d0,&
    7.516d0,6.969d0,6.564d0,6.216d0,5.927d0,5.68d0,4.71d0,3.86d0,3.065d0,2.405d0,&
    1.69d0,1.327d0,1.0953d0,0.9779d0,0.7431d0,0.35986d0,0.17586d0,0.0096131d0,0.0018802d0,0.0008661d0/
    DATA (ScatteringFactors(42,I),I=1,NQ) /&
    42d0,41.958d0,41.831d0,41.625d0,41.346d0,41.003d0,40.606d0,40.164d0,39.686d0,39.181d0,&
    38.656d0,38.117d0,37.569d0,37.016d0,36.461d0,35.907d0,35.355d0,34.806d0,34.263d0,33.725d0,&
    33.195d0,32.157d0,31.153d0,30.665d0,30.188d0,29.263d0,28.382d0,27.543d0,26.749d0,26.368d0,&
    25.998d0,25.289d0,24.62d0,23.989d0,23.394d0,23.109d0,22.832d0,22.3d0,21.796d0,20.638d0,&
    19.595d0,18.635d0,17.732d0,16.036d0,14.448d0,12.968d0,11.621d0,10.43d0,9.404d0,8.542d0,&
    7.831d0,7.251d0,6.78d0,6.397d0,6.08d0,5.813d0,4.827d0,3.988d0,3.217d0,2.581d0,&
    1.766d0,1.373d0,1.1206d0,1.0029d0,0.7675d0,0.38152d0,0.18901d0,0.010628d0,0.0020961d0,0.0009688d0/
    DATA (ScatteringFactors(43,I),I=1,NQ) /&
    43d0,42.955d0,42.821d0,42.603d0,42.308d0,41.945d0,41.526d0,41.059d0,40.557d0,40.028d0,&
    39.48d0,38.921d0,38.355d0,37.787d0,37.221d0,36.658d0,36.1d0,35.548d0,35.003d0,34.466d0,&
    33.936d0,32.9d0,31.897d0,31.409d0,30.93d0,29.998d0,29.104d0,28.25d0,27.435d0,27.042d0,&
    26.66d0,25.925d0,25.229d0,24.571d0,23.949d0,23.651d0,23.361d0,22.806d0,22.28d0,21.08d0,&
    20.012d0,19.042d0,18.142d0,16.477d0,14.925d0,13.466d0,12.116d0,10.9d0,9.833d0,8.919d0,&
    8.154d0,7.521d0,7.004d0,6.582d0,6.234d0,5.946d0,4.93d0,4.11d0,3.35d0,2.69d0,&
    1.84d0,1.42d0,1.146d0,1.0279d0,0.7917d0,0.40348d0,0.2026d0,0.011722d0,0.0023313d0,0.0010811d0/
    DATA (ScatteringFactors(44,I),I=1,NQ) /&
    44d0,43.96d0,43.842d0,43.649d0,43.386d0,43.061d0,42.681d0,42.254d0,41.789d0,41.292d0,&
    40.77d0,40.229d0,39.674d0,39.108d0,38.536d0,37.959d0,37.381d0,36.803d0,36.228d0,35.655d0,&
    35.088d0,33.971d0,32.886d0,32.356d0,31.837d0,30.829d0,29.866d0,28.949d0,28.079d0,27.662d0,&
    27.257d0,26.48d0,25.749d0,25.062d0,24.415d0,24.106d0,23.807d0,23.235d0,22.696d0,21.476d0,&
    20.403d0,19.438d0,18.551d0,16.922d0,15.405d0,13.968d0,12.62d0,11.385d0,10.282d0,9.323d0,&
    8.506d0,7.823d0,7.258d0,6.794d0,6.412d0,6.097d0,5.04d0,4.23d0,3.485d0,2.82d0,&
    1.925d0,1.47d0,1.1715d0,1.0531d0,0.8164d0,0.4257d0,0.21664d0,0.012899d0,0.0025872d0,0.0012039d0/
    DATA (ScatteringFactors(45,I),I=1,NQ) /&
    45d0,44.961d0,44.847d0,44.66d0,44.405d0,44.088d0,43.717d0,43.299d0,42.842d0,42.351d0,&
    41.834d0,41.296d0,40.741d0,40.173d0,39.597d0,39.015d0,38.429d0,37.841d0,37.254d0,36.668d0,&
    36.086d0,34.937d0,33.815d0,33.267d0,32.728d0,31.68d0,30.675d0,29.717d0,28.807d0,28.37d0,&
    27.944d0,27.13d0,26.363d0,25.642d0,24.964d0,24.64d0,24.327d0,23.729d0,23.167d0,21.9d0,&
    20.798d0,19.82d0,18.932d0,17.326d0,15.845d0,14.44d0,13.107d0,11.866d0,10.74d0,9.743d0,&
    8.88d0,8.148d0,7.535d0,7.028d0,6.608d0,6.262d0,5.14d0,4.35d0,3.62d0,2.94d0,&
    2.012d0,1.52d0,1.1975d0,1.0784d0,0.8401d0,0.44815d0,0.2311d0,0.014163d0,0.0028652d0,0.0013378d0/
    DATA (ScatteringFactors(46,I),I=1,NQ) /&
    46d0,45.968d0,45.874d0,45.718d0,45.503d0,45.232d0,44.908d0,44.535d0,44.119d0,43.663d0,&
    43.172d0,42.651d0,42.105d0,41.538d0,40.954d0,40.357d0,39.75d0,39.137d0,38.52d0,37.902d0,&
    37.286d0,36.064d0,34.868d0,34.283d0,33.708d0,32.592d0,31.523d0,30.505d0,29.54d0,29.077d0,&
    28.628d0,27.769d0,26.961d0,26.202d0,25.491d0,25.153d0,24.825d0,24.201d0,23.617d0,22.307d0,&
    21.177d0,20.186d0,19.296d0,17.711d0,16.266d0,14.893d0,13.58d0,12.342d0,11.2d0,10.173d0,&
    9.27d0,8.492d0,7.833d0,7.282d0,6.824d0,6.443d0,5.24d0,4.46d0,3.74d0,3.08d0,&
    2.1d0,1.575d0,1.2248d0,1.1038d0,0.8618d0,0.4708d0,0.24597d0,0.015518d0,0.0031668d0,0.0014338d0/
    DATA (ScatteringFactors(47,I),I=1,NQ) /&
    47d0,46.964d0,46.857d0,46.681d0,46.44d0,46.139d0,45.786d0,45.385d0,44.944d0,44.469d0,&
    43.964d0,43.435d0,42.886d0,42.322d0,41.744d0,41.157d0,40.563d0,39.964d0,39.361d0,38.758d0,&
    38.154d0,36.955d0,35.774d0,35.192d0,34.619d0,33.498d0,32.416d0,31.378d0,30.387d0,29.91d0,&
    29.444d0,28.551d0,27.707d0,26.911d0,26.163d0,25.805d0,25.459d0,24.8d0,24.181d0,22.795d0,&
    21.607d0,20.575d0,19.661d0,18.069d0,16.651d0,15.316d0,14.035d0,12.813d0,11.669d0,10.623d0,&
    9.687d0,8.869d0,8.165d0,7.569d0,7.069d0,6.651d0,5.351d0,4.566d0,3.862d0,3.207d0,&
    2.206d0,1.635d0,1.2543d0,1.1303d0,0.8822d0,0.4936d0,0.26123d0,0.016968d0,0.0034933d0,0.0016425d0/
    DATA (ScatteringFactors(48,I),I=1,NQ) /&
    48d0,47.962d0,47.848d0,47.66d0,47.404d0,47.085d0,46.71d0,46.287d0,45.822d0,45.324d0,&
    44.797d0,44.248d0,43.683d0,43.104d0,42.517d0,41.923d0,41.325d0,40.726d0,40.126d0,39.527d0,&
    38.93d0,37.746d0,36.581d0,36.007d0,35.44d0,34.329d0,33.251d0,32.21d0,31.21d0,30.725d0,&
    30.252d0,29.338d0,28.468d0,27.644d0,26.865d0,26.492d0,26.129d0,25.436d0,24.784d0,23.32d0,&
    22.063d0,20.978d0,20.027d0,18.405d0,17d0,15.698d0,14.451d0,13.253d0,12.116d0,11.06d0,&
    10.101d0,9.249d0,8.505d0,7.867d0,7.326d0,6.871d0,5.461d0,4.665d0,3.977d0,3.33d0,&
    2.304d0,1.698d0,1.2859d0,1.158d0,0.9021d0,0.51653d0,0.27687d0,0.018518d0,0.0038464d0,0.001815d0/
    DATA (ScatteringFactors(49,I),I=1,NQ) /&
    49d0,48.957d0,48.828d0,48.618d0,48.332d0,47.98d0,47.57d0,47.112d0,46.614d0,46.086d0,&
    45.534d0,44.964d0,44.383d0,43.793d0,43.199d0,42.603d0,42.006d0,41.41d0,40.817d0,40.226d0,&
    39.639d0,38.478d0,37.337d0,36.774d0,36.218d0,35.125d0,34.059d0,33.025d0,32.025d0,31.538d0,&
    31.06d0,30.134d0,29.247d0,28.401d0,27.596d0,27.209d0,26.832d0,26.108d0,25.425d0,23.881d0,&
    22.552d0,21.405d0,20.408d0,18.736d0,17.329d0,16.053d0,14.84d0,13.67d0,12.548d0,11.492d0,&
    10.518d0,9.639d0,8.86d0,8.184d0,7.603d0,7.11d0,5.577d0,4.761d0,4.087d0,3.449d0,&
    2.406d0,1.746d0,1.3196d0,1.187d0,0.9218d0,0.53954d0,0.29287d0,0.020111d0,0.0042217d0,0.002002d0/
    DATA (ScatteringFactors(50,I),I=1,NQ) /&
    50d0,49.955d0,49.821d0,49.601d0,49.303d0,48.934d0,48.504d0,48.022d0,47.498d0,46.942d0,&
    46.361d0,45.764d0,45.155d0,44.541d0,43.924d0,43.309d0,42.696d0,42.088d0,41.486d0,40.891d0,&
    40.302d0,39.145d0,38.016d0,37.462d0,36.915d0,35.841d0,34.794d0,33.775d0,32.786d0,32.303d0,&
    31.828d0,30.902d0,30.011d0,29.154d0,28.334d0,27.938d0,27.551d0,26.805d0,26.096d0,24.482d0,&
    23.081d0,21.868d0,20.815d0,19.073d0,17.646d0,16.384d0,15.201d0,14.062d0,12.962d0,11.913d0,&
    10.933d0,10.034d0,9.227d0,8.516d0,7.897d0,7.367d0,5.702d0,4.853d0,4.192d0,3.565d0,&
    2.509d0,1.835d0,1.3552d0,1.217d0,0.9407d0,0.56262d0,0.30921d0,0.021932d0,0.0046388d0,0.0022047d0/
    DATA (ScatteringFactors(51,I),I=1,NQ) /&
    51d0,50.955d0,50.819d0,50.596d0,50.293d0,49.915d0,49.474d0,48.977d0,48.434d0,47.856d0,&
    47.25d0,46.625d0,45.988d0,45.344d0,44.699d0,44.056d0,43.419d0,42.789d0,42.168d0,41.556d0,&
    40.955d0,39.783d0,38.652d0,38.1d0,37.556d0,36.495d0,35.465d0,34.464d0,33.491d0,33.016d0,&
    32.547d0,31.631d0,30.745d0,29.888d0,29.063d0,28.663d0,28.27d0,27.511d0,26.784d0,25.113d0,&
    23.646d0,22.366d0,21.253d0,19.424d0,17.958d0,16.696d0,15.537d0,14.429d0,13.355d0,12.321d0,&
    11.341d0,10.431d0,9.602d0,8.861d0,8.208d0,7.642d0,5.836d0,4.945d0,4.295d0,3.678d0,&
    2.615d0,1.909d0,1.3922d0,1.2478d0,0.9591d0,0.58573d0,0.32587d0,0.023805d0,0.0050817d0,0.002424d0/
    DATA (ScatteringFactors(52,I),I=1,NQ) /&
    52d0,51.954d0,51.818d0,51.594d0,51.288d0,50.906d0,50.458d0,49.951d0,49.395d0,48.8d0,&
    48.174d0,47.526d0,46.863d0,46.193d0,45.519d0,44.848d0,44.182d0,43.526d0,42.879d0,42.245d0,&
    41.623d0,40.419d0,39.267d0,38.709d0,38.163d0,37.102d0,36.079d0,35.09d0,34.131d0,33.663d0,&
    33.202d0,32.299d0,31.424d0,30.575d0,29.753d0,29.352d0,28.959d0,28.194d0,27.458d0,25.748d0,&
    24.226d0,22.885d0,21.711d0,19.783d0,18.262d0,16.986d0,15.841d0,14.759d0,13.712d0,12.698d0,&
    11.726d0,10.811d0,9.966d0,9.201d0,8.518d0,7.921d0,5.98d0,5.04d0,4.39d0,3.78d0,&
    2.722d0,1.99d0,1.4309d0,1.2799d0,0.9179d0,0.60884d0,0.34283d0,0.025795d0,0.0055581d0,0.0026611d0/
    DATA (ScatteringFactors(53,I),I=1,NQ) /&
    53d0,52.955d0,52.82d0,52.597d0,52.292d0,51.911d0,51.46d0,50.95d0,50.387d0,49.781d0,&
    49.142d0,48.476d0,47.793d0,47.099d0,46.4d0,45.702d0,45.008d0,44.323d0,43.648d0,42.987d0,&
    42.34d0,41.091d0,39.904d0,39.333d0,38.776d0,37.702d0,36.675d0,35.69d0,34.741d0,34.279d0,&
    33.824d0,32.936d0,32.075d0,31.238d0,30.427d0,30.03d0,29.64d0,28.877d0,28.141d0,26.412d0,&
    24.851d0,23.459d0,22.228d0,20.193d0,18.599d0,17.293d0,16.15d0,15.09d0,14.072d0,13.082d0,&
    12.125d0,11.214d0,10.36d0,9.576d0,8.868d0,8.239d0,6.142d0,5.132d0,4.478d0,3.891d0,&
    2.828d0,2.067d0,1.4121d0,1.3138d0,0.9973d0,0.63193d0,0.36008d0,0.027907d0,0.0060699d0,0.002917d0/
    DATA (ScatteringFactors(54,I),I=1,NQ) /&
    54d0,53.956d0,53.821d0,53.601d0,53.297d0,52.917d0,52.467d0,51.954d0,51.388d0,50.775d0,&
    50.125d0,49.447d0,48.747d0,48.033d0,47.311d0,46.588d0,45.868d0,45.155d0,44.453d0,43.763d0,&
    43.088d0,41.788d0,40.557d0,39.967d0,39.393d0,38.294d0,37.251d0,36.259d0,35.31d0,34.85d0,&
    34.399d0,33.52d0,32.671d0,31.847d0,31.047d0,30.656d0,30.271d0,29.517d0,28.785d0,27.054d0,&
    25.47d0,24.038d0,22.758d0,20.618d0,18.943d0,17.591d0,16.438d0,15.39d0,14.396d0,13.432d0,&
    12.494d0,11.592d0,10.736d0,9.94d0,9.212d0,8.556d0,6.315d0,5.229d0,4.566d0,3.991d0,&
    2.935d0,2.15d0,1.5151d0,1.3495d0,1.0163d0,0.65497d0,0.37759d0,0.030144d0,0.0066192d0,0.003193d0/
    DATA (ScatteringFactors(55,I),I=1,NQ) /&
    55d0,54.932d0,54.732d0,54.417d0,54.008d0,53.527d0,52.996d0,52.43d0,51.839d0,51.229d0,&
    50.603d0,49.963d0,49.309d0,48.645d0,47.971d0,47.291d0,46.606d0,45.921d0,45.237d0,44.559d0,&
    43.888d0,42.578d0,41.32d0,40.713d0,40.121d0,38.982d0,37.904d0,36.881d0,35.909d0,35.44d0,&
    34.981d0,34.094d0,33.241d0,32.419d0,31.624d0,31.236d0,30.854d0,30.107d0,29.382d0,27.661d0,&
    26.072d0,24.619d0,23.303d0,21.072d0,19.31d0,17.9d0,16.722d0,15.676d0,14.7d0,13.759d0,&
    12.845d0,11.956d0,11.104d0,10.303d0,9.558d0,8.881d0,6.502d0,5.332d0,4.651d0,4.087d0,&
    3.041d0,2.237d0,1.5626d0,1.3864d0,1.0339d0,0.67795d0,0.39535d0,0.032511d0,0.0072081d0,0.0034903d0/
    DATA (ScatteringFactors(56,I),I=1,NQ) /&
    56d0,55.925d0,55.703d0,55.35d0,54.888d0,54.345d0,53.743d0,53.106d0,52.45d0,51.786d0,&
    51.122d0,50.46d0,49.802d0,49.146d0,48.492d0,47.839d0,47.186d0,46.533d0,45.882d0,45.232d0,&
    44.586d0,43.309d0,42.064d0,41.456d0,40.859d0,39.702d0,38.598d0,37.546d0,36.545d0,36.063d0,&
    35.593d0,34.685d0,33.818d0,32.986d0,32.187d0,31.798d0,31.415d0,30.67d0,29.948d0,28.238d0,&
    26.652d0,25.189d0,23.851d0,21.547d0,19.701d0,18.224d0,17.008d0,15.953d0,14.988d0,14.067d0,&
    13.175d0,12.305d0,11.461d0,10.661d0,9.907d0,9.213d0,6.704d0,5.44d0,4.735d0,4.178d0,&
    3.146d0,2.325d0,1.6106d0,1.4241d0,1.0512d0,0.70083d0,0.41335d0,0.035014d0,0.0078387d0,0.0038103d0/
    DATA (ScatteringFactors(57,I),I=1,NQ) /&
    57d0,56.926d0,56.708d0,56.36d0,55.9d0,55.351d0,54.736d0,54.076d0,53.388d0,52.687d0,&
    51.982d0,51.278d0,50.58d0,49.888d0,49.202d0,48.523d0,47.849d0,47.182d0,46.519d0,45.862d0,&
    45.212d0,43.932d0,42.686d0,42.078d0,41.481d0,40.321d0,39.212d0,38.153d0,37.145d0,36.659d0,&
    36.185d0,35.27d0,34.397d0,33.562d0,32.76d0,32.37d0,31.988d0,31.243d0,30.523d0,28.817d0,&
    27.231d0,25.759d0,24.401d0,22.031d0,20.106d0,18.561d0,17.3d0,16.227d0,15.265d0,14.362d0,&
    13.489d0,12.636d0,11.807d0,11.009d0,10.253d0,9.55d0,6.917d0,5.55d0,4.82d0,4.27d0,&
    3.24d0,2.41d0,1.6608d0,1.4633d0,1.0683d0,0.7236d0,0.43155d0,0.031056d0,0.0085133d0,0.0041543d0/
    DATA (ScatteringFactors(58,I),I=1,NQ) /&
    58d0,57.928d0,57.715d0,57.375d0,56.924d0,56.385d0,55.779d0,55.127d0,54.446d0,53.75d0,&
    53.047d0,52.345d0,51.646d0,50.952d0,50.263d0,49.579d0,48.901d0,48.227d0,47.557d0,46.892d0,&
    46.233d0,44.933d0,43.663d0,43.042d0,42.432d0,41.244d0,40.104d0,39.014d0,37.975d0,37.474d0,&
    36.985d0,36.04d0,35.139d0,34.277d0,33.451d0,33.051d0,32.658d0,31.893d0,31.154d0,29.409d0,&
    27.791d0,26.289d0,24.901d0,22.469d0,20.481d0,18.881d0,17.583d0,16.491d0,15.526d0,14.633d0,&
    13.776d0,12.939d0,12.123d0,11.333d0,10.576d0,9.868d0,7.117d0,5.663d0,4.91d0,4.36d0,&
    3.34d0,2.49d0,1.7128d0,1.5038d0,1.0858d0,0.74625d0,0.44995d0,0.040442d0,0.0092343d0,0.0045238d0/
    DATA (ScatteringFactors(59,I),I=1,NQ) /&
    59d0,58.929d0,58.722d0,58.392d0,57.956d0,57.439d0,56.861d0,56.242d0,55.599d0,54.943d0,&
    54.281d0,53.617d0,52.952d0,52.288d0,51.623d0,50.957d0,50.289d0,49.62d0,48.95d0,48.28d0,&
    47.61d0,46.278d0,44.967d0,44.323d0,43.688d0,42.448d0,41.256d0,40.113d0,39.022d0,38.496d0,&
    37.982d0,36.989d0,36.042d0,35.137d0,34.269d0,33.849d0,33.437d0,32.635d0,31.862d0,30.04d0,&
    28.358d0,26.803d0,25.37d0,22.867d0,20.824d0,19.182d0,17.854d0,16.745d0,15.776d0,14.888d0,&
    14.042d0,13.218d0,12.414d0,11.631d0,10.878d0,10.166d0,7.333d0,5.8d0,5d0,4.445d0,&
    3.435d0,2.58d0,1.7656d0,1.545d0,1.1037d0,0.76875d0,0.46852d0,0.043377d0,0.010004d0,0.0049204d0/
    DATA (ScatteringFactors(60,I),I=1,NQ) /&
    60d0,59.931d0,59.728d0,59.404d0,58.977d0,58.468d0,57.899d0,57.288d0,56.651d0,56d0,&
    55.342d0,54.68d0,54.017d0,53.354d0,52.689d0,52.022d0,51.353d0,50.682d0,50.009d0,49.334d0,&
    48.66d0,47.317d0,45.989d0,45.336d0,44.69d0,43.428d0,42.21d0,41.04d0,39.92d0,39.379d0,&
    38.851d0,37.83d0,36.854d0,35.922d0,35.029d0,34.596d0,34.171d0,33.347d0,32.553d0,30.683d0,&
    28.96d0,27.367d0,25.899d0,23.325d0,21.214d0,19.513d0,18.139d0,17.003d0,16.024d0,15.138d0,&
    14.303d0,13.493d0,12.704d0,11.932d0,11.185d0,10.473d0,7.567d0,5.93d0,5.09d0,4.525d0,&
    3.53d0,2.67d0,1.8203d0,1.5874d0,1.1217d0,0.79109d0,0.48726d0,0.046466d0,0.010825d0,0.0053457d0/
    DATA (ScatteringFactors(61,I),I=1,NQ) /&
    61d0,60.932d0,60.734d0,60.417d0,59.998d0,59.497d0,58.936d0,58.333d0,57.703d0,57.057d0,&
    56.403d0,55.744d0,55.084d0,54.422d0,53.758d0,53.091d0,52.422d0,51.749d0,51.074d0,50.398d0,&
    49.72d0,48.367d0,47.026d0,46.364d0,45.71d0,44.427d0,43.186d0,41.991d0,40.844d0,40.289d0,&
    39.747d0,38.697d0,37.694d0,36.735d0,35.815d0,35.37d0,34.933d0,34.085d0,33.269d0,31.349d0,&
    29.581d0,27.948d0,26.442d0,23.796d0,21.616d0,19.853d0,18.43d0,17.262d0,16.266d0,15.378d0,&
    14.551d0,13.755d0,12.98d0,12.22d0,11.481d0,10.773d0,7.817d0,6.088d0,5.18d0,4.6d0,&
    3.625d0,2.77d0,1.8776d0,1.6318d0,1.1402d0,0.81325d0,0.50614d0,0.049713d0,0.0117d0,0.0058012d0/
    DATA (ScatteringFactors(62,I),I=1,NQ) /&
    62d0,61.934d0,61.74d0,61.428d0,61.017d0,60.525d0,59.972d0,59.377d0,58.753d0,58.113d0,&
    57.463d0,56.809d0,56.151d0,55.491d0,54.828d0,54.163d0,53.493d0,52.821d0,52.145d0,51.467d0,&
    50.786d0,49.426d0,48.074d0,47.406d0,46.743d0,45.443d0,44.18d0,42.961d0,41.789d0,41.221d0,&
    40.666d0,39.589d0,38.559d0,37.573d0,36.627d0,36.169d0,35.72d0,34.848d0,34.008d0,32.036d0,&
    30.222d0,28.547d0,27.002d0,24.281d0,22.03d0,20.202d0,18.728d0,17.523d0,16.507d0,15.613d0,&
    14.79d0,14.005d0,13.243d0,12.497d0,11.767d0,11.064d0,8.083d0,6.25d0,5.28d0,4.675d0,&
    3.72d0,2.865d0,1.9374d0,1.6783d0,1.1601d0,0.83523d0,0.52514d0,0.053123d0,0.012632d0,0.0062889d0/
    DATA (ScatteringFactors(63,I),I=1,NQ) /&
    63d0,62.936d0,62.746d0,62.441d0,62.036d0,61.552d0,61.007d0,60.419d0,59.801d0,59.166d0,&
    58.521d0,57.869d0,57.214d0,56.555d0,55.893d0,55.228d0,54.559d0,53.886d0,53.21d0,52.53d0,&
    51.847d0,50.48d0,49.119d0,48.444d0,47.775d0,46.458d0,45.176d0,43.935d0,42.74d0,42.16d0,&
    41.591d0,40.489d0,39.433d0,38.421d0,37.451d0,36.98d0,36.519d0,35.623d0,34.761d0,32.737d0,&
    30.877d0,29.161d0,27.576d0,24.781d0,22.459d0,20.565d0,19.035d0,17.789d0,16.747d0,15.841d0,&
    15.02d0,14.245d0,13.494d0,12.763d0,12.044d0,11.345d0,8.348d0,6.435d0,5.378d0,4.75d0,&
    3.812d0,2.965d0,1.9984d0,1.726d0,1.1811d0,0.85701d0,0.54426d0,0.056101d0,0.013624d0,0.0068105d0/
    DATA (ScatteringFactors(64,I),I=1,NQ) /&
    64d0,63.936d0,63.749d0,63.447d0,63.044d0,62.557d0,62.004d0,61.4d0,60.762d0,60.102d0,&
    59.427d0,58.746d0,58.061d0,57.375d0,56.69d0,56.005d0,55.321d0,54.637d0,53.953d0,53.27d0,&
    52.588d0,51.227d0,49.878d0,49.209d0,48.546d0,47.24d0,45.965d0,44.729d0,43.533d0,42.951d0,&
    42.38d0,41.272d0,40.207d0,39.184d0,38.203d0,37.726d0,37.259d0,36.352d0,35.479d0,33.428d0,&
    31.543d0,29.802d0,28.192d0,25.335d0,22.94d0,20.97d0,19.372d0,18.072d0,16.995d0,16.072d0,&
    15.247d0,14.477d0,13.741d0,13.022d0,12.317d0,11.631d0,8.683d0,6.588d0,5.49d0,4.83d0,&
    3.905d0,3.07d0,2.0603d0,1.7147d0,1.2034d0,0.87858d0,0.56345d0,0.060451d0,0.014678d0,0.007368d0/
    DATA (ScatteringFactors(65,I),I=1,NQ) /&
    65d0,64.938d0,64.755d0,64.461d0,64.071d0,63.603d0,63.073d0,62.499d0,61.894d0,61.27d0,&
    60.634d0,59.989d0,59.34d0,58.686d0,58.029d0,57.366d0,56.699d0,56.028d0,55.351d0,54.67d0,&
    53.985d0,52.61d0,51.234d0,50.549d0,49.868d0,48.523d0,47.208d0,45.929d0,44.69d0,44.087d0,&
    43.496d0,42.346d0,41.241d0,40.179d0,39.16d0,38.665d0,38.18d0,37.237d0,36.329d0,34.199d0,&
    32.243d0,30.438d0,28.772d0,25.822d0,23.353d0,21.323d0,19.675d0,18.338d0,17.234d0,16.296d0,&
    15.465d0,14.697d0,13.968d0,13.259d0,12.564d0,11.886d0,8.983d0,6.775d0,5.61d0,4.915d0,&
    3.99d0,3.17d0,2.123d0,1.8243d0,1.2268d0,0.89993d0,0.58275d0,0.064379d0,0.015799d0,0.0079635d0/
    DATA (ScatteringFactors(66,I),I=1,NQ) /&
    66d0,65.939d0,65.76d0,65.471d0,65.088d0,64.627d0,64.105d0,63.538d0,62.94d0,62.321d0,&
    61.689d0,61.049d0,60.403d0,59.752d0,59.097d0,58.437d0,57.771d0,57.101d0,56.425d0,55.744d0,&
    55.059d0,53.681d0,52.3d0,51.611d0,50.926d0,49.57d0,48.24d0,46.944d0,45.686d0,45.073d0,&
    44.471d0,43.299d0,42.171d0,41.086d0,40.042d0,39.536d0,39.039d0,38.073d0,37.143d0,34.958d0,&
    32.953d0,31.103d0,29.394d0,26.366d0,23.821d0,21.721d0,20.011d0,18.623d0,17.483d0,16.522d0,&
    15.68d0,14.913d0,14.19d0,13.491d0,12.808d0,12.141d0,9.267d0,6.963d0,5.72d0,5d0,&
    4.075d0,3.27d0,2.1867d0,1.8746d0,1.2503d0,0.92106d0,0.60211d0,0.068488d0,0.016986d0,0.008599d0/
    DATA (ScatteringFactors(67,I),I=1,NQ) /&
    67d0,66.94d0,66.763d0,66.476d0,66.093d0,65.627d0,65.096d0,64.513d0,63.895d0,63.251d0,&
    62.591d0,61.921d0,61.247d0,60.569d0,59.891d0,59.212d0,58.532d0,57.851d0,57.169d0,56.486d0,&
    55.803d0,54.435d0,53.07d0,52.39d0,51.714d0,50.375d0,49.059d0,47.772d0,46.52d0,45.908d0,&
    45.305d0,44.131d0,42.996d0,41.903d0,40.849d0,40.337d0,39.834d0,38.856d0,37.914d0,35.699d0,&
    33.664d0,31.786d0,30.049d0,26.958d0,24.343d0,22.167d0,20.385d0,18.934d0,17.746d0,16.753d0,&
    15.895d0,15.123d0,14.406d0,13.718d0,13.047d0,12.392d0,9.533d0,7.163d0,5.85d0,5.09d0,&
    4.155d0,3.355d0,2.252d0,1.9259d0,1.2738d0,0.94195d0,0.62152d0,0.072785d0,0.018247d0,0.0092769d0/
    DATA (ScatteringFactors(68,I),I=1,NQ) /&
    68d0,67.941d0,67.769d0,67.491d0,67.12d0,66.673d0,66.166d0,65.613d0,65.028d0,64.42d0,&
    63.798d0,63.167d0,62.528d0,61.884d0,61.234d0,60.578d0,59.917d0,59.249d0,58.576d0,57.897d0,&
    57.213d0,55.833d0,54.445d0,53.75d0,53.058d0,51.683d0,50.329d0,49.004d0,47.712d0,47.081d0,&
    46.459d0,45.246d0,44.075d0,42.945d0,41.857d0,41.327d0,40.808d0,39.797d0,38.822d0,36.531d0,&
    34.425d0,32.483d0,30.688d0,27.497d0,24.8d0,22.556d0,20.718d0,19.221d0,17.998d0,16.98d0,&
    16.107d0,15.329d0,14.612d0,13.929d0,13.267d0,12.621d0,9.783d0,7.375d0,5.98d0,5.18d0,&
    4.235d0,3.44d0,2.3168d0,1.9772d0,1.2979d0,0.9626d0,0.64096d0,0.077273d0,0.019583d0,0.0099994d0/
    DATA (ScatteringFactors(69,I),I=1,NQ) /&
    69d0,68.943d0,68.773d0,68.5d0,68.136d0,67.696d0,67.195d0,66.649d0,66.07d0,65.468d0,&
    64.852d0,64.224d0,63.589d0,62.948d0,62.301d0,61.648d0,60.989d0,60.324d0,59.653d0,58.975d0,&
    58.292d0,56.912d0,55.521d0,54.825d0,54.13d0,52.748d0,51.384d0,50.046d0,48.739d0,48.099d0,&
    47.469d0,46.237d0,45.046d0,43.896d0,42.786d0,42.246d0,41.715d0,40.682d0,39.686d0,37.342d0,&
    35.187d0,33.198d0,31.359d0,28.086d0,25.311d0,22.995d0,21.089d0,19.535d0,18.266d0,17.215d0,&
    16.321d0,15.533d0,14.815d0,14.137d0,13.483d0,12.847d0,10.033d0,7.588d0,6.11d0,5.28d0,&
    4.31d0,3.52d0,2.3808d0,2.028d0,1.3223d0,0.983d0,0.66043d0,0.081957d0,0.020998d0,0.010769d0/
    DATA (ScatteringFactors(70,I),I=1,NQ) /&
    70d0,69.944d0,69.777d0,69.509d0,69.151d0,68.717d0,68.223d0,67.684d0,67.112d0,66.516d0,&
    65.904d0,65.281d0,64.65d0,64.012d0,63.368d0,62.718d0,62.062d0,61.399d0,60.729d0,60.053d0,&
    59.371d0,57.992d0,56.601d0,55.903d0,55.206d0,53.817d0,52.444d0,51.095d0,49.774d0,49.127d0,&
    48.488d0,47.239d0,46.029d0,44.859d0,43.728d0,43.178d0,42.637d0,41.583d0,40.565d0,38.169d0,&
    35.964d0,33.929d0,32.045d0,28.69d0,25.837d0,23.447d0,21.474d0,19.86d0,18.542d0,17.454d0,&
    16.536d0,15.735d0,15.013d0,14.338d0,13.691d0,13.064d0,10.267d0,7.788d0,6.25d0,5.38d0,&
    4.38d0,3.6d0,2.4457d0,2.0795d0,1.3472d0,1.0032d0,0.67992d0,0.086843d0,0.022496d0,0.011588d0/
    DATA (ScatteringFactors(71,I),I=1,NQ) /&
    71d0,70.944d0,70.778d0,70.509d0,70.148d0,69.707d0,69.202d0,68.646d0,68.051d0,67.429d0,&
    66.789d0,66.137d0,65.477d0,64.813d0,64.146d0,63.478d0,62.807d0,62.134d0,61.46d0,60.783d0,&
    60.103d0,58.739d0,57.369d0,56.683d0,55.998d0,54.634d0,53.282d0,51.95d0,50.642d0,49.998d0,&
    49.363d0,48.117d0,46.906d0,45.731d0,44.593d0,44.038d0,43.492d0,42.427d0,41.398d0,38.97d0,&
    36.733d0,34.666d0,32.752d0,29.334d0,26.413d0,23.95d0,21.902d0,20.219d0,18.842d0,17.709d0,&
    16.759d0,15.939d0,15.208d0,14.534d0,13.894d0,13.277d0,10.5d0,8.013d0,6.4d0,5.49d0,&
    4.45d0,3.68d0,2.5134d0,2.1334d0,1.3734d0,1.023d0,0.69941d0,0.091934d0,0.024081d0,0.01246d0/
    DATA (ScatteringFactors(72,I),I=1,NQ) /&
    72d0,71.945d0,71.783d0,71.518d0,71.161d0,70.723d0,70.217d0,69.656d0,69.052d0,68.416d0,&
    67.757d0,67.083d0,66.4d0,65.711d0,65.019d0,64.326d0,63.634d0,62.942d0,62.251d0,61.56d0,&
    60.87d0,59.492d0,58.119d0,57.434d0,56.752d0,55.396d0,54.054d0,52.733d0,51.435d0,50.796d0,&
    50.164d0,48.924d0,47.717d0,46.543d0,45.405d0,44.849d0,44.301d0,43.232d0,42.197d0,39.752d0,&
    37.494d0,35.404d0,33.465d0,29.992d0,27.008d0,24.473d0,22.352d0,20.598d0,19.159d0,17.975d0,&
    16.988d0,16.145d0,15.403d0,14.727d0,14.091d0,13.481d0,10.733d0,8.238d0,6.56d0,5.6d0,&
    4.52d0,3.755d0,2.5806d0,2.1874d0,1.401d0,1.0427d0,0.71889d0,0.097236d0,0.025756d0,0.013381d0/
    DATA (ScatteringFactors(73,I),I=1,NQ) /&
    73d0,72.946d0,72.788d0,72.529d0,72.177d0,71.745d0,71.242d0,70.68d0,70.072d0,69.428d0,&
    68.758d0,68.069d0,67.367d0,66.658d0,65.944d0,65.229d0,64.515d0,63.802d0,63.09d0,62.382d0,&
    61.675d0,60.271d0,58.88d0,58.189d0,57.502d0,56.141d0,54.799d0,53.479d0,52.185d0,51.548d0,&
    50.918d0,49.683d0,48.479d0,47.308d0,46.171d0,45.615d0,45.068d0,43.998d0,42.962d0,40.508d0,&
    38.238d0,36.132d0,34.175d0,30.658d0,27.618d0,25.016d0,22.823d0,20.998d0,19.494d0,18.256d0,&
    17.228d0,16.356d0,15.598d0,14.916d0,14.282d0,13.679d0,10.95d0,8.48d0,6.74d0,5.71d0,&
    4.585d0,3.825d0,2.6457d0,2.2404d0,1.4298d0,1.0621d0,0.73834d0,0.10275d0,0.027526d0,0.014372d0/
    DATA (ScatteringFactors(74,I),I=1,NQ) /&
    74d0,73.948d0,73.793d0,73.539d0,73.194d0,72.767d0,72.269d0,71.711d0,71.103d0,70.455d0,&
    69.778d0,69.078d0,68.363d0,67.637d0,66.906d0,66.172d0,65.437d0,64.703d0,63.972d0,63.243d0,&
    62.519d0,61.082d0,59.663d0,58.961d0,58.265d0,56.888d0,55.536d0,54.21d0,52.912d0,52.274d0,&
    51.644d0,50.408d0,49.205d0,48.036d0,46.9d0,46.344d0,45.797d0,44.728d0,43.691d0,41.236d0,&
    38.96d0,36.846d0,34.878d0,31.327d0,28.238d0,25.576d0,23.313d0,21.418d0,19.847d0,18.552d0,&
    17.478d0,16.575d0,15.796d0,15.104d0,14.469d0,13.871d0,11.167d0,8.706d0,6.9d0,5.84d0,&
    4.65d0,3.9d0,2.71d0,2.2933d0,1.4598d0,1.0812d0,0.75171d0,0.10849d0,0.029394d0,0.015418d0/
    DATA (ScatteringFactors(75,I),I=1,NQ) /&
    75d0,74.949d0,74.797d0,74.548d0,74.209d0,73.788d0,73.295d0,72.74d0,72.132d0,71.482d0,&
    70.799d0,70.091d0,69.365d0,68.625d0,67.878d0,67.126d0,66.372d0,65.619d0,64.868d0,64.121d0,&
    63.378d0,61.906d0,60.457d0,59.742d0,59.034d0,57.637d0,56.27d0,54.932d0,53.627d0,52.986d0,&
    52.354d0,51.114d0,49.91d0,48.739d0,47.603d0,47.048d0,46.501d0,45.432d0,44.396d0,41.94d0,&
    39.662d0,37.544d0,35.569d0,31.993d0,28.865d0,26.148d0,23.821d0,21.856d0,20.219d0,18.864d0,&
    17.742d0,16.801d0,15.998d0,15.293d0,14.653d0,14.057d0,11.383d0,8.938d0,7.08d0,5.96d0,&
    4.715d0,3.97d0,2.775d0,2.3471d0,1.4912d0,1.1001d0,0.77716d0,0.11445d0,0.031361d0,0.016529d0/
    DATA (ScatteringFactors(76,I),I=1,NQ) /&
    76d0,75.95d0,75.801d0,75.538d0,75.225d0,74.81d0,74.323d0,73.772d0,73.167d0,72.518d0,&
    71.832d0,71.119d0,70.384d0,69.634d0,68.874d0,68.107d0,67.337d0,66.566d0,65.797d0,65.031d0,&
    64.269d0,62.761d0,61.278d0,60.548d0,59.825d0,58.403d0,57.013d0,55.658d0,54.339d0,53.692d0,&
    53.055d0,51.807d0,50.596d0,49.422d0,48.283d0,47.726d0,47.179d0,46.109d0,45.072d0,42.617d0,&
    40.34d0,38.222d0,36.244d0,32.654d0,29.495d0,26.732d0,24.345d0,22.314d0,20.61d0,19.194d0,&
    18.019d0,17.038d0,16.206d0,15.483d0,14.835d0,14.239d0,11.583d0,9.163d0,7.27d0,6.08d0,&
    4.788d0,4.035d0,2.8399d0,2.4014d0,1.5244d0,1.1186d0,0.7965d0,0.12064d0,0.033446d0,0.017107d0/
    DATA (ScatteringFactors(77,I),I=1,NQ) /&
    77d0,76.951d0,76.806d0,76.567d0,76.24d0,75.832d0,75.352d0,74.806d0,74.206d0,73.558d0,&
    72.872d0,72.156d0,71.416d0,70.658d0,69.887d0,69.108d0,68.324d0,67.538d0,66.752d0,65.969d0,&
    65.189d0,63.645d0,62.127d0,61.38d0,60.641d0,59.189d0,57.773d0,56.395d0,55.056d0,54.401d0,&
    53.756d0,52.496d0,51.274d0,50.091d0,48.946d0,48.387d0,47.837d0,46.765d0,45.726d0,43.269d0,&
    40.994d0,38.878d0,36.901d0,33.305d0,30.125d0,27.323d0,24.882d0,22.789d0,21.019d0,19.541d0,&
    18.312d0,17.287d0,16.422d0,15.678d0,15.018d0,14.418d0,11.783d0,9.4d0,7.46d0,6.21d0,&
    4.86d0,4.105d0,2.903d0,2.4551d0,1.5594d0,1.137d0,0.81579d0,0.12707d0,0.035639d0,0.018957d0/
    DATA (ScatteringFactors(78,I),I=1,NQ) /&
    78d0,77.955d0,77.82d0,77.599d0,77.295d0,76.914d0,76.462d0,75.946d0,75.373d0,74.751d0,&
    74.086d0,73.386d0,72.656d0,71.902d0,71.13d0,70.343d0,69.546d0,68.742d0,67.934d0,67.125d0,&
    66.317d0,64.709d0,63.125d0,62.344d0,61.571d0,60.056d0,58.582d0,57.152d0,55.769d0,55.094d0,&
    54.432d0,53.141d0,51.897d0,50.697d0,49.54d0,48.977d0,48.424d0,47.347d0,46.308d0,43.86d0,&
    41.601d0,39.502d0,37.539d0,33.958d0,30.766d0,27.93d0,25.437d0,23.281d0,21.445d0,19.902d0,&
    18.616d0,17.545d0,16.644d0,15.875d0,15.202d0,14.595d0,11.983d0,9.62d0,7.65d0,6.34d0,&
    4.935d0,4.175d0,2.9035d0,2.5077d0,1.5962d0,1.155d0,0.83501d0,0.13374d0,0.037949d0,0.020282d0/
    DATA (ScatteringFactors(79,I),I=1,NQ) /&
    79d0,78.957d0,78.826d0,78.609d0,78.311d0,77.936d0,77.491d0,76.981d0,76.414d0,75.797d0,&
    75.135d0,74.437d0,73.706d0,72.95d0,72.173d0,71.38d0,70.575d0,69.761d0,68.941d0,68.119d0,&
    67.296d0,65.657d0,64.039d0,63.241d0,62.452d0,60.902d0,59.395d0,57.935d0,56.523d0,55.835d0,&
    55.16d0,53.846d0,52.581d0,51.363d0,50.191d0,49.622d0,49.063d0,47.976d0,46.929d0,44.469d0,&
    42.207d0,40.11d0,38.153d0,34.581d0,31.387d0,28.53d0,25.998d0,23.789d0,21.892d0,20.287d0,&
    18.943d0,17.821d0,16.88d0,16.081d0,15.388d0,14.77d0,12.168d0,9.826d0,7.878d0,6.489d0,&
    5.01d0,4.244d0,3.0236d0,2.5607d0,1.635d0,1.1728d0,0.85415d0,0.14064d0,0.040382d0,0.021686d0/
    DATA (ScatteringFactors(80,I),I=1,NQ) /&
    80d0,79.556d0,79.819d0,79.595d0,79.286d0,78.899d0,78.439d0,77.913d0,77.33d0,76.696d0,&
    76.018d0,75.303d0,74.559d0,73.79d0,73.001d0,72.198d0,71.385d0,70.564d0,69.74d0,68.914d0,&
    68.088d0,66.447d0,64.828d0,64.029d0,63.239d0,61.687d0,60.177d0,58.711d0,57.292d0,56.6d0,&
    55.92d0,54.595d0,53.318d0,52.088d0,50.902d0,50.326d0,49.761d0,48.661d0,47.601d0,45.113d0,&
    42.829d0,40.718d0,38.753d0,35.176d0,31.98d0,29.112d0,26.554d0,24.303d0,22.354d0,20.692d0,&
    19.29d0,18.116d0,17.131d0,16.298d0,15.581d0,14.949d0,12.36d0,10.049d0,8.081d0,6.644d0,&
    5.09d0,4.31d0,3.0833d0,2.6137d0,1.6746d0,1.1903d0,0.87322d0,0.1478d0,0.042943d0,0.023114d0/
    DATA (ScatteringFactors(81,I),I=1,NQ) /&
    81d0,80.95d0,80.799d0,80.553d0,80.217d0,79.798d0,79.305d0,78.748d0,78.134d0,77.473d0,&
    76.773d0,76.042d0,75.284d0,74.507d0,73.715d0,72.912d0,72.101d0,71.285d0,70.467d0,69.648d0,&
    68.83d0,67.205d0,65.6d0,64.807d0,64.022d0,62.478d0,60.97d0,59.503d0,58.079d0,57.383d0,&
    56.698d0,55.362d0,54.072d0,52.826d0,51.625d0,51.041d0,50.467d0,49.352d0,48.276d0,45.753d0,&
    43.442d0,41.313d0,39.337d0,35.755d0,32.561d0,29.687d0,27.109d0,24.824d0,22.827d0,21.11d0,&
    19.652d0,18.424d0,17.394d0,16.524d0,15.78d0,15.131d0,12.53d0,10.27d0,8.29d0,6.8d0,&
    5.175d0,4.374d0,3.1425d0,2.6664d0,1.7142d0,1.2076d0,0.89221d0,0.15521d0,0.045637d0,0.024749d0/
    DATA (ScatteringFactors(82,I),I=1,NQ) /&
    82d0,81.949d0,81.792d0,81.536d0,81.186d0,80.75d0,80.237d0,79.656d0,79.018d0,78.332d0,&
    77.607d0,76.851d0,76.071d0,75.274d0,74.464d0,73.645d0,72.822d0,71.997d0,71.172d0,70.349d0,&
    69.53d0,67.907d0,66.31d0,65.523d0,64.743d0,63.21d0,61.712d0,60.253d0,58.833d0,58.138d0,&
    57.453d0,56.116d0,54.82d0,53.567d0,52.356d0,51.766d0,51.187d0,50.058d0,48.969d0,46.411d0,&
    44.069d0,41.914d0,39.921d0,36.322d0,33.127d0,30.252d0,27.662d0,25.35d0,23.313d0,21.546d0,&
    20.034d0,18.754d0,17.674d0,16.764d0,15.989d0,15.317d0,12.724d0,10.482d0,8.495d0,6.973d0,&
    5.26d0,4.441d0,3.2009d0,2.7184d0,1.7534d0,1.2246d0,0.9111d0,0.16288d0,0.04847d0,0.026415d0/
    DATA (ScatteringFactors(83,I),I=1,NQ) /&
    83d0,82.947d0,82.784d0,82.518d0,82.154d0,81.7d0,81.167d0,80.563d0,79.901d0,79.189d0,&
    78.438d0,77.657d0,76.852d0,76.032d0,75.202d0,74.365d0,73.527d0,72.689d0,71.855d0,71.026d0,&
    70.203d0,68.578d0,66.987d0,66.204d0,65.43d0,63.909d0,62.425d0,60.977d0,59.566d0,58.875d0,&
    58.193d0,56.859d0,55.563d0,54.306d0,53.089d0,52.495d0,51.91d0,50.771d0,49.669d0,47.077d0,&
    44.7d0,42.517d0,40.501d0,36.879d0,33.68d0,30.805d0,28.208d0,25.875d0,23.804d0,21.992d0,&
    20.429d0,19.097d0,17.969d0,17.017d0,16.207d0,15.51d0,12.896d0,10.69d0,8.704d0,7.145d0,&
    5.351d0,4.505d0,3.2579d0,2.7695d0,1.7927d0,1.2414d0,0.92989d0,0.17081d0,0.051448d0,0.028179d0/
    DATA (ScatteringFactors(84,I),I=1,NQ) /&
    84d0,83.944d0,83.778d0,83.506d0,83.134d0,82.669d0,82.121d0,81.501d0,80.819d0,80.086d0,&
    79.312d0,78.506d0,77.677d0,76.831d0,75.976d0,75.117d0,74.257d0,73.4d0,72.549d0,71.706d0,&
    70.871d0,69.232d0,67.634d0,66.852d0,66.08d0,64.567d0,63.093d0,61.658d0,60.26d0,59.575d0,&
    58.899d0,57.573d0,56.283d0,55.029d0,53.811d0,53.215d0,52.629d0,51.483d0,50.373d0,47.752d0,&
    45.343d0,43.127d0,41.085d0,37.43d0,34.22d0,31.344d0,28.744d0,26.397d0,24.298d0,22.446d0,&
    20.836d0,19.453d0,18.277d0,17.281d0,16.435d0,15.711d0,13.06d0,10.9d0,8.91d0,7.32d0,&
    5.44d0,4.567d0,3.3136d0,2.8196d0,1.8317d0,1.2579d0,0.94859d0,0.17901d0,0.054578d0,0.030044d0/
    DATA (ScatteringFactors(85,I),I=1,NQ) /&
    85d0,84.944d0,84.776d0,84.502d0,84.125d0,83.654d0,83.098d0,82.466d0,81.77d0,81.02d0,&
    80.226d0,79.398d0,78.545d0,77.674d0,76.794d0,75.908d0,75.023d0,74.143d0,73.269d0,72.405d0,&
    71.553d0,69.885d0,68.269d0,67.481d0,66.706d0,65.193d0,63.725d0,62.301d0,60.915d0,60.236d0,&
    59.566d0,58.253d0,56.974d0,55.728d0,54.515d0,53.921d0,53.335d0,52.189d0,51.075d0,48.435d0,&
    45.997d0,43.75d0,41.678d0,37.98d0,34.751d0,31.872d0,29.271d0,26.915d0,24.794d0,22.909d0,&
    21.256d0,19.826d0,18.602d0,17.562d0,16.677d0,15.922d0,13.23d0,11.09d0,9.12d0,7.5d0,&
    5.54d0,4.63d0,3.3681d0,2.8688d0,1.8702d0,1.2741d0,0.96717d0,0.18749d0,0.057865d0,0.032016d0/
    DATA (ScatteringFactors(86,I),I=1,NQ) /&
    86d0,85.945d0,85.777d0,85.502d0,85.123d0,84.649d0,84.087d0,83.448d0,82.742d0,81.979d0,&
    81.169d0,80.322d0,79.448d0,78.554d0,77.648d0,76.737d0,75.826d0,74.92d0,74.021d0,73.133d0,&
    72.258d0,70.552d0,68.907d0,68.109d0,67.325d0,65.802d0,64.332d0,62.912d0,61.535d0,60.862d0,&
    60.198d0,58.898d0,57.631d0,56.397d0,55.194d0,54.604d0,54.021d0,52.879d0,51.767d0,49.119d0,&
    46.659d0,44.384d0,42.281d0,38.533d0,35.277d0,32.389d0,29.787d0,27.426d0,25.291d0,23.379d0,&
    21.689d0,20.215d0,18.944d0,17.859d0,16.934d0,16.143d0,13.386d0,11.282d0,9.329d0,7.686d0,&
    5.65d0,4.702d0,3.4218d0,2.9173d0,1.9084d0,1.2901d0,0.98565d0,0.19623d0,0.061316d0,0.0341d0/
    DATA (ScatteringFactors(87,I),I=1,NQ) /&
    87d0,86.922d0,86.694d0,86.332d0,85.854d0,85.286d0,84.647d0,83.955d0,83.222d0,82.457d0,&
    81.666d0,80.852d0,80.018d0,79.167d0,78.303d0,77.43d0,76.55d0,75.667d0,74.785d0,73.907d0,&
    73.035d0,71.32d0,69.653d0,68.841d0,68.043d0,66.491d0,64.996d0,63.556d0,62.167d0,61.489d0,&
    60.823d0,59.52d0,58.256d0,57.026d0,55.829d0,55.242d0,54.663d0,53.527d0,52.42d0,49.777d0,&
    47.31d0,45.017d0,42.891d0,39.095d0,35.804d0,32.9d0,30.292d0,27.926d0,25.779d0,23.845d0,&
    22.123d0,20.608d0,19.295d0,18.165d0,17.199d0,16.377d0,13.55d0,11.46d0,9.53d0,7.878d0,&
    5.755d0,4.768d0,3.4742d0,2.9655d0,1.948d0,1.3059d0,1.004d0,0.20527d0,0.064938d0,0.036302d0/
    DATA (ScatteringFactors(88,I),I=1,NQ) /&
    88d0,87.915d0,87.664d0,87.263d0,86.734d0,86.104d0,85.397d0,84.638d0,83.845d0,83.03d0,&
    82.202d0,81.368d0,80.528d0,79.685d0,78.839d0,77.99d0,77.138d0,76.285d0,75.431d0,74.578d0,&
    73.728d0,72.043d0,70.389d0,69.576d0,68.775d0,67.21d0,65.696d0,64.235d0,62.826d0,62.14d0,&
    61.466d0,60.151d0,58.879d0,57.646d0,56.448d0,55.862d0,55.284d0,54.151d0,53.048d0,50.413d0,&
    47.948d0,45.646d0,43.504d0,39.664d0,36.335d0,33.408d0,30.79d0,28.418d0,26.263d0,24.312d0,&
    22.564d0,21.014d0,19.66d0,18.488d0,17.481d0,16.623d0,13.7d0,11.64d0,9.73d0,8.07d0,&
    5.87d0,4.84d0,3.5255d0,3.0135d0,1.9896d0,1.3214d0,1.0223d0,0.21458d0,0.068739d0,0.038629d0/
    DATA (ScatteringFactors(89,I),I=1,NQ) /&
    89d0,88.915d0,88.664d0,88.26d0,87.723d0,87.077d0,86.346d0,85.553d0,84.719d0,83.859d0,&
    82.985d0,82.105d0,81.225d0,80.348d0,79.474d0,78.605d0,77.739d0,76.879d0,76.023d0,75.172d0,&
    74.326d0,72.654d0,71.014d0,70.208d0,69.412d0,67.855d0,66.345d0,64.884d0,63.473d0,62.785d0,&
    62.11d0,60.792d0,59.517d0,58.282d0,57.084d0,56.497d0,55.919d0,54.787d0,53.684d0,51.05d0,&
    48.58d0,46.268d0,44.11d0,40.229d0,36.863d0,33.912d0,31.283d0,28.906d0,26.744d0,24.779d0,&
    23.008d0,21.427d0,20.036d0,18.823d0,17.776d0,16.88d0,13.86d0,11.815d0,9.93d0,8.255d0,&
    5.933d0,4.91d0,3.5153d0,3.0615d0,2.084d0,1.3366d0,1.0404d0,0.22419d0,0.072725d0,0.041085d0/
    DATA (ScatteringFactors(90,I),I=1,NQ) /&
    90d0,89.916d0,89.669d0,89.269d0,88.735d0,88.085d0,87.344d0,86.533d0,85.672d0,84.779d0,&
    83.867d0,82.946d0,82.025d0,81.107d0,80.196d0,79.294d0,78.4d0,77.516d0,76.642d0,75.777d0,&
    74.922d0,73.242d0,71.602d0,70.798d0,70.005d0,68.454d0,66.951d0,65.497d0,64.091d0,63.405d0,&
    62.731d0,61.416d0,60.143d0,58.91d0,57.713d0,57.127d0,56.55d0,55.419d0,54.317d0,51.684d0,&
    49.211d0,46.889d0,44.716d0,40.795d0,37.391d0,34.413d0,31.77d0,29.387d0,27.219d0,25.244d0,&
    23.454d0,21.846d0,20.421d0,19.17d0,18.083d0,17.149d0,14.02d0,11.98d0,10.13d0,8.44d0,&
    6.118d0,4.982d0,3.6231d0,3.1089d0,2.0806d0,1.3516d0,1.0584d0,0.2341d0,0.076906d0,0.043679d0/
    DATA (ScatteringFactors(91,I),I=1,NQ) /&
    91d0,90.919d0,90.678d0,90.29d0,89.772d0,89.144d0,88.427d0,87.644d0,86.813d0,85.95d0,&
    85.066d0,84.17d0,83.269d0,82.366d0,81.463d0,80.563d0,79.665d0,78.771d0,77.881d0,76.995d0,&
    76.115d0,74.375d0,72.668d0,71.829d0,71.001d0,69.38d0,67.81d0,66.294d0,64.832d0,64.121d0,&
    63.423d0,62.066d0,60.758d0,59.495d0,58.274d0,57.679d0,57.093d0,55.948d0,54.836d0,52.191d0,&
    49.719d0,47.405d0,45.241d0,41.333d0,37.93d0,34.946d0,32.292d0,29.897d0,27.714d0,25.72d0,&
    23.905d0,22.266d0,20.807d0,19.518d0,18.394d0,17.423d0,14.18d0,12.15d0,10.32d0,8.63d0,&
    6.25d0,5.055d0,3.6696d0,3.1555d0,2.1272d0,1.3664d0,1.0763d0,0.24431d0,0.081287d0,0.046418d0/
    DATA (ScatteringFactors(92,I),I=1,NQ) /&
    92d0,91.922d0,91.687d0,91.307d0,90.798d0,90.18d0,89.474d0,88.699d0,87.874d0,87.014d0,&
    86.13d0,85.232d0,84.326d0,83.417d0,82.505d0,81.595d0,80.685d0,79.779d0,78.875d0,77.975d0,&
    77.08d0,75.308d0,73.568d0,72.712d0,71.866d0,70.211d0,68.607d0,67.058d0,65.564d0,64.838d0,&
    64.126d0,62.742d0,61.409d0,60.125d0,58.886d0,58.283d0,57.689d0,56.531d0,55.41d0,52.748d0,&
    50.268d0,47.95d0,45.784d0,41.869d0,38.454d0,35.458d0,32.794d0,30.391d0,28.199d0,26.192d0,&
    24.36d0,22.699d0,21.207d0,19.886d0,18.723d0,17.713d0,14.341d0,12.294d0,10.495d0,8.823d0,&
    6.378d0,5.136d0,3.7151d0,3.2016d0,2.1746d0,1.381d0,1.094d0,0.25483d0,0.085879d0,0.049307d0/
    DATA (ScatteringFactors(93,I),I=1,NQ) /&
    93d0,92.922d0,92.691d0,92.318d0,91.817d0,91.208d0,90.51d0,89.742d0,88.923d0,88.067d0,&
    87.186d0,86.288d0,85.38d0,84.467d0,83.55d0,82.632d0,81.715d0,80.799d0,79.885d0,78.973d0,&
    78.066d0,76.267d0,74.496d0,73.624d0,72.763d0,71.074d0,69.436d0,67.853d0,66.326d0,65.584d0,&
    64.857d0,63.443d0,62.083d0,60.775d0,59.514d0,58.901d0,58.298d0,57.124d0,55.989d0,53.303d0,&
    50.808d0,48.483d0,46.312d0,42.39d0,38.966d0,35.961d0,33.289d0,30.879d0,28.68d0,26.662d0,&
    24.813d0,23.128d0,21.609d0,20.253d0,19.055d0,18.012d0,14.503d0,12.475d0,10.695d0,9.008d0,&
    6.489d0,5.206d0,3.7603d0,3.2478d0,2.2229d0,1.3953d0,1.1116d0,0.26566d0,0.09069d0,0.052357d0/
    DATA (ScatteringFactors(94,I),I=1,NQ) /&
    94d0,93.924d0,93.701d0,93.34d0,92.857d0,92.271d0,91.601d0,90.866d0,90.082d0,89.261d0,&
    88.413d0,87.547d0,86.665d0,85.772d0,84.87d0,83.961d0,83.044d0,82.123d0,81.198d0,80.271d0,&
    79.343d0,77.493d0,75.663d0,74.759d0,73.865d0,72.11d0,70.408d0,68.763d0,67.178d0,66.409d0,&
    65.655d0,64.193d0,62.789d0,61.442d0,60.147d0,59.518d0,58.901d0,57.702d0,56.544d0,53.819d0,&
    51.302d0,48.967d0,46.794d0,42.879d0,39.465d0,36.465d0,33.793d0,31.379d0,29.172d0,27.142d0,&
    25.275d0,23.566d0,22.019d0,20.63d0,19.398d0,18.319d0,14.664d0,12.656d0,10.895d0,9.193d0,&
    6.602d0,5.275d0,3.8056d0,3.2943d0,2.2716d0,1.4094d0,1.1291d0,0.27681d0,0.09573d0,0.055573d0/
    DATA (ScatteringFactors(95,I),I=1,NQ) /&
    95d0,94.926d0,94.706d0,94.352d0,93.877d0,93.299d0,92.638d0,91.91d0,91.131d0,90.315d0,&
    89.47d0,88.605d0,87.723d0,86.829d0,85.924d0,85.011d0,84.09d0,83.163d0,82.231d0,81.296d0,&
    80.36d0,78.49d0,76.636d0,75.719d0,74.811d0,73.027d0,71.293d0,69.615d0,67.997d0,67.212d0,&
    66.441d0,64.947d0,63.513d0,62.137d0,60.816d0,60.175d0,59.546d0,58.325d0,57.148d0,54.385d0,&
    51.842d0,49.49d0,47.307d0,43.38d0,39.958d0,36.952d0,34.276d0,31.858d0,29.648d0,27.611d0,&
    25.733d0,24.006d0,22.435d0,21.018d0,19.754d0,18.64d0,14.826d0,12.838d0,11.095d0,9.378d0,&
    6.713d0,5.345d0,3.8505d0,3.3404d0,2.3202d0,1.4233d0,1.1465d0,0.28828d0,0.10101d0,0.058967d0/
    DATA (ScatteringFactors(96,I),I=1,NQ) /&
    96d0,95.926d0,95.708d0,95.354d0,94.877d0,94.294d0,93.623d0,92.879d0,92.081d0,91.241d0,&
    90.371d0,89.479d0,88.573d0,87.656d0,86.731d0,85.802d0,84.869d0,83.934d0,82.998d0,82.062d0,&
    81.126d0,79.263d0,77.419d0,76.507d0,75.603d0,73.824d0,72.091d0,70.409d0,68.783d0,67.991d0,&
    67.214d0,65.705d0,64.254d0,62.859d0,61.519d0,60.869d0,60.231d0,58.992d0,57.798d0,54.998d0,&
    52.427d0,50.052d0,47.85d0,43.894d0,40.449d0,37.426d0,34.74d0,32.318d0,30.106d0,28.068d0,&
    26.184d0,24.446d0,22.857d0,21.415d0,20.121d0,18.975d0,14.988d0,13.019d0,11.295d0,9.563d0,&
    6.825d0,5.414d0,3.8949d0,3.3861d0,2.3684d0,1.437d0,1.1637d0,0.30009d0,0.10653d0,0.062546d0/
    DATA (ScatteringFactors(97,I),I=1,NQ) /&
    97d0,96.928d0,96.713d0,96.365d0,95.895d0,95.32d0,94.656d0,93.92d0,93.129d0,92.294d0,&
    91.429d0,90.54d0,89.635d0,88.718d0,87.793d0,86.862d0,85.926d0,84.988d0,84.047d0,83.105d0,&
    82.163d0,80.285d0,78.421d0,77.498d0,76.582d0,74.777d0,73.016d0,71.303d0,69.645d0,68.838d0,&
    68.045d0,66.503d0,65.02d0,63.595d0,62.226d0,61.562d0,60.91d0,59.646d0,58.43d0,55.581d0,&
    52.974d0,50.574d0,48.354d0,44.38d0,40.926d0,37.898d0,35.209d0,32.786d0,30.572d0,28.53d0,&
    26.639d0,24.889d0,23.281d0,21.815d0,20.496d0,19.315d0,15.15d0,13.2d0,11.495d0,9.748d0,&
    6.937d0,5.484d0,3.9391d0,3.4314d0,2.4159d0,1.4504d0,1.1808d0,0.31223d0,0.11232d0,0.06632d0/
    DATA (ScatteringFactors(98,I),I=1,NQ) /&
    98d0,97.929d0,97.718d0,97.375d0,96.912d0,96.344d0,95.688d0,94.961d0,94.176d0,93.347d0,&
    92.486d0,91.601d0,90.699d0,89.783d0,88.858d0,87.926d0,86.989d0,86.048d0,85.103d0,84.157d0,&
    83.21d0,81.318d0,79.437d0,78.504d0,77.577d0,75.749d0,73.96d0,72.219d0,70.531d0,69.707d0,&
    68.898d0,67.325d0,65.81d0,64.354d0,62.954d0,62.276d0,61.61d0,60.319d0,59.078d0,56.176d0,&
    53.528d0,51.098d0,48.858d0,44.859d0,41.395d0,38.361d0,35.671d0,33.247d0,31.033d0,28.989d0,&
    27.093d0,25.332d0,23.708d0,22.221d0,20.872d0,19.665d0,15.311d0,13.381d0,11.695d0,9.933d0,&
    7.049d0,5.553d0,3.9832d0,3.4762d0,2.4621d0,1.4637d0,1.1917d0,0.32471d0,0.11837d0,0.0703d0/
    DATA (ScatteringFactors(99,I),I=1,NQ) /&
    99d0,98.903d0,98.678d0,98.293d0,97.784d0,97.196d0,96.507d0,95.75d0,94.972d0,94.162d0,&
    93.322d0,92.479d0,91.625d0,90.751d0,89.854d0,88.943d0,88.025d0,87.096d0,86.155d0,85.202d0,&
    84.24d0,82.287d0,80.322d0,79.343d0,78.37d0,76.453d0,74.598d0,72.793d0,71.043d0,70.189d0,&
    69.349d0,67.712d0,66.134d0,64.632d0,63.188d0,62.486d0,61.797d0,60.457d0,59.165d0,56.118d0,&
    53.323d0,50.739d0,48.363d0,44.182d0,40.667d0,37.673d0,35.065d0,32.708d0,30.527d0,28.507d0,&
    26.545d0,24.73d0,23.036d0,21.468d0,20.029d0,18.726d0,14.395d0,12.4d0,10.406d0,9d0,&
    6.189d0,4.778d0,4.0269d0,3.5203d0,2.5011d0,1.4768d0,1.2146d0,0.33754d0,0.12471d0,0.074497d0/
    DATA (ScatteringFactors(100,I),I=1,NQ) /&
    100d0,99.903d0,99.682d0,99.302d0,98.799d0,98.217d0,97.235d0,96.184d0,95.712d0,95.207d0,&
    94.372d0,93.533d0,92.683d0,91.811d0,90.916d0,90.007d0,89.09d0,88.162d0,87.221d0,86.268d0,&
    85.304d0,83.344d0,81.367d0,80.38d0,79.398d0,77.463d0,75.585d0,73.754d0,71.975d0,71.106d0,&
    70.251d0,68.583d0,66.973d0,65.441d0,63.967d0,63.25d0,62.547d0,61.178d0,59.859d0,56.752d0,&
    53.904d0,51.277d0,48.865d0,44.627d0,41.016d0,38.065d0,35.531d0,33.171d0,30.924d0,28.909d0,&
    26.973d0,25.155d0,23.449d0,21.863d0,20.4d0,19.066d0,14.511d0,12.549d0,10.527d0,9.118d0,&
    6.301d0,4.842d0,4.0707d0,3.5641d0,2.5509d0,1.4896d0,1.2313d0,0.35073d0,0.13134d0,0.078923d0/
    REAL(KIND(1d0))::sintholam,temp,log10sintholam
    INTEGER,PARAMETER::inq=62
    REAL(KIND(1d0)),DIMENSION(inq:NQ)::log10QO4PI,log10sf
    INTEGER::init=0,nmin,nmax
    SAVE log10QO4PI,init
    IF(Z.GT.ZMAX.OR.Z.LE.0)THEN
       WRITE(*,*)"ERROR: the atomic form factor is not available with Z=",Z
       STOP
    ENDIF
    IF(q.LT.0d0)THEN
       WRITE(*,*)"ERROR: q < 0 !"
       STOP
    ENDIF
    IF(init.EQ.0)THEN
       DO I=inq,NQ
          log10QO4PI(I)=DLOG10(QO4PI(I))
       ENDDO
       init=1
    ENDIF
    ! GeV to angstrom^{-1}
    sintholam=q/FOURPI*1d5/0.1973d0
    IF(sintholam.GT.QO4PI(NQ))THEN
       AtomicFormFactor=1d0
       RETURN
    ENDIF
    IF(sintholam.LE.QO4PI(inq))THEN
       CALL SPLINE_INTERPOLATE(QO4PI(1:inq),ScatteringFactors(Z,1:inq),inq,sintholam,temp)
    ELSE
       J=0
       DO I=inq,NQ
          IF(sintholam.LE.QO4PI(I))THEN
             J=I
             EXIT
          ENDIF
       ENDDO
       IF(J.EQ.0)THEN
          J=NQ
       ENDIF
       IF(ScatteringFactors(Z,J-1).EQ.0d0)THEN
          temp=0d0
       ELSEIF(ScatteringFactors(Z,J).EQ.0d0)THEN
          temp=MAX((sintholam-QO4PI(J-1))/(QO4PI(J)-QO4PI(J-1))*ScatteringFactors(Z,J-1),0d0)
       ELSE
          J=MAX(J,3+inq)
          IF(J-inq+1.GE.5)THEN
             nmin=J-3
             nmax=J
          ELSE
             nmin=inq
             nmax=J
          ENDIF
          DO I=nmin,nmax
             log10sf(I)=DLOG10(MAX(ScatteringFactors(Z,I),1D-8))
          ENDDO
          log10sintholam=DLOG10(sintholam)
          CALL SPLINE_INTERPOLATE(log10QO4PI(nmin:nmax),log10sf(nmin:nmax),nmax-nmin+1,log10sintholam,temp)
          temp=10d0**temp
       ENDIF
    ENDIF
    temp=MAX(temp,0d0)
    ! we have used the approximation that for the atomic size, the nucleus can be approximated
    ! as a Dirac delta function in space since R_{A atom} >> R_A
    ! We also do not want to absorb Z into the form factor
    AtomicFormFactor=(ScatteringFactors(Z,1)-temp)/DBLE(Z)
    RETURN
  END FUNCTION AtomicFormFactor

  FUNCTION AtomicIncoherentStructureFunction(Z,q)
    ! charge-neutral atomic incoherent structure function
    ! this is for the incoherent scattering of photon with the 
    ! electrons in atoms (not for nucleus)
    ! see table 1 in /Users/huasheng/Physics/Books/AtomicPhysics/Atomicformfactors_incoherentscatteringfunctions_photonscatteringcrosssections.pdf
    IMPLICIT NONE
    INTEGER,INTENT(IN)::Z  ! atomic number
    REAL(KIND(1d0)),INTENT(IN)::q ! in unit of GeV
    REAL(KIND(1d0))::AtomicIncoherentStructureFunction ! it is dimensionless
    REAL(KIND(1d0)),PARAMETER::FOURPI=12.5663706143591729538505735331d0
    INTEGER,PARAMETER::ZMAX=100,NQ=42
    INTEGER::I,J
    ! mean atomic scattering factors for free atoms
    ! this uses Hartree-Fock or Dirac-Slater wave functions
    ! the following Fortran output can be found in <<Atomic form factors/Output into Fortran for gamma-UPC>>
    ! in "/Users/huasheng/Physics/FLibatM/jpsi_resummation/CSS/quarkonium.nb"
    REAL(KIND(1d0)),DIMENSION(NQ)::QO4PI ! in unit of angstrom^{-1}
    DATA QO4PI /0.d0,0.005d0,0.01d0,0.015d0,0.02d0,0.025d0,0.03d0,0.04d0,0.05d0,0.07d0,&
         0.09d0,0.1d0,0.125d0,0.15d0,0.175d0,0.2d0,0.25d0,0.3d0,0.4d0,0.5d0,&
         0.6d0,0.7d0,0.8d0,0.9d0,1.d0,1.25d0,1.5d0,2.d0,2.5d0,3.d0,&
         3.5d0,4.d0,5.d0,6.d0,7.d0,8.d0,10.d0,15.d0,20.d0,50.d0,&
         80.d0,100.d0/
    REAL(KIND(1d0)),DIMENSION(ZMAX,NQ)::ScatteringFactors
        DATA (ScatteringFactors(1,I),I=1,NQ) /&
    0d0,0.001d0,0.004d0,0.01d0,0.017d0,0.027d0,0.037d0,0.068d0,0.103d0,0.19d0,&
    0.29d0,0.343d0,0.471d0,0.59d0,0.687d0,0.769d0,0.878d0,0.937d0,0.983d0,0.995d0,&
    0.998d0,0.999d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,&
    1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,1d0,&
    1d0,1d0/
    DATA (ScatteringFactors(2,I),I=1,NQ) /&
    0d0,0.001d0,0.003d0,0.007d0,0.013d0,0.021d0,0.03d0,0.052d0,0.081d0,0.153d0,&
    0.245d0,0.296d0,0.432d0,0.584d0,0.736d0,0.881d0,1.146d0,1.362d0,1.657d0,1.818d0,&
    1.902d0,1.947d0,1.97d0,1.983d0,1.99d0,1.997d0,1.999d0,2d0,2d0,2d0,&
    2d0,2d0,2d0,2d0,2d0,2d0,2d0,2d0,2d0,2d0,&
    2d0,2d0/
    DATA (ScatteringFactors(3,I),I=1,NQ) /&
    0d0,0.007d0,0.026d0,0.058d0,0.102d0,0.154d0,0.216d0,0.354d0,0.5d0,0.765d0,&
    0.963d0,1.033d0,1.172d0,1.246d0,1.33d0,1.418d0,1.605d0,1.795d0,2.143d0,2.417d0,&
    2.613d0,2.746d0,2.834d0,2.891d0,2.928d0,2.973d0,2.989d0,2.998d0,2.999d0,3d0,&
    3d0,3d0,3d0,3d0,3d0,3d0,3d0,3d0,3d0,3d0,&
    3d0,3d0/
    DATA (ScatteringFactors(4,I),I=1,NQ) /&
    0d0,0.004d0,0.018d0,0.04d0,0.07d0,0.108d0,0.154d0,0.266d0,0.398d0,0.7d0,&
    1.016d0,1.171d0,1.503d0,1.774d0,1.966d0,2.121d0,2.322d0,2.47d0,2.744d0,3.005d0,&
    3.237d0,3.429d0,3.579d0,3.693d0,3.777d0,3.9d0,3.954d0,3.989d0,3.997d0,3.999d0,&
    4d0,4d0,4d0,4d0,4d0,4d0,4d0,4d0,4d0,4d0,&
    4d0,4d0/
    DATA (ScatteringFactors(5,I),I=1,NQ) /&
    0d0,0.004d0,0.016d0,0.035d0,0.062d0,0.096d0,0.137d0,0.238d0,0.36d0,0.651d0,&
    0.978d0,1.147d0,1.548d0,1.931d0,2.256d0,2.531d0,2.933d0,3.19d0,3.499d0,3.732d0,&
    3.948d0,4.146d0,4.32d0,4.469d0,4.59d0,4.792d0,4.896d0,4.973d0,4.992d0,4.997d0,&
    4.999d0,5d0,5d0,5d0,5d0,5d0,5d0,5d0,5d0,5d0,&
    5d0,5d0/
    DATA (ScatteringFactors(6,I),I=1,NQ) /&
    0d0,0.004d0,0.013d0,0.03d0,0.052d0,0.08d0,0.116d0,0.202d0,0.309d0,0.569d0,&
    0.876d0,1.039d0,1.448d0,1.866d0,2.253d0,2.604d0,3.198d0,3.643d0,4.184d0,4.478d0,&
    4.69d0,4.812d0,5.051d0,5.208d0,5.348d0,5.615d0,5.781d0,5.93d0,5.977d0,5.992d0,&
    5.997d0,5.999d0,6d0,6d0,6d0,6d0,6d0,6d0,6d0,6d0,&
    6d0,6d0/
    DATA (ScatteringFactors(7,I),I=1,NQ) /&
    0d0,0.003d0,0.013d0,0.029d0,0.052d0,0.08d0,0.115d0,0.202d0,0.31d0,0.58d0,&
    0.904d0,1.08d0,1.54d0,2.003d0,2.447d0,2.858d0,3.559d0,4.097d0,4.792d0,5.182d0,&
    5.437d0,5.635d0,5.809d0,5.968d0,6.113d0,6.415d0,6.63d0,6.86d0,6.947d0,6.979d0,&
    6.991d0,6.996d0,6.999d0,7d0,6.9999d0,7d0,7d0,7d0,7d0,7d0,&
    7d0,7d0/
    DATA (ScatteringFactors(8,I),I=1,NQ) /&
    0d0,0.003d0,0.011d0,0.025d0,0.045d0,0.07d0,0.1d0,0.176d0,0.271d0,0.514d0,&
    0.812d0,0.977d0,1.42d0,1.885d0,2.35d0,2.799d0,3.614d0,4.293d0,5.257d0,5.828d0,&
    6.115d0,6.411d0,6.596d0,6.755d0,6.901d0,7.216d0,7.462d0,7.764d0,7.9d0,7.957d0,&
    7.981d0,7.991d0,7.998d0,7.999d0,7.9998d0,8d0,8d0,8d0,8d0,8d0,&
    8d0,8d0/
    DATA (ScatteringFactors(9,I),I=1,NQ) /&
    0d0,0.002d0,0.01d0,0.022d0,0.04d0,0.062d0,0.089d0,0.156d0,0.242d0,0.461d0,&
    0.735d0,0.888d0,1.308d0,1.761d0,2.227d0,2.691d0,3.569d0,4.347d0,5.552d0,6.339d0,&
    6.832d0,7.151d0,7.376d0,7.552d0,7.703d0,8.024d0,8.288d0,8.648d0,8.834d0,8.923d0,&
    8.963d0,8.982d0,8.995d0,8.998d0,8.9995d0,9d0,9d0,9d0,9d0,9d0,&
    9d0,9d0/
    DATA (ScatteringFactors(10,I),I=1,NQ) /&
    0d0,0.002d0,0.009d0,0.02d0,0.036d0,0.056d0,0.08d0,0.141d0,0.218d0,0.418d0,&
    0.669d0,0.812d0,1.205d0,1.637d0,2.088d0,2.547d0,3.442d0,4.269d0,5.644d0,6.64d0,&
    7.32d0,7.774d0,8.085d0,8.312d0,8.49d0,8.836d0,9.113d0,9.518d0,9.752d0,9.875d0,&
    9.937d0,9.967d0,9.991d0,9.997d0,9.9989d0,10d0,10d0,10d0,10d0,10d0,&
    10d0,10d0/
    DATA (ScatteringFactors(11,I),I=1,NQ) /&
    0d0,0.009d0,0.036d0,0.079d0,0.138d0,0.209d0,0.291d0,0.476d0,0.674d0,1.049d0,&
    1.364d0,1.503d0,1.828d0,2.16d0,2.516d0,2.891d0,3.667d0,4.431d0,5.804d0,6.903d0,&
    7.724d0,8.313d0,8.729d0,9.028d0,9.252d0,9.646d0,9.939d0,10.376d0,10.654d0,10.813d0,&
    10.9d0,10.946d0,10.983d0,10.994d0,10.998d0,10.999d0,11d0,11d0,11d0,11d0,&
    11d0,11d0/
    DATA (ScatteringFactors(12,I),I=1,NQ) /&
    0d0,0.01d0,0.04d0,0.09d0,0.157d0,0.241d0,0.339d0,0.57d0,0.831d0,1.372d0,&
    1.858d0,2.066d0,2.491d0,2.829d0,3.135d0,3.444d0,4.096d0,4.771d0,6.064d0,7.181d0,&
    8.086d0,8.784d0,9.304d0,9.689d0,9.975d0,10.449d0,10.766d0,11.229d0,11.543d0,11.738d0,&
    11.852d0,11.916d0,11.972d0,11.99d0,11.996d0,11.998d0,12d0,12d0,12d0,12d0,&
    12d0,12d0/
    DATA (ScatteringFactors(13,I),I=1,NQ) /&
    0d0,0.01d0,0.039d0,0.087d0,0.153d0,0.235d0,0.332d0,0.564d0,0.832d0,1.419d0,&
    1.997d0,2.264d0,2.851d0,3.324d0,3.712d0,4.047d0,4.653d0,5.25d0,6.435d0,7.523d0,&
    8.459d0,9.225d0,9.83d0,10.296d0,10.652d0,11.233d0,11.592d0,12.083d0,12.425d0,12.652d0,&
    12.794d0,12.879d0,12.957d0,12.984d0,12.993d0,12.997d0,12.999d0,13d0,13d0,13d0,&
    13d0,13d0/
    DATA (ScatteringFactors(14,I),I=1,NQ) /&
    0d0,0.009d0,0.035d0,0.079d0,0.139d0,0.215d0,0.305d0,0.523d0,0.782d0,1.372d0,&
    1.992d0,2.293d0,2.988d0,3.587d0,4.092d0,4.52d0,5.218d0,5.808d0,5.903d0,7.937d0,&
    8.867d0,9.667d0,10.33d0,10.864d0,11.286d0,11.99d0,12.408d0,12.937d0,13.302d0,13.558d0,&
    13.726d0,13.832d0,13.937d0,13.975d0,13.99d0,13.995d0,13.999d0,14d0,14d0,14d0,&
    14d0,14d0/
    DATA (ScatteringFactors(15,I),I=1,NQ) /&
    0d0,0.008d0,0.032d0,0.071d0,0.126d0,0.194d0,0.277d0,0.477d0,0.719d0,1.284d0,&
    1.899d0,2.206d0,2.944d0,3.611d0,4.206d0,4.732d0,5.611d0,6.312d0,7.435d0,8.419d0,&
    9.323d0,10.131d0,10.827d0,11.411d0,11.888d0,12.716d0,13.209d0,13.79d0,14.117d0,14.457d0,&
    14.65d0,14.778d0,14.911d0,14.963d0,14.984d0,14.993d0,14.998d0,15d0,15d0,15d0,&
    15d0,15d0/
    DATA (ScatteringFactors(16,I),I=1,NQ) /&
    0d0,0.007d0,0.029d0,0.065d0,0.114d0,0.177d0,0.253d0,0.439d0,0.666d0,1.212d0,&
    1.831d0,2.151d0,2.94d0,3.68d0,4.354d0,4.96d0,5.984d0,6.795d0,8.002d0,8.96d0,&
    9.829d0,10.626d0,11.336d0,11.952d0,12.472d0,13.414d0,13.99d0,14.641d0,15.051d0,15.351d0,&
    15.567d0,15.716d0,15.88d0,15.948d0,15.977d0,15.989d0,15.997d0,16d0,16d0,16d0,&
    16d0,16d0/
    DATA (ScatteringFactors(17,I),I=1,NQ) /&
    0d0,0.007d0,0.026d0,0.059d0,0.104d0,0.162d0,0.232d0,0.404d0,0.617d0,1.138d0,&
    1.744d0,2.065d0,2.877d0,3.665d0,4.4d0,5.074d0,6.24d0,7.182d0,8.553d0,9.539d0,&
    10.382d0,11.158d0,11.867d0,12.499d0,13.05d0,14.088d0,14.75d0,15.487d0,15.924d0,16.243d0,&
    16.479d0,16.648d0,16.843d0,16.93d0,16.968d0,16.985d0,16.996d0,17d0,17d0,17d0,&
    17d0,17d0/
    DATA (ScatteringFactors(18,I),I=1,NQ) /&
    0d0,0.006d0,0.024d0,0.054d0,0.096d0,0.149d0,0.213d0,0.373d0,0.571d0,1.063d0,&
    1.644d0,1.956d0,2.76d0,3.558d0,4.32d0,5.033d0,6.303d0,7.371d0,8.998d0,10.106d0,&
    10.967d0,11.126d0,12.424d0,13.061d0,13.629d0,14.145d0,15.489d0,16.324d0,16.795d0,17.132d0,&
    17.386d0,17.573d0,17.8d0,17.907d0,17.956d0,17.978d0,17.994d0,18d0,18d0,18d0,&
    18d0,18d0/
    DATA (ScatteringFactors(19,I),I=1,NQ) /&
    0d0,0.016d0,0.062d0,0.138d0,0.238d0,0.358d0,0.495d0,0.794d0,1.105d0,1.692d0,&
    2.233d0,2.5d0,3.19d0,3.905d0,4.616d0,5.301d0,6.555d0,7.652d0,9.405d0,10.65d0,&
    11.568d0,12.329d0,13.014d0,13.645d0,14.22d0,15.393d0,16.212d0,17.152d0,17.664d0,18.02d0,&
    18.29d0,18.494d0,18.752d0,18.88d0,18.941d0,18.97d0,18.992d0,18.999d0,19d0,19d0,&
    19d0,19d0/
    DATA (ScatteringFactors(20,I),I=1,NQ) /&
    0d0,0.018d0,0.072d0,0.16d0,0.278d0,0.423d0,0.59d0,0.968d0,1.375d0,2.152d0,&
    2.817d0,3.105d0,3.162d0,4.401d0,5.048d0,5.69d0,6.899d0,7.981d0,9.79d0,11.151d0,&
    12.163d0,12.953d0,13.635d0,14.256d0,14.83d0,16.038d0,16.921d0,17.97d0,18.531d0,18.906d0,&
    19.191d0,19.411d0,19.698d0,19.849d0,19.924d0,19.961d0,19.989d0,19.999d0,20d0,20d0,&
    20d0,20d0/
    DATA (ScatteringFactors(21,I),I=1,NQ) /&
    0d0,0.017d0,0.067d0,0.149d0,0.26d0,0.396d0,0.555d0,0.92d0,1.321d0,2.122d0,&
    2.826d0,3.136d0,3.834d0,4.492d0,5.148d0,5.801d0,7.046d0,8.169d0,10.071d0,11.561d0,&
    12.684d0,13.545d0,14.256d0,14.885d0,15.46d0,16.694d0,17.63d0,18.182d0,19.397d0,19.794d0,&
    20.093d0,20.326d0,20.641d0,20.813d0,20.903d0,20.949d0,20.985d0,20.999d0,21d0,21d0,&
    21d0,21d0/
    DATA (ScatteringFactors(22,I),I=1,NQ) /&
    0d0,0.016d0,0.063d0,0.139d0,0.243d0,0.372d0,0.522d0,0.872d0,1.263d0,2.063d0,&
    2.189d0,3.114d0,3.846d0,4.523d0,5.192d0,5.86d0,7.144d0,8.312d0,10.304d0,11.901d0,&
    13.14d0,14.093d0,14.856d0,15.509d0,16.095d0,17.353d0,18.334d0,19.585d0,20.259d0,20.682d0,&
    20.994d0,21.239d0,21.58d0,21.774d0,21.879d0,21.935d0,21.98d0,21.998d0,22d0,22d0,&
    22d0,22d0/
    DATA (ScatteringFactors(23,I),I=1,NQ) /&
    0d0,0.015d0,0.059d0,0.13d0,0.229d0,0.351d0,0.493d0,0.822d0,1.206d0,1.996d0,&
    2.733d0,3.067d0,3.82d0,4.51d0,5.185d0,5.858d0,7.167d0,8.375d0,10.454d0,12.156d0,&
    13.514d0,14.574d0,15.413d0,16.111d0,16.721d0,18.01d0,19.032d0,20.379d0,21.116d0,21.569d0,&
    21.896d0,22.152d0,22.515d0,22.732d0,22.853d0,22.919d0,22.974d0,22.998d0,23d0,23d0,&
    23d0,23d0/
    DATA (ScatteringFactors(24,I),I=1,NQ) /&
    0d0,0.012d0,0.046d0,0.104d0,0.181d0,0.278d0,0.392d0,0.661d0,0.968d0,1.631d0,&
    2.291d0,2.609d0,3.377d0,4.123d0,4.857d0,5.571d0,6.948d0,8.206d0,10.415d0,12.264d0,&
    13.77d0,14.96d0,15.902d0,16.67d0,17.323d0,18.666d0,19.73d0,21.168d0,21.97d0,22.456d0,&
    22.798d0,23.065d0,23.449d0,23.686d0,23.823d0,23.9d0,23.967d0,23.997d0,24d0,24d0,&
    24d0,24d0/
    DATA (ScatteringFactors(25,I),I=1,NQ) /&
    0d0,0.013d0,0.052d0,0.117d0,0.205d0,0.315d0,0.444d0,0.751d0,1.104d0,1.866d0,&
    2.606d0,2.949d0,3.728d0,4.435d0,5.116d0,5.191d0,7.122d0,8.38d0,10.604d0,12.486d0,&
    14.062d0,15.346d0,16.376d0,17.211d0,17.91d0,19.312d0,20.411d0,21.938d0,22.812d0,23.337d0,&
    23.698d0,23.976d0,24.38d0,24.636d0,24.79d0,24.879d0,24.959d0,24.996d0,25d0,25d0,&
    25d0,25d0/
    DATA (ScatteringFactors(26,I),I=1,NQ) /&
    0d0,0.012d0,0.05d0,0.111d0,0.195d0,0.3d0,0.423d0,0.718d0,1.06d0,1.806d0,&
    2.544d0,2.891d0,3.684d0,4.405d0,5.096d0,5.781d0,7.138d0,8.432d0,10.133d0,12.687d0,&
    14.343d0,15.116d0,16.831d0,17.137d0,18.488d0,19.959d0,21.097d0,22.704d0,23.65d0,24.216d0,&
    24.598d0,24.881d0,25.31d0,25.585d0,25.755d0,25.856d0,25.949d0,25.995d0,25.999d0,26d0,&
    26d0,26d0/
    DATA (ScatteringFactors(27,I),I=1,NQ) /&
    0d0,0.012d0,0.047d0,0.106d0,0.186d0,0.286d0,0.404d0,0.688d0,1.019d0,1.75d0,&
    2.483d0,2.832d0,3.636d0,4.369d0,5.069d0,5.764d0,7.143d0,8.469d0,10.844d0,12.861d0,&
    14.596d0,16.05d0,17.249d0,18.229d0,19.035d0,20.596d0,21.171d0,23.462d0,24.48d0,25.092d0,&
    25.497d0,25.799d0,26.238d0,26.531d0,26.717d0,26.83d0,26.938d0,26.994d0,26.999d0,27d0,&
    27d0,27d0/
    DATA (ScatteringFactors(28,I),I=1,NQ) /&
    0d0,0.011d0,0.045d0,0.101d0,0.177d0,0.273d0,0.387d0,0.66d0,0.981d0,1.696d0,&
    2.423d0,2.772d0,3.582d0,4.322d0,5.029d0,5.726d0,7.115d0,8.461d0,10.894d0,12.98d0,&
    14.78d0,16.317d0,17.602d0,18.664d0,19.543d0,21.21d0,22.445d0,24.211d0,25.302d0,25.962d0,&
    26.394d0,26.71d0,27.166d0,27.475d0,27.676d0,27.802d0,27.926d0,27.992d0,27.999d0,28d0,&
    28d0,28d0/
    DATA (ScatteringFactors(29,I),I=1,NQ) /&
    0d0,0.009d0,0.036d0,0.081d0,0.143d0,0.22d0,0.312d0,0.534d0,0.796d0,1.393d0,&
    2.029d0,2.348d0,3.139d0,3.919d0,4.692d0,5.455d0,5.931d0,8.31d0,10.178d0,12.942d0,&
    14.847d0,16.494d0,17.885d0,19.043d0,20.002d0,21.802d0,23.107d0,24.957d0,26.119d0,26.83d0,&
    27.291d0,27.622d0,28.095d0,28.418d0,28.634d0,28.772d0,28.912d0,28.99d0,28.999d0,29d0,&
    29d0,29d0/
    DATA (ScatteringFactors(30,I),I=1,NQ) /&
    0d0,0.01d0,0.041d0,0.093d0,0.163d0,0.251d0,0.357d0,0.612d0,0.912d0,1.596d0,&
    2.307d0,2.654d0,3.47d0,4.22d0,4.932d0,5.631d0,7.024d0,8.388d0,10.901d0,13.094d0,&
    15.02d0,16.109d0,18.163d0,19.395d0,20.427d0,22.365d0,23.745d0,25.683d0,26.919d0,27.687d0,&
    28.181d0,28.53d0,29.021d0,29.358d0,29.588d0,29.739d0,29.896d0,29.988d0,29.998d0,30d0,&
    30d0,30d0/
    DATA (ScatteringFactors(31,I),I=1,NQ) /&
    0d0,0.011d0,0.044d0,0.098d0,0.173d0,0.266d0,0.377d0,0.644d0,0.959d0,1.672d0,&
    2.422d0,2.791d0,3.673d0,4.483d0,5.233d0,5.939d0,7.287d0,8.599d0,11.082d0,13.29d0,&
    15.233d0,16.947d0,18.445d0,19.734d0,20.831d0,22.907d0,24.37d0,26.4d0,27.71d0,28.536d0,&
    29.067d0,29.436d0,29.947d0,30.297d0,30.541d0,30.705d0,30.879d0,30.985d0,30.998d0,31d0,&
    31d0,31d0/
    DATA (ScatteringFactors(32,I),I=1,NQ) /&
    0d0,0.011d0,0.042d0,0.095d0,0.167d0,0.258d0,0.367d0,0.63d0,0.944d0,1.669d0,&
    2.449d0,2.839d0,3.781d0,4.659d0,5.472d0,6.229d0,7.619d0,8.912d0,11.338d0,13.536d0,&
    15.486d0,17.215d0,18.741d0,20.074d0,21.224d0,23.43d0,24.983d0,27.109d0,28.492d0,29.377d0,&
    29.947d0,30.34d0,30.872d0,31.236d0,31.492d0,31.668d0,31.86d0,31.982d0,31.997d0,32d0,&
    32d0,32d0/
    DATA (ScatteringFactors(33,I),I=1,NQ) /&
    0d0,0.01d0,0.04d0,0.09d0,0.159d0,0.246d0,0.35d0,0.604d0,0.909d0,1.621d0,&
    2.4d0,2.793d0,3.758d0,4.675d0,5.543d0,6.365d0,7.878d0,9.236d0,11.658d0,13.828d0,&
    15.775d0,17.511d0,19.056d0,20.42d0,21.612d0,23.938d0,25.583d0,27.81d0,29.264d0,30.209d0,&
    30.822d0,31.241d0,31.796d0,32.113d0,32.442d0,32.629d0,32.84d0,32.979d0,32.996d0,33d0,&
    33d0,33d0/
    DATA (ScatteringFactors(34,I),I=1,NQ) /&
    0d0,0.01d0,0.038d0,0.086d0,0.152d0,0.235d0,0.335d0,0.581d0,0.879d0,1.59d0,&
    2.388d0,2.799d0,3.818d0,4.794d0,5.717d0,6.589d0,8.186d0,9.601d0,12.033d0,14.168d0,&
    16.098d0,17.835d0,19.391d0,20.778d0,22.003d0,24.434d0,26.171d0,28.504d0,30.028d0,31.034d0,&
    31.691d0,32.137d0,32.719d0,33.109d0,33.39d0,33.589d0,33.818d0,33.975d0,33.996d0,34d0,&
    34d0,34d0/
    DATA (ScatteringFactors(35,I),I=1,NQ) /&
    0d0,0.009d0,0.036d0,0.082d0,0.144d0,0.224d0,0.32d0,0.557d0,0.846d0,1.548d0,&
    2.351d0,2.771d0,3.826d0,4.851d0,5.826d0,6.748d0,8.442d0,9.94d0,12.44d0,14.552d0,&
    16.456d0,18.185d0,19.747d0,21.149d0,22.399d0,24.92d0,26.147d0,29.19d0,30.785d0,31.85d0,&
    32.554d0,33.03d0,33.641d0,34.045d0,34.337d0,34.547d0,34.794d0,34.97d0,34.995d0,35d0,&
    35d0,35d0/
    DATA (ScatteringFactors(36,I),I=1,NQ) /&
    0d0,0.009d0,0.035d0,0.078d0,0.138d0,0.214d0,0.306d0,0.533d0,0.812d0,1.494d0,&
    2.286d0,2.703d0,3.764d0,4.805d0,5.805d0,6.76d0,8.546d0,10.157d0,12.828d0,14.969d0,&
    16.849d0,18.562d0,20.123d0,21.535d0,22.804d0,25.401d0,27.313d0,29.87d0,31.534d0,32.659d0,&
    33.41d0,33.919d0,34.562d0,34.98d0,35.283d0,35.504d0,35.769d0,35.965d0,35.994d0,36d0,&
    36d0,36d0/
    DATA (ScatteringFactors(37,I),I=1,NQ) /&
    0d0,0.02d0,0.078d0,0.172d0,0.296d0,0.446d0,0.615d0,0.986d0,1.372d0,2.122d0,&
    2.852d0,3.225d0,4.189d0,5.172d0,6.135d0,7.062d0,8.812d0,10.431d0,13.206d0,15.41d0,&
    17.282d0,18.974d0,20.526d0,21.94d0,23.221d0,25.88d0,27.871d0,30.543d0,32.277d0,33.461d0,&
    34.259d0,34.803d0,35.482d0,35.915d0,36.228d0,36.459d0,36.742d0,36.959d0,36.992d0,37d0,&
    37d0,37d0/
    DATA (ScatteringFactors(38,I),I=1,NQ) /&
    0d0,0.023d0,0.091d0,0.202d0,0.351d0,0.533d0,0.74d0,1.206d0,1.701d0,2.642d0,&
    3.457d0,3.831d0,4.738d0,5.653d0,6.57d0,7.464d0,9.159d0,10.746d0,13.576d0,15.86d0,&
    17.745d0,19.42d0,20.956d0,22.367d0,23.654d0,26.361d0,28.423d0,31.21d0,33.014d0,34.255d0,&
    35.103d0,35.682d0,36.399d0,36.848d0,37.172d0,37.413d0,37.713d0,37.953d0,37.991d0,38d0,&
    38d0,38d0/
    DATA (ScatteringFactors(39,I),I=1,NQ) /&
    0d0,0.022d0,0.087d0,0.194d0,0.338d0,0.514d0,0.718d0,1.186d0,1.694d0,2.701d0,&
    3.594d0,3.999d0,4.948d0,5.874d0,6.796d0,7.7d0,9.413d0,11.01d0,13.899d0,16.279d0,&
    18.215d0,19.891d0,21.416d0,22.82d0,24.11d0,26.849d0,28.91d0,31.87d0,33.145d0,35.043d0,&
    35.94d0,36.557d0,37.316d0,37.782d0,38.116d0,38.366d0,38.684d0,38.946d0,38.989d0,39d0,&
    39d0,39d0/
    DATA (ScatteringFactors(40,I),I=1,NQ) /&
    0d0,0.021d0,0.083d0,0.184d0,0.322d0,0.492d0,0.689d0,1.147d0,1.654d0,2.688d0,&
    3.633d0,4.064d0,5.065d0,6.019d0,6.959d0,7.879d0,9.621d0,11.236d0,14.176d0,16.658d0,&
    18.612d0,20.373d0,21.895d0,23.294d0,24.583d0,27.347d0,29.517d0,32.522d0,34.47d0,35.825d0,&
    36.771d0,37.426d0,38.23d0,38.715d0,39.059d0,39.318d0,39.653d0,39.938d0,39.987d0,40d0,&
    40d0,40d0/
    DATA (ScatteringFactors(41,I),I=1,NQ) /&
    0d0,0.017d0,0.068d0,0.152d0,0.265d0,0.406d0,0.571d0,0.958d0,1.395d0,2.325d0,&
    3.236d0,3.672d0,4.722d0,5.735d0,6.724d0,7.684d0,9.508d0,11.213d0,14.317d0,16.949d0,&
    19.081d0,20.844d0,22.386d0,23.787d0,25.077d0,27.86d0,30.067d0,33.167d0,35.188d0,36.601d0,&
    37.596d0,38.291d0,39.142d0,39.647d0,40.002d0,40.21d0,40.621d0,40.93d0,40.985d0,41d0,&
    41d0,41d0/
    DATA (ScatteringFactors(42,I),I=1,NQ) /&
    0d0,0.016d0,0.065d0,0.144d0,0.252d0,0.387d0,0.545d0,0.918d0,1.344d0,2.264d0,&
    3.181d0,3.625d0,4.692d0,5.72d0,6.72d0,7.69d0,9.532d0,11.26d0,14.44d0,17.196d0,&
    19.455d0,21.3d0,22.877d0,24.288d0,25.581d0,28.378d0,30.62d0,33.808d0,35.901d0,37.37d0,&
    38.415d0,39.15d0,40.051d0,40.578d0,40.945d0,41.221d0,41.587d0,41.92d0,41.983d0,42d0,&
    42d0,42d0/
    DATA (ScatteringFactors(43,I),I=1,NQ) /&
    0d0,0.018d0,0.072d0,0.16d0,0.28d0,0.431d0,0.607d0,1.024d0,1.502d0,2.527d0,&
    3.522d0,3.987d0,5.066d0,6.067d0,7.035d0,7.984d0,9.806d0,11.512d0,14.653d0,17.456d0,&
    19.816d0,21.748d0,23.37d0,24.797d0,26.093d0,28.901d0,31.173d0,34.447d0,36.61d0,38.134d0,&
    39.226d0,40.003d0,40.958d0,41.509d0,41.886d0,42.171d0,42.553d0,42.911d0,42.98d0,43d0,&
    43d0,43d0/
    DATA (ScatteringFactors(44,I),I=1,NQ) /&
    0d0,0.015d0,0.059d0,0.131d0,0.23d0,0.354d0,0.501d0,0.85d0,1.256d0,2.16d0,&
    3.096d0,3.559d0,4.69d0,5.783d0,6.84d0,7.857d0,9.765d0,11.531d0,14.782d0,17.685d0,&
    20.15d0,22.173d0,23.855d0,25.312d0,26.621d0,29.444d0,31.74d0,35.081d0,37.311d0,38.891d0,&
    40.033d0,40.851d0,41.861d0,42.438d0,42.828d0,43.121d0,43.518d0,43.9d0,43.978d0,44d0,&
    44d0,44d0/
    DATA (ScatteringFactors(45,I),I=1,NQ) /&
    0d0,0.014d0,0.056d0,0.126d0,0.221d0,0.34d0,0.482d0,0.82d0,1.215d0,2.103d0,&
    3.034d0,3.499d0,4.642d0,5.753d0,6.828d0,7.863d0,9.802d0,11.591d0,14.883d0,17.858d0,&
    20.428d0,22.557d0,24.318d0,25.819d0,27.148d0,29.991d0,32.309d0,35.715d0,38.009d0,39.643d0,&
    40.833d0,41.693d0,42.761d0,43.366d0,43.769d0,44.07d0,44.481d0,44.889d0,44.975d0,45d0,&
    45d0,45d0/
    DATA (ScatteringFactors(46,I),I=1,NQ) /&
    0d0,0.01d0,0.039d0,0.088d0,0.156d0,0.242d0,0.346d0,0.604d0,0.923d0,1.705d0,&
    2.619d0,3.103d0,4.334d0,5.536d0,6.668d0,7.725d0,9.654d0,11.441d0,14.824d0,17.943d0,&
    20.653d0,22.904d0,24.756d0,26.316d0,27.677d0,30.549d0,32.888d0,36.349d0,38.703d0,40.389d0,&
    41.627d0,42.529d0,43.658d0,44.293d0,44.71d0,45.019d0,45.444d0,45.877d0,45.971d0,46d0,&
    46d0,46d0/
    DATA (ScatteringFactors(47,I),I=1,NQ) /&
    0d0,0.013d0,0.052d0,0.117d0,0.205d0,0.316d0,0.448d0,0.765d0,1.139d0,1.99d0,&
    2.901d0,3.362d0,4.506d0,5.631d0,6.727d0,7.785d0,9.77d0,11.598d0,14.969d0,18.082d0,&
    20.858d0,23.212d0,25.162d0,26.792d0,28.195d0,31.106d0,33.465d0,36.983d0,39.395d0,41.131d0,&
    42.415d0,43.359d0,44.55d0,45.217d0,45.65d0,45.968d0,46.406d0,46.864d0,46.967d0,47d0,&
    47d0,47d0/
    DATA (ScatteringFactors(48,I),I=1,NQ) /&
    0d0,0.015d0,0.059d0,0.132d0,0.232d0,0.358d0,0.507d0,0.866d0,1.288d0,2.237d0,&
    3.22d0,3.7d0,4.843d0,5.921d0,6.961d0,7.98d0,9.956d0,11.812d0,15.185d0,18.263d0,&
    21.064d0,23.501d0,25.546d0,27.252d0,28.705d0,31.666d0,34.046d0,37.618d0,40.085d0,41.87d0,&
    43.198d0,44.184d0,45.437d0,46.139d0,46.589d0,46.915d0,47.368d0,47.85d0,47.963d0,48d0,&
    48d0,48d0/
    DATA (ScatteringFactors(49,I),I=1,NQ) /&
    0d0,0.015d0,0.062d0,0.138d0,0.242d0,0.372d0,0.527d0,0.898d0,1.334d0,2.316d0,&
    3.344d0,3.852d0,5.07d0,6.207d0,7.276d0,8.297d0,10.244d0,12.083d0,15.444d0,18.489d0,&
    21.288d0,23.779d0,25.906d0,27.691d0,29.203d0,32.229d0,34.634d0,38.255d0,40.774d0,42.605d0,&
    43.917d0,45.003d0,46.321d0,47.06d0,47.526d0,47.863d0,48.328d0,48.835d0,48.959d0,49d0,&
    49d0,49d0/
    DATA (ScatteringFactors(50,I),I=1,NQ) /&
    0d0,0.015d0,0.06d0,0.134d0,0.236d0,0.365d0,0.518d0,0.886d0,1.323d0,2.322d0,&
    3.386d0,3.911d0,5.204d0,6.416d0,7.55d0,8.615d0,10.589d0,12.415d0,15.746d0,18.76d0,&
    21.541d0,24.059d0,26.252d0,28.113d0,29.687d0,32.794d0,35.226d0,38.894d0,41.462d0,43.338d0,&
    44.751d0,45.811d0,47.2d0,47.977d0,48.463d0,48.81d0,49.288d0,49.82d0,49.954d0,50d0,&
    50d0,50d0/
    DATA (ScatteringFactors(51,I),I=1,NQ) /&
    0d0,0.015d0,0.058d0,0.13d0,0.228d0,0.353d0,0.501d0,0.86d0,1.287d0,2.273d0,&
    3.336d0,3.811d0,5.187d0,6.453d0,7.662d0,8.811d0,10.908d0,12.117d0,16.088d0,19.067d0,&
    21.823d0,24.349d0,26.59d0,28.518d0,30.151d0,33.358d0,35.822d0,39.536d0,42.151d0,44.069d0,&
    45.522d0,46.626d0,48.075d0,48.892d0,49.399d0,49.756d0,50.248d0,50.804d0,50.949d0,51d0,&
    51d0,51d0/
    DATA (ScatteringFactors(52,I),I=1,NQ) /&
    0d0,0.014d0,0.056d0,0.125d0,0.22d0,0.341d0,0.485d0,0.837d0,1.261d0,2.255d0,&
    3.349d0,3.907d0,5.284d0,6.61d0,7.876d0,9.076d0,11.26d0,13.171d0,16.466d0,19.401d0,&
    22.134d0,24.655d0,26.927d0,28.912d0,30.613d0,33.918d0,36.422d0,40.181d0,42.84d0,44.798d0,&
    46.29d0,47.431d0,48.945d0,49.804d0,50.333d0,50.702d0,51.207d0,51.787d0,51.943d0,51.999d0,&
    52d0,52d0/
    DATA (ScatteringFactors(53,I),I=1,NQ) /&
    0d0,0.013d0,0.054d0,0.12d0,0.213d0,0.33d0,0.47d0,0.813d0,1.23d0,2.221d0,&
    3.331d0,3.903d0,5.328d0,6.709d0,8.03d0,9.287d0,11.579d0,13.564d0,16.876d0,19.777d0,&
    22.471d0,24.98d0,27.269d0,29.298d0,31.056d0,34.474d0,37.024d0,40.827d0,43.529d0,45.526d0,&
    47.054d0,48.233d0,49.811d0,50.714d0,51.266d0,51.647d0,52.165d0,52.77d0,52.937d0,52.999d0,&
    53d0,53d0/
    DATA (ScatteringFactors(54,I),I=1,NQ) /&
    0d0,0.013d0,0.052d0,0.116d0,0.205d0,0.318d0,0.454d0,0.788d0,1.194d0,2.168d0,&
    3.27d0,3.841d0,5.275d0,6.677d0,8.033d0,9.34d0,11.771d0,13.892d0,17.307d0,20.175d0,&
    22.833d0,25.324d0,27.619d0,29.68d0,31.488d0,35.023d0,37.628d0,41.477d0,44.22d0,46.254d0,&
    47.817d0,49.03d0,50.673d0,51.62d0,52.197d0,52.591d0,53.123d0,53.751d0,53.931d0,53.999d0,&
    54d0,54d0/
    DATA (ScatteringFactors(55,I),I=1,NQ) /&
    0d0,0.027d0,0.105d0,0.231d0,0.396d0,0.594d0,0.815d0,1.296d0,1.793d0,2.782d0,&
    3.794d0,4.32d0,5.672d0,7.023d0,8.339d0,9.615d0,12.035d0,14.217d0,17.753d0,20.612d0,&
    23.228d0,25.691d0,27.981d0,30.064d0,31.914d0,35.565d0,38.232d0,42.129d0,44.912d0,46.981d0,&
    48.517d0,49.824d0,51.53d0,52.523d0,53.127d0,53.534d0,54.081d0,54.732d0,54.924d0,55d0,&
    55d0,55d0/
    DATA (ScatteringFactors(56,I),I=1,NQ) /&
    0d0,0.031d0,0.124d0,0.272d0,0.47d0,0.71d0,0.981d0,1.578d0,2.196d0,3.356d0,&
    4.398d0,4.902d0,6.115d0,7.468d0,8.74d0,9.976d0,12.343d0,14.544d0,18.201d0,21.078d0,&
    23.654d0,26.083d0,28.359d0,30.453d0,32.336d0,36.1d0,38.836d0,42.784d0,45.605d0,47.109d0,&
    49.336d0,50.615d0,52.383d0,53.424d0,54.055d0,54.477d0,55.038d0,55.713d0,55.911d0,56d0,&
    56d0,56d0/
    DATA (ScatteringFactors(57,I),I=1,NQ) /&
    0d0,0.03d0,0.119d0,0.262d0,0.455d0,0.69d0,0.958d0,1.56d0,2.199d0,3.433d0,&
    4.542d0,5.068d0,6.367d0,7.671d0,8.956d0,10.204d0,12.583d0,14.814d0,18.609d0,21.555d0,&
    24.109d0,26.502d0,28.759d0,30.854d0,32.758d0,36.622d0,39.438d0,43.443d0,46.301d0,48.436d0,&
    50.094d0,51.403d0,53.233d0,54.32d0,54.981d0,55.419d0,55.995d0,56.692d0,56.909d0,57d0,&
    57d0,57d0/
    DATA (ScatteringFactors(58,I),I=1,NQ) /&
    0d0,0.029d0,0.116d0,0.256d0,0.445d0,0.675d0,0.938d0,1.53d0,2.162d0,3.39d0,&
    4.499d0,5.025d0,6.324d0,7.634d0,8.93d0,10.19d0,12.592d0,14.844d0,18.713d0,21.737d0,&
    24.334d0,26.75d0,29.025d0,31.143d0,33.083d0,37.075d0,40.001d0,44.104d0,47.006d0,49.175d0,&
    50.86d0,52.191d0,54.083d0,55.211d0,55.906d0,56.361d0,56.951d0,57.671d0,57.901d0,58d0,&
    58d0,58d0/
    DATA (ScatteringFactors(59,I),I=1,NQ) /&
    0d0,0.029d0,0.116d0,0.257d0,0.445d0,0.673d0,0.933d0,1.513d0,2.124d0,3.296d0,&
    4.362d0,4.876d0,6.171d0,7.489d0,8.791d0,10.051d0,12.447d0,14.688d0,18.554d0,21.644d0,&
    24.331d0,26.818d0,29.145d0,31.311d0,33.304d0,37.451d0,40.515d0,44.157d0,47.716d0,49.921d0,&
    51.634d0,52.996d0,54.934d0,56.113d0,56.832d0,57.303d0,57.908d0,58.65d0,58.892d0,59d0,&
    59d0,59d0/
    DATA (ScatteringFactors(60,I),I=1,NQ) /&
    0d0,0.029d0,0.114d0,0.252d0,0.437d0,0.662d0,0.918d0,1.491d0,2.097d0,3.261d0,&
    4.334d0,4.849d0,6.144d0,7.463d0,8.77d0,10.036d0,12.439d0,14.688d0,18.603d0,21.759d0,&
    24.493d0,27.012d0,29.364d0,31.556d0,33.581d0,37.848d0,41.033d0,45.404d0,48.419d0,50.66d0,&
    52.4d0,53.788d0,55.779d0,57.003d0,57.753d0,58.243d0,58.865d0,59.627d0,59.884d0,59.999d0,&
    60d0,60d0/
    DATA (ScatteringFactors(61,I),I=1,NQ) /&
    0d0,0.028d0,0.112d0,0.248d0,0.429d0,0.65d0,0.903d0,1.469d0,2.071d0,3.237d0,&
    4.304d0,4.818d0,6.111d0,7.432d0,8.742d0,10.014d0,12.426d0,14.682d0,18.641d0,21.856d0,&
    24.634d0,27.183d0,29.559d0,31.776d0,33.832d0,38.212d0,41.523d0,46.04d0,49.119d0,51.398d0,&
    53.166d0,54.579d0,56.621d0,57.892d0,58.674d0,59.181d0,59.821d0,60.605d0,60.874d0,60.999d0,&
    61d0,61d0/
    DATA (ScatteringFactors(62,I),I=1,NQ) /&
    0d0,0.028d0,0.11d0,0.243d0,0.422d0,0.64d0,0.888d0,1.448d0,2.045d0,3.207d0,&
    4.272d0,4.786d0,6.016d0,7.395d0,8.708d0,9.984d0,12.401d0,14.661d0,18.652d0,21.919d0,&
    24.741d0,27.325d0,29.73d0,31.975d0,34.062d0,38.554d0,41.993d0,46.669d0,49.816d0,52.136d0,&
    53.932d0,55.369d0,57.461d0,58.777d0,59.592d0,60.119d0,60.777d0,61.581d0,61.864d0,61.999d0,&
    62d0,62d0/
    DATA (ScatteringFactors(63,I),I=1,NQ) /&
    0d0,0.027d0,0.108d0,0.239d0,0.415d0,0.629d0,0.874d0,1.428d0,2.02d0,3.176d0,&
    4.24d0,4.754d0,6.039d0,7.359d0,8.611d0,9.954d0,12.374d0,14.643d0,18.662d0,21.976d0,&
    24.84d0,27.456d0,29.889d0,32.159d0,34.275d0,38.866d0,42.432d0,47.281d0,50.508d0,52.872d0,&
    54.698d0,56.159d0,58.298d0,59.66d0,60.508d0,61.056d0,61.732d0,62.557d0,62.854d0,62.999d0,&
    63d0,63d0/
    DATA (ScatteringFactors(64,I),I=1,NQ) /&
    0d0,0.026d0,0.104d0,0.23d0,0.4d0,0.609d0,0.85d0,1.401d0,2.001d0,3.2d0,&
    4.307d0,4.831d0,6.12d0,7.42d0,8.126d0,10.007d0,12.452d0,14.749d0,18.869d0,22.269d0,&
    25.155d0,27.766d0,30.194d0,32.463d0,34.583d0,39.224d0,42.884d0,47.891d0,51.193d0,53.6d0,&
    55.456d0,56.942d0,59.13d0,60.538d0,61.42d0,61.99d0,62.687d0,63.533d0,63.843d0,63.999d0,&
    64d0,64d0/
    DATA (ScatteringFactors(65,I),I=1,NQ) /&
    0d0,0.026d0,0.102d0,0.226d0,0.394d0,0.6d0,0.838d0,1.383d0,1.978d0,3.173d0,&
    4.278d0,4.802d0,6.088d0,7.387d0,8.694d0,9.979d0,12.436d0,14.748d0,18.914d0,22.377d0,&
    25.315d0,27.961d0,30.414d0,32.703d0,34.842d0,39.553d0,43.315d0,48.494d0,51.88d0,54.335d0,&
    56.222d0,57.731d0,59.964d0,61.415d0,62.332d0,62.924d0,63.641d0,64.508d0,64.832d0,64.999d0,&
    65d0,65d0/
    DATA (ScatteringFactors(66,I),I=1,NQ) /&
    0d0,0.026d0,0.103d0,0.227d0,0.395d0,0.6d0,0.836d0,1.37d0,1.946d0,3.087d0,&
    4.144d0,4.654d0,5.931d0,7.245d0,8.569d0,9.867d0,12.335d0,14.647d0,18.199d0,22.282d0,&
    25.284d0,28.001d0,30.512d0,32.845d0,35.021d0,39.832d0,43.713d0,49.082d0,52.567d0,55.013d0,&
    56.993d0,58.526d0,60.8d0,62.293d0,63.244d0,63.858d0,64.596d0,65.483d0,65.82d0,65.999d0,&
    66d0,66d0/
    DATA (ScatteringFactors(67,I),I=1,NQ) /&
    0d0,0.025d0,0.101d0,0.224d0,0.389d0,0.591d0,0.824d0,1.352d0,1.923d0,3.058d0,&
    4.113d0,4.621d0,5.894d0,7.204d0,8.528d0,9.83d0,12.308d0,14.628d0,18.81d0,22.338d0,&
    25.383d0,28.137d0,30.678d0,33.039d0,35.24d0,40.121d0,44.101d0,49.656d0,53.241d0,55.803d0,&
    57.756d0,59.314d0,61.631d0,63.166d0,64.151d0,64.189d0,65.55d0,66.458d0,66.808d0,66.999d0,&
    67d0,67d0/
    DATA (ScatteringFactors(68,I),I=1,NQ) /&
    0d0,0.025d0,0.099d0,0.22d0,0.383d0,0.582d0,0.812d0,1.335d0,1.901d0,3.03d0,&
    4.082d0,4.588d0,5.856d0,7.164d0,8.488d0,9.192d0,12.279d0,14.609d0,18.817d0,22.387d0,&
    25.471d0,28.258d0,30.83d0,33.216d0,35.44d0,40.387d0,44.463d0,50.21d0,53.906d0,56.528d0,&
    58.518d0,60.101d0,62.46d0,64.037d0,65.057d0,65.119d0,66.504d0,67.432d0,67.795d0,67.999d0,&
    68d0,68d0/
    DATA (ScatteringFactors(69,I),I=1,NQ) /&
    0d0,0.025d0,0.098d0,0.217d0,0.317d0,0.514d0,0.801d0,1.318d0,1.579d0,3.002d0,&
    4.051d0,4.555d0,5.819d0,7.122d0,8.445d0,9.751d0,12.245d0,14.581d0,18.809d0,22.412d0,&
    25.532d0,28.352d0,30.954d0,33.369d0,35.62d0,40.637d0,44.808d0,50.752d0,54.565d0,57.25d0,&
    59.278d0,60.887d0,63.288d0,64.905d0,65.96d0,66.647d0,67.457d0,68.405d0,68.783d0,68.999d0,&
    69d0,69d0/
    DATA (ScatteringFactors(70,I),I=1,NQ) /&
    0d0,0.024d0,0.096d0,0.214d0,0.372d0,0.566d0,0.79d0,1.301d0,1.859d0,2.975d0,&
    4.02d0,4.527d0,5.782d0,7.088d0,8.403d0,9.72d0,12.211d0,14.565d0,18.81d0,22.443d0,&
    25.596d0,28.446d0,31.077d0,33.519d0,35.794d0,40.869d0,45.131d0,51.27d0,55.21d0,57.966d0,&
    60.035d0,61.672d0,64.115d0,65.771d0,66.861d0,67.573d0,68.409d0,69.378d0,69.769d0,69.999d0,&
    70d0,70d0/
    DATA (ScatteringFactors(71,I),I=1,NQ) /&
    0d0,0.024d0,0.093d0,0.207d0,0.362d0,0.553d0,0.774d0,1.286d0,1.854d0,3.019d0,&
    4.114d0,4.634d0,5.9d0,7.176d0,8.471d0,9.762d0,12.254d0,14.607d0,18.919d0,22.64d0,&
    25.838d0,28.699d0,31.332d0,33.778d0,36.059d0,41.159d0,45.471d0,51.782d0,55.846d0,58.672d0,&
    60.785d0,62.451d0,64.936d0,66.633d0,67.759d0,68.497d0,69.361d0,70.351d0,70.755d0,70.998d0,&
    71d0,71d0/
    DATA (ScatteringFactors(72,I),I=1,NQ) /&
    0d0,0.023d0,0.09d0,0.2d0,0.35d0,0.535d0,0.751d0,1.256d0,1.823d0,3.009d0,&
    4.142d0,4.679d0,5.97d0,7.245d0,8.532d0,9.815d0,12.305d0,14.667d0,19.037d0,22.85d0,&
    26.11d0,28.989d0,31.624d0,34.07d0,36.354d0,41.465d0,45.817d0,52.285d0,56.415d0,59.375d0,&
    61.532d0,63.227d0,65.755d0,67.492d0,68.653d0,69.419d0,70.311d0,71.323d0,71.741d0,71.998d0,&
    72d0,72d0/
    DATA (ScatteringFactors(73,I),I=1,NQ) /&
    0d0,0.022d0,0.087d0,0.193d0,0.337d0,0.517d0,0.728d0,1.222d0,1.783d0,2.974d0,&
    4.128d0,4.616d0,5.983d0,7.257d0,8.535d0,9.811d0,12.304d0,14.683d0,19.123d0,23.042d0,&
    26.385d0,29.298d0,31.938d0,34.385d0,36.669d0,41.786d0,46.168d0,52.777d0,57.095d0,60.072d0,&
    62.277d0,64.002d0,66.572d0,68.348d0,69.545d0,70.339d0,71.261d0,72.296d0,72.726d0,72.998d0,&
    73d0,73d0/
    DATA (ScatteringFactors(74,I),I=1,NQ) /&
    0d0,0.021d0,0.083d0,0.186d0,0.326d0,0.5d0,0.705d0,1.188d0,1.741d0,2.929d0,&
    4.095d0,4.649d0,5.967d0,7.238d0,8.506d0,9.773d0,12.268d0,14.611d0,19.193d0,23.224d0,&
    26.662d0,29.621d0,32.213d0,34.719d0,37.003d0,42.123d0,46.526d0,53.262d0,57.707d0,60.764d0,&
    63.018d0,64.775d0,67.388d0,69.201d0,70.435d0,71.256d0,72.21d0,73.261d0,73.711d0,73.998d0,&
    74d0,74d0/
    DATA (ScatteringFactors(75,I),I=1,NQ) /&
    0d0,0.02d0,0.081d0,0.18d0,0.315d0,0.484d0,0.684d0,1.156d0,1.7d0,2.881d0,&
    4.054d0,4.615d0,5.945d0,7.219d0,8.478d0,9.739d0,12.222d0,14.635d0,19.217d0,23.361d0,&
    26.914d0,29.945d0,32.624d0,35.075d0,37.357d0,42.474d0,46.895d0,53.739d0,58.311d0,61.452d0,&
    63.756d0,65.546d0,68.201d0,70.053d0,71.322d0,72.172d0,73.158d0,74.239d0,74.696d0,74.998d0,&
    75d0,75d0/
    DATA (ScatteringFactors(76,I),I=1,NQ) /&
    0d0,0.02d0,0.078d0,0.174d0,0.306d0,0.47d0,0.664d0,1.126d0,1.661d0,2.837d0,&
    4.02d0,4.59d0,5.947d0,7.244d0,8.521d0,9.791d0,12.294d0,14.715d0,19.321d0,23.533d0,&
    27.177d0,30.274d0,32.981d0,35.437d0,37.719d0,42.839d0,47.266d0,54.211d0,58.908d0,62.134d0,&
    64.491d0,66.314d0,69.013d0,70.902d0,72.206d0,73.085d0,74.106d0,75.21d0,75.68d0,75.998d0,&
    76d0,76d0/
    DATA (ScatteringFactors(77,I),I=1,NQ) /&
    0d0,0.019d0,0.076d0,0.169d0,0.297d0,0.457d0,0.646d0,1.098d0,1.624d0,2.791d0,&
    3.98d0,4.558d0,5.938d0,7.258d0,8.554d0,9.841d0,12.367d0,14.803d0,19.427d0,23.694d0,&
    27.429d0,30.602d0,33.349d0,35.817d0,38.1d0,43.211d0,47.651d0,54.678d0,59.496d0,62.811d0,&
    65.221d0,67.081d0,69.824d0,71.149d0,73.088d0,73.997d0,75.052d0,76.181d0,76.663d0,76.997d0,&
    77d0,77d0/
    DATA (ScatteringFactors(78,I),I=1,NQ) /&
    0d0,0.016d0,0.064d0,0.142d0,0.25d0,0.385d0,0.546d0,0.933d0,1.39d0,2.435d0,&
    3.556d0,4.123d0,5.532d0,6.917d0,8.27d0,9.589d0,12.13d0,14.583d0,19.331d0,23.766d0,&
    27.638d0,30.91d0,33.713d0,36.204d0,38.491d0,43.61d0,48.049d0,55.142d0,60.077d0,63.483d0,&
    65.953d0,67.847d0,70.633d0,72.593d0,73.968d0,74.906d0,75.991d0,77.152d0,77.647d0,77.997d0,&
    78d0,78d0/
    DATA (ScatteringFactors(79,I),I=1,NQ) /&
    0d0,0.016d0,0.062d0,0.138d0,0.243d0,0.375d0,0.531d0,0.909d0,1.357d0,2.388d0,&
    3.502d0,4.068d0,5.483d0,6.875d0,8.236d0,9.56d0,12.102d0,14.551d0,19.321d0,23.828d0,&
    27.815d0,31.197d0,34.072d0,36.597d0,38.896d0,44.011d0,48.456d0,55.605d0,60.651d0,64.149d0,&
    66.678d0,68.61d0,71.441d0,73.436d0,74.845d0,75.813d0,76.941d0,78.123d0,78.63d0,78.997d0,&
    79d0,79d0/
    DATA (ScatteringFactors(80,I),I=1,NQ) /&
    0d0,0.017d0,0.069d0,0.155d0,0.273d0,0.421d0,0.597d0,1.02d0,1.519d0,2.647d0,&
    3.828d0,4.409d0,5.815d0,7.159d0,8.471d0,9.766d0,12.31d0,14.772d0,19.489d0,23.96d0,&
    28.005d0,31.48d0,34.429d0,36.994d0,39.307d0,44.422d0,48.866d0,56.066d0,61.218d0,64.808d0,&
    67.399d0,69.37d0,72.248d0,74.278d0,75.72d0,76.718d0,77.884d0,79.093d0,79.612d0,79.997d0,&
    80d0,80d0/
    DATA (ScatteringFactors(81,I),I=1,NQ) /&
    0d0,0.018d0,0.073d0,0.162d0,0.285d0,0.439d0,0.621d0,1.059d0,1.573d0,2.737d0,&
    3.965d0,4.575d0,6.056d0,7.46d0,8.8d0,10.096d0,12.608d0,15.041d0,19.702d0,24.127d0,&
    28.197d0,31.15d0,34.117d0,37.39d0,39.724d0,44.844d0,49.283d0,56.529d0,61.179d0,65.461d0,&
    68.116d0,70.129d0,73.053d0,75.117d0,76.592d0,77.62d0,78.826d0,80.063d0,80.594d0,80.996d0,&
    81d0,81d0/
    DATA (ScatteringFactors(82,I),I=1,NQ) /&
    0d0,0.018d0,0.072d0,0.16d0,0.281d0,0.434d0,0.615d0,1.053d0,1.571d0,2.755d0,&
    4.022d0,4.658d0,6.212d0,7.694d0,9.099d0,10.437d0,12.964d0,15.368d0,19.964d0,24.331d0,&
    28.408d0,32.02d0,35.121d0,37.79d0,40.151d0,45.277d0,49.713d0,56.993d0,62.335d0,66.108d0,&
    68.829d0,70.885d0,73.858d0,75.956d0,77.463d0,78.521d0,79.767d0,81.033d0,81.576d0,81.996d0,&
    82d0,82d0/
    DATA (ScatteringFactors(83,I),I=1,NQ) /&
    0d0,0.017d0,0.07d0,0.156d0,0.274d0,0.424d0,0.601d0,1.03d0,1.54d0,2.715d0,&
    3.981d0,4.622d0,6.21d0,7.757d0,9.25d0,10.676d0,13.314d0,15.734d0,20.268d0,24.586d0,&
    28.642d0,32.294d0,35.462d0,38.19d0,40.585d0,45.722d0,50.156d0,57.459d0,62.887d0,66.151d0,&
    69.538d0,71.638d0,74.662d0,76.793d0,78.332d0,79.419d0,80.706d0,82.002d0,82.558d0,82.996d0,&
    83d0,83d0/
    DATA (ScatteringFactors(84,I),I=1,NQ) /&
    0d0,0.017d0,0.068d0,0.152d0,0.267d0,0.414d0,0.588d0,1.013d0,1.522d0,2.712d0,&
    4.011d0,4.682d0,6.335d0,7.945d0,9.496d0,10.976d0,13.694d0,16.133d0,20.606d0,24.863d0,&
    28.893d0,32.568d0,35.195d0,38.581d0,41.016d0,46.111d0,50.602d0,57.93d0,63.435d0,67.387d0,&
    70.242d0,72.389d0,75.465d0,77.628d0,79.199d0,80.315d0,81.644d0,82.972d0,83.539d0,83.995d0,&
    84d0,84d0/
    DATA (ScatteringFactors(85,I),I=1,NQ) /&
    0d0,0.018d0,0.066d0,0.148d0,0.26d0,0.403d0,0.574d0,0.992d0,1.497d0,2.691d0,&
    4.019d0,4.702d0,6.408d0,8.016d0,9.687d0,11.229d0,14.056d0,16.546d0,20.918d0,25.171d0,&
    29.169d0,32.854d0,36.127d0,38.971d0,41.45d0,46.643d0,51.062d0,58.404d0,63.979d0,68.018d0,&
    70.943d0,73.137d0,76.266d0,78.463d0,80.064d0,81.21d0,82.588d0,83.941d0,84.52d0,84.995d0,&
    85d0,85d0/
    DATA (ScatteringFactors(86,I),I=1,NQ) /&
    0d0,0.016d0,0.064d0,0.144d0,0.254d0,0.393d0,0.56d0,0.969d0,1.465d0,2.645d0,&
    3.968d0,4.652d0,6.371d0,8.065d0,9.722d0,11.329d0,14.317d0,16.929d0,21.382d0,25.507d0,&
    29.469d0,33.153d0,36.46d0,39.358d0,41.885d0,47.118d0,51.533d0,58.884d0,64.521d0,68.646d0,&
    71.639d0,73.882d0,77.067d0,79.296d0,80.927d0,82.102d0,83.515d0,84.91d0,85.501d0,85.995d0,&
    86d0,86d0/
    DATA (ScatteringFactors(87,I),I=1,NQ) /&
    0d0,0.031d0,0.122d0,0.268d0,0.461d0,0.69d0,0.945d0,1.5d0,2.078d0,3.252d0,&
    4.482d0,5.124d0,6.711d0,8.415d0,10.03d0,11.609d0,14.614d0,17.303d0,21.816d0,25.878d0,&
    29.791d0,33.463d0,36.795d0,39.74d0,42.315d0,47.602d0,52.013d0,59.368d0,65.062d0,69.268d0,&
    72.331d0,74.624d0,77.866d0,80.128d0,81.789d0,82.992d0,84.449d0,85.879d0,86.481d0,86.994d0,&
    87d0,87d0/
    DATA (ScatteringFactors(88,I),I=1,NQ) /&
    0d0,0.036d0,0.144d0,0.317d0,0.547d0,0.823d0,1.135d0,1.817d0,2.52d0,3.848d0,&
    5.082d0,5.696d0,7.266d0,8.858d0,10.427d0,11.964d0,14.934d0,17.672d0,22.27d0,26.283d0,&
    30.139d0,33.79d0,37.136d0,40.121d0,42.744d0,48.093d0,52.502d0,59.858d0,65.602d0,69.885d0,&
    73.018d0,75.363d0,78.664d0,80.96d0,82.649d0,83.881d0,85.381d0,86.847d0,87.461d0,87.994d0,&
    88d0,88d0/
    DATA (ScatteringFactors(89,I),I=1,NQ) /&
    0d0,0.035d0,0.14d0,0.308d0,0.534d0,0.808d0,1.119d0,1.814d0,2.548d0,3.962d0,&
    5.265d0,5.898d0,7.488d0,9.092d0,10.674d0,12.22d0,15.203d0,17.992d0,22.715d0,26.712d0,&
    30.511d0,34.133d0,37.483d0,40.498d0,43.164d0,48.588d0,53d0,60.356d0,66.142d0,70.497d0,&
    73.702d0,76.099d0,79.461d0,81.79d0,83.508d0,84.767d0,86.311d0,87.816d0,88.441d0,88.993d0,&
    89d0,89d0/
    DATA (ScatteringFactors(90,I),I=1,NQ) /&
    0d0,0.034d0,0.135d0,0.299d0,0.519d0,0.787d0,1.095d0,1.792d0,2.54d0,4.016d0,&
    5.383d0,6.039d0,7.663d0,9.289d0,10.889d0,12.446d0,15.442d0,18.271d0,23.139d0,27.162d0,&
    30.909d0,34.496d0,37.842d0,40.879d0,43.582d0,49.085d0,53.51d0,60.86d0,66.682d0,71.107d0,&
    74.381d0,76.832d0,80.257d0,82.619d0,84.365d0,85.652d0,87.241d0,88.784d0,89.421d0,89.993d0,&
    90d0,90d0/
    DATA (ScatteringFactors(91,I),I=1,NQ) /&
    0d0,0.034d0,0.134d0,0.296d0,0.514d0,0.778d0,1.081d0,1.76d0,2.485d0,3.904d0,&
    5.228d0,5.873d0,7.498d0,9.143d0,10.767d0,12.345d0,15.364d0,18.189d0,23.062d0,27.163d0,&
    30.988d0,34.631d0,38.032d0,41.137d0,43.916d0,49.565d0,54.03d0,61.374d0,67.219d0,71.707d0,&
    75.05d0,77.558d0,81.053d0,83.451d0,85.224d0,86.537d0,88.17d0,89.752d0,90.4d0,90.999d0,&
    91d0,91d0/
    DATA (ScatteringFactors(92,I),I=1,NQ) /&
    0d0,0.033d0,0.132d0,0.291d0,0.505d0,0.766d0,1.064d0,1.737d0,2.457d0,3.875d0,&
    5.203d0,5.851d0,7.482d0,9.136d0,10.771d0,12.359d0,15.387d0,18.222d0,23.157d0,27.321d0,&
    31.178d0,34.842d0,38.27d0,41.418d0,44.254d0,50.036d0,54.551d0,61.894d0,67.76d0,72.306d0,&
    75.717d0,78.282d0,81.846d0,84.28d0,86.081d0,87.42d0,89.097d0,90.719d0,91.379d0,91.992d0,&
    92d0,92d0/
    DATA (ScatteringFactors(93,I),I=1,NQ) /&
    0d0,0.033d0,0.13d0,0.287d0,0.499d0,0.756d0,1.052d0,1.72d0,2.438d0,3.86d0,&
    5.195d0,5.846d0,7.484d0,9.143d0,10.784d0,12.375d0,15.403d0,18.242d0,23.23d0,27.456d0,&
    31.349d0,35.038d0,38.495d0,41.683d0,44.573d0,50.496d0,55.072d0,62.42d0,68.304d0,72.902d0,&
    76.38d0,79.001d0,82.637d0,85.108d0,86.936d0,88.302d0,90.023d0,91.687d0,92.358d0,92.991d0,&
    93d0,93d0/
    DATA (ScatteringFactors(94,I),I=1,NQ) /&
    0d0,0.033d0,0.131d0,0.29d0,0.503d0,0.762d0,1.057d0,1.719d0,2.424d0,3.811d0,&
    5.124d0,5.774d0,7.422d0,9.089d0,10.72d0,12.288d0,15.254d0,18.039d0,23.007d0,27.315d0,&
    31.301d0,35.059d0,38.58d0,41.84d0,44.812d0,50.924d0,55.593d0,62.956d0,68.849d0,73.494d0,&
    77.035d0,79.714d0,83.426d0,85.937d0,87.792d0,89.183d0,90.948d0,92.655d0,93.337d0,93.99d0,&
    94d0,94d0/
    DATA (ScatteringFactors(95,I),I=1,NQ) /&
    0d0,0.033d0,0.129d0,0.286d0,0.496d0,0.751d0,1.043d0,1.699d0,2.399d0,3.782d0,&
    5.094d0,5.745d0,7.392d0,9.065d0,10.7d0,12.277d0,15.246d0,18.033d0,23.027d0,27.38d0,&
    31.401d0,35.19d0,38.74d0,42.04d0,45.067d0,51.338d0,56.106d0,63.495d0,69.399d0,74.088d0,&
    77.688d0,80.425d0,84.213d0,86.763d0,88.646d0,90.062d0,91.87d0,93.622d0,94.316d0,94.99d0,&
    95d0,95d0/
    DATA (ScatteringFactors(96,I),I=1,NQ) /&
    0d0,0.032d0,0.124d0,0.276d0,0.479d0,0.728d0,1.014d0,1.667d0,2.381d0,3.192d0,&
    5.134d0,5.783d0,7.432d0,9.116d0,10.744d0,12.355d0,15.38d0,18.189d0,23.222d0,27.609d0,&
    31.662d0,35.48d0,39.058d0,42.384d0,45.435d0,51.155d0,56.572d0,64.037d0,69.951d0,74.697d0,&
    78.34d0,81.132d0,84.998d0,87.587d0,89.497d0,90.944d0,92.791d0,94.589d0,95.294d0,95.989d0,&
    96d0,96d0/
    DATA (ScatteringFactors(97,I),I=1,NQ) /&
    0d0,0.031d0,0.123d0,0.272d0,0.473d0,0.719d0,1.003d0,1.649d0,2.36d0,3.766d0,&
    5.11d0,5.159d0,7.41d0,9.1d0,10.741d0,12.36d0,15.408d0,18.246d0,23.331d0,27.763d0,&
    31.857d0,35.115d0,39.33d0,42.69d0,45.772d0,52.158d0,57.033d0,64.588d0,70.508d0,75.304d0,&
    78.986d0,81.836d0,85.78d0,88.411d0,90.348d0,91.825d0,93.711d0,95.555d0,96.272d0,96.988d0,&
    96.999d0,97d0/
    DATA (ScatteringFactors(98,I),I=1,NQ) /&
    0d0,0.032d0,0.124d0,0.274d0,0.475d0,0.721d0,1.003d0,1.64d0,2.329d0,3.69d0,&
    4.996d0,5.644d0,7.29d0,8.994d0,10.647d0,12.264d0,15.308d0,18.181d0,23.332d0,27.82d0,&
    31.967d0,35.874d0,39.535d0,42.938d0,46.059d0,52.527d0,57.476d0,65.147d0,71.01d0,75.91d0,&
    79.626d0,82.534d0,86.559d0,89.236d0,91.2d0,92.146d0,94.631d0,96.522d0,97.25d0,97.988d0,&
    97.999d0,98d0/
    DATA (ScatteringFactors(99,I),I=1,NQ) /&
    0d0,0.031d0,0.122d0,0.27d0,0.469d0,0.712d0,0.99d0,1.62d0,2.305d0,3.658d0,&
    4.96d0,5.606d0,7.242d0,8.954d0,10.609d0,12.233d0,15.292d0,18.195d0,23.397d0,27.931d0,&
    32.119d0,36.066d0,39.764d0,43.201d0,46.354d0,52.886d0,57.915d0,65.707d0,71.635d0,76.517d0,&
    80.266d0,83.231d0,87.335d0,90.057d0,92.05d0,93.586d0,95.548d0,97.488d0,98.228d0,98.987d0,&
    98.999d0,99d0/
    DATA (ScatteringFactors(100,I),I=1,NQ) /&
    0d0,0.031d0,0.12d0,0.266d0,0.463d0,0.702d0,0.978d0,1.602d0,2.282d0,3.627d0,&
    4.923d0,5.567d0,7.203d0,8.91d0,10.567d0,12.199d0,15.272d0,18.202d0,23.453d0,28.029d0,&
    32.257d0,36.24d0,39.973d0,43.442d0,46.625d0,53.218d0,58.337d0,66.27d0,72.204d0,77.125d0,&
    80.904d0,83.926d0,88.109d0,90.817d0,92.899d0,94.465d0,96.464d0,98.453d0,99.206d0,99.986d0,&
    99.999d0,100d0/
    REAL(KIND(1d0))::sintholam,temp
    IF(Z.GT.ZMAX.OR.Z.LE.0)THEN
       WRITE(*,*)"ERROR: the atomic form factor is not available with Z=",Z
       STOP
    ENDIF
    IF(q.LT.0d0)THEN
       WRITE(*,*)"ERROR: q < 0 !"
       STOP
    ENDIF
    ! GeV to angstrom^{-1}
    sintholam=q/FOURPI*1d5/0.1973d0
    IF(sintholam.GT.QO4PI(NQ))THEN
       AtomicIncoherentStructureFunction=DBLE(Z)
       RETURN
    ELSEIF(sintholam.EQ.0d0)THEN
       AtomicIncoherentStructureFunction=0d0
       RETURN
    ENDIF
    CALL SPLINE_INTERPOLATE(QO4PI,ScatteringFactors(Z,1:NQ),NQ,sintholam,temp)
    temp=MAX(temp,0d0)
    AtomicIncoherentStructureFunction=temp
    RETURN
  END FUNCTION AtomicIncoherentStructureFunction

  FUNCTION GetASymbol(nuclearA,nuclearZ)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ
    CHARACTER(len=7)::GetASymbol,GetASymbol2
    CHARACTER(len=8)::fmt
    CHARACTER(len=5)::x1
    SELECT CASE(nuclearZ)
    CASE(1)
       GetASymbol="H      "
    CASE(2)
       GetASymbol="He     "
    CASE(3)
       GetASymbol="Li     "
    CASE(4)
       GetASymbol="Be     "
    CASE(5)
       GetASymbol="B      "
    CASE(6)
       GetASymbol="C      "
    CASE(7)
       GetASymbol="N      "
    CASE(8)
       GetASymbol="O      "
    CASE(9)
       GetASymbol="F      "
    CASE(10)
       GetASymbol="Ne     "
    CASE(11)
       GetASymbol="Na     "
    CASE(12)
       GetASymbol="Mg     "
    CASE(13)
       GetASymbol="Al     "
    CASE(14)
       GetASymbol="Si     "
    CASE(15)
       GetASymbol="P      "
    CASE(16)
       GetASymbol="S      "
    CASE(17)
       GetASymbol="Cl     "
    CASE(18)
       GetASymbol="Ar     "
    CASE(19)
       GetASymbol="K      "
    CASE(20)
       GetASymbol="Ca     "
    CASE(21)
       GetASymbol="Sc     "
    CASE(22)
       GetASymbol="Ti     "
    CASE(23)
       GetASymbol="V      "
    CASE(24)
       GetASymbol="Cr     "
    CASE(25)
       GetASymbol="Mn     "
    CASE(26)
       GetASymbol="Fe     "
    CASE(27)
       GetASymbol="Co     "
    CASE(28)
       GetASymbol="Ni     "
    CASE(29)
       GetASymbol="Cu     "
    CASE(30)
       GetASymbol="Zn     "
    CASE(31)
       GetASymbol="Ga     "
    CASE(32)
       GetASymbol="Ge     "
    CASE(33)
       GetASymbol="As     "
    CASE(34)
       GetASymbol="Se     "
    CASE(35)
       GetASymbol="Br     "
    CASE(36)
       GetASymbol="Kr     "
    CASE(37)
       GetASymbol="Rb     "
    CASE(38)
       GetASymbol="Sr     "
    CASE(39)
       GetASymbol="Y      "
    CASE(40)
       GetASymbol="Zr     "
    CASE(41)
       GetASymbol="Nb     "
    CASE(42)
       GetASymbol="Mo     "
    CASE(43)
       GetASymbol="Tc     "
    CASE(44)
       GetASymbol="Ru     "
    CASE(45)
       GetASymbol="Rh     "
    CASE(46)
       GetASymbol="Pd     "
    CASE(47)
       GetASymbol="Ag     "
    CASE(48)
       GetASymbol="Cd     "
    CASE(49)
       GetASymbol="In     "
    CASE(50)
       GetASymbol="Sn     "
    CASE(51)
       GetASymbol="Sb     "
    CASE(52)
       GetASymbol="Te     "
    CASE(53)
       GetASymbol="I      "
    CASE(54)
       GetASymbol="Xe     "
    CASE(55)
       GetASymbol="Cs     "
    CASE(56)
       GetASymbol="Ba     "
    CASE(57)
       GetASymbol="La     "
    CASE(58)
       GetASymbol="Ce     "
    CASE(59)
       GetASymbol="Pr     "
    CASE(60)
       GetASymbol="Nd     "
    CASE(61)
       GetASymbol="Pm     "
    CASE(62)
       GetASymbol="Sm     "
    CASE(63)
       GetASymbol="Eu     "
    CASE(64)
       GetASymbol="Gd     "
    CASE(65)
       GetASymbol="Tb     "
    CASE(66)
       GetASymbol="Dy     "
    CASE(67)
       GetASymbol="Ho     "
    CASE(68)
       GetASymbol="Er     "
    CASE(69)
       GetASymbol="Tm     "
    CASE(70)
       GetASymbol="Yb     "
    CASE(71)
       GetASymbol="Lu     "
    CASE(72)
       GetASymbol="Hf     "
    CASE(73)
       GetASymbol="Ta     "
    CASE(74)
       GetASymbol="W      "
    CASE(75)
       GetASymbol="Re     "
    CASE(76)
       GetASymbol="Os     "
    CASE(77)
       GetASymbol="Ir     "
    CASE(78)
       GetASymbol="Pt     "
    CASE(79)
       GetASymbol="Au     "
    CASE(80)
       GetASymbol="Hg     "
    CASE(81)
       GetASymbol="Tl     "
    CASE(82)
       GetASymbol="Pb     "
    CASE(83)
       GetASymbol="Bi     "
    CASE(84)
       GetASymbol="Po     "
    CASE(85)
       GetASymbol="At     "
    CASE(86)
       GetASymbol="Rn     "
    CASE(87)
       GetASymbol="Fr     "
    CASE(88)
       GetASymbol="Ra     "
    CASE(89)
       GetASymbol="Ac     "
    CASE(90)
       GetASymbol="Th     "
    CASE(91)
       GetASymbol="Pa     "
    CASE(92)
       GetASymbol="U      "
    CASE(93)
       GetASymbol="Np     "
    CASE(94)
       GetASymbol="Pu     "
    CASE(95)
       GetASymbol="Am     "
    CASE(96)
       GetASymbol="Cm     "
    CASE(97)
       GetASymbol="Bk     "
    CASE(98)
       GetASymbol="Cf     "
    CASE(99)
       GetASymbol="Es     "
    CASE(100)
       GetASymbol="Fm     "
    CASE(101)
       GetASymbol="Md     "
    CASE(102)
       GetASymbol="No     "
    CASE(103)
       GetASymbol="Lr     "
    CASE(104)
       GetASymbol="Rf     "
    CASE(105)
       GetASymbol="Db     "
    CASE(106)
       GetASymbol="Sg     "
    CASE(107)
       GetASymbol="Bh     "
    CASE(108)
       GetASymbol="Hs     "
    CASE(109)
       GetASymbol="Mt     "
    CASE(110)
       GetASymbol="Ds     "
    CASE(111)
       GetASymbol="Rg     "
    CASE(112)
       GetASymbol="Cn     "
    CASE(113)
       GetASymbol="Nh     "
    CASE(114)
       GetASymbol="Fl     "
    CASE(115)
       GetASymbol="Mc     "
    CASE(116)
       GetASymbol="Lv     "
    CASE(117)
       GetASymbol="Ts     "
    CASE(118)
       GetASymbol="Og     "
    CASE(119)
       GetASymbol="Uue    "
    CASE(120)
       GetASymbol="Ubn    "
    CASE(121)
       GetASymbol="Ubu    "
    CASE(122)
       GetASymbol="Ubb    "
    CASE(123)
       GetASymbol="Mu     "
    CASE(124)
       GetASymbol="Ubq    "
    CASE DEFAULT
       WRITE(*,*)"ERROR:Unknown the atomic number Z of nuclear = ",NuclearZ
       STOP
    END SELECT
    GetASymbol2=GetASymbol
    fmt='(I5)'
    WRITE(x1,fmt)nuclearA
    x1=adjustl(x1)
    GetASymbol=TRIM(GetASymbol2)//TRIM(x1)
    RETURN
  END FUNCTION GetASymbol
  
END MODULE OpticalGlauber_Geometry
