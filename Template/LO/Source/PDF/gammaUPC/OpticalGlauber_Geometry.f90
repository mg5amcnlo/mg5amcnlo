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
    INTEGER,PARAMETER::data_len=41
    CHARACTER(len=5),DIMENSION(data_len)::ion_names=(/'H2   ','Li7  ','Be9  ','B10  ','B11  ','C13  ',&
         'C14  ','N14  ','N15  ','O16  ','Ne20 ','Mg24 ','Mg25 ','Al27 ','Si28 ',&
         'Si29 ','Si30 ','P31  ','Cl35 ','Cl37 ','Ar40 ','K39  ','Ca40 ','Ca48 ',&
         'Ni58 ','Ni60 ','Ni61 ','Ni62 ','Ni64 ','Cu63 ','Kr78 ','Ag110','Sb122','Xe129',&
         'Xe132','Nd142','Er166','W186 ','Au197','Pb207','Pb208'/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_A=(/2d0,7d0,9d0,10d0,11d0,13d0,&
         14d0,14d0,15d0,16d0,20d0,24d0,25d0,27d0,28d0,&
         29d0,30d0,31d0,35d0,37d0,40d0,39d0,40d0,48d0,&
         58d0,60d0,61d0,62d0,64d0,63d0,78d0,110d0,122d0,129d0,&
         132d0,142d0,166d0,186d0,197d0,207d0,208d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_Z=(/1d0,3d0,4d0,5d0,5d0,6d0,&
         6d0,7d0,7d0,8d0,10d0,12d0,12d0,13d0,14d0,&
         14d0,14d0,15d0,17d0,17d0,18d0,19d0,20d0,20d0,&
         28d0,28d0,28d0,28d0,28d0,29d0,36d0,47d0,51d0,54d0,&
         54d0,60d0,68d0,74d0,79d0,82d0,82d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_R=(/0.01d0,1.77d0,1.791d0,1.71d0,1.69d0,1.635d0,&
         1.73d0,2.570d0,2.334d0,2.608d0,2.791d0,3.108d0,3.22d0,3.07d0,3.340d0,&
         3.338d0,3.338d0,3.369d0,3.476d0,3.554d0,3.766d0,3.743d0,3.766d0,3.7369d0,&
         4.3092d0,4.4891d0,4.4024d0,4.4425d0,4.5211d0,4.214d0,4.5d0,5.33d0,5.32d0,5.36d0,&
         5.4d0,5.6135d0,5.98d0,6.58d0,6.38d0,6.62d0,6.624d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_aa=(/0.5882d0,0.327d0,0.611d0,0.837d0,0.811d0,1.403d0,&
         1.38d0,0.5052d0,0.498d0,0.513d0,0.698d0,0.607d0,0.58d0,0.519d0,0.580d0,&
         0.547d0,0.547d0,0.582d0,0.599d0,0.588d0,0.586d0,0.595d0,0.586d0,0.5245d0,&
         0.5169d0,0.5369d0,0.5401d0,0.5386d0,0.5278d0,0.586d0,0.5d0,0.535d0,0.57d0,0.59d0,&
         0.61d0,0.5868d0,0.446d0,0.480d0,0.535d0,0.546d0,0.549d0/)
    REAL(KIND(1d0)),DIMENSION(data_len)::ion_w=(/0d0,0d0,0d0,0d0,0d0,0d0,&
         0d0,-0.180d0,0.139d0,-0.051d0,-0.168d0,-0.163d0,-0.236d0,0d0,-0.233d0,&
         -0.203d0,-0.203d0,-0.173d0,-0.10d0,-0.13d0,-0.161d0,-0.201d0,-0.161d0,-0.030d0,&
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
       GetASymbol="Tin    "
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
