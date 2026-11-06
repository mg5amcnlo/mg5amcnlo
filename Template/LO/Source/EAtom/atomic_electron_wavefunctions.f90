MODULE atomic_electron_wavefunctions
  IMPLICIT NONE
  ! Tables of the atom currently loaded (see eatom_load). Columns of
  ! eatom_grid: 2*i-1 = gtilde, 2*i = ftilde of shell i, on eatom_pgrid (GeV).
  INTEGER,SAVE::eatom_loadedA=-1,eatom_loadedZ=-1
  INTEGER,SAVE::eatom_NP=0,eatom_NSHELL=0
  INTEGER,DIMENSION(:),ALLOCATABLE,SAVE::eatom_shells
  REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::eatom_nume
  REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::eatom_pgrid
  REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::eatom_grid
  ! cubic-spline coefficients of each column, computed once in eatom_load
  REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::eatom_splb,eatom_splc,eatom_spld
  ! below eatom_lowp(j) (above eatom_highp(j)) column j is A*p**pow with
  ! (A,pow)=eatom_lowfit(j,1:2) (eatom_highfit(j,1:2))
  REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::eatom_lowp,eatom_highp
  REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::eatom_lowfit,eatom_highfit
CONTAINS

  SUBROUTINE eatom_WF(nuclearA,nuclearZ,HEL,p,wf)
    ! similar as IXXXXX in aloha_function.f
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ,HEL
    ! in units of GeV
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p
    COMPLEX(KIND(1d0)),DIMENSION(6),INTENT(OUT)::wf
    INTEGER,PARAMETER::offset=10000
    ! ishell=Sign(kappa)*(100*n+|kappa|)
    INTEGER::ishell
    REAL(KIND(1d0))::m
    ishell=ABS(HEL)-offset
    m=SIGN(1000d0,DBLE(HEL))
    CALL Atomic_electron_WF(nuclearA,nuclearZ,ishell,m,p,wf)
    RETURN
  END SUBROUTINE eatom_WF

  SUBROUTINE Atomic_electron_WF(nuclearA,nuclearZ,ishell,m,p,wf)
    ! similar as IXXXXX in aloha_function.f
    USE spherical_harmonic
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ
    ! ishell=Sign(kappa)*(100*n+|kappa|)
    INTEGER,INTENT(IN)::ishell
    ! m=-j,...,j
    ! if |m|>=999, we take Y_{kappa m}(theta, phi)=SQRT((2j+1)/(2))*Y_{(-)SGN(m)*0.5d0}(theta, phi)
    REAL(KIND(1d0)),INTENT(IN)::m
    ! in units of GeV
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::p
    COMPLEX(KIND(1d0)),DIMENSION(6),INTENT(OUT)::wf
    INTEGER::n,kappa,twoj,l,lp
    COMPLEX(KIND(1d0)),DIMENSION(2)::Y
    REAL(KIND(1d0))::theta,phi,pt,pp
    REAL(KIND(1d0))::costh,sinth,nume,gtilde,ftilde
    COMPLEX(KIND(1d0))::expiphi,expmiphi,pref,pref2,pref3,pref4
    REAL(KIND(1d0)),PARAMETER::pipi=3.14159265358979323846264338327950288d0
    ! 1/Sqrt[2]
    REAL(KIND(1d0)),PARAMETER::oosqrttwo=0.707106781186547524400844362105d0
    INTEGER::i
    wf(1)=dcmplx(p(0),p(3))*(-1)
    wf(2)=dcmplx(p(1),p(2))*(-1)
    CALL GetShellQuantumNumbers(ishell,n,kappa,twoj,l,lp)

    ! we have put sqrt(2*E) in with the same dimension as IXXXXX
    pref=(2d0*pipi)**(1.5d0)*(-dcmplx(0d0,1d0))**l*dsqrt(2d0*p(0))

    pt=p(1)**2+p(2)**2
    pp=DSQRT(pt+p(3)**2)
    pt=DSQRT(pt)
    IF(pp.EQ.0d0)THEN
       theta=0d0
       phi=0d0
    ELSE
       ! ATAN2 keeps full precision near the poles and the sign of p_y
       theta=ATAN2(pt,p(3))
       IF(pt.EQ.0d0)THEN
          phi=0d0
       ELSE
          phi=ATAN2(p(2),p(1))
       ENDIF
    ENDIF

    IF(ABS(m).GE.999d0)THEN
       Y=SpinorSphericalHarmonicY(-1,SIGN(0.5d0,m),theta,phi)
       Y(1:2)=DSQRT(dble(twoj+1)/(2d0))*Y(1:2)
    ELSE
       Y=SpinorSphericalHarmonicY(kappa,m,theta,phi)
    ENDIF

    CALL Get_atomic_electron_gftilde(nuclearA,nuclearZ,ishell,pp,nume,gtilde,ftilde)

    costh=COS(theta)
    sinth=SIN(theta)
    expiphi=EXP(dcmplx(0d0,phi))
    expmiphi=EXP(dcmplx(0d0,-phi))

    wf(3)=Y(1)*gtilde*pref
    wf(4)=Y(2)*gtilde*pref
    wf(5)=(Y(1)*costh+Y(2)*sinth*expmiphi)*ftilde*pref
    wf(6)=(Y(1)*sinth*expiphi-Y(2)*costh)*ftilde*pref

    ! convert from Pauli-Dirac representation to ALOHA chiral representation
    ! need to multiply {{1/Sqrt[2], 0, -(1/Sqrt[2]), 0},
    !                   {0, 1/Sqrt[2], 0, -(1/Sqrt[2])},
    !                   {1/Sqrt[2], 0, 1/Sqrt[2], 0},
    !                   {0, 1/Sqrt[2], 0, 1/Sqrt[2]}}
    pref=wf(3)-wf(5)
    pref2=wf(4)-wf(6)
    pref3=wf(3)+wf(5)
    pref4=wf(4)+wf(6)
    wf(3)=pref*oosqrttwo
    wf(4)=pref2*oosqrttwo
    wf(5)=pref3*oosqrttwo
    wf(6)=pref4*oosqrttwo

    IF(ABS(m).GE.999d0)THEN
       ! in the averaged m case, we also include
       ! the averaged electron number square root
       ! in the wf definition
       DO i=3,6
          wf(i)=wf(i)*DSQRT(nume)/DSQRT(dble(twoj+1))
       ENDDO
    ENDIF

    RETURN
  END SUBROUTINE Atomic_electron_WF

  SUBROUTINE eatom_load(nuclearA,nuclearZ)
    ! Load the gtilde/ftilde tables of atom (A,Z) into the module data,
    ! fit the low- and high-p power laws and compute the spline
    ! coefficients once. Reloads if a different atom is requested.
    USE C12_gftilde
    USE Pb208_gftilde
    USE W186_gftilde
    USE U238_gftilde
    USE interpolation
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ
    REAL(KIND(1d0)),PARAMETER::alphaEW=0.0072973525205055605d0
    INTEGER,DIMENSION(:),ALLOCATABLE::i_low_p,i_high_p
    INTEGER::i,j,k,nlow,nhigh,n,kappa,twoj,l,lp
    REAL(KIND(1d0))::xpower
    IF(nuclearA.EQ.eatom_loadedA.AND.nuclearZ.EQ.eatom_loadedZ)RETURN
    IF(ALLOCATED(eatom_shells))THEN
       DEALLOCATE(eatom_shells,eatom_nume,eatom_pgrid,eatom_grid)
       DEALLOCATE(eatom_splb,eatom_splc,eatom_spld)
       DEALLOCATE(eatom_lowp,eatom_highp,eatom_lowfit,eatom_highfit)
    ENDIF
    IF(nuclearA.EQ.12.AND.nuclearZ.EQ.6)THEN
       eatom_NP=C12_NP
       eatom_NSHELL=C12_NSHELL
    ELSEIF(nuclearA.EQ.186.AND.nuclearZ.EQ.74)THEN
       eatom_NP=W186_NP
       eatom_NSHELL=W186_NSHELL
    ELSEIF(nuclearA.EQ.208.AND.nuclearZ.EQ.82)THEN
       eatom_NP=Pb208_NP
       eatom_NSHELL=Pb208_NSHELL
    ELSEIF(nuclearA.EQ.238.AND.nuclearZ.EQ.92)THEN
       eatom_NP=U238_NP
       eatom_NSHELL=U238_NSHELL
    ELSE
       WRITE(*,*)"ERROR: Do not known (A,Z)=",nuclearA,nuclearZ
       STOP
    ENDIF
    ALLOCATE(eatom_shells(eatom_NSHELL),eatom_nume(eatom_NSHELL))
    ALLOCATE(eatom_pgrid(eatom_NP),eatom_grid(eatom_NP,2*eatom_NSHELL))
    ALLOCATE(eatom_splb(eatom_NP,2*eatom_NSHELL),eatom_splc(eatom_NP,2*eatom_NSHELL))
    ALLOCATE(eatom_spld(eatom_NP,2*eatom_NSHELL))
    ALLOCATE(eatom_lowp(2*eatom_NSHELL),eatom_highp(2*eatom_NSHELL))
    ALLOCATE(eatom_lowfit(2*eatom_NSHELL,2),eatom_highfit(2*eatom_NSHELL,2))
    ALLOCATE(i_low_p(2*eatom_NSHELL),i_high_p(2*eatom_NSHELL))
    IF(nuclearA.EQ.12.AND.nuclearZ.EQ.6)THEN
       eatom_shells=C12_all_shells
       eatom_nume=C12_nume_shells
       eatom_lowp=C12_low_p
       i_low_p=C12_i_low_p
       eatom_highp=C12_high_p
       i_high_p=C12_i_high_p
       eatom_pgrid=C12_P_grid
       eatom_grid=C12_gftilde_grid
    ELSEIF(nuclearA.EQ.186.AND.nuclearZ.EQ.74)THEN
       eatom_shells=W186_all_shells
       eatom_nume=W186_nume_shells
       eatom_lowp=W186_low_p
       i_low_p=W186_i_low_p
       eatom_highp=W186_high_p
       i_high_p=W186_i_high_p
       eatom_pgrid=W186_P_grid
       eatom_grid=W186_gftilde_grid
    ELSEIF(nuclearA.EQ.208.AND.nuclearZ.EQ.82)THEN
       eatom_shells=Pb208_all_shells
       eatom_nume=Pb208_nume_shells
       eatom_lowp=Pb208_low_p
       i_low_p=Pb208_i_low_p
       eatom_highp=Pb208_high_p
       i_high_p=Pb208_i_high_p
       eatom_pgrid=Pb208_P_grid
       eatom_grid=Pb208_gftilde_grid
    ELSE
       eatom_shells=U238_all_shells
       eatom_nume=U238_nume_shells
       eatom_lowp=U238_low_p
       i_low_p=U238_i_low_p
       eatom_highp=U238_high_p
       i_high_p=U238_i_high_p
       eatom_pgrid=U238_P_grid
       eatom_grid=U238_gftilde_grid
    ENDIF
    ! high p: gtilde~ftilde~p**(-2-sqrt(1-Z**2*alpha**2))
    IF(1d0-alphaEW**2*nuclearZ**2.GE.0d0)THEN
       xpower=-2d0-DSQRT(1d0-alphaEW**2*nuclearZ**2)
    ELSE
       xpower=-2d0
    ENDIF
    DO i=1,eatom_NSHELL
       CALL GetShellQuantumNumbers(eatom_shells(i),n,kappa,twoj,l,lp)
       DO k=0,1
          j=2*i-1+k
          ! low p: gtilde~p**l, ftilde~p**lp
          IF(k.EQ.0)THEN
             eatom_lowfit(j,2)=DBLE(l)
          ELSE
             eatom_lowfit(j,2)=DBLE(lp)
          ENDIF
          nlow=i_low_p(j)
          eatom_lowfit(j,1)=Fit_powerlaw_A(nlow,eatom_lowfit(j,2),&
               eatom_pgrid(1:nlow),eatom_grid(1:nlow,j))
          nhigh=eatom_NP-i_high_p(j)+1
          eatom_highfit(j,2)=xpower
          eatom_highfit(j,1)=Fit_powerlaw_A(nhigh,xpower,&
               eatom_pgrid(i_high_p(j):eatom_NP),eatom_grid(i_high_p(j):eatom_NP,j))
          CALL spline(eatom_pgrid,eatom_grid(1:eatom_NP,j),eatom_splb(1:eatom_NP,j),&
               eatom_splc(1:eatom_NP,j),eatom_spld(1:eatom_NP,j),eatom_NP)
       ENDDO
    ENDDO
    DEALLOCATE(i_low_p,i_high_p)
    eatom_loadedA=nuclearA
    eatom_loadedZ=nuclearZ
    RETURN
  END SUBROUTINE eatom_load

  FUNCTION eatom_column(j,p)
    ! gtilde (j odd) or ftilde (j even) of column j at |p| (GeV)
    USE interpolation
    IMPLICIT NONE
    INTEGER,INTENT(IN)::j
    REAL(KIND(1d0)),INTENT(IN)::p
    REAL(KIND(1d0))::eatom_column
    IF(p.LT.eatom_lowp(j))THEN
       eatom_column=eatom_lowfit(j,1)*p**eatom_lowfit(j,2)
    ELSEIF(p.GT.eatom_highp(j))THEN
       eatom_column=eatom_highfit(j,1)*p**eatom_highfit(j,2)
    ELSE
       eatom_column=ispline(p,eatom_pgrid,eatom_grid(1:eatom_NP,j),&
            eatom_splb(1:eatom_NP,j),eatom_splc(1:eatom_NP,j),&
            eatom_spld(1:eatom_NP,j),eatom_NP)
    ENDIF
    RETURN
  END FUNCTION eatom_column

  SUBROUTINE Get_atomic_electron_gftilde(nuclearA,nuclearZ,ishell,p,nume,gtilde,ftilde)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ
    ! ishell=Sign(kappa)*(100*n+|kappa|)
    INTEGER,INTENT(IN)::ishell
    ! in units of GeV
    REAL(KIND(1d0)),INTENT(IN)::p
    ! average number of electron in the shell
    REAL(KIND(1d0)),INTENT(OUT)::nume
    ! in units of GeV**(-3/2) with normalization
    ! 1=2*Pi**2*Int[(ftilde**2+gtilde**2)/(2*Pi)**3,d3p] with p from 0 to infinity
    REAL(KIND(1d0)),INTENT(OUT)::gtilde,ftilde
    INTEGER::i,ish
    CALL eatom_load(nuclearA,nuclearZ)
    ish=-1
    DO i=1,eatom_NSHELL
       IF(ishell.EQ.eatom_shells(i))THEN
          ish=i
          EXIT
       ENDIF
    ENDDO
    IF(ish.EQ.-1)THEN
       WRITE(*,*)"ERROR: shell cannot find: ishell=",ishell
       STOP
    ENDIF
    nume=eatom_nume(ish)
    gtilde=eatom_column(2*ish-1,p)
    ftilde=eatom_column(2*ish,p)
    RETURN
  END SUBROUTINE Get_atomic_electron_gftilde

  SUBROUTINE eatom_get_shells(nuclearA,nuclearZ,maxshell,nshell,shells,nume)
    ! occupied shells (ishell codes) and their mean electron numbers
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ,maxshell
    INTEGER,INTENT(OUT)::nshell
    INTEGER,DIMENSION(maxshell),INTENT(OUT)::shells
    REAL(KIND(1d0)),DIMENSION(maxshell),INTENT(OUT)::nume
    CALL eatom_load(nuclearA,nuclearZ)
    nshell=eatom_NSHELL
    IF(nshell.GT.maxshell)THEN
       WRITE(*,*)"ERROR: eatom_get_shells needs maxshell >=",nshell
       STOP
    ENDIF
    shells(1:nshell)=eatom_shells(1:nshell)
    nume(1:nshell)=eatom_nume(1:nshell)
    RETURN
  END SUBROUTINE eatom_get_shells

  SUBROUTINE eatom_table_range(nuclearA,nuclearZ,pmin,pmax)
    ! momentum range (GeV) covered by the tables
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ
    REAL(KIND(1d0)),INTENT(OUT)::pmin,pmax
    CALL eatom_load(nuclearA,nuclearZ)
    pmin=eatom_pgrid(1)
    pmax=eatom_pgrid(eatom_NP)
    RETURN
  END SUBROUTINE eatom_table_range

  FUNCTION eatom_radial_density(nuclearA,nuclearZ,ishell,p)
    ! radial momentum density n(p) in GeV**(-1):
    !   n(p) = sum_shells nume_i p**2 (gtilde_i**2+ftilde_i**2)   (ishell=0)
    !   n(p) = nume_i p**2 (gtilde_i**2+ftilde_i**2)              (one shell)
    ! with Int_0^infinity n(p) dp = Z (sum) or nume_i (one shell)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nuclearA,nuclearZ,ishell
    REAL(KIND(1d0)),INTENT(IN)::p
    REAL(KIND(1d0))::eatom_radial_density
    INTEGER::i
    REAL(KIND(1d0))::g,f
    CALL eatom_load(nuclearA,nuclearZ)
    eatom_radial_density=0d0
    DO i=1,eatom_NSHELL
       IF(ishell.NE.0.AND.ishell.NE.eatom_shells(i))CYCLE
       g=eatom_column(2*i-1,p)
       f=eatom_column(2*i,p)
       eatom_radial_density=eatom_radial_density+eatom_nume(i)*p**2*(g**2+f**2)
    ENDDO
    RETURN
  END FUNCTION eatom_radial_density

  ! just fit A in the form of y=A*x**l with l known
  FUNCTION Fit_powerlaw_A(n,l,x,y)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n
    REAL(KIND(1d0)),INTENT(IN)::l
    REAL(KIND(1d0)),DIMENSION(n),INTENT(IN)::x,y
    REAL(KIND(1d0))::Fit_powerlaw_A
    REAL(KIND(1d0))::numerator,denominator,fi
    INTEGER::i
    numerator=0d0
    denominator=0d0
    do i = 1, n
       fi=x(i)**l
       numerator=numerator+y(i)*fi
       denominator=denominator+fi*fi
    end do

    if(denominator.eq.0d0)then
       WRITE(*,*)"ERROR: denominator is zero in Fit_powerlaw_A"
       stop
    end if

    Fit_powerlaw_A=numerator/denominator
    RETURN
  END FUNCTION Fit_powerlaw_A

  SUBROUTINE GetShellQuantumNumbers(ishell,n,kappa,twoj,l,lp)
    IMPLICIT NONE
    ! ishell=Sign(kappa)*(100*n+|kappa|)
    INTEGER,INTENT(IN)::ishell
    ! lp=l-Sign(kappa)
    ! twoj=2*j
    INTEGER,INTENT(OUT)::n,kappa,twoj,l,lp
    kappa=SIGN(MOD(ABS(ishell),100),ishell)
    n=ABS(ishell)/100
    twoj=2*ABS(kappa)-1
    IF(kappa.GT.0)THEN
       l=kappa
       lp=l-1
    ELSEIF(kappa.LT.0)THEN
       l=-1-kappa
       lp=l+1
    ELSE
       WRITE(*,*)"ERROR: kappa = 0"
       STOP
    ENDIF
    RETURN
  END SUBROUTINE GetShellQuantumNumbers

  FUNCTION GetShellSymbol(ishell)
    IMPLICIT NONE
    ! ishell=Sign(kappa)*(100*n+|kappa|)
    INTEGER,INTENT(IN)::ishell
    CHARACTER(len=7)::GetShellSymbol,GetShellSymbol2
    CHARACTER(len=8)::fmt
    CHARACTER(len=5)::x1
    INTEGER::kappa,nn
    kappa=SIGN(MOD(ABS(ishell),100),ishell)
    nn=ABS(ishell)/100
    IF(nn.LE.0)THEN
       WRITE(*,*)"ERROR: n <= 0 in GetShellSymbol"
       STOP
    ENDIF
    SELECT CASE(kappa)
    CASE(-1)
       GetShellSymbol2="s      "
    CASE(-2)
       GetShellSymbol2="p      "
    CASE(1)
       GetShellSymbol2="p-     "
    CASE(-3)
       GetShellSymbol2="d      "
    CASE(2)
       GetShellSymbol2="d-     "
    CASE(-4)
       GetShellSymbol2="f      "
    CASE(3)
       GetShellSymbol2="f-     "
    CASE(-5)
       GetShellSymbol2="g      "
    CASE(4)
       GetShellSymbol2="g-     "
    CASE(-6)
       GetShellSymbol2="h      "
    CASE(5)
       GetShellSymbol2="h-     "
    CASE DEFAULT
       WRITE(*,*)"ERROR:Unknown kappa = ",kappa
       STOP
    END SELECT
    fmt='(I5)'
    WRITE(x1,fmt)nn
    x1=adjustl(x1)
    GetShellSymbol=TRIM(x1)//TRIM(GetShellSymbol2)
    RETURN
  END FUNCTION GetShellSymbol
  
  ! This is originally from gamma-UPC (arXiv:2207.03012)
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

END MODULE atomic_electron_wavefunctions
