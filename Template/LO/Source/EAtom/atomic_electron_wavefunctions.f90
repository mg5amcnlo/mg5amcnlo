MODULE atomic_electron_wavefunctions
  IMPLICIT NONE
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
       theta=ACOS(p(3)/pp)
       IF(pt.EQ.0d0)THEN
          phi=0d0
       ELSE
          phi=ACOS(p(1)/pt)
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

  SUBROUTINE Get_atomic_electron_gftilde(nuclearA,nuclearZ,ishell,p,nume,gtilde,ftilde)
    USE C12_gftilde
    USE Pb208_gftilde
    USE W186_gftilde
    USE interpolation
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
    CHARACTER(len=7)::Aname
    INTEGER,SAVE::init=0
    CHARACTER(len=100)::datafile
    LOGICAL::lexist
    INTEGER,PARAMETER::iunit=30784
    INTEGER::Aval,Zval
    INTEGER,SAVE::NP,NSHELL
    INTEGER,DIMENSION(:),ALLOCATABLE,SAVE::all_shells
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::nume_shells
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::P_grid
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::gftilde_grid
    INTEGER::i,j,igt,ift
    INTEGER,DIMENSION(:),ALLOCATABLE::i_low_p,i_high_p
    ! where low p fit starts when p<low_p
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::low_p
    ! where high p fit starts when p>high_p
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE,SAVE::high_p
    REAL(KIND(1d0)),PARAMETER::alphaEW=0.0072973525205055605d0
    INTEGER::N_LOW,N_HIGH,n,kappa,twoj,l,lp
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::X_LOW,Y_LOW,X_HIGH,Y_HIGH
    REAL(KIND(1d0))::xpower,xA
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::gtilde_LOW,ftilde_LOW
    REAL(KIND(1d0)),DIMENSION(:,:),ALLOCATABLE,SAVE::gtilde_HIGH,ftilde_HIGH
    IF(init.EQ.0)THEN
       Aval=nuclearA
       Zval=nuclearZ
       IF(nuclearA.EQ.12.AND.nuclearZ.EQ.6)THEN
          ! C12
          NP=C12_NP
          NSHELL=C12_NSHELL
          ALLOCATE(all_shells(NSHELL))
          all_shells(1:NSHELL)=C12_all_shells(1:NSHELL)
          ALLOCATE(nume_shells(NSHELL))
          nume_shells(1:NSHELL)=C12_nume_shells(1:NSHELL)
          ALLOCATE(low_p(2*NSHELL))
          ALLOCATE(i_low_p(2*NSHELL))
          low_p(1:(2*NSHELL))=C12_low_p(1:(2*NSHELL))
          i_low_p(1:(2*NSHELL))=C12_i_low_p(1:(2*NSHELL))
          ALLOCATE(high_p(2*NSHELL))
          ALLOCATE(i_high_p(2*NSHELL))
          high_p(1:(2*NSHELL))=C12_high_p(1:(2*NSHELL))
          i_high_p(1:(2*NSHELL))=C12_i_high_p(1:(2*NSHELL))
          ALLOCATE(P_grid(NP))
          P_grid(1:NP)=C12_P_grid(1:NP)
          ALLOCATE(gftilde_grid(NP,2*NSHELL))
          DO i=1,NP
             gftilde_grid(i,1:(2*NSHELL))=C12_gftilde_grid(i,1:(2*NSHELL))
          ENDDO
       ELSEIF(nuclearA.EQ.186.AND.nuclearZ.EQ.74)THEN
          ! W186
          NP=W186_NP
          NSHELL=W186_NSHELL
          ALLOCATE(all_shells(NSHELL))
          all_shells(1:NSHELL)=W186_all_shells(1:NSHELL)
          ALLOCATE(nume_shells(NSHELL))
          nume_shells(1:NSHELL)=W186_nume_shells(1:NSHELL)
          ALLOCATE(low_p(2*NSHELL))
          ALLOCATE(i_low_p(2*NSHELL))
          low_p(1:(2*NSHELL))=W186_low_p(1:(2*NSHELL))
          i_low_p(1:(2*NSHELL))=W186_i_low_p(1:(2*NSHELL))
          ALLOCATE(high_p(2*NSHELL))
          ALLOCATE(i_high_p(2*NSHELL))
          high_p(1:(2*NSHELL))=W186_high_p(1:(2*NSHELL))
          i_high_p(1:(2*NSHELL))=W186_i_high_p(1:(2*NSHELL))
          ALLOCATE(P_grid(NP))
          P_grid(1:NP)=W186_P_grid(1:NP)
          ALLOCATE(gftilde_grid(NP,2*NSHELL))
          DO i=1,NP
             gftilde_grid(i,1:(2*NSHELL))=W186_gftilde_grid(i,1:(2*NSHELL))
          ENDDO
       ELSEIF(nuclearA.EQ.208.AND.nuclearZ.EQ.82)THEN
          ! Pb208
          NP=Pb208_NP
          NSHELL=Pb208_NSHELL
          ALLOCATE(all_shells(NSHELL))
          all_shells(1:NSHELL)=Pb208_all_shells(1:NSHELL)
          ALLOCATE(nume_shells(NSHELL))
          nume_shells(1:NSHELL)=Pb208_nume_shells(1:NSHELL)
          ALLOCATE(low_p(2*NSHELL))
          ALLOCATE(i_low_p(2*NSHELL))
          low_p(1:(2*NSHELL))=Pb208_low_p(1:(2*NSHELL))
          i_low_p(1:(2*NSHELL))=Pb208_i_low_p(1:(2*NSHELL))
          ALLOCATE(high_p(2*NSHELL))
          ALLOCATE(i_high_p(2*NSHELL))
          high_p(1:(2*NSHELL))=Pb208_high_p(1:(2*NSHELL))
          i_high_p(1:(2*NSHELL))=Pb208_i_high_p(1:(2*NSHELL))
          ALLOCATE(P_grid(NP))
          P_grid(1:NP)=Pb208_P_grid(1:NP)
          ALLOCATE(gftilde_grid(NP,2*NSHELL))
          DO i=1,NP
             gftilde_grid(i,1:(2*NSHELL))=Pb208_gftilde_grid(i,1:(2*NSHELL))
          ENDDO
       ELSE
          WRITE(*,*)"ERROR: Do not known (A,Z)=",nuclearA,nuclearZ
          STOP
       ENDIF
       N_LOW=0
       N_HIGH=0
       DO j=1,2*NSHELL
          IF(NP-i_high_p(j)+1.GT.N_HIGH)N_HIGH=NP-i_high_p(j)+1
          IF(i_low_p(j).GT.N_LOW)N_LOW=i_low_p(j)
       ENDDO
       ALLOCATE(X_LOW(N_LOW))
       ALLOCATE(Y_LOW(N_LOW))
       ALLOCATE(X_HIGH(N_HIGH))
       ALLOCATE(Y_HIGH(N_HIGH))
       ALLOCATE(gtilde_LOW(NSHELL,2))
       ALLOCATE(ftilde_LOW(NSHELL,2))
       ALLOCATE(gtilde_HIGH(NSHELL,2))
       ALLOCATE(ftilde_HIGH(NSHELL,2))
       DO i=1,NSHELL
          CALL GetShellQuantumNumbers(all_shells(i),n,kappa,twoj,l,lp)
          ! for low p
          ! gtilde~p**l, ftilde~p**lp
          xpower=l
          gtilde_LOW(i,2)=xpower
          DO j=1,i_low_p(2*i-1)
             X_LOW(j)=P_grid(j)
             Y_LOW(j)=gftilde_grid(j,2*i-1)
          ENDDO
          xA=Fit_powerlaw_A(i_low_p(2*i-1),xpower,X_LOW(1:i_low_p(2*i-1)),&
               Y_LOW(1:i_low_p(2*i-1)))
          gtilde_LOW(i,1)=xA
          
          xpower=lp
          ftilde_LOW(i,2)=xpower
          DO j=1,i_low_p(2*i)
             X_LOW(j)=P_grid(j)
             Y_LOW(j)=gftilde_grid(j,2*i)
          ENDDO
          xA=Fit_powerlaw_A(i_low_p(2*i),xpower,X_LOW(1:i_low_p(2*i)),&
               Y_LOW(1:i_low_p(2*i)))
          ftilde_LOW(i,1)=xA

          ! for high p
          ! gtilde~ftilde~p**(-2-sqrt(1-Z**2*alpha**2))
          IF(1d0-alphaEW**2*nuclearZ**2.GE.0d0)THEN
             xpower=-2d0-DSQRT(1d0-alphaEW**2*nuclearZ**2)
          ELSE
             xpower=-2d0
          ENDIF
          gtilde_HIGH(i,2)=xpower
          ftilde_HIGH(i,2)=xpower

          DO j=1,NP-i_high_p(2*i-1)+1
             X_HIGH(j)=P_grid(i_high_p(2*i-1)+j-1)
             Y_HIGH(j)=gftilde_grid(i_high_p(2*i-1)+j-1,2*i-1)
          ENDDO
          xA=Fit_powerlaw_A(NP-i_high_p(2*i-1)+1,xpower,&
               X_HIGH(1:(NP-i_high_p(2*i-1)+1)),&
               Y_HIGH(1:(NP-i_high_p(2*i-1)+1)))
          gtilde_HIGH(i,1)=xA

          DO j=1,NP-i_high_p(2*i)+1
             X_HIGH(j)=P_grid(i_high_p(2*i)+j-1)
             Y_HIGH(j)=gftilde_grid(i_high_p(2*i)+j-1,2*i)
          ENDDO
          xA=Fit_powerlaw_A(NP-i_high_p(2*i)+1,xpower,&
               X_HIGH(1:(NP-i_high_p(2*i)+1)),&
               Y_HIGH(1:(NP-i_high_p(2*i)+1)))
          ftilde_HIGH(i,1)=xA
       ENDDO
       init=1
    ENDIF
    gtilde=0d0
    ftilde=0d0
    ! we first need to decide which shell
    igt=-1
    ift=-1
    DO i=1,NSHELL
       IF(ishell.EQ.all_shells(i))THEN
          igt=2*i-1
          ift=2*i
          EXIT
       ENDIF
    ENDDO
    IF(igt.EQ.-1.or.ift.EQ.-1)THEN
       WRITE(*,*)"ERROR: shell cannot find: ishell=",ishell
       STOP
    ENDIF
    nume=nume_shells((igt+1)/2)
    IF(p.LT.low_p(igt))THEN
       ! we use the low p fit
       gtilde=gtilde_LOW((igt+1)/2,1)*p**(gtilde_LOW((igt+1)/2,2))
    ELSEIF(p.GT.high_p(igt))THEN
       ! we use the high p fit
       gtilde=gtilde_HIGH((igt+1)/2,1)*p**(gtilde_HIGH((igt+1)/2,2))
    ELSE
       ! otherwise, we use the grid to interpolate
       CALL SPLINE_INTERPOLATE(P_grid,gftilde_grid(1:NP,igt),NP,p,gtilde)
    ENDIF
    IF(p.LT.low_p(ift))THEN
       ! we use the low p fit
       ftilde=ftilde_LOW((ift+1)/2,1)*p**(ftilde_LOW((ift+1)/2,2))
    ELSEIF(p.GT.high_p(ift))THEN
       ! we use the high p fit
       ftilde=ftilde_HIGH((ift+1)/2,1)*p**(ftilde_HIGH((ift+1)/2,2))
    ELSE
       ! otherwise, we use the grid to interpolate
       CALL SPLINE_INTERPOLATE(P_grid,gftilde_grid(1:NP,ift),NP,p,ftilde)
    ENDIF
    RETURN
  END SUBROUTINE Get_atomic_electron_gftilde

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
