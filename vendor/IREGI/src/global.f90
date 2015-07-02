MODULE global
  IMPLICIT NONE
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! One should change the following lines correspondingly!
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  INTEGER,PARAMETER::MAXRANK_IREGI=7,MAXNLOOP_IREGI=7,MAXINDICES_IREGI=10
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! END of changing                                      !
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  INTEGER,PARAMETER::DBL=SELECTED_REAL_KIND(p=13),ICHECK=0
  INTEGER,PARAMETER::LINT=SELECTED_INT_KIND(32) ! this is used for very long integer like xiarray_arg1 in funlib.f90
  ! x1+x2+...+xn==i
  ! C(i+n-1)^(n-1)=(i+n-1)!/(n-1)!/i!
  TYPE xiarraytype
     INTEGER,DIMENSION(:,:),ALLOCATABLE::xiarray_i_n
  END TYPE xiarraytype
  TYPE( xiarraytype ),DIMENSION(:),ALLOCATABLE::xiarray
  INTEGER,DIMENSION(0:MAXRANK_IREGI,2:MAXNLOOP_IREGI-1)::ntot_xiarray
  TYPE factor_xiarraytype
     INTEGER,DIMENSION(:),ALLOCATABLE::factor_xiarray_i_n
  END TYPE factor_xiarraytype
  ! (MAXRANK_IREGI+1)*(MAXNLOOP_IREGI-2)
  TYPE( factor_xiarraytype ),DIMENSION(:),ALLOCATABLE::factor_xiarray
  REAL(KIND(1d0))::MU_R_IREGI=1d3
  LOGICAL::STABLE_IREGI=.TRUE.
  LOGICAL::print_banner=.FALSE.
  ! Sum((3+i)*(2+i)*(1+i)/6,{i,0,k})=Length(factor_xiarray_k_5)
  ! f[i,n]=(i+n-1)!/(n-1)!/i!
  ! nmax=5,rmax=6,NLOOPLINE=nmax+1
  ! see syntensor in ti_reduce.f90
  ! xiarraymax2=(f[0,nmax])+(f[1,nmax])+(f[2,nmax]+f[0,nmax])
  ! +(f[3,nmax]+f[1,nmax])+(f[4,nmax]+f[2,nmax]+f[0,nmax])
  ! +(f[5,nmax]+f[3,nmax]+f[1,nmax])+(f[6,nmax]+f[4,nmax]+f[2,nmax]+f[0,nmax])
  ! when rmax=5,nmax=5 -> xiarraymax2=314
  ! when rmax=6,nmax=5 -> xiarraymax2=610
  !INTEGER,PARAMETER::xiarraymax=210,xiarraymax2=610,xiarraymax3=210
  REAL(KIND(1d0)),PARAMETER::EPS=1d-10
  REAL(KIND(1d0)),DIMENSION(0:3,0:3)::metric
  INTEGER::scalarlib=1 ! 1: QCDLoop, 2: OneLoop
  REAL(KIND(1d0))::ZEROTHR_IREGI=1d-6
  LOGICAL::check=.FALSE.,ONSHELL_IREGI=.FALSE.,ML5_CONVENTION=.FALSE.
  INTEGER::nmass
  REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI)::massref
  ! factorial_pair(i,j)=Gamma(i+j)/Gamma(i)/Gamma(j+1)*Gamma(1)
  REAL(KIND(1d0)),DIMENSION(MAXINDICES_IREGI,0:MAXRANK_IREGI)::factorial_pair
  !INTEGER::CURRENT_PS=1
!  LOGICAL::assigninv=.FALSE.
!  REAL(KIND(1d0))::ep12,ep22,ep32,ep42,es12,es23
  TYPE ibppave_node
     !INTEGER :: ITERATION=0
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:MAXNLOOP_IREGI) :: indices
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI) :: M2L
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3) :: PCL
     COMPLEX(KIND(1d0)),DIMENSION(1:4) :: value
     TYPE( ibppave_node ), POINTER :: parent
     TYPE( ibppave_node ), POINTER :: left
     TYPE( ibppave_node ), POINTER :: right
  END TYPE ibppave_node
  LOGICAL::RECYCLING=.FALSE. !,OPTIMIZATION=.TRUE.
  TYPE(ibppave_node),POINTER::ibp_save,pave_save,shiftpaveden_save !,pave_UV_save
  TYPE ibppave_node_array
     TYPE(ibppave_node),POINTER::ptr
  END TYPE ibppave_node_array
  TYPE(ibppave_node_array),DIMENSION(MAXNLOOP_IREGI)::ibppave_save_array !,pave_S1_save
  !TYPE(ibppave_node_array),DIMENSION(0:10)::pave_S_save
  TYPE ibppave_node2
     !INTEGER :: ITERATION=0
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:MAXNLOOP_IREGI) :: indices
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI) :: M2L
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,MAXNLOOP_IREGI) :: PijMatrix
     COMPLEX(KIND(1d0)),DIMENSION(1:4) :: value
     TYPE( ibppave_node2 ), POINTER :: parent
     TYPE( ibppave_node2 ), POINTER :: left
     TYPE( ibppave_node2 ), POINTER :: right
  END TYPE ibppave_node2
  TYPE(ibppave_node2),POINTER::ibp_save2,pave_save2,shiftpaveden_save2
  TYPE cibppave_node
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:MAXNLOOP_IREGI) :: indices
     COMPLEX(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI) :: M2L
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3) :: PCL
     COMPLEX(KIND(1d0)),DIMENSION(1:4) :: value
     TYPE( cibppave_node ), POINTER :: parent
     TYPE( cibppave_node ), POINTER :: left
     TYPE( cibppave_node ), POINTER :: right
  END TYPE cibppave_node
  TYPE(cibppave_node),POINTER::cibp_save,cpave_save,cshiftpaveden_save
  TYPE xyzmatrices_node
     !NLOOPLINE,PCL,M2L,XMATRIX,YMATRIX,ZMATRIX,detY,detZ
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3)::PCL
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI)::M2L
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,MAXNLOOP_IREGI)::XMATRIX,YMATRIX
     REAL(KIND(1d0)),DIMENSION(2:MAXNLOOP_IREGI,2:MAXNLOOP_IREGI)::ZMATRIX
     REAL(KIND(1d0))::detY,detZ
     TYPE( xyzmatrices_node ), POINTER :: parent
     TYPE( xyzmatrices_node ), POINTER :: left
     TYPE( xyzmatrices_node ), POINTER :: right
  END TYPE xyzmatrices_node
  TYPE(xyzmatrices_node),POINTER::xyzmatrices_save
  TYPE cxyzmatrices_node
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3)::PCL
     COMPLEX(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI)::M2L
     COMPLEX(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,MAXNLOOP_IREGI)::XMATRIX,YMATRIX
     COMPLEX(KIND(1d0)),DIMENSION(2:MAXNLOOP_IREGI,2:MAXNLOOP_IREGI)::ZMATRIX
     COMPLEX(KIND(1d0))::detY,detZ
     TYPE( cxyzmatrices_node ), POINTER :: parent
     TYPE( cxyzmatrices_node ), POINTER :: left
     TYPE( cxyzmatrices_node ), POINTER :: right
  END TYPE cxyzmatrices_node
  TYPE(cxyzmatrices_node),POINTER::cxyzmatrices_save
  TYPE rsmatrices_node
     ! NLOOPLINE,PCL,M2L,rmatrix,smatrix,rdet,sdet
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3)::PCL
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI)::M2L
     REAL(KIND(1d0)),DIMENSION(0:MAXNLOOP_IREGI,0:MAXNLOOP_IREGI)::smatrix
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,MAXNLOOP_IREGI)::rmatrix
     REAL(KIND(1d0))::detR,detS
     TYPE(rsmatrices_node),POINTER::parent
     TYPE(rsmatrices_node),POINTER::left
     TYPE(rsmatrices_node),POINTER::right
  END TYPE rsmatrices_node
  TYPE(rsmatrices_node),POINTER::rsmatrices_save
  TYPE crsmatrices_node
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,0:3)::PCL
     COMPLEX(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI)::M2L
     COMPLEX(KIND(1d0)),DIMENSION(0:MAXNLOOP_IREGI,0:MAXNLOOP_IREGI)::smatrix
     COMPLEX(KIND(1d0)),DIMENSION(MAXNLOOP_IREGI,MAXNLOOP_IREGI)::rmatrix
     COMPLEX(KIND(1d0))::detR,detS
     TYPE(crsmatrices_node),POINTER::parent
     TYPE(crsmatrices_node),POINTER::left
     TYPE(crsmatrices_node),POINTER::right
  END TYPE crsmatrices_node
  TYPE(crsmatrices_node),POINTER::crsmatrices_save
  SAVE
END MODULE global
