MODULE global
  IMPLICIT NONE
  INTEGER::DBL=SELECTED_REAL_KIND(p=13),ICHECK=0
  ! x1+x2+...+xn==i
  ! C(i+n-1)^(n-1)=(i+n-1)!/(n-1)!/i!
  INTEGER,DIMENSION(0:10,1)::x1array  ! i from 0 to 10,i.e.,x1=i
  INTEGER,DIMENSION(1,2)::xiarray_0_2 ! x1+x2=0
  INTEGER,DIMENSION(2,2)::xiarray_1_2 ! x1+x2=1
  INTEGER,DIMENSION(3,2)::xiarray_2_2 ! x1+x2=2
  INTEGER,DIMENSION(4,2)::xiarray_3_2 ! x1+x2=3
  INTEGER,DIMENSION(5,2)::xiarray_4_2 ! x1+x2=4
  INTEGER,DIMENSION(6,2)::xiarray_5_2 ! x1+x2=5
  INTEGER,DIMENSION(7,2)::xiarray_6_2 ! x1+x2=6
  INTEGER,DIMENSION(1,3)::xiarray_0_3 ! x1+x2+x3=0
  INTEGER,DIMENSION(3,3)::xiarray_1_3 ! x1+x2+x3=1
  INTEGER,DIMENSION(6,3)::xiarray_2_3 ! x1+x2+x3=2
  INTEGER,DIMENSION(10,3)::xiarray_3_3 ! x1+x2+x3=3
  INTEGER,DIMENSION(15,3)::xiarray_4_3 ! x1+x2+x3=4
  INTEGER,DIMENSION(21,3)::xiarray_5_3 ! x1+x2+x3=5
  INTEGER,DIMENSION(28,3)::xiarray_6_3 ! x1+x2+x3=6
  INTEGER,DIMENSION(1,4)::xiarray_0_4 ! x1+x2+x3+x4=0
  INTEGER,DIMENSION(4,4)::xiarray_1_4 ! x1+x2+x3+x4=1
  INTEGER,DIMENSION(10,4)::xiarray_2_4 ! x1+x2+x3+x4=2
  INTEGER,DIMENSION(20,4)::xiarray_3_4 ! x1+x2+x3+x4=3
  INTEGER,DIMENSION(35,4)::xiarray_4_4 ! x1+x2+x3+x4=4
  INTEGER,DIMENSION(56,4)::xiarray_5_4 ! x1+x2+x3+x4=5
  INTEGER,DIMENSION(84,4)::xiarray_6_4 ! x1+x2+x3+x4=6
  INTEGER,DIMENSION(1,5)::xiarray_0_5 ! x1+x2+x3+x4+x5=0
  INTEGER,DIMENSION(5,5)::xiarray_1_5 ! x1+x2+x3+x4+x5=1
  INTEGER,DIMENSION(15,5)::xiarray_2_5 ! x1+x2+x3+x4+x5=2
  INTEGER,DIMENSION(35,5)::xiarray_3_5 ! x1+x2+x3+x4+x5=3
  INTEGER,DIMENSION(70,5)::xiarray_4_5 ! x1+x2+x3+x4+x5=4
  INTEGER,DIMENSION(126,5)::xiarray_5_5 ! x1+x2+x3+x4+x5=5
  INTEGER,DIMENSION(210,5)::xiarray_6_5 ! x1+x2+x3+x4+x5=6
  INTEGER,DIMENSION(0:6,2:5)::ntot_xiarray
  REAL(KIND(1d0)),DIMENSION(1)::factor_xiarray_0_2
  REAL(KIND(1d0)),DIMENSION(2)::factor_xiarray_1_2
  REAL(KIND(1d0)),DIMENSION(3)::factor_xiarray_2_2
  REAL(KIND(1d0)),DIMENSION(4)::factor_xiarray_3_2
  REAL(KIND(1d0)),DIMENSION(5)::factor_xiarray_4_2
  REAL(KIND(1d0)),DIMENSION(6)::factor_xiarray_5_2
  REAL(KIND(1d0)),DIMENSION(7)::factor_xiarray_6_2
  REAL(KIND(1d0)),DIMENSION(1)::factor_xiarray_0_3
  REAL(KIND(1d0)),DIMENSION(3)::factor_xiarray_1_3
  REAL(KIND(1d0)),DIMENSION(6)::factor_xiarray_2_3
  REAL(KIND(1d0)),DIMENSION(10)::factor_xiarray_3_3
  REAL(KIND(1d0)),DIMENSION(15)::factor_xiarray_4_3
  REAL(KIND(1d0)),DIMENSION(21)::factor_xiarray_5_3
  REAL(KIND(1d0)),DIMENSION(28)::factor_xiarray_6_3
  REAL(KIND(1d0)),DIMENSION(1)::factor_xiarray_0_4
  REAL(KIND(1d0)),DIMENSION(4)::factor_xiarray_1_4
  REAL(KIND(1d0)),DIMENSION(10)::factor_xiarray_2_4
  REAL(KIND(1d0)),DIMENSION(20)::factor_xiarray_3_4
  REAL(KIND(1d0)),DIMENSION(35)::factor_xiarray_4_4
  REAL(KIND(1d0)),DIMENSION(56)::factor_xiarray_5_4
  REAL(KIND(1d0)),DIMENSION(84)::factor_xiarray_6_4
  REAL(KIND(1d0)),DIMENSION(1)::factor_xiarray_0_5
  REAL(KIND(1d0)),DIMENSION(5)::factor_xiarray_1_5
  REAL(KIND(1d0)),DIMENSION(15)::factor_xiarray_2_5
  REAL(KIND(1d0)),DIMENSION(35)::factor_xiarray_3_5
  REAL(KIND(1d0)),DIMENSION(70)::factor_xiarray_4_5
  REAL(KIND(1d0)),DIMENSION(126)::factor_xiarray_5_5
  REAL(KIND(1d0)),DIMENSION(210)::factor_xiarray_6_5
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
  INTEGER,PARAMETER::xiarraymax=210,xiarraymax2=610,xiarraymax3=210
  REAL(KIND(1d0)),PARAMETER::EPS=1d-10
  REAL(KIND(1d0)),DIMENSION(0:3,0:3)::metric
  INTEGER::scalarlib=1 ! 1: QCDLoop, 2: OneLoop
  REAL(KIND(1d0))::ZEROTHR_IREGI=1d-6
  LOGICAL::check=.FALSE.,ONSHELL_IREGI=.FALSE.,ML5_CONVENTION=.FALSE.
  INTEGER::nmass
  REAL(KIND(1d0)),DIMENSION(20)::massref
  ! factorial_pair(i,j)=Gamma(i+j)/Gamma(i)/Gamma(j+1)*Gamma(1)
  REAL(KIND(1d0)),DIMENSION(10,0:10)::factorial_pair
  !INTEGER::CURRENT_PS=1
!  LOGICAL::assigninv=.FALSE.
!  REAL(KIND(1d0))::ep12,ep22,ep32,ep42,es12,es23
  TYPE ibppave_node
     !INTEGER :: ITERATION=0
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:10) :: indices
     REAL(KIND(1d0)),DIMENSION(10) :: M2L
     REAL(KIND(1d0)),DIMENSION(10,0:3) :: PCL
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
  TYPE(ibppave_node_array),DIMENSION(10)::ibppave_save_array !,pave_S1_save
  !TYPE(ibppave_node_array),DIMENSION(0:10)::pave_S_save
  TYPE ibppave_node2
     !INTEGER :: ITERATION=0
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:10) :: indices
     REAL(KIND(1d0)),DIMENSION(10) :: M2L
     REAL(KIND(1d0)),DIMENSION(10,10) :: PijMatrix
     COMPLEX(KIND(1d0)),DIMENSION(1:4) :: value
     TYPE( ibppave_node2 ), POINTER :: parent
     TYPE( ibppave_node2 ), POINTER :: left
     TYPE( ibppave_node2 ), POINTER :: right
  END TYPE ibppave_node2
  TYPE(ibppave_node2),POINTER::ibp_save2,pave_save2,shiftpaveden_save2
  TYPE cibppave_node
     INTEGER :: NLOOPLINE
     LOGICAL :: stable
     INTEGER,DIMENSION(0:10) :: indices
     COMPLEX(KIND(1d0)),DIMENSION(10) :: M2L
     REAL(KIND(1d0)),DIMENSION(10,0:3) :: PCL
     COMPLEX(KIND(1d0)),DIMENSION(1:4) :: value
     TYPE( cibppave_node ), POINTER :: parent
     TYPE( cibppave_node ), POINTER :: left
     TYPE( cibppave_node ), POINTER :: right
  END TYPE cibppave_node
  TYPE(cibppave_node),POINTER::cibp_save,cpave_save,cshiftpaveden_save
  TYPE xyzmatrices_node
     !NLOOPLINE,PCL,M2L,XMATRIX,YMATRIX,ZMATRIX,detY,detZ
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(10,0:3)::PCL
     REAL(KIND(1d0)),DIMENSION(10)::M2L
     REAL(KIND(1d0)),DIMENSION(10,10)::XMATRIX,YMATRIX
     REAL(KIND(1d0)),DIMENSION(2:10,2:10)::ZMATRIX
     REAL(KIND(1d0))::detY,detZ
     TYPE( xyzmatrices_node ), POINTER :: parent
     TYPE( xyzmatrices_node ), POINTER :: left
     TYPE( xyzmatrices_node ), POINTER :: right
  END TYPE xyzmatrices_node
  TYPE(xyzmatrices_node),POINTER::xyzmatrices_save
  TYPE cxyzmatrices_node
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(10,0:3)::PCL
     COMPLEX(KIND(1d0)),DIMENSION(10)::M2L
     COMPLEX(KIND(1d0)),DIMENSION(10,10)::XMATRIX,YMATRIX
     COMPLEX(KIND(1d0)),DIMENSION(2:10,2:10)::ZMATRIX
     COMPLEX(KIND(1d0))::detY,detZ
     TYPE( cxyzmatrices_node ), POINTER :: parent
     TYPE( cxyzmatrices_node ), POINTER :: left
     TYPE( cxyzmatrices_node ), POINTER :: right
  END TYPE cxyzmatrices_node
  TYPE(cxyzmatrices_node),POINTER::cxyzmatrices_save
  TYPE rsmatrices_node
     ! NLOOPLINE,PCL,M2L,rmatrix,smatrix,rdet,sdet
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(10,0:3)::PCL
     REAL(KIND(1d0)),DIMENSION(10)::M2L
     REAL(KIND(1d0)),DIMENSION(0:10,0:10)::smatrix
     REAL(KIND(1d0)),DIMENSION(10,10)::rmatrix
     REAL(KIND(1d0))::detR,detS
     TYPE(rsmatrices_node),POINTER::parent
     TYPE(rsmatrices_node),POINTER::left
     TYPE(rsmatrices_node),POINTER::right
  END TYPE rsmatrices_node
  TYPE(rsmatrices_node),POINTER::rsmatrices_save
  TYPE crsmatrices_node
     INTEGER::NLOOPLINE
     REAL(KIND(1d0)),DIMENSION(10,0:3)::PCL
     COMPLEX(KIND(1d0)),DIMENSION(10)::M2L
     COMPLEX(KIND(1d0)),DIMENSION(0:10,0:10)::smatrix
     COMPLEX(KIND(1d0)),DIMENSION(10,10)::rmatrix
     COMPLEX(KIND(1d0))::detR,detS
     TYPE(crsmatrices_node),POINTER::parent
     TYPE(crsmatrices_node),POINTER::left
     TYPE(crsmatrices_node),POINTER::right
  END TYPE crsmatrices_node
  TYPE(crsmatrices_node),POINTER::crsmatrices_save
  SAVE
END MODULE global
