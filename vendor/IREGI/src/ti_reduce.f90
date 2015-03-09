MODULE ti_reduce
  USE global
  USE si_reduce
  USE pave_reduce
  USE funlib
  IMPLICIT NONE
CONTAINS
  SUBROUTINE tensor_integral_reduce(IMODE,NLOOPLINE,MAXRANK,NCOEFS,PDEN_IN,M2L_IN,MU,TICOEFS,TSTABLE,CTMODE)
    ! CTMODE=1 ---> DO NOTHING
    ! CTMODE=2 ---> ROTATE THE LOOP PROPAGATOR DIRECTION,i.e D0...DN-1 to DN-1...D0
    ! IMODE=0, IBP reduction
    ! IMODE=1, PaVe reduction
    ! IMODE=2, PaVe reduction with stablility improved by IBP reduction 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,MAXRANK,IMODE,NCOEFS
    INTEGER,INTENT(IN),OPTIONAL::CTMODE
    REAL(KIND(1d0)),INTENT(IN)::MU
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PDEN_IN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PDEN
    REAL(KIND(1d0)),DIMENSION(0:3)::PDEN_N
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L_IN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(OUT)::TICOEFS
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4)::TICOEFS_SAVE
    LOGICAL,INTENT(OUT)::TSTABLE
    INTEGER,DIMENSION(1,1)::sol11
    REAL(KIND(1d0)),DIMENSION(1)::factor1
    INTEGER::first=0,i,j,jk,numzerp,n,r,nr,init,ntot
    INTEGER,DIMENSION(NLOOPLINE)::zerp
    ! f[i,n]=(i+n-1)!/(n-1)!/i!
    ! nmax=5,rmax=6,NLOOPLINE=nmax+1
    ! see syntensor in ti_reduce.f90
    ! xiarraymax2=(f[0,nmax])+(f[1,nmax])+(f[2,nmax]+f[0,nmax])
    ! +(f[3,nmax]+f[1,nmax])+(f[4,nmax]+f[2,nmax]+f[0,nmax])
    ! +(f[5,nmax]+f[3,nmax]+f[1,nmax])+(f[6,nmax]+f[4,nmax]+f[2,nmax]+f[0,nmax])
    ! when rmax=5,nmax=5 -> xiarraymax2=314
    ! when rmax=6,nmax=5 -> xiarraymax2=610
    INTEGER::xiarraymax2,xiarraymax,xiarraymax3
    !REAL(KIND(1d0)),DIMENSION(xiarraymax2)::syfactor
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::syfactor 
    !INTEGER,DIMENSION(xiarraymax2,-1:NLOOPLINE)::sy
    INTEGER,DIMENSION(:,:),ALLOCATABLE::sy 
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    REAL(KIND(1d0))::temp
    INTEGER::ctmode_local
    LOGICAL::do_shift
    SAVE first,xiarraymax,xiarraymax2,xiarraymax3,syfactor,sy
    IF(PRESENT(CTMODE))THEN
       ctmode_local=CTMODE
    ELSE
       ctmode_local=1
    ENDIF
    MU_R_IREGI=MU
    IF(first.EQ.0)THEN
       IF(.NOT.print_banner)THEN
          INCLUDE "banner.inc"
          print_banner=.TRUE.
       ENDIF
       xiarraymax2=0
       DO i=0,MAXRANK_IREGI
          j=i/2+1
          DO jk=1,j
             xiarraymax2=xiarraymax2+&
                  xiarray_arg1(2*jk-2+MOD(i,2),MAXNLOOP_IREGI-1)
          ENDDO
       ENDDO
       xiarraymax=xiarray_arg1(MAXRANK_IREGI,MAXNLOOP_IREGI-1)
       xiarraymax3=xiarray_arg1(MAXRANK_IREGI,4)
       IF(.NOT.ALLOCATED(syfactor))THEN
          ALLOCATE(syfactor(xiarraymax2))
       ENDIF
       IF(.NOT.ALLOCATED(sy))THEN
          ALLOCATE(sy(xiarraymax2,-1:MAXNLOOP_IREGI))
       ENDIF
       ! initialization xiarray and metric
       CALL all_Integers(1,1,1,sol11,factor1)
       DO i=0,3
          DO j=0,3
             IF(i.NE.j)THEN
                metric(i,j)=0d0
             ELSEIF(i.EQ.0)THEN
                metric(i,j)=1d0
             ELSE
                metric(i,j)=-1d0
             ENDIF
          ENDDO
       END DO
       first=1
    ENDIF
    IF(ctmode_local.EQ.2)THEN
       PDEN_N(0:3)=PDEN_IN(NLOOPLINE,0:3)
       DO I=1,NLOOPLINE
          PDEN(NLOOPLINE-I+1,0:3)=PDEN_IN(I,0:3)-PDEN_N(0:3)
          M2L(NLOOPLINE-I+1)=M2L_IN(I)
       ENDDO
    ELSE
       PDEN_N(0:3)=PDEN_IN(1,0:3)
       DO I=1,NLOOPLINE
          PDEN(I,0:3)=PDEN_IN(I,0:3)-PDEN_N(0:3)
          M2L(1:NLOOPLINE)=M2L_IN(1:NLOOPLINE)
       ENDDO
    ENDIF
    numzerp=0
    temp=(ABS(PDEN_N(0))+ABS(PDEN_N(1))+ABS(PDEN_N(2))+ABS(PDEN_N(3)))/4d0
    IF(temp.LT.EPS)THEN
       do_shift=.FALSE.
    ELSE
       do_shift=.TRUE.
    ENDIF
    DO i=1,NLOOPLINE
       temp=(ABS(PDEN(i,0))+ABS(PDEN(i,1))+ABS(PDEN(i,2))+ABS(PDEN(i,3)))/4d0
       IF(temp.LT.EPS)THEN
          zerp(i)=0
          numzerp=numzerp+1
       ELSE
          zerp(i)=1
       ENDIF
    ENDDO
    n=NLOOPLINE-numzerp
    IF(n.GE.MAXNLOOP_IREGI.OR.MAXRANK.GT.MAXRANK_IREGI)THEN
       WRITE(*,100)"ERROR: out of range of tensor_integral_reduce (N<=",MAXNLOOP_IREGI,",R<=",MAXRANK_IREGI,")"
       STOP
    ENDIF
    CALL sytensor(n,MAXRANK,xiarraymax,xiarraymax2,ntot,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2))
    CALL symmetry(IMODE,NLOOPLINE,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2),&
         numzerp,zerp,PDEN,M2L,NCOEFS,TICOEFS)
    TSTABLE=STABLE_IREGI
    IF(TSTABLE.AND.do_shift)THEN
       ! do momentum shift
       TICOEFS_SAVE(0:NCOEFS-1,1:4)=TICOEFS(0:NCOEFS-1,1:4)
       PDEN_N(0:3)=-PDEN_N(0:3)
       CALL SHIFT_MOM(PDEN_N,NCOEFS,TICOEFS_SAVE,TICOEFS)
    ENDIF
    RETURN
100 FORMAT(2X,A50,I2,A4,I2,A1)
  END SUBROUTINE tensor_integral_reduce

  SUBROUTINE tensor_integral_reduce2(IMODE,NLOOPLINE,MAXRANK,NCOEFS,PDEN,PijMatrix,M2L,MU,TICOEFS,TSTABLE)
    ! IMODE=0, IBP reduction 
    ! IMODE=1, PaVe reduction 
    ! IMODE=2, PaVe reduction with stablility improved by IBP reduction     
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,MAXRANK,IMODE,NCOEFS
    REAL(KIND(1d0)),INTENT(IN)::MU
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PDEN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(OUT)::TICOEFS
    LOGICAL,INTENT(OUT)::TSTABLE
    INTEGER,DIMENSION(1,1)::sol11
    REAL(KIND(1d0)),DIMENSION(1)::factor1
    INTEGER::first=0,i,j,jk,numzerp,n,r,nr,init,ntot
    INTEGER,DIMENSION(NLOOPLINE)::zerp
    ! f[i,n]=(i+n-1)!/(n-1)!/i!
    ! nmax=5,rmax=6,NLOOPLINE=nmax+1
    ! see syntensor in ti_reduce.f90 
    ! xiarraymax2=(f[0,nmax])+(f[1,nmax])+(f[2,nmax]+f[0,nmax])
    ! +(f[3,nmax]+f[1,nmax])+(f[4,nmax]+f[2,nmax]+f[0,nmax])
    ! +(f[5,nmax]+f[3,nmax]+f[1,nmax])+(f[6,nmax]+f[4,nmax]+f[2,nmax]+f[0,nmax])
    ! when rmax=5,nmax=5 -> xiarraymax2=314
    ! when rmax=6,nmax=5 -> xiarraymax2=610 
    INTEGER::xiarraymax2,xiarraymax,xiarraymax3
    !REAL(KIND(1d0)),DIMENSION(xiarraymax2)::syfactor
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::syfactor
    !INTEGER,DIMENSION(xiarraymax2,-1:NLOOPLINE)::sy
    INTEGER,DIMENSION(:,:),ALLOCATABLE::sy
    REAL(KIND(1d0))::temp
    SAVE first,xiarraymax,xiarraymax2,xiarraymax3,syfactor,sy
    MU_R_IREGI=MU
    IF(first.EQ.0)THEN
       IF(.NOT.print_banner)THEN
          INCLUDE "banner.inc"
          print_banner=.TRUE.
       ENDIF
       xiarraymax2=0
       DO i=0,MAXRANK_IREGI
          j=i/2+1
          DO jk=1,j
             xiarraymax2=xiarraymax2+&
                  xiarray_arg1(2*jk-2+MOD(i,2),MAXNLOOP_IREGI-1)
          ENDDO
       ENDDO
       xiarraymax=xiarray_arg1(MAXRANK_IREGI,MAXNLOOP_IREGI-1)
       xiarraymax3=xiarray_arg1(MAXRANK_IREGI,4)
       IF(.NOT.ALLOCATED(syfactor))THEN
          ALLOCATE(syfactor(xiarraymax2))
       ENDIF
       IF(.NOT.ALLOCATED(sy))THEN
          ALLOCATE(sy(xiarraymax2,-1:MAXNLOOP_IREGI))
       ENDIF
       ! initialization xiarray and metric 
       CALL all_Integers(1,1,1,sol11,factor1)
       DO i=0,3
          DO j=0,3
             IF(i.NE.j)THEN
                metric(i,j)=0d0
             ELSEIF(i.EQ.0)THEN
                metric(i,j)=1d0
             ELSE
                metric(i,j)=-1d0
             ENDIF
          ENDDO
       END DO
       first=1
    ENDIF
    numzerp=0
    DO i=1,NLOOPLINE
       temp=(ABS(PDEN(i,0))+ABS(PDEN(i,1))+ABS(PDEN(i,2))+ABS(PDEN(i,3)))/4d0
       IF(temp.LT.EPS)THEN
          zerp(i)=0
          numzerp=numzerp+1
       ELSE
          zerp(i)=1
       ENDIF
    ENDDO
    n=NLOOPLINE-numzerp
    IF(n.GE.MAXNLOOP_IREGI.OR.MAXRANK.GT.MAXRANK_IREGI)THEN
       WRITE(*,100)"ERROR: out of range of tensor_integral_reduce2 (N<=",MAXNLOOP_IREGI,",R<=",MAXRANK_IREGI,")"
       STOP
    ENDIF
    CALL sytensor(n,MAXRANK,xiarraymax,xiarraymax2,ntot,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2))
    CALL symmetry2(IMODE,NLOOPLINE,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2),&
         numzerp,zerp,PDEN,PijMatrix,M2L,NCOEFS,TICOEFS)
    TSTABLE=STABLE_IREGI
    RETURN
100 FORMAT(2X,A51,I2,A4,I2,A1)
  END SUBROUTINE tensor_integral_reduce2

  SUBROUTINE sytensor(n,rmax,xiarraymax,xiarraymax2,ntot,sy,syfactor)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n,rmax,xiarraymax,xiarraymax2
    INTEGER,INTENT(OUT)::ntot
    INTEGER,DIMENSION(xiarraymax2,-1:n),INTENT(OUT)::sy
    !INTEGER,DIMENSION(*,*),INTENT(OUT)::sy
    INTEGER::r,i,i0,i0max,rm2x0,nntot,init
    INTEGER,DIMENSION(xiarraymax,n)::sol
    REAL(KIND(1d0)),DIMENSION(xiarraymax2),INTENT(OUT)::syfactor
    REAL(KIND(1d0)),DIMENSION(xiarraymax)::factor
    IF(n.EQ.0)THEN
       init=1
       DO r=0,rmax
          IF(MOD(r,2).NE.0)CYCLE
          i0max=r/2
          sy(init,-1)=r
          sy(init,0)=i0max
          syfactor(init)=1d0/pi**(r-i0max)/(-2)**i0max
          init=init+1
       ENDDO
       ntot=init-1
       RETURN
    ENDIF
    ! 2*x0+x1+x2+...+xn=r
    init=1
    DO r=0,rmax
       i0max=r/2
       DO i=0,i0max
          rm2x0=r-2*i
          IF(n.LT.2)THEN
             nntot=1
          ELSE
             nntot=ntot_xiarray(rm2x0,n)
          ENDIF
          CALL all_integers(n,nntot,rm2x0,sol(1:nntot,1:n),factor(1:nntot))
          sy(init:init+nntot-1,-1)=r
          sy(init:init+nntot-1,0)=i
          sy(init:init+nntot-1,1:n)=sol(1:nntot,1:n)
          ! canonical form,i.e., all initial indices of propagators are 1 at first
          syfactor(init:init+nntot-1)=1d0/factor(1:nntot)*DBLE(factorial(rm2x0))&
               /pi**(r-i)/(-2)**i
          init=init+nntot
       ENDDO
    ENDDO
    ntot=init-1
    RETURN
  END SUBROUTINE sytensor

  SUBROUTINE symmetry2(IMODE,NLOOPLINE,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
       sy,syfactor,numzerp,zerp,PDEN,PijMatrix,M2L,NCOEFS,coefs)
    ! IMODE=0, IBP reduction
    ! IMODE=1, PaVe reduction
    ! IMODE=2, PaVe reduction with stablility improved by IBP reduction
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ntot,numzerp,NLOOPLINE,IMODE,NCOEFS,xiarraymax,xiarraymax2,xiarraymax3
    INTEGER,DIMENSION(xiarraymax2,-1:NLOOPLINE-numzerp),INTENT(IN)::sy
    !INTEGER,DIMENSION(*,*),INTENT(IN)::sy
    REAL(KIND(1d0)),DIMENSION(xiarraymax2),INTENT(IN)::syfactor
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::zerp
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PDEN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER::idim
    INTEGER,DIMENSION(NLOOPLINE)::indices
    INTEGER,DIMENSION(0:NLOOPLINE)::paveindices
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(OUT)::coefs
    INTEGER,DIMENSION(xiarraymax,4)::sol
    REAL(KIND(1d0)),DIMENSION(xiarraymax)::factor
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE-numzerp,0:3)::PCLL
    INTEGER::i,j,k,init,r,nntot,nt
    LOGICAL::lzero
    INTEGER,DIMENSION(0:NLOOPLINE)::nj
    INTEGER,DIMENSION(0:NLOOPLINE,1:MAXRANK_IREGI)::jlist
    REAL(KIND(1d0))::res
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::scalar
    REAL(KIND(1d0)),DIMENSION(xiarraymax3)::coco
    !REAL(KIND(1d0)),DIMENSION(xiarraymax)::coco
    INTEGER,DIMENSION(MAXRANK_IREGI)::lor
    coefs(0:NCOEFS-1,1:4)=DCMPLX(0d0)
    j=1
    DO i=1,NLOOPLINE
       IF(zerp(i).NE.0)THEN
          PCLL(j,0:3)=PDEN(i,0:3)
          j=j+1
       ENDIF
    ENDDO
    nt=NLOOPLINE-numzerp
    DO i=1,ntot
       r=sy(i,-1)
       nntot=ntot_xiarray(r,4)
       CALL all_integers(4,nntot,r,sol(1:nntot,1:4),factor(1:nntot))
       lzero=.TRUE.
       coco(1:nntot)=0d0
       DO j=1,nntot
          res=0d0
          init=1
          IF(sol(j,1).GE.1)THEN
             lor(init:init+sol(j,1)-1)=0
             init=init+sol(j,1)
          ENDIF
          IF(sol(j,2).GE.1)THEN
             lor(init:init+sol(j,2)-1)=1
             init=init+sol(j,2)
          ENDIF
          IF(sol(j,3).GE.1)THEN
             lor(init:init+sol(j,3)-1)=2
             init=init+sol(j,3)
          ENDIF
          IF(sol(j,4).GE.1)THEN
             lor(init:init+sol(j,4)-1)=3
             init=init+sol(j,4)
          ENDIF
          nj(0:nt)=0
          jlist(0:nt,1:MAXRANK_IREGI)=0
          CALL recursive_symmetry(nt,1,r,PCLL(1:nt,0:3),sy(i,-1:nt),lor(1:r),&
               nj(0:nt),jlist(0:nt,1:MAXRANK_IREGI),res)
          lzero=lzero.AND.(ABS(res).LT.EPS)
          IF(.NOT.ML5_CONVENTION)THEN
             coco(j)=res*factor(j)
          ELSE
             coco(j)=res
          ENDIF
       ENDDO
       IF(.NOT.lzero)THEN
          IF(IMODE.EQ.0)THEN
             ! IBP reduction
             indices(1:NLOOPLINE)=1 ! canonical form
             idim=2*(r-sy(i,0))
             init=1
             DO j=1,NLOOPLINE
                IF(zerp(j).NE.1)CYCLE
                indices(j)=indices(j)+sy(i,init)
                init=init+1
             ENDDO
             scalar(1:4)=scalar_integral_reduce2(NLOOPLINE,idim,indices,PijMatrix,M2L)
             IF(.NOT.STABLE_IREGI)THEN
                WRITE(*,*)"IREGI:WARNING, it detects unstable case, some integrals may set to be 0."
                RETURN
             ENDIF
             DO j=1,nntot
                init=calc_pos(sol(j,1:4))
                coefs(init,1:4)=coefs(init,1:4)+coco(j)*syfactor(i)*scalar(1:4)
             ENDDO
          ELSE
             ! PaVe reduction
             paveindices(0)=2*sy(i,0)
             init=1
             DO j=1,NLOOPLINE
                IF(zerp(j).NE.1)THEN
                   paveindices(j)=0
                ELSE
                   paveindices(j)=sy(i,init)
                   init=init+1
                ENDIF
             ENDDO
             IF(IMODE.EQ.1)THEN
                ! pure PaVe reduction
                scalar(1:4)=pavefun_reduce2(NLOOPLINE,paveindices,PijMatrix,M2L)
             ELSE
                ! optimized PaVe reduction
                ! the stability has been improved by IBP reduction
                scalar(1:4)=pave_opt_reduce2(NLOOPLINE,paveindices,PijMatrix,M2L)                
             ENDIF             
             IF(.NOT.STABLE_IREGI)THEN
                WRITE(*,*)"IREGI:WARNING, it detects unstable case, some integrals may set to be 0."
                RETURN
             ENDIF
             DO j=1,nntot
                init=calc_pos(sol(j,1:4))
                coefs(init,1:4)=coefs(init,1:4)+coco(j)*scalar(1:4)
             ENDDO
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE symmetry2

  SUBROUTINE symmetry(IMODE,NLOOPLINE,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
       sy,syfactor,numzerp,zerp,PDEN,M2L,NCOEFS,coefs)
    ! IMODE=0, IBP reduction  
    ! IMODE=1, PaVe reduction
    ! IMODE=2, PaVe reduction with stablility improved by IBP reduction 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ntot,numzerp,NLOOPLINE,IMODE,NCOEFS,xiarraymax,xiarraymax2,xiarraymax3
    INTEGER,DIMENSION(xiarraymax2,-1:NLOOPLINE-numzerp),INTENT(IN)::sy
    !INTEGER,DIMENSION(*,*),INTENT(IN)::sy 
    REAL(KIND(1d0)),DIMENSION(xiarraymax2),INTENT(IN)::syfactor
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::zerp
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PDEN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER::idim
    INTEGER,DIMENSION(NLOOPLINE)::indices
    INTEGER,DIMENSION(0:NLOOPLINE)::paveindices
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(OUT)::coefs
    INTEGER,DIMENSION(xiarraymax,4)::sol
    REAL(KIND(1d0)),DIMENSION(xiarraymax)::factor
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE-numzerp,0:3)::PCLL
    INTEGER::i,j,k,init,r,nntot,nt
    LOGICAL::lzero
    INTEGER,DIMENSION(0:NLOOPLINE)::nj
    INTEGER,DIMENSION(0:NLOOPLINE,1:MAXRANK_IREGI)::jlist
    REAL(KIND(1d0))::res
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::scalar
    REAL(KIND(1d0)),DIMENSION(xiarraymax3)::coco
    !REAL(KIND(1d0)),DIMENSION(84)::coco
    INTEGER,DIMENSION(MAXRANK_IREGI)::lor
    coefs(0:NCOEFS-1,1:4)=DCMPLX(0d0)
    j=1
    DO i=1,NLOOPLINE
       IF(zerp(i).NE.0)THEN
          PCLL(j,0:3)=PDEN(i,0:3)
          j=j+1
       ENDIF
    ENDDO
    nt=NLOOPLINE-numzerp
    DO i=1,ntot
       r=sy(i,-1)
!       IF(r.EQ.0)THEN
!          init=0
!       ENDIF
       nntot=ntot_xiarray(r,4)
       CALL all_integers(4,nntot,r,sol(1:nntot,1:4),factor(1:nntot))
       lzero=.TRUE.
       coco(1:nntot)=0d0
       DO j=1,nntot
          res=0d0
          init=1
          IF(sol(j,1).GE.1)THEN
             lor(init:init+sol(j,1)-1)=0
             init=init+sol(j,1)
          ENDIF
          IF(sol(j,2).GE.1)THEN
             lor(init:init+sol(j,2)-1)=1
             init=init+sol(j,2)
          ENDIF
          IF(sol(j,3).GE.1)THEN
             lor(init:init+sol(j,3)-1)=2
             init=init+sol(j,3)
          ENDIF
          IF(sol(j,4).GE.1)THEN
             lor(init:init+sol(j,4)-1)=3
             init=init+sol(j,4)
          ENDIF
          nj(0:nt)=0
          jlist(0:nt,1:MAXRANK_IREGI)=0
          CALL recursive_symmetry(nt,1,r,PCLL(1:nt,0:3),sy(i,-1:nt),lor(1:r),&
               nj(0:nt),jlist(0:nt,1:MAXRANK_IREGI),res)
          lzero=lzero.AND.(ABS(res).LT.EPS)
          IF(.NOT.ML5_CONVENTION)THEN
             coco(j)=res*factor(j)
          ELSE
             coco(j)=res
          ENDIF
          !coco(j)=res*factor(j)
       ENDDO
       IF(.NOT.lzero)THEN
          IF(IMODE.EQ.0)THEN
             ! IBP reduction
             indices(1:NLOOPLINE)=1 ! canonical form
             idim=2*(r-sy(i,0))
             init=1
             DO j=1,NLOOPLINE
                IF(zerp(j).NE.1)CYCLE
                indices(j)=indices(j)+sy(i,init)
                init=init+1
             ENDDO
             scalar(1:4)=scalar_integral_reduce(NLOOPLINE,idim,indices,PDEN,M2L)
             IF(.NOT.STABLE_IREGI)THEN
                WRITE(*,*)"IREGI:WARNING, it detects unstable case, some integrals may set to be 0."
                RETURN
             ENDIF
             DO j=1,nntot
                init=calc_pos(sol(j,1:4))
                coefs(init,1:4)=coefs(init,1:4)+coco(j)*syfactor(i)*scalar(1:4)
             ENDDO
          ELSE
             ! PaVe reduction
             paveindices(0)=2*sy(i,0)
             init=1
             DO j=1,NLOOPLINE
                IF(zerp(j).NE.1)THEN
                   paveindices(j)=0
                ELSE
                   paveindices(j)=sy(i,init)
                   init=init+1
                ENDIF
             ENDDO
             IF(IMODE.EQ.1)THEN
                ! pure PaVe reduction
                scalar(1:4)=pavefun_reduce(NLOOPLINE,paveindices,PDEN,M2L)
             ELSE
                ! optimized PaVe reduction
                ! the stability has been improved by IBP reduction
                scalar(1:4)=pave_opt_reduce(NLOOPLINE,paveindices,PDEN,M2L)
             ENDIF
             IF(.NOT.STABLE_IREGI)THEN
                WRITE(*,*)"IREGI:WARNING, it detects unstable case, some integrals may set to be 0."
                RETURN
             ENDIF
             DO j=1,nntot
                init=calc_pos(sol(j,1:4))
                coefs(init,1:4)=coefs(init,1:4)+coco(j)*scalar(1:4)
             ENDDO
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE symmetry

  RECURSIVE SUBROUTINE recursive_symmetry(n,s,r,PCL,sy,lor,njj,jjlist,res)
    ! jjlist is a list of indices for i-th nonzero propagator momentum
    IMPLICIT NONE
    INTEGER,INTENT(IN)::r,s,n ! n is the number of nonzero momenta in loop
    REAL(KIND(1d0)),INTENT(INOUT)::res
    REAL(KIND(1d0))::temp,temp2
    INTEGER,DIMENSION(0:n),INTENT(INOUT)::njj
    INTEGER,DIMENSION(0:n,MAXRANK_IREGI),INTENT(INOUT)::jjlist
    REAL(KIND(1d0)),DIMENSION(n,0:3),INTENT(IN)::PCL
    INTEGER,DIMENSION(-1:n),INTENT(IN)::sy
    INTEGER,DIMENSION(r),INTENT(IN)::lor
    INTEGER::i,j,nlen
    IF(r.LT.s)THEN
       IF(njj(0).EQ.0)THEN
          temp=1d0
       ELSE
          temp=gfunction(njj(0),jjlist(0,1:njj(0)),r,lor)
       ENDIF
       IF(ABS(temp).LT.EPS)THEN
          res=res+0d0
          RETURN
       ENDIF
       DO i=1,n
          nlen=njj(i)
          DO j=1,nlen
             temp2=PCL(i,lor(jjlist(i,j)))
             IF(ABS(temp2).LT.EPS)THEN
                temp=0d0
                res=res+temp
                RETURN
             ENDIF
             temp=temp*temp2
          ENDDO
       ENDDO
       res=res+temp
       RETURN
    ELSE
       IF(njj(0).LT.2*sy(0))THEN
          njj(0)=njj(0)+1
          jjlist(0,njj(0))=s
          CALL recursive_symmetry(n,s+1,r,PCL,sy,lor,njj,jjlist,res)
          njj(0)=njj(0)-1
       ENDIF
       DO i=1,n
          IF(njj(i).LT.sy(i))THEN
             njj(i)=njj(i)+1
             jjlist(i,njj(i))=s
             CALL recursive_symmetry(n,s+1,r,PCL,sy,lor,njj,jjlist,res)
             njj(i)=njj(i)-1
          ENDIF
       ENDDO
       RETURN
    ENDIF
  END SUBROUTINE recursive_symmetry

  RECURSIVE FUNCTION calc_gfunction(n,lor) RESULT(gf)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n
    INTEGER,DIMENSION(n),INTENT(IN)::lor
    INTEGER::i
    REAL(KIND(1d0))::gf,temp
    INTEGER,DIMENSION(n)::lor2
    IF(n.EQ.0)THEN
       gf=1d0
       RETURN
    ENDIF
    IF(MOD(n,2).NE.0)THEN
       gf=0d0
       RETURN
    ENDIF
    gf=0d0
    lor2(2:n-1)=lor(3:n)
    DO i=2,n
       IF(i.NE.2)lor2(i-1)=lor(i-1)
       temp=metric(lor(1),lor(i))
       IF(ABS(temp).LT.EPS)CYCLE
       gf=gf+temp*calc_gfunction(n-2,lor2(2:n-1))
    ENDDO
    RETURN
  END FUNCTION calc_gfunction

  FUNCTION gfunction(nnj,jlist,nn,lor2)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::nn,nnj
    INTEGER,DIMENSION(nn),INTENT(IN)::lor2
    INTEGER,DIMENSION(nnj),INTENT(IN)::jlist
    INTEGER::ifirst=0,i,j,k,l,t
    INTEGER::maxgrank,maxgrank2 ! maxrank2=2*maxrank*4
    REAL(KIND(1d0)),DIMENSION(:,:,:,:),ALLOCATABLE::gfsave
    INTEGER,DIMENSION(:),ALLOCATABLE::lor
    INTEGER::n
    INTEGER,DIMENSION(0:3)::indexnum
    REAL(KIND(1d0))::gfunction
    SAVE ifirst,gfsave
    IF(ifirst.EQ.0)THEN
       maxgrank=MAXRANK_IREGI/2
       maxgrank2=2*maxgrank*4
       IF(.NOT.ALLOCATED(gfsave))THEN
          ALLOCATE(gfsave(0:maxgrank,0:maxgrank,0:maxgrank,0:maxgrank))
       ENDIF
       IF(.NOT.ALLOCATED(lor))THEN
          ALLOCATE(lor(maxgrank2))
       ENDIF
       DO i=0,maxgrank
          DO j=0,maxgrank
             DO k=0, maxgrank
                DO l=0, maxgrank
                   n=2*i+2*j+2*k+2*l
                   IF(n.EQ.0)THEN
                      gfsave(i,j,k,l)=1d0
                      CYCLE
                   ENDIF
                   t=1
                   IF(i.GE.1)THEN
                      lor(t:t+2*i-1)=0
                      t=t+2*i
                   ENDIF
                   IF(j.GE.1)THEN
                      lor(t:t+2*j-1)=1
                      t=t+2*j
                   ENDIF
                   IF(k.GE.1)THEN
                      lor(t:t+2*k-1)=2
                      t=t+2*k
                   ENDIF
                   IF(l.GE.1)THEN
                      lor(t:t+2*l-1)=3
                      t=t+2*l
                   ENDIF
                   gfsave(i,j,k,l)=calc_gfunction(n,lor(1:n))
                END DO
             ENDDO
          END DO
       ENDDO
       ifirst=1
    ENDIF
    IF(nnj.EQ.0)THEN
       gfunction=1d0
       RETURN
    ENDIF
    IF(MOD(nnj,2).NE.0)THEN
       gfunction=0d0
       RETURN
    ENDIF
    indexnum(0:3)=0
    DO i=1,nnj
       indexnum(lor2(jlist(i)))=indexnum(lor2(jlist(i)))+1
    ENDDO
    DO i=0,3
       IF(MOD(indexnum(i),2).NE.0)THEN
          gfunction=0d0
          RETURN
       ELSE
          indexnum(i)=indexnum(i)/2
       ENDIF
    ENDDO
    gfunction=gfsave(indexnum(0),indexnum(1),indexnum(2),indexnum(3))
    RETURN
  END FUNCTION gfunction

  FUNCTION calc_pos(pos)
    IMPLICIT NONE
    INTEGER,DIMENSION(0:3),INTENT(IN)::pos
    INTEGER::calc_pos
    INTEGER,PARAMETER::maxrank=MAXRANK_IREGI
    INTEGER,DIMENSION(0:maxrank,0:maxrank,0:maxrank,0:maxrank)::possave
    INTEGER::ifirst=0,i,j,k,l,rk,init
    SAVE possave,ifirst
    IF(ifirst.EQ.0)THEN
       init=0
       DO rk=0,maxrank
          DO i=rk,0,-1
             DO j=rk-i,0,-1
                DO k=rk-i-j,0,-1
                   l=rk-i-j-k
                   possave(i,j,k,l)=init
                   init=init+1
                ENDDO
             ENDDO
          ENDDO
       ENDDO
       ifirst=1
    ENDIF
    calc_pos=possave(pos(0),pos(1),pos(2),pos(3))
    RETURN
  END FUNCTION calc_pos
END MODULE ti_reduce
