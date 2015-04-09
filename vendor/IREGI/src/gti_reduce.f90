MODULE gti_reduce
  USE global
  USE si_reduce
  USE pave_reduce
  USE funlib
  USE ti_reduce
  IMPLICIT NONE
CONTAINS
  SUBROUTINE general_ti_reduce(IMODE,NLOOPLINE,idim_init,indices_init,MAXRANK,NCOEFS,PDEN,M2L,MU,TICOEFS,TSTABLE)
    ! IMODE=0, IBP reduction
    ! IMODE=1, PaVe reduction
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,MAXRANK,IMODE,NCOEFS,idim_init
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::indices_init
    REAL(KIND(1d0)),INTENT(IN)::MU
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PDEN
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
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    REAL(KIND(1d0))::temp
    INTEGER,DIMENSION(NLOOPLINE)::indices_init2
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
       ! initialization xiarray and metric, factorial_pair
       CALL all_Integers(1,1,1,sol11,factor1)
       CALL calc_factorial_pair
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
    IF(MOD(idim_init,2).NE.0)THEN
       WRITE(*,*)"ERROR: the initial idim=d-4+2*eps should be even in general_ti_reduce ",idim_init
       STOP
    ENDIF
    numzerp=0
    DO i=1,NLOOPLINE
       temp=(ABS(PDEN(i,0))+ABS(PDEN(i,1))+ABS(PDEN(i,2))+ABS(PDEN(i,3)))/4d0
       IF(temp.LT.EPS)THEN
          zerp(i)=0
          numzerp=numzerp+1
       ELSE
          zerp(i)=1
          indices_init2(i-numzerp)=indices_init(i)
       ENDIF
    ENDDO
    n=NLOOPLINE-numzerp
    IF(n.GE.MAXNLOOP_IREGI.OR.MAXRANK.GT.MAXRANK_IREGI)THEN
       WRITE(*,100)"ERROR: out of range of general_ti_reduce (N<=",MAXNLOOP_IREGI,",R<=",MAXRANK_IREGI,")"
       STOP
    ENDIF
    CALL general_sytensor(n,indices_init2(1:n),MAXRANK,xiarraymax,xiarraymax2,ntot,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2))
    CALL general_symmetry(IMODE,NLOOPLINE,idim_init,indices_init,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
         sy(1:xiarraymax2,-1:n),&
         syfactor(1:xiarraymax2),numzerp,zerp,PDEN,M2L,NCOEFS,TICOEFS)
    TSTABLE=STABLE_IREGI
    RETURN
100 FORMAT(2X,A45,I2,A4,I2,A1)
  END SUBROUTINE general_ti_reduce

  SUBROUTINE general_ti_reduce2(IMODE,NLOOPLINE,idim_init,indices_init,MAXRANK,NCOEFS,PDEN,PijMatrix,M2L,MU,TICOEFS,TSTABLE)
    ! IMODE=0, IBP reduction 
    ! IMODE=1, PaVe reduction 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,MAXRANK,IMODE,NCOEFS,idim_init
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::indices_init
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
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::syfactor
    !INTEGER,DIMENSION(xiarraymax2,-1:NLOOPLINE)::sy
    INTEGER,DIMENSION(:,:),ALLOCATABLE::sy
    REAL(KIND(1d0))::temp
    INTEGER,DIMENSION(NLOOPLINE)::indices_init2
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
       ! initialization xiarray and metric,factorial_pair 
       CALL all_Integers(1,1,1,sol11,factor1)
       CALL calc_factorial_pair
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
    IF(MOD(idim_init,2).NE.0)THEN
       WRITE(*,*)"ERROR: the initial idim=d-4+2*eps should be even in general_ti_reduce2 ",idim_init
       STOP
    ENDIF
    numzerp=0
    DO i=1,NLOOPLINE
       temp=(ABS(PDEN(i,0))+ABS(PDEN(i,1))+ABS(PDEN(i,2))+ABS(PDEN(i,3)))/4d0
       IF(temp.LT.EPS)THEN
          zerp(i)=0
          numzerp=numzerp+1
       ELSE
          zerp(i)=1
          indices_init2(i-numzerp)=indices_init(i)
       ENDIF
    ENDDO
    n=NLOOPLINE-numzerp
    IF(n.GE.MAXNLOOP_IREGI.OR.MAXRANK.GT.MAXRANK_IREGI)THEN
       WRITE(*,100)"ERROR: out of range of general_ti_reduce2 (N<=",MAXNLOOP_IREGI,",R<=",MAXRANK_IREGI,")"
       STOP
    ENDIF
    CALL general_sytensor(n,indices_init2(1:n),MAXRANK,xiarraymax,xiarraymax2,ntot,&
         sy(1:xiarraymax2,-1:n),syfactor(1:xiarraymax2))
    CALL general_symmetry2(IMODE,NLOOPLINE,idim_init,indices_init,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
         sy(1:xiarraymax2,-1:n),&
         syfactor(1:xiarraymax2),numzerp,zerp,PDEN,PijMatrix,M2L,NCOEFS,TICOEFS)
    TSTABLE=STABLE_IREGI
    RETURN
100 FORMAT(2X,A46,I2,A4,I2,A1)
  END SUBROUTINE general_ti_reduce2

  SUBROUTINE general_sytensor(n,indices_init2,rmax,xiarraymax,xiarraymax2,ntot,sy,syfactor)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n,rmax,xiarraymax2,xiarraymax
    INTEGER,DIMENSION(n),INTENT(IN)::indices_init2
    INTEGER,INTENT(OUT)::ntot
    INTEGER,DIMENSION(xiarraymax2,-1:n),INTENT(OUT)::sy
    !INTEGER,DIMENSION(*,*),INTENT(OUT)::sy
    INTEGER::r,i,j,k,i0,i0max,rm2x0,nntot,init
    INTEGER,DIMENSION(xiarraymax,n)::sol
    REAL(KIND(1d0)),DIMENSION(xiarraymax2),INTENT(OUT)::syfactor
    REAL(KIND(1d0)),DIMENSION(xiarraymax)::factor
    REAL(KIND(1d0)),DIMENSION(xiarraymax)::factor_temp
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
          ! general form, the following can be optimized
          factor_temp(1:nntot)=1d0
          DO j=1,n
             DO k=1,nntot
                factor_temp(k)=factor_temp(k)&
                     *factorial_pair(indices_init2(j),sol(k,j))
             ENDDO
          ENDDO
          syfactor(init:init+nntot-1)=1d0/factor(1:nntot)*DBLE(factorial(rm2x0))&
               /pi**(r-i)/(-2)**i
          syfactor(init:init+nntot-1)=syfactor(init:init+nntot-1)*factor_temp(1:nntot)
          init=init+nntot
       ENDDO
    ENDDO
    ntot=init-1
    RETURN
  END SUBROUTINE general_sytensor

  SUBROUTINE general_symmetry2(IMODE,NLOOPLINE,idim_init,indices_init,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
       sy,syfactor,numzerp,zerp,PDEN,PijMatrix,M2L,NCOEFS,coefs)
    ! IMODE=0, IBP reduction
    ! IMODE=1, PaVe reduction
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ntot,numzerp,NLOOPLINE,IMODE,NCOEFS,idim_init,xiarraymax2,xiarraymax,xiarraymax3
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::indices_init
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
    INTEGER::i,j,k,init,r,nntot,nt,nindtot,nindtot0
    LOGICAL::lzero
    INTEGER,DIMENSION(0:NLOOPLINE)::nj
    INTEGER,DIMENSION(0:NLOOPLINE,1:MAXRANK_IREGI)::jlist
    REAL(KIND(1d0))::res,factemp
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::scalar
    !REAL(KIND(1d0)),DIMENSION(xiarraymax)::coco
    REAL(KIND(1d0)),DIMENSION(xiarraymax3)::coco
    INTEGER,DIMENSION(MAXRANK_IREGI)::lor
    coefs(0:NCOEFS-1,1:4)=DCMPLX(0d0)
    j=1
    nindtot0=0
    DO i=1,NLOOPLINE
       IF(zerp(i).NE.0)THEN
          PCLL(j,0:3)=PDEN(i,0:3)
          j=j+1
       ENDIF
       nindtot0=nindtot0+indices_init(i)
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
          !coco(j)=res*factor(j)
       ENDDO
       IF(.NOT.lzero)THEN
          indices(1:NLOOPLINE)=indices_init(1:NLOOPLINE) ! general form
          idim=2*(r-sy(i,0))+idim_init
          init=1
          nindtot=nindtot0
          DO j=1,NLOOPLINE
             IF(zerp(j).NE.1)CYCLE
             indices(j)=indices(j)+sy(i,init)
             nindtot=nindtot+sy(i,init)
             init=init+1
          ENDDO
          IF(IMODE.EQ.0)THEN
             ! IBP reduction
             scalar(1:4)=scalar_integral_reduce2(NLOOPLINE,idim,indices,PijMatrix,M2L)
          ELSE
             IF(idim-2*nindtot+2*NLOOPLINE.GE.0.AND.indices(1).EQ.1)THEN
                ! PaVe reduction 
                ! it can be improved as Gamma(ni+di)/Gamma(ni)*1/Gamma(ni+di), where 1/Gamma(ni+di) from factemp
                CALL IBP2PAVE(NLOOPLINE,idim,indices,nindtot,factemp,paveindices)
                scalar(1:4)=pavefun_reduce2(NLOOPLINE,paveindices,PijMatrix,M2L)*factemp
             ELSE
                scalar(1:4)=scalar_integral_reduce2(NLOOPLINE,idim,indices,PijMatrix,M2L)
             ENDIF
          ENDIF
          IF(.NOT.STABLE_IREGI)RETURN
          DO j=1,nntot
             init=calc_pos(sol(j,1:4))
             coefs(init,1:4)=coefs(init,1:4)+coco(j)*syfactor(i)*scalar(1:4)
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE general_symmetry2

  SUBROUTINE general_symmetry(IMODE,NLOOPLINE,idim_init,indices_init,ntot,xiarraymax,xiarraymax2,xiarraymax3,&
       sy,syfactor,numzerp,zerp,PDEN,M2L,NCOEFS,coefs)
    ! IMODE=0, IBP reduction  
    ! IMODE=1, PaVe reduction 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ntot,numzerp,NLOOPLINE,IMODE,NCOEFS,idim_init,xiarraymax2,xiarraymax,xiarraymax3
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::indices_init
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
    INTEGER::i,j,k,init,r,nntot,nt,nindtot,nindtot0
    LOGICAL::lzero
    INTEGER,DIMENSION(0:NLOOPLINE)::nj
    INTEGER,DIMENSION(0:NLOOPLINE,1:MAXRANK_IREGI)::jlist
    REAL(KIND(1d0))::res,factemp
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::scalar
    !REAL(KIND(1d0)),DIMENSION(xiarraymax)::coco
    REAL(KIND(1d0)),DIMENSION(xiarraymax3)::coco
    INTEGER,DIMENSION(MAXRANK_IREGI)::lor
    coefs(0:NCOEFS-1,1:4)=DCMPLX(0d0)
    j=1
    nindtot0=0
    DO i=1,NLOOPLINE
       IF(zerp(i).NE.0)THEN
          PCLL(j,0:3)=PDEN(i,0:3)
          j=j+1
       ENDIF
       nindtot0=nindtot0+indices_init(i)
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
          indices(1:NLOOPLINE)=indices_init(1:NLOOPLINE) ! general form
          idim=2*(r-sy(i,0))+idim_init
          init=1
          nindtot=nindtot0
          DO j=1,NLOOPLINE
             IF(zerp(j).NE.1)CYCLE
             indices(j)=indices(j)+sy(i,init)
             nindtot=nindtot+sy(i,init)
             init=init+1
          ENDDO
          IF(IMODE.EQ.0)THEN
             ! IBP reduction
             scalar(1:4)=scalar_integral_reduce(NLOOPLINE,idim,indices,PDEN,M2L)
          ELSE
             ! PaVe reduction 
             ! it can be improved as Gamma(ni+di)/Gamma(ni)*1/Gamma(ni+di), where 1/Gamma(ni+di) from factemp
             IF(idim-2*nindtot+2*NLOOPLINE.GE.0.AND.indices(1).EQ.1)THEN
                CALL IBP2PAVE(NLOOPLINE,idim,indices,nindtot,factemp,paveindices)
                scalar(1:4)=pavefun_reduce(NLOOPLINE,paveindices,PDEN,M2L)*factemp
             ELSE
                scalar(1:4)=scalar_integral_reduce(NLOOPLINE,idim,indices,PDEN,M2L)
             ENDIF
          ENDIF
          IF(.NOT.STABLE_IREGI)RETURN
          DO j=1,nntot
             init=calc_pos(sol(j,1:4))
             coefs(init,1:4)=coefs(init,1:4)+coco(j)*syfactor(i)*scalar(1:4)
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE general_symmetry

END MODULE gti_reduce
