MODULE cpave_reduce
USE global
USE funlib
USE matrix_base
USE cmatrix_base
USE matrices
USE mis_warp
USE kinematics
USE binary_tree
USE pave_reduce
CONTAINS
  ! shift the momentum so that the moemntum of the first denominator is q
  ! Eq.(2.9) in hep - ph/0509141 v2
  ! ADD RECURSIVE TO MAKE IT WORK IN ML5
  RECURSIVE FUNCTION comp_shiftpaveden(NLOOPLINE,paveindices,PCL,M2L,ind1) RESULT(pave)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,ind1 ! ind1 is the second pave index before dropping
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL1
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pave
    REAL(KIND(1d0)),DIMENSION(0:3)::mom
    INTEGER::i2,i,ntot,j,i1
    INTEGER,DIMENSION(0:NLOOPLINE)::indices0,indices00
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::sumf
    REAL(KIND(1d0))::factor,factor0
    TYPE xitemptype
       INTEGER::dim1,dim2
       INTEGER,DIMENSION(:,:),ALLOCATABLE::xitempi
    END TYPE xitemptype
    TYPE(xitemptype),DIMENSION(MAXNLOOP_IREGI-1)::xitemp
    REAL(KIND(1d0)),DIMENSION(:),ALLOCATABLE::factor_xi
    LOGICAL::find
    INTEGER::init=0
    SAVE init,xitemp,factor_xi
    TYPE(cibppave_node),POINTER::item
    pave(1:4)=DCMPLX(0d0)
    IF(.NOT.STABLE_IREGI)RETURN
    IF(RECYCLING)THEN
       PCL1(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL1(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%stable=.TRUE.
       item%indices(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL1(1:NLOOPLINE,0:3)
       CALL cibppave_bt_search(item,cshiftpaveden_save,find)
       IF(find)THEN
          pave(1:4)=item%value(1:4)
          STABLE_IREGI=item%stable
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    IF(init.EQ.0)THEN
       DO i=1,MAXNLOOP_IREGI-1
          xitemp(i)%dim2=i
          xitemp(i)%dim1=xiarray_arg1(MAXRANK_IREGI,i)
          IF(.NOT.ALLOCATED(xitemp(i)%xitempi))THEN
             ALLOCATE(xitemp(i)%xitempi(xitemp(i)%dim1,xitemp(i)%dim2))
          ENDIF
       ENDDO
       IF(.NOT.ALLOCATED(factor_xi))THEN
          j=xiarray_arg1(MAXRANK_IREGI,MAXNLOOP_IREGI-1)
          ALLOCATE(factor_xi(j))
       ENDIF
       init=1
    ENDIF
    mom(0:3)=PCL(1,0:3)
    IF(NLOOPLINE.GE.2)THEN
       indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
       i2=paveindices(1)
       IF(i2.EQ.0)THEN
          indices0(1)=ind1
          IF(.NOT.RECYCLING)THEN
             PCL1(1,0:3)=0d0
             DO i=2,NLOOPLINE
                PCL1(i,0:3)=PCL(i,0:3)-mom(0:3)
             ENDDO
          ENDIF
          pave(1:4)=comp_pavefun_reduce(NLOOPLINE,indices0,PCL1,M2L)
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ELSE
          pave(1:4)=DCMPLX(0d0)
          factor0=DBLE(factorial(i2))
          IF(.NOT.RECYCLING)THEN
             PCL1(1,0:3)=0d0
             DO i=2,NLOOPLINE
                PCL1(i,0:3)=PCL(i,0:3)-mom(0:3)
             ENDDO
          ENDIF
          indices0(1)=ind1
          DO i=0,i2
             ! solve for sumf
             sumf(1:4)=DCMPLX(0d0)
             SELECT CASE(NLOOPLINE-1)
             CASE(1)
                ntot=1
                CALL all_integers(NLOOPLINE-1,ntot,i,xitemp(1)%xitempi(1:ntot,1:1),&
                     factor_xi(1:ntot))
                DO j=1,ntot
                   indices00(0:NLOOPLINE)=indices0(0:NLOOPLINE)
                   indices00(2:NLOOPLINE)=indices00(2:NLOOPLINE)&
                        +xitemp(1)%xitempi(j,1:1)
                   sumf(1:4)=sumf(1:4)+factor_xi(j)*&
                        comp_pavefun_reduce(NLOOPLINE,indices00,PCL1,M2L)
                ENDDO
             CASE DEFAULT
                ntot=ntot_xiarray(i,NLOOPLINE-1)
                CALL all_integers(NLOOPLINE-1,ntot,i,&
                     xitemp(NLOOPLINE-1)%xitempi(1:ntot,1:NLOOPLINE-1),&
                     factor_xi(1:ntot))
                DO j=1,ntot
                   indices00(0:NLOOPLINE)=indices0(0:NLOOPLINE)
                   indices00(2:NLOOPLINE)=indices00(2:NLOOPLINE)&
                        +xitemp(NLOOPLINE-1)%xitempi(j,1:NLOOPLINE-1)
                   sumf(1:4)=sumf(1:4)+factor_xi(j)*&
                        comp_pavefun_reduce(NLOOPLINE,indices00,PCL1,M2L)
                ENDDO
             END SELECT
             factor=factor0/DBLE(factorial(i2-i))/DBLE(factorial(i))
             pave(1:4)=pave(1:4)+factor*sumf(1:4)
          ENDDO
          pave(1:4)=pave(1:4)*(-1)**i2
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
    ELSE
       i1=paveindices(1)
       indices0(0)=paveindices(0)
       indices0(1)=0
       PCL1(1,0:3)=0d0
       pave(1:4)=(-1d0)**i1*comp_pavefun_reduce(NLOOPLINE,indices0,PCL1,M2L)
       IF(RECYCLING)item%value(1:4)=pave(1:4)
       RETURN
    ENDIF
  END FUNCTION comp_shiftpaveden

  RECURSIVE FUNCTION comp_pavefun_reduce(NLOOPLINE,paveindices,PCL,M2L) RESULT(pave)
    IMPLICIT NONE
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pave
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    LOGICAL::indices_zero
    INTEGER::i,j,k,lind,nindtot,n,ind1,ii,jj,pos,postemp,kk,ll,ss,PP
    INTEGER,DIMENSION(0:NLOOPLINE)::indices0
    INTEGER,DIMENSION(0:NLOOPLINE-1)::indices1
    REAL(KIND(1d0))::P02
    COMPLEX(KIND(1d0))::prefactor,temp
    COMPLEX(KIND(1d0)),PARAMETER::imag=DCMPLX(0d0,1d0)
    REAL(KIND(1d0)),PARAMETER::pi=3.141592653589793d0
    REAL(KIND(1d0)),PARAMETER::eulergamma=0.5772156649015329d0
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    REAL(KIND(1d0)),DIMENSION(0:3)::P0
    COMPLEX(KIND(1d0)),DIMENSION(2:NLOOPLINE,2:NLOOPLINE)::ZMATRIX
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE)::XMATRIX,YMATRIX
    COMPLEX(KIND(1d0))::detY,detZ,yy1,xx2,xx1,xx3,xx4,xx5
    REAL(KIND(1d0))::maxtemp
    COMPLEX(KIND(1d0))::x1,x2,cprefactor
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL0
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE-1,0:3)::PCL1
    COMPLEX(KIND(1d0)),DIMENSION(1:NLOOPLINE-1)::M2L1
    COMPLEX(KIND(1d0)),DIMENSION(2:NLOOPLINE,2:NLOOPLINE)::InvZMATRIX
    LOGICAL::OK_FLAG
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pavetemp
    COMPLEX(KIND(1d0)),DIMENSION(2,2)::xxList2
    COMPLEX(KIND(1d0)),DIMENSION(3,3)::xxList3
    INTEGER::ll1,ll2,ll3
    INTEGER,DIMENSION(MAXNLOOP_IREGI*MAXINDICES_IREGI)::llarray
    TYPE(cibppave_node),POINTER::item
    LOGICAL::find
    pave(1:4)=DCMPLX(0d0)
    IF(.NOT.STABLE_IREGI)RETURN
    IF(RECYCLING)THEN
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%stable=.TRUE.
       item%indices(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL(1:NLOOPLINE,0:3)
       CALL cibppave_bt_search(item,cpave_save,find)
       IF(find)THEN
          pave(1:4)=item%value(1:4)
          STABLE_IREGI=item%stable
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    indices_zero=.TRUE.
    lind=-1
    nindtot=0
    DO i=0,NLOOPLINE
       IF(paveindices(i).LT.0)THEN
          pave(1:4)=DCMPLX(0d0)
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
       IF(paveindices(i).NE.0.AND.indices_zero)THEN
          indices_zero=.FALSE.
          lind=i
       ENDIF
       nindtot=nindtot+paveindices(i)
       indices0(i)=paveindices(i)
    ENDDO
    ! the end of recursion
    IF(NLOOPLINE.LE.4.AND.indices_zero)THEN
       pave(1:4)=I0CC1(NLOOPLINE,PCL,M2L)
       IF(RECYCLING)item%value(1:4)=pave(1:4)
       RETURN
    ENDIF
    ! 1-point functions Eq.(3.4) in hep-ph/0509141
    IF(NLOOPLINE.EQ.1)THEN
       n=paveindices(0)/2
       prefactor=(M2L(1)/DCMPLX(2d0))**(n)/DCMPLX(factorial(n+1))
       pave(1:4)=I0CC1(NLOOPLINE,PCL,M2L)
       temp=DCMPLX(0d0)
       DO i=1,n
          temp=temp+DCMPLX(1d0)/DCMPLX(i+1)
       ENDDO
       pave(4)=imag*pi**2*M2L(1)*temp+pave(4)
       pave(1:4)=prefactor*pave(1:4)
       IF(RECYCLING)item%value(1:4)=pave(1:4)
       RETURN
    ENDIF
    ! 2-point functions
    IF(NLOOPLINE.EQ.2)THEN
       ! Eqs.(4.1-4.13)
       P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
       P02=scalarprod(P0,P0)
       IF(ABS(M2L(1)).LT.EPS.AND.ABS(M2L(2)).LT.EPS&
            .AND.ABS(P02).LT.EPS)THEN
          IF(lind.EQ.0)THEN
             pave(1:4)=DCMPLX(0d0)
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ENDIF
          pave(1:4)=I0CC1(NLOOPLINE,PCL,M2L)
          DO i=0,paveindices(2)-1
             prefactor=DCMPLX(1d0)/DCMPLX(2+i)
             pave(4)=-pave(4)*prefactor&
                  -2d0*prefactor**2*(pave(1)+pave(2))&
                  -4d0*prefactor**3*pave(3)
             pave(1)=-pave(1)*prefactor
             pave(2)=-pave(2)*prefactor-2d0*prefactor**2*pave(3)
             pave(3)=-prefactor*pave(3)
             pave(1:4)=pave(1:4)*(i+1)
          ENDDO
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
       IF(lind.EQ.0)THEN
          ! Eq.(5.10), (5.40), (5.50), (5.53)
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
               XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detZ).GE.EPS)THEN
             ! Eq.(5.10)
             indices1(0)=paveindices(0)-2
             indices0(0)=indices0(0)-2
             indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
             ind1=paveindices(1)
             pave(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)&
                  +2d0*M2L(1)*comp_pavefun_reduce(NLOOPLINE,indices0,&
                  PCL,M2L)
             indices0(2)=indices0(2)+1
             pave(1:4)=pave(1:4)+comp_pave_f(1,NLOOPLINE,PCL,M2L)*&
                  comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
             pave(4)=pave(4)+4d0*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
             pave(1:4)=pave(1:4)/2d0/DCMPLX(paveindices(0)+paveindices(2)+1)
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSE
             yy1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
             IF(ABS(yy1).GE.EPS)THEN
                ! Eq.(5.40)
                pave(1:4)=comp_pave_S(paveindices(0)+paveindices(2),0,&
                     NLOOPLINE,paveindices,PCL,M2L)
                pave(4)=pave(4)+4d0*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
                pave(1:4)=pave(1:4)/2d0&
                     /DCMPLX(2+paveindices(0)+2*paveindices(2))
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                ! Eq.(5.50)
                indices0(0)=indices0(0)-2
                indices0(2)=indices0(2)+2
                pave(1:4)=comp_pave_S1(indices0(0)+indices0(2),1,&
                     NLOOPLINE,indices0,PCL,M2L)
                pave(1:4)=pave(1:4)/(DBLE(2*(1+paveindices(2))))
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ENDIF
          ENDIF
       ENDIF
       IF(lind.EQ.2)THEN
          ii=paveindices(2)
          P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
          P02=scalarprod(P0,P0)
          IF(ii.GT.3.OR.(ii.GT.0.AND.ABS(P02).LT.EPS))THEN
             ! Eq.(4.8)
             IF(ABS(M2L(1)).GE.EPS)THEN
                IF(ABS(P02).LT.EPS.AND.ABS(M2L(1)-M2L(2)).GE.EPS)THEN
                   x1=(M2L(1)-imag*EPS*10d0)&
                        /(M2L(1)-M2L(2))
                   x2=DCMPLX(0d0)
                ELSEIF(ABS(P02).LT.EPS.AND.ABS(M2L(1)-M2L(2)).LT.EPS)THEN
                   x1=DCMPLX(0d0)
                   x2=DCMPLX(0d0)
                ELSEIF(ABS(P02).GE.EPS)THEN
                   x1=((DCMPLX(P02)+M2L(1)-M2L(2)))&
                        +SQRT((M2L(2)-M2L(1)-DCMPLX(P02))**2&
                        -4d0*P02*(M2L(1)-imag*10*EPS))&
                        /DCMPLX(2d0*P02)
                   x2=((DCMPLX(P02)+M2L(1)-M2L(2)))&
                        -SQRT((M2L(2)-M2L(1)-DCMPLX(P02))**2&
                        -4d0*P02*(M2L(1)-imag*10*EPS))&
                        /DCMPLX(2d0*P02)
                ENDIF
                cprefactor=imag*pi**2*(-1)**ii/DBLE(ii+1)
                pave(1)=DCMPLX(1d0)
                pave(2:3)=DCMPLX(0d0)
                pave(4)=-eulergamma-DLOG(pi)+LOG(MU_R_IREGI**2/M2L(1))&
                     -AuxiliaryFunction(ii,x1)&
                     -AuxiliaryFunction(ii,x2)
                pave(1:4)=cprefactor*pave(1:4)
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                cprefactor=imag*pi**2*(-1)**ii/DBLE(ii+1)
                IF(ABS(DCMPLX(P02)-M2L(2)).GE.EPS)THEN
                   IF(ABS(P02).LT.EPS)THEN
                      pave(1)=DCMPLX(1d0)
                      pave(2:3)=DCMPLX(0d0)
                      pave(4)=-eulergamma-DLOG(pi)&
                           +LOG(DCMPLX(MU_R_IREGI**2)/(M2L(2)-imag*10*EPS))&
                           +DCMPLX(1d0)/DCMPLX(DBLE(ii+1))
                      pave(1:4)=pave(1:4)*cprefactor
                      IF(RECYCLING)item%value(1:4)=pave(1:4)
                      RETURN
                   ENDIF
                   pave(1)=DCMPLX(1d0)
                   pave(2:3)=DCMPLX(0d0)
                   pave(4)=-eulergamma-DLOG(pi)&
                        +LOG(DCMPLX(MU_R_IREGI**2)/(M2L(2)-DCMPLX(P02)-imag*10*EPS))&
                        +DCMPLX(1d0)/DCMPLX(DBLE(ii+1))&
                        -AuxiliaryFunction(ii,DCMPLX(1d0)&
                        -(M2L(2)-imag*10*EPS)/DCMPLX(P02))
                   pave(1:4)=pave(1:4)*cprefactor
                   IF(RECYCLING)item%value(1:4)=pave(1:4)
                   RETURN
                ELSE
                   IF(ABS(P02).GE.EPS)THEN
                      pave(1)=DCMPLX(1d0)
                      pave(2:3)=DCMPLX(0d0)
                      pave(4)=-eulergamma-DLOG(pi)&
                           +LOG(DCMPLX(MU_R_IREGI**2)/(DCMPLX(P02)-imag*10*EPS))&
                           +DCMPLX(2d0)/DCMPLX(DBLE(1+ii))
                      pave(1:4)=cprefactor*pave(1:4)
                      IF(RECYCLING)item%value(1:4)=pave(1:4)
                      RETURN
                   ELSE
                      pave(1:4)=0d0
                      IF(RECYCLING)item%value(1:4)=pave(1:4)
                      RETURN
                   ENDIF
                ENDIF
             ENDIF
          ELSE
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detZ).GE.EPS)THEN
                ! Eq.(5.11)
                CALL CMNXNINV(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                     InvZMATRIX(2:NLOOPLINE,2:NLOOPLINE),OK_FLAG)
                IF(.NOT.OK_FLAG)THEN
                   STABLE_IREGI=.FALSE.
                   IF(RECYCLING)THEN
                      item%value(1:4)=DCMPLX(0d0)
                      item%stable=.FALSE.
                   ENDIF
                   RETURN
                ENDIF
                indices0(0)=2
                indices0(2)=indices0(2)-2
                pave(1:4)=InvZMATRIX(2,2)&
                     *comp_pave_S(ii,1,NLOOPLINE,paveindices,PCL,M2L)
                IF(ii.NE.1)THEN
                   pave(1:4)=pave(1:4)-InvZMATRIX(2,2)*2d0*(ii-1)*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
                IF(ABS(xx2).GE.EPS)THEN
                   ! Eq.(5.38)
                   indices0(0)=0
                   indices0(1)=0
                   indices0(2)=ii+2
                   pave(1:4)=comp_pave_S1(ii+1,1,NLOOPLINE,indices0,PCL,M2L)
                   indices0(0)=2
                   indices0(1)=0
                   indices0(2)=ii-1
                   pave(1:4)=pave(1:4)-2d0*ii&
                        *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   pave(1:4)=-1d0/xx2*pave(1:4)
                   IF(RECYCLING)item%value(1:4)=pave(1:4)
                   RETURN
                ELSE
                   ! Eq.(5.53)
                   xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,2,2)
                   IF(ABS(xx2).GE.EPS)THEN
                      indices0(0)=2
                      pave(1:4)=2d0*(3+ii)*comp_pavefun_reduce(NLOOPLINE,indices0,&
                           PCL,M2L)
                      pave(4)=pave(4)-4d0*comp_pave_UV(NLOOPLINE,indices0,PCL,M2L)
                      indices1(0)=paveindices(0)
                      indices1(1)=paveindices(2)
                      ind1=paveindices(1)
                      pave(1:4)=pave(1:4)&
                           -comp_shiftpaveden(NLOOPLINE-1,indices1,&
                           PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                      pave(1:4)=1d0/xx2*pave(1:4)
                      IF(RECYCLING)item%value(1:4)=pave(1:4)
                      RETURN
                   ELSE
                      STABLE_IREGI=.FALSE.
                      IF(RECYCLING)THEN
                         item%value(1:4)=DCMPLX(0d0)
                         item%stable=.FALSE.
                      ENDIF
                      RETURN
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDIF
    ENDIF
    ! 3-point functions
    IF(NLOOPLINE.EQ.3)THEN
       IF(lind.GE.2)THEN
          ! Eqs.(5.11,5.38,5.53)
          ii=paveindices(2)
          jj=paveindices(3)
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detZ).GE.EPS)THEN
             ! Eq.(5.11)
             CALL CMNXNINV(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                  InvZMATRIX(2:NLOOPLINE,2:NLOOPLINE),OK_FLAG)
             IF(.NOT.OK_FLAG)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             IF(ii.GT.0)THEN
                pave(1:4)=DCMPLX(0d0)
                maxtemp=MAXVAL(ABS(INVZMATRIX(2:NLOOPLINE,2:NLOOPLINE)))
                maxtemp=MAX(maxtemp,1d-99)
                DO i=2,NLOOPLINE
                   IF(ABS(InvZMATRIX(2,i)/maxtemp).LT.EPS)CYCLE
                   indices0(0:3)=paveindices(0:3)
                   indices0(2)=indices0(2)-1
                   indices0(i)=indices0(i)+1
                   pave(1:4)=pave(1:4)+InvZMATRIX(2,i)&
                        *comp_pave_S(ii+jj,i-1,NLOOPLINE,indices0,PCL,M2L)
                   IF(i.EQ.2)THEN
                      IF(ABS(DBLE(ii-1)).GE.EPS)THEN
                         indices0(0:3)=paveindices(0:3)
                         indices0(0)=2
                         indices0(2)=indices0(2)-2
                         pave(1:4)=pave(1:4)-InvZMATRIX(2,i)*2d0*(ii-1)&
                              *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      ENDIF
                   ELSE
                      IF(ABS(DBLE(jj)).GE.EPS)THEN
                         indices0(0:3)=paveindices(0:3)
                         indices0(0)=2
                         indices0(2)=indices0(2)-1
                         indices0(3)=indices0(3)-1
                         pave(1:4)=pave(1:4)-InvZMATRIX(2,i)*2d0*jj*&
                              comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      ENDIF
                   ENDIF
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                pave(1:4)=DCMPLX(0d0)
                maxtemp=MAXVAL(ABS(INVZMATRIX(2:NLOOPLINE,2:NLOOPLINE)))
                maxtemp=MAX(maxtemp,1d-99)
                DO i=2,NLOOPLINE
                   IF(ABS(InvZMATRIX(3,i)/maxtemp).LT.EPS)CYCLE
                   indices0(0:3)=paveindices(0:3)
                   indices0(3)=indices0(3)-1
                   indices0(i)=indices0(i)+1
                   pave(1:4)=pave(1:4)+InvZMATRIX(3,i)&
                        *comp_pave_S(jj,i-1,NLOOPLINE,indices0,PCL,M2L)
                   IF(i.NE.NLOOPLINE.OR.ABS(jj-1).LT.EPS)CYCLE
                   indices0(0:3)=paveindices(0:3)
                   indices0(0)=2
                   indices0(3)=indices0(3)-2
                   pave(1:4)=pave(1:4)-InvZMATRIX(3,i)*2d0*(jj-1)&
                        *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ENDIF
          ELSE
             pos=1
             xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
             DO i=3,NLOOPLINE
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,3)
                IF(ABS(xx1).GT.ABS(xx2))THEN
                   xx2=xx1
                   pos=i-1
                ENDIF
             ENDDO
             IF(ABS(xx2).GE.EPS)THEN
                ! Eq.(5.38)
                pave(1:4)=DCMPLX(0d0)
                DO i=1,2
                   xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                        ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),pos,i)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   indices0(0:3)=paveindices(0:3)
                   indices0(i+1)=indices0(i+1)+1
                   pave(1:4)=pave(1:4)+xx1*comp_pave_S1(ii+jj+1,i,NLOOPLINE,&
                        indices0,PCL,M2L)
                   IF(i.EQ.1)THEN
                      IF(ii.NE.0)THEN
                         indices0(0:3)=paveindices(0:3)
                         indices0(0)=2
                         indices0(2)=indices0(2)-1
                         pave(1:4)=pave(1:4)-2d0*ii*xx1&
                              *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      ENDIF
                   ELSE
                      IF(jj.NE.0)THEN
                         indices0(0:3)=paveindices(0:3)
                         indices0(0)=2
                         indices0(3)=indices0(3)-1
                         pave(1:4)=pave(1:4)-2d0*jj*xx1&
                              *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      ENDIF
                   ENDIF
                ENDDO
                pave(1:4)=-pave(1:4)/xx2
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                ! Eq.(5.53)
                xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,2,2)
                pos=1
                postemp=1
                DO i=2,3
                   DO j=2,3
                      IF(i.EQ.2.AND.j.EQ.2)CYCLE
                      xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i,j)
                      IF(ABS(xx1).GT.ABS(xx2))THEN
                         xx2=xx1
                         pos=i-1
                         postemp=j-1
                      ENDIF
                   ENDDO
                ENDDO
                IF(ABS(xx2).GE.EPS)THEN
                   pave(1:4)=DCMPLX(0d0)
                   xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                        ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                        pos,postemp)
                   IF(ABS(xx1).GE.EPS)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=2
                      pave(1:4)=pave(1:4)-xx1/xx2*2d0*(2+ii+jj)&
                           *comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      pave(4)=pave(4)-4d0*xx1/xx2&
                           *comp_pave_UV(NLOOPLINE,indices0,PCL,M2L)
                      indices1(0)=paveindices(0)
                      indices1(1:2)=paveindices(2:3)
                      ind1=paveindices(1)
                      pave(1:4)=pave(1:4)-xx1/xx2*&
                           comp_shiftpaveden(NLOOPLINE-1,indices1,&
                           PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                   ENDIF
                   DO i=1,2
                      DO j=1,2
                         xx1=SIGNED_CMINOR22(NLOOPLINE-1,&
                              ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                              pos,i,postemp,j)*comp_pave_f(i,NLOOPLINE,PCL,M2L)
                         IF(ABS(x1).LT.EPS)CYCLE
                         indices0(0:3)=paveindices(0:3)
                         indices0(j+1)=indices0(j+1)+1
                         pave(1:4)=pave(1:4)+xx1/xx2*comp_pave_S1(ii+jj+1,j,&
                              NLOOPLINE,indices0,PCL,M2L)
                         IF(j.EQ.1)THEN
                            IF(ii.NE.0)THEN
                               indices0(0:3)=paveindices(0:3)
                               indices0(0)=2
                               indices0(2)=indices0(2)-1
                               pave(1:4)=pave(1:4)-xx1/xx2*2d0*ii*&
                                    comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                            ENDIF
                         ELSE
                            IF(jj.NE.0)THEN
                               indices0(0:3)=paveindices(0:3)
                               indices0(0)=2
                               indices0(3)=indices0(3)-1
                               pave(1:4)=pave(1:4)-xx1/xx2*2d0*jj*&
                                    comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                            ENDIF
                         ENDIF
                      ENDDO
                   ENDDO
                   IF(RECYCLING)item%value(1:4)=pave(1:4)
                   RETURN
                ELSE
                   STABLE_IREGI=.FALSE.
                   IF(RECYCLING)THEN
                      item%value(1:4)=DCMPLX(0d0)
                      item%stable=.FALSE.
                   ENDIF
                   RETURN
                ENDIF
             ENDIF
          ENDIF
       ENDIF
       ! Eqs.(5.10,5.40,5.50,5.53)
       kk=paveindices(0)
       ii=paveindices(2)
       jj=paveindices(3)
       CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
            XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
       IF(ABS(detZ).GE.EPS)THEN
          ! Eq.(5.10)
          indices1(0)=paveindices(0)-2
          indices1(1:2)=paveindices(2:3)
          ind1=paveindices(1)
          pave(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
               PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
          IF(ABS(M2L(1)).GE.EPS)THEN
             indices0(0:3)=paveindices(0:3)
             indices0(0)=indices0(0)-2
             pave(1:4)=pave(1:4)+2d0*M2L(1)*comp_pavefun_reduce(NLOOPLINE,&
                  indices0,PCL,M2L)
          ENDIF
          DO i=1,2
             xx1=comp_pave_f(i,NLOOPLINE,PCL,M2L)
             IF(ABS(xx1).LT.EPS)CYCLE
             indices0(0:3)=paveindices(0:3)
             indices0(0)=indices0(0)-2
             indices0(i+1)=indices0(i+1)+1
             pave(1:4)=pave(1:4)+xx1*comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
          ENDDO
          pave(4)=pave(4)+4d0*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
          pave(1:4)=pave(1:4)/(2d0*(ii+jj+kk))
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ELSE
          pos=1
          postemp=1
          xx2=SIGNED_CMINOR11(NLOOPLINE-1,&
               ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),1,1)
          xxList2(1,1)=xx2
          DO i=1,2
             DO j=1,2
                IF(i.EQ.1.AND.j.EQ.1)CYCLE
                xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                     ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                     i,j)
                xxList2(i,j)=xx1
                IF(ABS(xx1).GT.ABS(xx2))THEN
                   pos=i
                   postemp=j
                   xx2=xx1
                ENDIF
             ENDDO
          ENDDO
          yy1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
          xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,3)
          IF(ABS(yy1).LT.ABS(xx1))THEN
             yy1=xx1
          ENDIF
          IF(ABS(yy1).GE.EPS.AND.ABS(xx2).GE.EPS)THEN
             ! Eq.(5.40)
             pave(1:4)=comp_pave_S(kk+ii+jj,0,NLOOPLINE,&
                  paveindices,PCL,M2L)
             pave(4)=pave(4)+4d0*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
             DO i=1,2
                IF(ABS(xxList2(i,postemp)).GE.EPS)THEN
                   indices0(0:3)=paveindices(0:3)
                   indices0(0)=indices0(0)-2
                   indices0(i+1)=indices0(i+1)+1
                   indices0(pos+1)=indices0(pos+1)+1
                   pave(1:4)=pave(1:4)+1d0/xx2*&
                        xxList2(i,postemp)&
                        *comp_pave_S1(kk+ii+jj,i,NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(ABS(xxList2(pos,postemp)).GE.EPS)THEN
                   indices0(0:3)=paveindices(0:3)
                   indices0(0)=indices0(0)-2
                   indices0(i+1)=indices0(i+1)+2
                   pave(1:4)=pave(1:4)-1d0/xx2*&
                        xxList2(pos,postemp)*comp_pave_S1(&
                        kk+ii+jj,i,NLOOPLINE,indices0,PCL,M2L)
                ENDIF
             ENDDO
             DO i=1,2
                DO j=1,2
                   xx1=SIGNED_CMINOR22(NLOOPLINE-1,&
                        ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                        pos,i,postemp,j)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   xx3=comp_pave_f(i,NLOOPLINE,PCL,M2L)
                   IF(ABS(xx3).GE.EPS)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=indices0(0)-2
                      indices0(j+1)=indices0(j+1)+1
                      pave(1:4)=pave(1:4)-xx3*xx1/xx2*&
                           comp_pave_S1(ii+jj+kk-1,j,NLOOPLINE,&
                           indices0,PCL,M2L)
                   ENDIF
                   IF(ii.NE.0.AND.i.EQ.1)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(2)=indices0(2)-1
                      indices0(j+1)=indices0(j+1)+1
                      pave(1:4)=pave(1:4)-2d0*ii*xx1/xx2*&
                           comp_pave_S1(kk+ii+jj,j,NLOOPLINE,&
                           indices0,PCL,M2L)
                   ENDIF
                   IF(jj.NE.0.AND.i.EQ.2)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(3)=indices0(3)-1
                      indices0(j+1)=indices0(j+1)+1
                      pave(1:4)=pave(1:4)-2d0*jj*xx1/xx2*&
                           comp_pave_S1(kk+ii+jj,j,NLOOPLINE,&
                           indices0,PCL,M2L)
                   ENDIF
                   xx4=comp_pave_f(j,NLOOPLINE,PCL,M2L)
                   IF(ABS(xx3*xx4).GE.EPS)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=indices0(0)-2
                      pave(1:4)=pave(1:4)+xx1*xx3*xx4/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   xx5=DCMPLX(0d0)
                   IF(i.EQ.1)xx5=xx5+xx4
                   IF(j.EQ.1)xx5=xx5+xx3
                   IF(ii.NE.0.AND.ABS(xx5).GE.EPS)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(2)=indices0(2)-1
                      pave(1:4)=pave(1:4)+2d0*ii*xx5*xx1/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   xx5=DCMPLX(0d0)
                   IF(i.EQ.2)xx5=xx5+xx4
                   IF(j.EQ.2)xx5=xx5+xx3
                   IF(jj.NE.0.AND.ABS(xx5).GE.EPS)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(3)=indices0(3)-1
                      pave(1:4)=pave(1:4)+2d0*jj*xx5*xx1/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(ii.NE.0.AND.ii.NE.1.AND.i.EQ.1.AND.j.EQ.1)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=indices0(0)+2
                      indices0(2)=indices0(2)-2
                      pave(1:4)=pave(1:4)+4d0*ii*(ii-1)*xx1/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(jj.NE.0.AND.jj.NE.1.AND.i.EQ.2.AND.j.EQ.2)THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=indices0(0)+2
                      indices0(3)=indices0(3)-2
                      pave(1:4)=pave(1:4)+4d0*jj*(jj-1)*xx1/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(ii.NE.0.AND.jj.NE.0.AND.((i.EQ.1.AND.j.EQ.2)&
                        .OR.(i.EQ.2.OR.j.EQ.1)))THEN
                      indices0(0:3)=paveindices(0:3)
                      indices0(0)=indices0(0)+2
                      indices0(2)=indices0(2)-1
                      indices0(3)=indices0(3)-1
                      pave(1:4)=pave(1:4)+4d0*ii*jj*xx1/xx2*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                ENDDO
             ENDDO
             pave(1:4)=pave(1:4)/(2d0*DBLE(1+kk+2*ii+2*jj))
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSEIF(ABS(yy1).LT.EPS.AND.ABS(xx2).GE.EPS)THEN
             ! Eq.(5.50)
             ll1=pos
             IF(paveindices(pos+1).LT.paveindices(postemp+1))ll1=postemp
             ll2=3-ll1
             IF(paveindices(ll2+1).NE.0.AND.&
                  ABS(xxList2(pos+postemp-ll1,ll2)).GE.EPS)THEN
                indices0(0:3)=paveindices(0:3)
                indices0(ll1+1)=indices0(ll1+1)+1
                indices0(ll2+1)=indices0(ll2+1)-1
                pave(1:4)=-2d0*paveindices(ll2+1)&
                     *xxList2(pos+postemp-ll1,ll2)*&
                     comp_pavefun_reduce(NLOOPLINE,indices0,&
                     PCL,M2L)
             ENDIF
             DO i=1,2
                IF(ABS(xxList2(pos+postemp-ll1,i)).LT.EPS)CYCLE
                indices0(0:3)=paveindices(0:3)
                indices0(0)=indices0(0)-2
                indices0(ll1+1)=indices0(ll1+1)+1
                indices0(i+1)=indices0(i+1)+1
                pave(1:4)=pave(1:4)+xxList2(pos+postemp-ll1,i)*&
                     comp_pave_S1(kk+ii+jj,i,NLOOPLINE,indices0,PCL,M2L)
             ENDDO
             pave(1:4)=pave(1:4)/xx2/(2d0*(1+paveindices(ll1+1)))
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSE
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
       ENDIF
    ENDIF
    ! 4-point functions
    IF(NLOOPLINE.EQ.4)THEN
       IF(lind.GE.2)THEN
          ! Eqs.(5.11,5.38,5.53)
          ii=paveindices(2)
          jj=paveindices(3)
          kk=paveindices(4)
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
               XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detZ).GE.EPS)THEN
             ! Eq.(5.11)
             CALL CMNXNINV(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                  InvZMATRIX(2:NLOOPLINE,2:NLOOPLINE),OK_FLAG)
             IF(.NOT.OK_FLAG)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             IF(ii.GT.0)THEN
                pave(1:4)=DCMPLX(0d0)
                maxtemp=MAXVAL(ABS(INVZMATRIX(2:NLOOPLINE,2:NLOOPLINE)))
                maxtemp=MAX(maxtemp,1d-99)
                DO i=1,3
                   xx1=InvZMATRIX(2,i+1)
                   IF(ABS(xx1/maxtemp).LT.EPS)CYCLE
                   indices0(0:4)=paveindices(0:4)
                   indices0(2)=indices0(2)-1
                   indices0(i+1)=indices0(i+1)+1
                   pave(1:4)=pave(1:4)+xx1*&
                        comp_pave_S(ii+jj+kk,i,NLOOPLINE,&
                        indices0,PCL,M2L)
                   IF(ii.NE.1.AND.i.EQ.1)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(2)=indices0(2)-2
                      pave(1:4)=pave(1:4)-xx1*2d0*(ii-1)*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(jj.NE.0.AND.i.EQ.2)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(2)=indices0(2)-1
                      indices0(3)=indices0(3)-1
                      pave(1:4)=pave(1:4)-xx1*2d0*jj*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(kk.NE.0.AND.i.EQ.3)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(2)=indices0(2)-1
                      indices0(4)=indices0(4)-1
                      pave(1:4)=pave(1:4)-xx1*2d0*kk*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSEIF(ii.EQ.0.AND.jj.GT.0)THEN
                pave(1:4)=DCMPLX(0d0)
                maxtemp=MAXVAL(ABS(INVZMATRIX(2:NLOOPLINE,2:NLOOPLINE)))
                maxtemp=MAX(maxtemp,1d-99)
                DO i=1,3
                   xx1=InvZMATRIX(3,i+1)
                   IF(ABS(xx1/maxtemp).LT.EPS)CYCLE
                   indices0(0:4)=paveindices(0:4)
                   indices0(3)=indices0(3)-1
                   indices0(i+1)=indices0(i+1)+1
                   pave(1:4)=pave(1:4)+xx1*comp_pave_S(jj+kk,i,NLOOPLINE,&
                        indices0,PCL,M2L)
                   IF(jj.NE.1.AND.i.EQ.2)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(3)=indices0(3)-2
                      pave(1:4)=pave(1:4)-2d0*xx1*(jj-1)*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(kk.NE.0.AND.i.EQ.3)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(3)=indices0(3)-1
                      indices0(4)=indices0(4)-1
                      pave(1:4)=pave(1:4)-2d0*kk*xx1*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSEIF(ii.EQ.0.AND.jj.EQ.0.AND.kk.GT.0)THEN
                pave(1:4)=DCMPLX(0d0)
                maxtemp=MAXVAL(ABS(INVZMATRIX(2:NLOOPLINE,2:NLOOPLINE)))
                maxtemp=MAX(maxtemp,1d-99)
                DO i=1,3
                   xx1=InvZMATRIX(4,i+1)
                   IF(ABS(xx1/maxtemp).LT.EPS)CYCLE
                   indices0(0:4)=paveindices(0:4)
                   indices0(4)=indices0(4)-1
                   indices0(i+1)=indices0(i+1)+1
                   pave(1:4)=pave(1:4)+xx1*comp_pave_S(kk,i,NLOOPLINE,&
                        indices0,PCL,M2L)
                   IF(kk.NE.1.AND.i.EQ.3)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(4)=indices0(4)-2
                      pave(1:4)=pave(1:4)-xx1*2d0*(kk-1)*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ENDIF
          ELSE
             xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
             pos=1
             DO i=3,4
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,i)
                IF(ABS(xx1).GT.ABS(xx2))THEN
                   xx2=xx1
                   pos=i-1
                ENDIF
             ENDDO
             IF(ABS(xx2).GE.EPS)THEN
                ! Eq.(5.38)
                pave(1:4)=DCMPLX(0d0)
                DO i=1,3
                   xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                        ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),pos,i)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   indices0(0:4)=paveindices(0:4)
                   indices0(i+1)=indices0(i+1)+1
                   pave(1:4)=pave(1:4)+xx1*comp_pave_S1(ii+jj+kk+1,i,NLOOPLINE,&
                        indices0,PCL,M2L)
                   IF(ii.NE.0.AND.i.EQ.1)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(2)=indices0(2)-1
                      pave(1:4)=pave(1:4)-2d0*ii*xx1*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(jj.NE.0.AND.i.EQ.2)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(3)=indices0(3)-1
                      pave(1:4)=pave(1:4)-2d0*jj*xx1*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                   IF(kk.NE.0.AND.i.EQ.3)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices0(4)=indices0(4)-1
                      pave(1:4)=pave(1:4)-2d0*kk*xx1*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                   ENDIF
                ENDDO
                pave(1:4)=pave(1:4)/(-xx2)
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                ! Eq.(5.53)
                xx2=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,2,2)
                pos=1
                postemp=1
                xxList3(1,1)=xx2
                DO i=2,4
                   DO j=2,4
                      IF(i.EQ.2.AND.j.EQ.2)CYCLE
                      xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i,j)
                      xxList3(i-1,j-1)=xx1
                      IF(ABS(xx1).GT.ABS(xx2))THEN
                         xx2=xx1
                         pos=i-1
                         postemp=j-1
                      ENDIF
                   ENDDO
                ENDDO
                IF(ABS(xx2).GE.EPS)THEN
                   xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                        ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),pos,postemp)
                   pave(1:4)=DCMPLX(0d0)
                   IF(ABS(xx1).GE.EPS)THEN
                      indices0(0:4)=paveindices(0:4)
                      indices0(0)=2
                      indices1(0)=paveindices(0)
                      indices1(1:3)=paveindices(2:4)
                      ind1=paveindices(1)
                      pave(1:4)=pave(1:4)+xx1/xx2*2d0*(1+ii+jj+kk)*&
                           comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                      pave(4)=pave(4)-4d0*xx1/xx2*comp_pave_UV(NLOOPLINE,indices0,PCL,M2L)
                      pave(1:4)=pave(1:4)-xx1/xx2*comp_shiftpaveden(NLOOPLINE-1,indices1,&
                           PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                   ENDIF
                   DO i=1,3
                      xx3=comp_pave_f(i,NLOOPLINE,PCL,M2L)
                      IF(ABS(xx3).LT.EPS)CYCLE
                      DO j=1,3
                         xx1=SIGNED_CMINOR22(NLOOPLINE-1,&
                              ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),pos,i,postemp,j)
                         IF(ABS(xx1).LT.EPS)CYCLE
                         indices0(0:4)=paveindices(0:4)
                         indices0(j+1)=indices0(j+1)+1
                         pave(1:4)=pave(1:4)+xx1*xx3/xx2*&
                              comp_pave_S1(ii+jj+kk+1,j,NLOOPLINE,indices0,PCL,M2L)
                         IF(ii.NE.0.AND.j.EQ.1)THEN
                            indices0(0:4)=paveindices(0:4)
                            indices0(0)=2
                            indices0(2)=indices0(2)-1
                            pave(1:4)=pave(1:4)-2d0*ii*xx1*xx3/xx2*&
                                 comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                         ENDIF
                         IF(jj.NE.0.AND.j.EQ.2)THEN
                            indices0(0:4)=paveindices(0:4)
                            indices0(0)=2
                            indices0(3)=indices0(3)-1
                            pave(1:4)=pave(1:4)-2d0*jj*xx1*xx3/xx2*&
                                 comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                         ENDIF
                         IF(kk.NE.0.AND.j.EQ.3)THEN
                            indices0(0:4)=paveindices(0:4)
                            indices0(0)=2
                            indices0(4)=indices0(4)-1
                            pave(1:4)=pave(1:4)-2d0*kk*xx1*xx3/xx2*&
                                 comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                         ENDIF
                      ENDDO
                   ENDDO
                   IF(RECYCLING)item%value(1:4)=pave(1:4)
                   RETURN
                ELSE
                   STABLE_IREGI=.FALSE.
                   IF(RECYCLING)THEN
                      item%value(1:4)=DCMPLX(0d0)
                      item%stable=.FALSE.
                   ENDIF
                   RETURN
                ENDIF
             ENDIF
          ENDIF
       ENDIF
       ! Eqs.(5.10,5.40,5.50,5.53)
       kk=paveindices(0)
       ii=paveindices(2)
       jj=paveindices(3)
       ll=paveindices(4)
       CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
            XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
       IF(ABS(detZ).GE.EPS)THEN
          ! Eq.(5.10)
          indices1(0)=paveindices(0)-2
          indices1(1:3)=paveindices(2:4)
          ind1=paveindices(1)
          pave(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
               PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
          pave(4)=pave(4)+4d0*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
          IF(ABS(M2L(1)).GE.EPS)THEN
             indices0(0:4)=paveindices(0:4)
             indices0(0)=indices0(0)-2
             pave(1:4)=pave(1:4)+2d0*M2L(1)*&
                  comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
          ENDIF
          DO i=1,3
             xx1=comp_pave_f(i,NLOOPLINE,PCL,M2L)
             IF(ABS(xx1).LT.EPS)CYCLE
             indices0(0:4)=paveindices(0:4)
             indices0(0)=indices0(0)-2
             indices0(i+1)=indices0(i+1)+1
             pave(1:4)=pave(1:4)+xx1*&
                  comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
          ENDDO
          pave(1:4)=pave(1:4)/DBLE(2*(ii+jj+kk+ll-1))
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
       xx2=SIGNED_CMINOR11(NLOOPLINE-1,&
            ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),1,1)
       xxList3(1,1)=xx2
       pos=1
       postemp=1
       DO i=1,3
          DO j=1,3
             IF(i.EQ.1.AND.j.EQ.1)CYCLE
             xx1=SIGNED_CMINOR11(NLOOPLINE-1,&
                  ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),i,j)
             xxList3(i,j)=xx1
             IF(ABS(xx1).GT.ABS(xx2))THEN
                xx2=xx2
                pos=i
                postemp=j
             ENDIF
          ENDDO
       ENDDO
       yy1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,2)
       DO i=3,4
          xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,1,i)
          IF(ABS(xx1).GT.ABS(yy1))THEN
             yy1=xx1
          ENDIF
       ENDDO
       IF(ABS(yy1).GE.EPS.AND.ABS(xx2).GE.EPS)THEN
          ! Eq.(5.40)
          pave(1:4)=DCMPLX(0d0)
          pave(4)=pave(4)+4d0*xx2*comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L)
          pave(1:4)=pave(1:4)+xx2*comp_pave_S(kk+ii+jj+ll,0,NLOOPLINE,&
               paveindices,PCL,M2L)
          DO i=1,3
             IF(ABS(xxList3(i,postemp)).GE.EPS)THEN
                indices0(0:4)=paveindices(0:4)
                indices0(0)=indices0(0)-2
                indices0(i+1)=indices0(i+1)+1
                indices0(pos+1)=indices0(pos+1)+1
                pave(1:4)=pave(1:4)+xxList3(i,postemp)*&
                     comp_pave_S1(kk+ii+jj+ll,i,NLOOPLINE,indices0,&
                     PCL,M2L)
             ENDIF
             IF(ABS(xxList3(pos,postemp)).GE.EPS)THEN
                indices0(0:4)=paveindices(0:4)
                indices0(0)=indices0(0)-2
                indices0(i+1)=indices0(i+1)+2
                pave(1:4)=pave(1:4)-xxList3(pos,postemp)*&
                     comp_pave_S1(kk+ii+jj+ll,i,NLOOPLINE,indices0,&
                     PCL,M2L)
             ENDIF
          ENDDO
          DO i=1,3
             xx3=comp_pave_f(i,NLOOPLINE,PCL,M2L)
             DO j=1,3
                xx1=SIGNED_CMINOR22(NLOOPLINE-1,&
                     ZMATRIX(2:NLOOPLINE,2:NLOOPLINE),&
                     pos,i,postemp,j)
                IF(ABS(xx1).LT.EPS)CYCLE
                IF(ABS(xx3).GE.EPS)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)-2
                   indices0(j+1)=indices0(j+1)+1
                   pave(1:4)=pave(1:4)-xx1*xx3*&
                        comp_pave_S1(ii+jj+kk+ll-1,j,NLOOPLINE,&
                        indices0,PCL,M2L)
                ENDIF
                IF(ii.NE.0.AND.i.EQ.1)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(2)=indices0(2)-1
                   indices0(j+1)=indices0(j+1)+1
                   pave(1:4)=pave(1:4)-xx1*2d0*ii*&
                        comp_pave_S1(ii+jj+kk+ll,j,NLOOPLINE,&
                        indices0,PCL,M2L)
                ENDIF
                IF(jj.NE.0.AND.i.EQ.2)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(3)=indices0(3)-1
                   indices0(j+1)=indices0(j+1)+1
                   pave(1:4)=pave(1:4)-xx1*2d0*jj*&
                        comp_pave_S1(ii+jj+kk+ll,j,NLOOPLINE,&
                        indices0,PCL,M2L)
                ENDIF
                IF(ll.NE.0.AND.i.EQ.3)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(4)=indices0(4)-1
                   indices0(j+1)=indices0(j+1)+1
                   pave(1:4)=pave(1:4)-xx1*2d0*ll*&
                        comp_pave_S1(ii+jj+kk+ll,j,NLOOPLINE,&
                        indices0,PCL,M2L)
                ENDIF
                xx4=comp_pave_f(j,NLOOPLINE,PCL,M2L)
                IF(ABS(xx3).GE.EPS.AND.ABS(xx4).GE.EPS)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)-2
                   pave(1:4)=pave(1:4)+xx1*xx3*xx4*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                xx5=DCMPLX(0d0)
                IF(j.EQ.1)xx5=xx5+xx3
                IF(i.EQ.1)xx5=xx5+xx4
                IF(ii.NE.0.AND.ABS(xx5).GE.EPS)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(2)=indices0(2)-1
                   pave(1:4)=pave(1:4)+xx5*xx1*2d0*ii*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                xx5=DCMPLX(0d0)
                IF(j.EQ.2)xx5=xx5+xx3
                IF(i.EQ.2)xx5=xx5+xx4
                IF(jj.NE.0.AND.ABS(xx5).GE.EPS)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(3)=indices0(3)-1
                   pave(1:4)=pave(1:4)+xx5*xx1*2d0*jj*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                xx5=DCMPLX(0d0)
                IF(j.EQ.3)xx5=xx5+xx3
                IF(i.EQ.3)xx5=xx5+xx4
                IF(ll.NE.0.AND.ABS(xx5).GE.EPS)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(4)=indices0(4)-1
                   pave(1:4)=pave(1:4)+xx5*xx1*2d0*ii*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(ii.NE.0.AND.ii.NE.1.AND.i.EQ.1.AND.j.EQ.1)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(2)=indices0(2)-2
                   pave(1:4)=pave(1:4)+4d0*ii*(ii-1)*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(jj.NE.0.AND.jj.NE.1.AND.i.EQ.2.AND.j.EQ.2)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(3)=indices0(3)-2
                   pave(1:4)=pave(1:4)+4d0*jj*(jj-1)*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(ll.NE.0.AND.ll.NE.1.AND.i.EQ.3.AND.j.EQ.3)THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(4)=indices0(4)-2
                   pave(1:4)=pave(1:4)+4d0*ll*(ll-1)*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(ii.NE.0.AND.jj.NE.0.AND.((i.EQ.1.AND.j.EQ.2)&
                     .OR.(i.EQ.2.AND.j.EQ.1)))THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(2)=indices0(2)-1
                   indices0(3)=indices0(3)-1
                   pave(1:4)=pave(1:4)+4d0*ii*jj*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(ii.NE.0.AND.ll.NE.0.AND.((i.EQ.1.AND.j.EQ.3)&
                     .OR.(i.EQ.3.AND.j.EQ.1)))THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(2)=indices0(2)-1
                   indices0(4)=indices0(4)-1
                   pave(1:4)=pave(1:4)+4d0*ii*jj*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
                IF(jj.NE.0.AND.ll.NE.0.AND.((i.EQ.2.AND.j.EQ.3)&
                     .OR.(i.EQ.3.AND.j.EQ.2)))THEN
                   indices0(0:4)=paveindices(0:4)
                   indices0(0)=indices0(0)+2
                   indices0(3)=indices0(3)-1
                   indices0(4)=indices0(4)-1
                   pave(1:4)=pave(1:4)+4d0*ll*jj*xx1*&
                        comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
                ENDIF
             ENDDO
          ENDDO
          pave(1:4)=pave(1:4)/DBLE(2d0*(kk+2*ii+2*jj+2*ll))/xx2
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ELSEIF(ABS(yy1).LT.EPS.AND.ABS(xx2).GE.EPS)THEN
          ! Eq.(5.50)
          ll1=pos
          IF(paveindices(pos+1).LT.paveindices(postemp+1))ll1=postemp
          SELECT CASE (ll1)
          CASE (1)
             ll2=2
             ll3=3
          CASE(2)
             ll2=1
             ll3=3
          CASE DEFAULT
             ll2=1
             ll3=2
          END SELECT
          pave(1:4)=DCMPLX(0d0)
          IF(paveindices(ll2+1).NE.0.AND.&
               ABS(xxList3(pos+postemp-ll1,ll2)).GE.EPS)THEN
             indices0(0:4)=paveindices(0:4)
             indices0(ll1+1)=indices0(ll1+1)+1
             indices0(ll2+1)=indices0(ll2+1)-1
             pave(1:4)=pave(1:4)-2d0*paveindices(ll2+1)*xxList3(pos+postemp-ll1,ll2)*&
                  comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
          ENDIF
          IF(paveindices(ll3+1).NE.0.AND.&
               ABS(xxList3(pos+postemp-ll1,ll3)).GE.EPS)THEN
             indices0(0:4)=paveindices(0:4)
             indices0(ll1+1)=indices0(ll1+1)+1
             indices0(ll3+1)=indices0(ll3+1)-1
             pave(1:4)=pave(1:4)-2d0*paveindices(ll3+1)*xxList3(pos+postemp-ll1,ll3)*&
                  comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
          ENDIF
          DO i=1,3
             xx1=xxList3(pos+postemp-ll1,i)
             IF(ABS(xx1).GE.EPS)THEN
                indices0(0:4)=paveindices(0:4)
                indices0(0)=indices0(0)-2
                indices0(ll1+1)=indices0(ll1+1)+1
                indices0(i+1)=indices0(i+1)+1
                pave(1:4)=pave(1:4)+xx1*&
                     comp_pave_S1(kk+ii+jj+ll,i,NLOOPLINE,indices0,PCL,M2L)
             ENDIF
          ENDDO
          pave(1:4)=pave(1:4)/(2d0*(1d0+paveindices(ll1+1)))/xx2
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ELSE
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
    ENDIF
    ! Eq.(3.6) in hep-ph/0212259v2 && Eq.(D.4) in hep-ph/0509141
    ! >=5 point scalar functions
    IF(indices_zero.AND.NLOOPLINE.GE.5)THEN
       CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
            XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
       IF(ABS(detY).GE.EPS)THEN
          pave(1:4)=DCMPLX(0d0)
          indices1(0:NLOOPLINE-1)=0
          PCL1(1:NLOOPLINE-1,0:3)=PCL(2:NLOOPLINE,0:3)
          M2L1(1:NLOOPLINE-1)=M2L(2:NLOOPLINE)
          DO i=1,NLOOPLINE
             IF(i.GT.1)THEN
                PCL1(i-1,0:3)=PCL(i-1,0:3)
                M2L1(i-1)=M2L(i-1)
             ENDIF
             xx1=CYNi(i,NLOOPLINE,YMATRIX)
             IF(ABS(xx1).LT.EPS)CYCLE
             pave(1:4)=pave(1:4)-xx1/detY*&
                  comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
          ENDDO
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ELSE
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
    ENDIF
    ! 5-point functions up to rank 5
    IF(NLOOPLINE.EQ.5)THEN
       IF(lind.GE.2)THEN
          ii=paveindices(2)
          jj=paveindices(3)
          kk=paveindices(4)
          ll=paveindices(5)
          IF(ii+jj+kk+ll.EQ.1)THEN
             ! Eq.(6.17)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).GE.EPS)THEN
                pave(1:4)=DCMPLX(0d0)
                indices1(0:NLOOPLINE-1)=0
                PCL1(1:NLOOPLINE-1,0:3)=PCL(2:NLOOPLINE,0:3)
                M2L1(1:NLOOPLINE-1)=M2L(2:NLOOPLINE)
                pavetemp(1:4)=comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                DO i=1,4
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,lind,i+1)
                   PCL1(i,0:3)=PCL(i,0:3)
                   M2L1(i)=M2L(i)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   pave(1:4)=pave(1:4)+xx1/detY*&
                        (comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)-&
                        pavetemp(1:4))
                ENDDO
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,lind,1)
                pave(1:4)=pave(1:4)-xx1/detY*pavetemp(1:4)
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
          ELSEIF(ii+jj+kk+ll.LT.6)THEN
             ! Generalization of Eq.(6.18), (6.19), (6.20), (6.21)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).GE.EPS)THEN
                ss=ii+jj+kk+ll
                IF(ss.GT.MAXNLOOP_IREGI*MAXINDICES_IREGI)THEN
                   WRITE(*,*)"ERROR: out of range of llarray in comp_pavefun_reduce"
                   STOP
                ENDIF
                j=1
                DO i=2,5
                   IF(paveindices(i).GE.1)THEN
                      llarray(j:j-1+paveindices(i))=i-1
                      j=j+paveindices(i)
                   END IF
                ENDDO
                pave(1:4)=DCMPLX(0d0)
                DO i=1,ss
                   DO j=1,4
                      xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,&
                           llarray(i)+1,j+1)
                      IF(ABS(xx1).GE.EPS)THEN
                         indices1(0)=paveindices(0)
                         indices1(1:4)=paveindices(2:5)
                         ind1=paveindices(1)
                         indices1(llarray(i))=indices1(llarray(i))-1
                         pave(1:4)=pave(1:4)-xx1/DBLE(ss)/detY*&
                              comp_shiftpaveden(NLOOPLINE-1,indices1,&
                              PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                         xx2=DCMPLX(DBLE(hdeltabar(4,indices1(1:4),j)))
                         IF(ABS(xx2).GE.EPS)THEN
                            indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                            indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                            indices1(0:j)=indices0(0:j)
                            PCL1(1:j,0:3)=PCL(1:j,0:3)
                            M2L1(1:j)=M2L(1:j)
                            IF(j.NE.4)THEN
                               indices1(j+1:NLOOPLINE-1)=indices0(j+2:NLOOPLINE)
                               PCL1(j+1:NLOOPLINE-1,0:3)=PCL(j+2:NLOOPLINE,0:3)
                               M2L1(j+1:NLOOPLINE-1)=M2L(j+2:NLOOPLINE)
                            ENDIF
                            pave(1:4)=pave(1:4)+xx1/DBLE(ss)/detY*xx2*&
                                 comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                         ENDIF
                      ENDIF
                   ENDDO
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,&
                        llarray(i)+1,1)
                   IF(ABS(xx1).GE.EPS)THEN
                      indices1(0)=paveindices(0)
                      indices1(1:4)=paveindices(2:5)
                      indices1(llarray(i))=indices1(llarray(i))-1
                      ind1=paveindices(1)
                      pave(1:4)=pave(1:4)-xx1/DBLE(ss)/detY*&
                           comp_shiftpaveden(NLOOPLINE-1,indices1,&
                           PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                   ENDIF
                   DO j=1,ss
                      IF(j.EQ.i)CYCLE
                      DO k=1,4
                         xx1=SIGNED_CMINOR22(NLOOPLINE,XMATRIX,&
                              llarray(i)+1,k+1,1,llarray(j)+1)
                         IF(ABS(xx1).LT.EPS)CYCLE
                         indices1(0)=2
                         indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
                         ind1=paveindices(1)
                         indices1(llarray(i))=indices1(llarray(i))-1
                         indices1(llarray(j))=indices1(llarray(j))-1
                         pave(1:4)=pave(1:4)+xx1*2d0/DBLE(ss)/detY*&
                              comp_shiftpaveden(NLOOPLINE-1,indices1,&
                              PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                         xx2=DCMPLX(DBLE(hdeltabar(4,indices1(1:4),k)))
                         IF(ABS(xx2).GE.EPS)THEN
                            indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                            indices0(0)=indices0(0)+2
                            indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                            indices0(llarray(j)+1)=indices0(llarray(j)+1)-1
                            indices1(0:k)=indices0(0:k)
                            PCL1(1:k,0:3)=PCL(1:k,0:3)
                            M2L1(1:k)=M2L(1:k)
                            IF(k.NE.4)THEN
                               indices1(k+1:NLOOPLINE-1)=indices0(k+2:NLOOPLINE)
                               PCL1(k+1:NLOOPLINE-1,0:3)=PCL(k+2:NLOOPLINE,0:3)
                               M2L1(k+1:NLOOPLINE-1)=M2L(k+2:NLOOPLINE)
                            ENDIF
                            pave(1:4)=pave(1:4)-xx1*2d0/DBLE(ss)/detY*xx2*&
                                 comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                         ENDIF
                      ENDDO
                   ENDDO
                ENDDO
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
          ENDIF
       ENDIF
       IF(lind.EQ.0.AND.paveindices(2).EQ.0.AND.paveindices(3).EQ.0&
            .AND.paveindices(4).EQ.0.AND.paveindices(5).EQ.0.AND.&
            paveindices(0).LT.6)THEN
          ! Generalization of Eq.(6.18) and Eq.(6.20)
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
               XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detY).GE.EPS)THEN
             pave(1:4)=DCMPLX(0d0)
             indices1(0)=paveindices(0)
             indices1(1:NLOOPLINE-1)=0
             ind1=0
             pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
             DO i=1,4
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,1)
                IF(ABS(xx1).LT.EPS)CYCLE
                PCL1(1:i,0:3)=PCL(1:i,0:3)
                M2L1(1:i)=M2L(1:i)
                IF(i.NE.4)THEN
                   PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                   M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                ENDIF
                pave(1:4)=pave(1:4)+xx1/detY*&
                     (comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)&
                     -pavetemp(1:4))
             ENDDO
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSE
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
       ENDIF
       IF(lind.EQ.0.AND.paveindices(0).EQ.2)THEN
          ii=paveindices(2)
          jj=paveindices(3)
          kk=paveindices(4)
          ll=paveindices(5)
          ss=ii+jj+kk+ll
          IF(ss.EQ.1)THEN
             ! Eq.(6.19)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).GE.EPS)THEN
                IF(ii.EQ.1)THEN
                   pos=1
                ELSEIF(jj.EQ.1)THEN
                   pos=2
                ELSEIF(kk.EQ.1)THEN
                   pos=3
                ELSE
                   pos=4
                ENDIF
                pave(1:4)=DCMPLX(0d0)
                indices1(0)=2
                indices1(1:4)=paveindices(2:5)
                ind1=paveindices(1)
                pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                     PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                DO i=1,4
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,1)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   pave(1:4)=pave(1:4)-2d0*xx1/3d0/detY*pavetemp(1:4)
                   IF(i.NE.pos)THEN
                      indices1(0:i)=paveindices(0:i)
                      PCL1(1:i,0:3)=PCL(1:i,0:3)
                      M2L1(1:i)=M2L(1:i)
                      IF(i.NE.4)THEN
                         indices1(i+1:NLOOPLINE-1)=paveindices(i+2:NLOOPLINE)
                         PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                         M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                      ENDIF
                      pave(1:4)=pave(1:4)+2d0*xx1/3d0/detY*&
                           comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                   ENDIF
                ENDDO
                indices1(0)=2
                indices1(1:NLOOPLINE-1)=0
                ind1=0
                pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                DO i=1,4
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,pos+1,i+1)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   PCL1(1:i,0:3)=PCL(1:i,0:3)
                   M2L1(1:i)=M2L(1:i)
                   IF(i.NE.4)THEN
                      PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                      M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                   ENDIF
                   pave(1:4)=pave(1:4)+xx1/3d0/detY*&
                        (comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)&
                        -pavetemp(1:4))
                ENDDO
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,pos+1,1)
                IF(ABS(xx1).GE.EPS)THEN
                   indices1(0)=2
                   indices1(1:4)=0
                   ind1=0
                   pave(1:4)=pave(1:4)-xx1/3d0/detY*pavetemp(1:4)
                ENDIF
                IF(RECYCLING)item%value(1:4)=pave(1:4)
                RETURN
             ELSE
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
          ELSEIF(ss.LT.4)THEN
             ! Generalization of Eq.(6.20) and Eq.(6.21)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).LT.EPS)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             ss=ss+2
             IF(ss.GT.MAXNLOOP_IREGI*MAXINDICES_IREGI+2)THEN
                WRITE(*,*)"ERROR: out of range of llarray in comp_pavefun_reduce"
                STOP
             ENDIF
             j=1
             DO i=2,5
                IF(paveindices(i).GE.1)THEN
                   llarray(j:j-1+paveindices(i))=i-1
                   j=j+paveindices(i)
                END IF
             ENDDO
             pave(1:4)=DCMPLX(0d0)
             DO i=1,ss-2
                DO j=1,4
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,llarray(i)+1,j+1)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   indices1(0)=paveindices(0)
                   indices1(1:4)=paveindices(2:5)
                   ind1=paveindices(1)
                   indices1(llarray(i))=indices1(llarray(i))-1
                   pave(1:4)=pave(1:4)-xx1/DBLE(ss)/detY*&
                        comp_shiftpaveden(NLOOPLINE-1,indices1,&
                        PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                   xx2=DCMPLX(DBLE(hdeltabar(4,indices1(1:4),j)))
                   IF(ABS(xx2).GE.EPS)THEN
                      indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                      indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                      indices1(0:j)=indices0(0:j)
                      PCL1(1:j,0:3)=PCL(1:j,0:3)
                      M2L1(1:j)=M2L(1:j)
                      IF(j.NE.4)THEN
                         indices1(j+1:NLOOPLINE-1)=indices0(j+2:NLOOPLINE)
                         PCL1(j+1:NLOOPLINE-1,0:3)=PCL(j+2:NLOOPLINE,0:3)
                         M2L1(j+1:NLOOPLINE-1)=M2L(j+2:NLOOPLINE)
                      ENDIF
                      pave(1:4)=pave(1:4)+xx1*xx2/DBLE(ss)/detY*&
                           comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                   ENDIF
                ENDDO
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,llarray(i)+1,1)
                IF(ABS(xx1).GE.EPS)THEN
                   indices1(0)=paveindices(0)
                   indices1(1:4)=paveindices(2:5)
                   ind1=paveindices(1)
                   indices1(llarray(i))=indices1(llarray(i))-1
                   pave(1:4)=pave(1:4)-xx1/DBLE(ss)/detY*&
                        comp_shiftpaveden(NLOOPLINE-1,indices1,&
                        PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                ENDIF
                DO j=1,ss-2
                   IF(j.EQ.i)CYCLE
                   DO k=1,4
                      xx1=SIGNED_CMINOR22(NLOOPLINE,XMATRIX,&
                           llarray(i)+1,k+1,1,llarray(j)+1)
                      IF(ABS(xx1).LT.EPS)CYCLE
                      indices1(0)=4
                      indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
                      ind1=paveindices(1)
                      indices1(llarray(i))=indices1(llarray(i))-1
                      indices1(llarray(j))=indices1(llarray(j))-1
                      pave(1:4)=pave(1:4)+xx1*2d0/DBLE(ss)/detY*&
                           comp_shiftpaveden(NLOOPLINE-1,indices1,&
                           PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                      xx2=DCMPLX(DBLE(hdeltabar(4,indices1(1:4),k)))
                      IF(ABS(xx2).GE.EPS)THEN
                         indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                         indices0(0)=indices0(0)+2
                         indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                         indices0(llarray(j)+1)=indices0(llarray(j)+1)-1
                         indices1(0:k)=indices0(0:k)
                         PCL1(1:k,0:3)=PCL(1:k,0:3)
                         M2L1(1:k)=M2L(1:k)
                         IF(k.NE.4)THEN
                            indices1(k+1:NLOOPLINE-1)=indices0(k+2:NLOOPLINE)
                            PCL1(k+1:NLOOPLINE-1,0:3)=PCL(k+2:NLOOPLINE,0:3)
                            M2L1(k+1:NLOOPLINE-1)=M2L(k+2:NLOOPLINE)
                         ENDIF
                         pave(1:4)=pave(1:4)-xx1*2d0/DBLE(ss)/detY*xx2*&
                              comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                      ENDIF
                   ENDDO
                ENDDO
             ENDDO
             indices1(0)=paveindices(0)
             indices1(1:4)=paveindices(2:5)
             ind1=paveindices(1)
             pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
             DO i=1,4
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,1)
                IF(ABS(xx1).LT.EPS)CYCLE
                pave(1:4)=pave(1:4)-2d0*xx1/DBLE(ss)/detY*pavetemp(1:4)
                xx2=DCMPLX(DBLE(hdeltabar(4,paveindices(2:5),i)))
                IF(ABS(xx2).GE.EPS)THEN
                   indices1(0:i)=paveindices(0:i)
                   PCL1(1:i,0:3)=PCL(1:i,0:3)
                   M2L1(1:i)=M2L(1:i)
                   IF(i.NE.4)THEN
                      indices1(i+1:NLOOPLINE-1)=paveindices(i+2:NLOOPLINE)
                      PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                      M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                   ENDIF
                   pave(1:4)=pave(1:4)+2d0*xx1/DBLE(ss)/detY*xx2*&
                        comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                ENDIF
             ENDDO
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ENDIF
       ENDIF
       IF(lind.EQ.0.AND.paveindices(0).EQ.4)THEN
          ii=paveindices(2)
          jj=paveindices(3)
          kk=paveindices(4)
          ll=paveindices(5)
          ss=ii+jj+kk+ll
          IF(ss.EQ.1)THEN
             ! Eq.(6.21)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).LT.EPS)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             IF(ii.EQ.1)THEN
                pos=1
             ELSEIF(jj.EQ.1)THEN
                pos=2
             ELSEIF(kk.EQ.1)THEN
                pos=3
             ELSE
                pos=4
             ENDIF
             pave(1:4)=DCMPLX(0d0)
             indices1(0)=paveindices(0)
             indices1(1:4)=paveindices(2:5)
             ind1=paveindices(1)
             pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
             DO i=1,4
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,1)
                IF(ABS(xx1).LT.EPS)CYCLE
                pave(1:4)=pave(1:4)-4d0*xx1/5d0/detY*pavetemp(1:4)
                IF(i.NE.pos)THEN
                   indices1(0:i)=paveindices(0:i)
                   PCL1(1:i,0:3)=PCL(1:i,0:3)
                   M2L1(1:i)=M2L(1:i)
                   IF(i.NE.4)THEN
                      indices1(i+1:NLOOPLINE-1)=paveindices(i+2:NLOOPLINE)
                      PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                      M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                   ENDIF
                   pave(1:4)=pave(1:4)+4d0*xx1/5d0/detY*&
                        comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                ENDIF
             ENDDO
             indices1(0)=paveindices(0)
             indices1(1:NLOOPLINE-1)=0
             ind1=0
             pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
             DO i=1,4
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,pos+1,i+1)
                IF(ABS(xx1).LT.EPS)CYCLE
                PCL1(1:i,0:3)=PCL(1:i,0:3)
                M2L1(1:i)=M2L(1:i)
                IF(i.NE.4)THEN
                   PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                   M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                ENDIF
                pave(1:4)=pave(1:4)+xx1/5d0/detY*(&
                     comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)&
                     -pavetemp(1:4))
             ENDDO
             xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,pos+1,1)
             IF(ABS(xx1).GE.EPS)THEN
                pave(1:4)=pave(1:4)-xx1/5d0/detY*pavetemp(1:4)
                pave(4)=pave(4)-xx1/5d0/detY*imag*pi**2/48d0
             ENDIF
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ENDIF
       ENDIF
    ENDIF
    ! >= 6 point functions
    IF(NLOOPLINE.GE.6)THEN
       ss=0
       DO i=2,NLOOPLINE
          ss=ss+paveindices(i)
          IF(paveindices(i).NE.0)pos=i-1
       ENDDO
       ! Generalization of Eq.(7.16)
       IF(lind.GE.2)THEN
          !ss=0
          !DO i=2,NLOOPLINE
          !   ss=ss+paveindices(i)
          !   IF(paveindices(i).NE.0)pos=i-1
          !ENDDO
          IF(ss.EQ.1)THEN
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).LT.EPS)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             pave(1:4)=DCMPLX(0d0)
             indices1(0:NLOOPLINE-1)=0
             pavetemp(1:4)=comp_pavefun_reduce(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE))
             DO i=1,NLOOPLINE-1
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,pos+1)
                IF(ABS(xx1).LT.EPS)CYCLE
                PCL1(1:i,0:3)=PCL(1:i,0:3)
                M2L1(1:i)=M2L(1:i)
                IF(i.NE.NLOOPLINE-1)THEN
                   PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                   M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                ENDIF
                pave(1:4)=pave(1:4)+xx1/detY*(&
                     comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)&
                     -pavetemp(1:4))
             ENDDO
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSEIF(ss.GE.2.AND.ss.LT.2*NLOOPLINE-4)THEN
             ! Generalization of Eq.(7.17)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).LT.EPS)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             IF(ss.GT.MAXNLOOP_IREGI*MAXINDICES_IREGI)THEN
                WRITE(*,*)"ERROR: out of range of llarray in comp_pavefun_reduce"
                STOP
             ENDIF
             j=1
             DO i=2,NLOOPLINE
                IF(paveindices(i).GE.1)THEN
                   llarray(j:j-1+paveindices(i))=i-1
                   j=j+paveindices(i)
                END IF
             ENDDO
             pave(1:4)=DCMPLX(0d0)
             kk=0
             DO i=1,ss
                postemp=llarray(i)+1
                IF(postemp.EQ.kk)CYCLE
                kk=postemp
                ll=paveindices(postemp)
                DO j=1,NLOOPLINE-1
                   xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,j+1,llarray(i)+1)
                   IF(ABS(xx1).LT.EPS)CYCLE
                   indices1(0)=paveindices(0)
                   indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
                   ind1=paveindices(1)
                   indices1(llarray(i))=indices1(llarray(i))-1
                   pave(1:4)=pave(1:4)-ll*xx1/DBLE(ss)/detY*&
                     comp_shiftpaveden(NLOOPLINE-1,indices1,&
                     PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                   xx2=DCMPLX(DBLE(hdeltabar(NLOOPLINE-1,indices1(1:NLOOPLINE-1),j)))
                   IF(ABS(xx2).GE.EPS)THEN
                      indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                      indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                      indices1(0:j)=indices0(0:j)
                      PCL1(1:j,0:3)=PCL(1:j,0:3)
                      M2L1(1:j)=M2L(1:j)
                      IF(j.NE.NLOOPLINE-1)THEN
                         indices1(j+1:NLOOPLINE-1)=indices0(j+2:NLOOPLINE)
                         PCL1(j+1:NLOOPLINE-1,0:3)=PCL(j+2:NLOOPLINE,0:3)
                         M2L1(j+1:NLOOPLINE-1)=M2L(j+2:NLOOPLINE)
                      ENDIF
                      pave(1:4)=pave(1:4)+ll*xx1*xx2/DBLE(ss)/detY*&
                           comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                   ENDIF
                ENDDO
             ENDDO
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ENDIF
       ENDIF
       IF(lind.EQ.0.AND.paveindices(0).EQ.2.AND.ss.LE.1)THEN
          IF(ss.EQ.0)THEN
             pave(1:4)=DCMPLX(0d0)
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ELSEIF(ss.EQ.1)THEN
             ! Eq.(7.18)
             CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
                  XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
             IF(ABS(detY).LT.EPS)THEN
                STABLE_IREGI=.FALSE.
                IF(RECYCLING)THEN
                   item%value(1:4)=DCMPLX(0d0)
                   item%stable=.FALSE.
                ENDIF
                RETURN
             ENDIF
             indices1(0)=2
             indices1(1:NLOOPLINE-1)=0
             ind1=0
             pave(1:4)=DCMPLX(0d0)
             pavetemp(1:4)=comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
             DO i=1,NLOOPLINE-1
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,i+1,pos+1)
                IF(ABS(xx1).LT.EPS)CYCLE
                PCL1(1:i,0:3)=PCL(1:i,0:3)
                M2L1(1:i)=M2L(1:i)
                IF(i.NE.NLOOPLINE-1)THEN
                   PCL1(i+1:NLOOPLINE-1,0:3)=PCL(i+2:NLOOPLINE,0:3)
                   M2L1(i+1:NLOOPLINE-1)=M2L(i+2:NLOOPLINE)
                ENDIF
                pave(1:4)=pave(1:4)+xx1/3d0/detY*&
                     (comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)&
                     -pavetemp(1:4))
             ENDDO
             IF(RECYCLING)item%value(1:4)=pave(1:4)
             RETURN
          ENDIF
       ELSEIF(lind.EQ.0.AND.ss+paveindices(0).LT.2*NLOOPLINE-4)THEN
          ! Generalization of Eq.(7.13) 
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
               XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detY).LT.EPS)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          PP=ss+paveindices(0)
          IF(PP.GT.MAXNLOOP_IREGI*MAXINDICES_IREGI)THEN
             WRITE(*,*)"ERROR: out of range of llarray in comp_pavefun_reduce"
             STOP
          ENDIF
          j=1
          DO i=2,NLOOPLINE
             IF(paveindices(i).GE.1)THEN
                llarray(j:j-1+paveindices(i))=i-1
                j=j+paveindices(i)
             END IF
          ENDDO
          pave(1:4)=DCMPLX(0d0)
          kk=0
          DO i=1,ss
             postemp=llarray(i)+1
             IF(postemp.EQ.kk)CYCLE
             kk=postemp
             ll=paveindices(postemp)
             DO j=1,NLOOPLINE-1
                xx1=SIGNED_CMINOR11(NLOOPLINE,XMATRIX,j+1,llarray(i)+1)
                IF(ABS(xx1).LT.EPS)CYCLE
                indices1(0)=paveindices(0)
                indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
                ind1=paveindices(1)
                indices1(llarray(i))=indices1(llarray(i))-1
                pave(1:4)=pave(1:4)-ll*xx1/DBLE(PP)/detY*&
                     comp_shiftpaveden(NLOOPLINE-1,indices1,&
                     PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
                xx2=DCMPLX(DBLE(hdeltabar(NLOOPLINE-1,indices1(1:NLOOPLINE-1),j)))
                IF(ABS(xx2).GE.EPS)THEN
                   indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                   indices0(llarray(i)+1)=indices0(llarray(i)+1)-1
                   indices1(0:j)=indices0(0:j)
                   PCL1(1:j,0:3)=PCL(1:j,0:3)
                   M2L1(1:j)=M2L(1:j)
                   IF(j.NE.NLOOPLINE-1)THEN
                      indices1(j+1:NLOOPLINE-1)=indices0(j+2:NLOOPLINE)
                      PCL1(j+1:NLOOPLINE-1,0:3)=PCL(j+2:NLOOPLINE,0:3)
                      M2L1(j+1:NLOOPLINE-1)=M2L(j+2:NLOOPLINE)
                   ENDIF
                   pave(1:4)=pave(1:4)+ll*xx1*xx2/DBLE(PP)/detY*&
                        comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
                ENDIF
             ENDDO
          ENDDO
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
       ! generalization of Eq.(D.3)
       IF(ss+paveindices(0).GE.2*NLOOPLINE-4)THEN
          CALL CXYZMATRICES(NLOOPLINE,PCL,M2L,&
               XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
          IF(ABS(detY).LT.EPS)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          PP=ss+paveindices(0)
          IF(PP.GT.MAXNLOOP_IREGI*MAXINDICES_IREGI)THEN
             WRITE(*,*)"ERROR: out of range of llarray in comp_pavefun_reduce"
             STOP
          ENDIF
          pave(1:4)=DCMPLX(0d0)
          xx1=CYNi(1,NLOOPLINE,YMATRIX)
          IF(ABS(xx1).GE.EPS)THEN
             indices1(0)=paveindices(0)
             indices1(1:NLOOPLINE-1)=paveindices(2:NLOOPLINE)
             ind1=paveindices(1)
             pave(1:4)=pave(1:4)-xx1/detY*comp_shiftpaveden(NLOOPLINE-1,indices1,&
                  PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
          ENDIF
          DO i=1,NLOOPLINE-1
             xx1=CYNi(i+1,NLOOPLINE,YMATRIX)
             xx2=DCMPLX(DBLE(hdeltabar(NLOOPLINE-1,paveindices(2:NLOOPLINE),i)))
             IF(ABS(xx1).GE.EPS.AND.ABS(xx2).GE.EPS)THEN
                indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
                indices1(0:j)=indices0(0:j)
                PCL1(1:j,0:3)=PCL(1:j,0:3)
                M2L1(1:j)=M2L(1:j)
                IF(j.NE.NLOOPLINE-1)THEN
                   indices1(j+1:NLOOPLINE-1)=indices0(j+2:NLOOPLINE)
                   PCL1(j+1:NLOOPLINE-1,0:3)=PCL(j+2:NLOOPLINE,0:3)
                   M2L1(j+1:NLOOPLINE-1)=M2L(j+2:NLOOPLINE)
                ENDIF
                pave(1:4)=pave(1:4)-xx2*xx1/detY*&
                     comp_pavefun_reduce(NLOOPLINE-1,indices1,PCL1,M2L1)
             ENDIF
          ENDDO
          IF(RECYCLING)item%value(1:4)=pave(1:4)
          RETURN
       ENDIF
    ENDIF
    STABLE_IREGI=.FALSE.
    pave(1:4)=DCMPLX(0d0)
    IF(RECYCLING)THEN
       item%value(1:4)=pave(1:4)
       item%stable=.FALSE.
    ENDIF
    RETURN
    WRITE(*,*)"ERROR: out of the range of comp_pavefun_reduce"
    STOP
  END FUNCTION comp_pavefun_reduce

  FUNCTION comp_pave_f(k,NLOOPLINE,PCL,M2L)
    ! Eq.(2.24) in hep-ph/0509141
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,k
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0))::comp_pave_f
    REAL(KIND(1d0)),DIMENSION(0:3)::PP
    IF(k.LE.0.OR.k.GE.NLOOPLINE)THEN
       WRITE(*,*)"ERROR:k is out of range in comp_pave_f!"
       STOP
    ENDIF
    PP(0:3)=PCL(k+1,0:3)-PCL(1,0:3)
    comp_pave_f=DCMPLX(scalarprod(PP(0:3),PP(0:3)))-M2L(k+1)+M2L(1)
    RETURN
  END FUNCTION comp_pave_f

  FUNCTION comp_pave_S(P,k,NLOOPLINE,paveindices,PCL,M2L) RESULT(pave)
    ! Eqs.(5.8-5.9) in hep-ph/0509141
    IMPLICIT NONE
    INTEGER,INTENT(IN)::k,P,NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pave
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pavetemp
    INTEGER,DIMENSION(0:NLOOPLINE)::indices0
    INTEGER,DIMENSION(0:NLOOPLINE-1)::indices1
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE-1,0:3)::PCL1
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE-1)::M2L1
    INTEGER::ind1
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    IF(P.LE.0)THEN
       WRITE(*,*)"ERROR: P is smaller than zero in comp_pave_S"
       STOP
    ENDIF
    IF(k.GE.NLOOPLINE.OR.k.LT.0)THEN
       WRITE(*,*)"ERROR: k is outside range in comp_pave_S"
       STOP
    ENDIF
    IF(k.GT.0)THEN
       ! Eq.(5.8) in hep-ph/0509141
       pave(1:4)=DCMPLX(0d0)
       IF(paveindices(k+1).EQ.1)THEN
          indices1(0:k)=paveindices(0:k)
          indices1(k+1:NLOOPLINE-1)=paveindices(k+2:NLOOPLINE)
          PCL1(1:k,0:3)=PCL(1:k,0:3)
          PCL1(k+1:NLOOPLINE-1,0:3)=PCL(k+2:NLOOPLINE,0:3)
          M2L1(1:k)=M2L(1:k)
          M2L1(k+1:NLOOPLINE-1)=M2L(k+2:NLOOPLINE)
          pave(1:4)=pave(1:4)+comp_pavefun_reduce(NLOOPLINE-1,indices1,&
               PCL1,M2L1)
       ENDIF
       indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
       indices0(k+1)=indices0(k+1)-1
       indices1(0)=indices0(0)
       indices1(1:NLOOPLINE-1)=indices0(2:NLOOPLINE)
       ind1=indices0(1)
       pave(1:4)=pave(1:4)-comp_shiftpaveden(NLOOPLINE-1,indices1,&
            PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
       pave(1:4)=pave(1:4)-comp_pave_f(k,NLOOPLINE,PCL,M2L)*&
            comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
       RETURN
    ELSE
       ! Eq.(5.9) in hep-ph/0509141
       indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
       indices0(0)=indices0(0)-2
       indices1(0)=indices0(0)
       indices1(1:NLOOPLINE-1)=indices0(2:NLOOPLINE)
       ind1=indices0(1)
       pave(1:4)=DCMPLX(2d0)*comp_shiftpaveden(NLOOPLINE-1,indices1,&
            PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
       IF(ABS(M2L(1)).GT.EPS)THEN
          pave(1:4)=pave(1:4)+DCMPLX(2d0)*M2L(1)*&
               comp_pavefun_reduce(NLOOPLINE,indices0,PCL,M2L)
       ENDIF
       RETURN
    ENDIF
  END FUNCTION comp_pave_S

  FUNCTION comp_pave_S1(P,k,NLOOPLINE,paveindices,PCL,M2L) RESULT(pave)
    ! Eq.(5.13) in hep-ph/0509141
    IMPLICIT NONE
    INTEGER,INTENT(IN)::k,P,NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::pave
    INTEGER,DIMENSION(0:NLOOPLINE)::indices0
    INTEGER,DIMENSION(0:NLOOPLINE-1)::indices1
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE-1,0:3)::PCL1
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE-1)::M2L1
    INTEGER::ind1
!    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    IF(P.LE.0)THEN
       WRITE(*,*)"ERROR: P is smaller than zero in comp_pave_S1"
       STOP
    ENDIF
    IF(k.LE.0.OR.k.GE.NLOOPLINE)THEN
       WRITE(*,*)"ERROR: k is outside range in comp_pave_S1"
       STOP
    ENDIF
    pave(1:4)=DCMPLX(0d0)
    IF(paveindices(k+1).EQ.1)THEN
       indices1(0:k)=paveindices(0:k)
       indices1(k+1:NLOOPLINE-1)=paveindices(k+2:NLOOPLINE)
       PCL1(1:k,0:3)=PCL(1:k,0:3)
       PCL1(k+1:NLOOPLINE-1,0:3)=PCL(k+2:NLOOPLINE,0:3)
       M2L1(1:k)=M2L(1:k)
       M2L1(k+1:NLOOPLINE-1)=M2L(k+2:NLOOPLINE)
       pave(1:4)=pave(1:4)+comp_pavefun_reduce(NLOOPLINE-1,indices1,&
            PCL1,M2L1)
    ENDIF
    indices0(0:NLOOPLINE)=paveindices(0:NLOOPLINE)
    indices0(k+1)=indices0(k+1)-1
    indices1(0)=indices0(0)
    indices1(1:NLOOPLINE-1)=indices0(2:NLOOPLINE)
    ind1=indices0(1)
    pave(1:4)=pave(1:4)-comp_shiftpaveden(NLOOPLINE-1,indices1,&
         PCL(2:NLOOPLINE,0:3),M2L(2:NLOOPLINE),ind1)
    RETURN
  END FUNCTION comp_pave_S1


  FUNCTION comp_pave_UV(NLOOPLINE,paveindices,PCL,M2L) RESULT(pave_UV)
    ! hep-ph/0509141
    ! More expressions are given in hep-ph/0609282: 
    ! rank 9 for point 4, rank 8 for point 3 and rank 10 for point 2.
    ! General formalism is also aviable. 
    ! However, I think the expressions in hep-ph/0509141 are sufficient and I will leave it.
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    LOGICAL::indices_zero
    INTEGER::lind,nindtot
    COMPLEX(KIND(1d0))::pave_UV
    INTEGER,DIMENSION(0:NLOOPLINE)::indices0
    COMPLEX(KIND(1d0)),PARAMETER::imag=DCMPLX(0d0,1d0)
    REAL(KIND(1d0)),PARAMETER::pi=3.141592653589793d0
    COMPLEX(KIND(1d0)),PARAMETER::ipi2=DCMPLX(0d0,9.869604401089358d0) ! imag*pi**2
    REAL(KIND(1d0)),DIMENSION(0:3)::P0
    REAL(KIND(1d0))::P02,P032,P021,P031,P041,P042,P043
    INTEGER::ii,jj,kk,ll,ss,i
    indices_zero=.TRUE.
    lind=-1
    nindtot=0
    DO i=0,NLOOPLINE
       IF(paveindices(i).LT.0)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
       IF(paveindices(i).NE.0.AND.indices_zero)THEN
          indices_zero=.FALSE.
          lind=i
       ENDIF
       nindtot=nindtot+paveindices(i)
       indices0(i)=paveindices(i)
    ENDDO
    ! scalar Eqs.(A.1-A.2)
    IF(indices_zero)THEN
       IF(NLOOPLINE.EQ.1)THEN
          pave_UV=ipi2*M2L(1)
       ELSEIF(NLOOPLINE.EQ.2)THEN
          pave_UV=ipi2
       ELSE
          pave_UV=DCMPLX(0d0)
       ENDIF
       RETURN
    ENDIF
    ! 1-point Eq.(A.1)
    IF(NLOOPLINE.EQ.1)THEN
       pave_UV=ipi2*M2L(1)**((paveindices(0)+2)/2)/&
            2**(paveindices(0)/2)/DBLE(factorial(paveindices(0)/2+1))
       RETURN
    ENDIF
    ! 2-point Eq.(A.2)
    IF(NLOOPLINE.EQ.2)THEN
       IF(lind.EQ.2)THEN
          SELECT CASE(paveindices(2))
          CASE(1)
             pave_UV=-ipi2/2d0
             RETURN
          CASE(2)
             pave_UV=ipi2/3d0
             RETURN
          CASE(3)
             pave_UV=-ipi2/4d0
             RETURN
          CASE(4)
             pave_UV=ipi2/5d0
             RETURN
          CASE(5)
             pave_UV=-ipi2/6d0
             RETURN
          CASE DEFAULT
             pave_UV=(-1)**paveindices(2)*ipi2&
                  /DBLE(paveindices(2)+1)
             RETURN
          END SELECT
       ENDIF
       IF(lind.EQ.0.AND.paveindices(0).EQ.2)THEN
          SELECT CASE(paveindices(2))
          CASE(0)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=-ipi2/12d0*(P02-3d0*M2L(1)-3d0*M2L(2))
             RETURN
          CASE(1)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=ipi2/24d0*(P02-2d0*M2L(1)-4d0*M2L(2))
             RETURN
          CASE(2)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=-ipi2/120d0*(3d0*P02-5d0*M2L(1)-15d0*M2L(2))
             RETURN
          CASE(3)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=ipi2/120d0*(2d0*P02-3d0*M2L(1)-12d0*M2L(2))
             RETURN
          CASE DEFAULT
             WRITE(*,*)"ERROR:out of range of comp_pave_UV"
             STOP
          END SELECT
       ENDIF
       IF(lind.EQ.0.AND.paveindices(0).EQ.4)THEN
          SELECT CASE(paveindices(2))
          CASE(0)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=ipi2/240d0*(P02**2-5d0*P02*&
                  (M2L(1)+M2L(2))+10d0*(M2L(1)**2+M2L(2)**2+M2L(1)*M2L(2)))
             RETURN
          CASE(1)
             P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
             P02=scalarprod(P0,P0)
             pave_UV=-ipi2/480d0*(P02**2-4d0*P02*M2L(1)-6d0*P02*M2L(2)&
                  +5d0*M2L(1)**2+15d0*M2L(2)**2+10d0*M2L(1)*M2L(2))
             RETURN
          CASE DEFAULT
             WRITE(*,*)"ERROR:out of range of comp_pave_UV"
             STOP
          END SELECT
       END IF
       IF(paveindices(0)+paveindices(1)+paveindices(2).LE.5)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
    ENDIF
    ! 3-point Eq.(A.3)
    IF(NLOOPLINE.EQ.3)THEN
       IF(lind.GE.2)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
       ii=paveindices(2)
       jj=paveindices(3)
       IF(paveindices(0).EQ.2)THEN
          SELECT CASE(ii)
          CASE(0)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/4d0
                RETURN
             CASE(1)
                pave_UV=-ipi2/12d0
                RETURN
             CASE(2)
                pave_UV=ipi2/24d0
                RETURN
             CASE(3)
                pave_UV=-ipi2/40d0
                RETURN
             CASE(4)
                pave_UV=ipi2/60d0
                RETURN
             CASE(5)
                pave_UV=-ipi2/84d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(1)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/12d0
                RETURN
             CASE(1)
                pave_UV=ipi2/48d0
                RETURN
             CASE(2)
                pave_UV=-ipi2/120d0
                RETURN
             CASE(3)
                pave_UV=ipi2/240d0
                RETURN
             CASE(4)
                pave_UV=-ipi2/420d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(2)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/24d0
                RETURN
             CASE(1)
                pave_UV=-ipi2/12d0
                RETURN
             CASE(2)
                pave_UV=ipi2/360d0
                RETURN
             CASE(3)
                pave_UV=-ipi2/840d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(3)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/40d0
                RETURN
             CASE(1)
                pave_UV=ipi2/240d0
                RETURN
             CASE(2)
                pave_UV=-ipi2/840d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(4)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/60d0
                RETURN
             CASE(1)
                pave_UV=-ipi2/420d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(5)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/84d0
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          END SELECT
       ENDIF
       P0(0:3)=PCL(3,0:3)-PCL(2,0:3)
       P032=scalarprod(P0,P0)
       P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
       P021=scalarprod(P0,P0)
       P0(0:3)=PCL(3,0:3)-PCL(1,0:3)
       P031=scalarprod(P0,P0)
       IF(paveindices(0).EQ.4)THEN
          SELECT CASE(ii)
          CASE(0)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/96d0*(P032+P021+P031)&
                     +ipi2/24d0*(M2L(1)+M2L(2)+M2L(3))
                RETURN
             CASE(1)
                pave_UV=ipi2/480d0*(2d0*P032-5d0*M2L(1)&
                     +P021-5d0*M2L(2)+2d0*P031-10d0*M2L(3))
                RETURN
             CASE(2)
                pave_UV=-ipi2/1440d0*(3d0*P032-6d0*M2L(1)+P021-6d0*M2L(2)&
                     +3d0*P031-18d0*M2L(3))
                RETURN
             CASE(3)
                pave_UV=ipi2/3360d0*(4d0*P032-7d0*M2L(1)+P021-7d0*M2L(2)&
                     +4d0*P031-28d0*M2L(3))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(1)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/480d0*(2d0*P032-5d0*M2L(1)&
                     +2d0*P021-10d0*M2L(2)+P031-5d0*M2L(3))
                RETURN
             CASE(1)
                pave_UV=-ipi2/1440d0*(2d0*P032-3d0*M2L(1)+P021-6d0*M2L(2)&
                     +P031-6d0*M2L(3))
                RETURN
             CASE(2)
                pave_UV=ipi2/10080d0*(6d0*P032-7d0*M2L(1)+2d0*P021-14d0*M2L(2)&
                     +3d0*P031-21d0*M2L(3))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(2)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/1440d0*(3d0*P032-6d0*M2L(1)+3d0*P021-18d0*M2L(2)&
                     +P031-6d0*M2L(3))
                RETURN
             CASE(1)
                pave_UV=ipi2/10080d0*(6d0*P032-7d0*M2L(1)+3d0*P021-21d0*M2L(2)&
                     +2d0*P031-14d0*M2L(3))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(3)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/3360d0*(4d0*P032-7d0*M2L(1)+4d0*P021-28d0*M2L(2)&
                     +P031-7d0*M2L(3))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          END SELECT
       ENDIF
       IF(paveindices(0).EQ.6)THEN
          SELECT CASE(ii)
          CASE(0)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=ipi2/5760d0*(2d0*P032**2-6d0*P032*M2L(1)&
                     +30d0*M2L(1)**2+2d0*P032*(P021-6d0*M2L(2)+P031-6d0*M2L(3))&
                     -6d0*M2L(1)*(2d0*P021-5d0*M2L(2)+2d0*P031-5d0*M2L(3))&
                     +2d0*(P021**2-6d0*P021*M2L(2)+15d0*M2L(2)**2)&
                     +2d0*(P031**2-6d0*P031*M2L(3)+15d0*M2L(3)**2)&
                     +(P021*P031-6d0*P021*M2L(3)+15d0*M2L(2)*M2L(3))&
                     +(P021*P031-6d0*P031*M2L(2)+15d0*M2L(2)*M2L(3)))
                RETURN
             CASE(1)
                pave_UV=-ipi2/20160d0*(3d0*P032**2-7d0*P032*M2L(1)+21d0*M2L(1)**2&
                     +P032*(2d0*(P021-7d0*M2L(2))+3d0*(P031-7d0*M2L(3)))&
                     -7d0*M2L(1)*((P021-3d0*M2L(2))+2d0*(P031-3d0*M2L(3)))&
                     +(P021**2-7d0*P021*M2L(2)+21d0*M2L(2)**2)&
                     +(P021*P031-7d0*P021*M2L(3)+21d0*M2L(2)*M2L(3))&
                     +(P021*P031-7d0*P031*M2L(2)+21d0*M2L(2)*M2L(3))&
                     +3d0*(P031**2-7d0*P031*M2L(3)+21d0*M2L(3)**2))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(1)
             SELECT CASE(jj)
             CASE(0)
                pave_UV=-ipi2/20160d0*(3d0*P032**2-7d0*P032*M2L(1)+21d0*M2L(1)**2&
                     +P032*(3d0*(P021-7d0*M2L(2))+2d0*(P031-7d0*M2L(3)))&
                     -7d0*M2L(1)*(2d0*(P021-3d0*M2L(2))+(P031-3d0*M2L(3)))&
                     +3d0*(P021**2-7d0*P021*M2L(2)+21d0*M2L(2)**2)&
                     +(P021*P031-7d0*P021*M2L(3)+21d0*M2L(2)*M2L(3))&
                     +(P021*P031-7d0*P031*M2L(2)+21d0*M2L(2)*M2L(3))&
                     +(P031**2-7d0*P031*M2L(3)+21d0*M2L(3)**2))
                RETURN
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          END SELECT
       END IF
       IF(paveindices(0)+paveindices(1)+paveindices(2)+paveindices(3).LE.7)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
    ENDIF
    ! 4-point Eq.(A.4)
    IF(NLOOPLINE.EQ.4)THEN
       IF(lind.GE.2)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
       ii=paveindices(2)
       jj=paveindices(3)
       kk=paveindices(4)
       IF(paveindices(0).EQ.4)THEN
          SELECT CASE(ii)
          CASE(0)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/24d0
                   RETURN
                CASE(1)
                   pave_UV=-ipi2/96d0
                   RETURN
                CASE(2)
                   pave_UV=ipi2/240d0
                   RETURN
                CASE(3)
                   pave_UV=-ipi2/480d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE(1)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/96d0
                   RETURN
                CASE(1)
                   pave_UV=ipi2/480d0
                   RETURN
                CASE(2)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP   
                END SELECT
             CASE(2)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/240d0
                   RETURN
                CASE(1)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE(3)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/480d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(1)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/96d0
                   RETURN
                CASE(1)
                   pave_UV=ipi2/480d0
                   RETURN
                CASE(2)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE(1)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/480d0
                   RETURN
                CASE(1)
                   pave_UV=-ipi2/2880d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE(2)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(2)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/240d0
                   RETURN
                CASE(1)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP   
                END SELECT
             CASE(1)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/1440d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(3)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/480d0
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE DEFAULT
             WRITE(*,*)"ERROR:out of range of comp_pave_UV"
             STOP
          END SELECT
       ENDIF
       P0(0:3)=PCL(3,0:3)-PCL(2,0:3)
       P032=scalarprod(P0,P0)
       P0(0:3)=PCL(2,0:3)-PCL(1,0:3)
       P021=scalarprod(P0,P0)
       P0(0:3)=PCL(3,0:3)-PCL(1,0:3)
       P031=scalarprod(P0,P0)
       P0(0:3)=PCL(4,0:3)-PCL(2,0:3)
       P042=scalarprod(P0,P0)
       P0(0:3)=PCL(4,0:3)-PCL(3,0:3)
       P043=scalarprod(P0,P0)
       P0(0:3)=PCL(4,0:3)-PCL(1,0:3)
       P041=scalarprod(P0,P0)
       IF(paveindices(0).EQ.6)THEN
          SELECT CASE(ii)
          CASE(0)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=-ipi2/960d0*(P032+P042+P043+P021+P031+P041)&
                        +ipi2/192d0*(M2L(1)+M2L(2)+M2L(3)+M2L(4))
                   RETURN
                CASE(1)
                   pave_UV=ipi2/5760d0*(P021+P031+2d0*P041+P032+2d0*P042+2d0*P043)&
                        -ipi2/960d0*(M2L(1)+M2L(2)+M2L(3)+2d0*M2L(4))
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE(1)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/5760d0*(P021+2d0*P031+P041+2d0*P032+P042+2d0*P043)&
                        -ipi2/960d0*(M2L(1)+M2L(2)+2d0*M2L(3)+M2L(4))
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE(1)
             SELECT CASE(jj)
             CASE(0)
                SELECT CASE(kk)
                CASE(0)
                   pave_UV=ipi2/5760d0*(2d0*P021+P031+P041+2d0*P032+2d0*P042+P043)&
                        -ipi2/960d0*(M2L(1)+2d0*M2L(2)+M2L(3)+M2L(4))
                   RETURN
                CASE DEFAULT
                   WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                   STOP
                END SELECT
             CASE DEFAULT
                WRITE(*,*)"ERROR:out of range of comp_pave_UV"
                STOP
             END SELECT
          CASE DEFAULT
             WRITE(*,*)"ERROR:out of range of comp_pave_UV"
             STOP
          END SELECT
       ENDIF
       IF(paveindices(0)+paveindices(1)+paveindices(2)&
            +paveindices(3)+paveindices(4).LE.7)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       END IF
    END IF
    ! 5-point Eq.(A.5)
    IF(NLOOPLINE.EQ.5)THEN
       IF(lind.GE.2)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
       IF(lind.EQ.0)THEN
          IF(paveindices(0).EQ.6)THEN
             ii=paveindices(2)
             jj=paveindices(3)
             kk=paveindices(4)
             ll=paveindices(5)
             IF(ii.EQ.0.AND.jj.EQ.0.AND.kk.EQ.0.AND.ll.EQ.0)THEN
                pave_UV=ipi2/192d0
                RETURN
             ENDIF
          ENDIF
       ENDIF
       IF(paveindices(0)+paveindices(1)+paveindices(2)&
            +paveindices(3)+paveindices(4)+paveindices(5).LE.6)THEN
          pave_UV=DCMPLX(0d0)
          RETURN
       ENDIF
    ENDIF
    WRITE(*,*)"ERROR:out of range of comp_pave_UV"
    STOP
  END FUNCTION comp_pave_UV
END MODULE cpave_reduce
