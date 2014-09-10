MODULE csi_reduce
USE funlib
USE linear_algebra
USE matrix_base
USE cmatrix_base
USE matrices
USE si_reduce
USE cpave_reduce
USE mis_warp
USE binary_tree
IMPLICIT NONE
!REAL(KIND(1d0)),PARAMETER::pi=3.141592653589793d0
!REAL(KIND(1d0)),PARAMETER::prefactor=0.3183098861837907d0 ! pi^(-1)
CONTAINS

  RECURSIVE FUNCTION comp_scalar_integral_reduce(NLOOPLINE,idim,indices,PCL_IN,M2L) RESULT(si)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE,idim ! dim=idim+d, d=4-2*eps
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL_IN
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER,DIMENSION(NLOOPLINE),INTENT(IN)::indices ! propagator indices
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::si ! epsUV^(-1),epsIR^(-1),epsIR^(-2),finite
    INTEGER::i,j,idim0,nindtot,lind
    INTEGER,DIMENSION(NLOOPLINE)::indices0,indices00
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL0
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE)::M2L0
    INTEGER,DIMENSION(NLOOPLINE-1)::indices1
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE-1,0:3)::PCL1
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE-1)::M2L1
    REAL(KIND(1d0)),DIMENSION(0:3)::r12
    REAL(KIND(1d0))::p2
    COMPLEX(KIND(1d0))::detR,detS,c
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE)::RMATRIX,INVRMATRIX
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPLINE,0:NLOOPLINE)::SMATRIX,INVSMATRIX
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPLINE)::z0
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE)::z
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,1:4)::siz0
    COMPLEX(KIND(1d0)),DIMENSION(1:4)::sitemp
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPLINE,1:4)::siz1
    LOGICAL::indices_one,ok_flag,triv_flag
    REAL(KIND(1d0))::maxtemp
    COMPLEX(KIND(1d0))::temp
    REAL(KIND(1d0))::factor
    LOGICAL,DIMENSION(NLOOPLINE)::lnonzero
    LOGICAL,DIMENSION(0:NLOOPLINE)::lnonzero2
    INTEGER,DIMENSION(0:NLOOPLINE)::paveindices
    TYPE(cibppave_node),POINTER::item
    LOGICAL::find
    REAL(KIND(1d0)),DIMENSION(0:3)::PMOM
    si(1:4)=DCMPLX(0d0)
    IF(.NOT.STABLE_IREGI)RETURN
    ! trivial 0-point function
    IF(NLOOPLINE.EQ.0)THEN
       si(1:4)=DCMPLX(0d0)
       RETURN
    ENDIF
    ! remove (..,0,...)
    DO i=1,NLOOPLINE
       IF(indices(i).EQ.0)THEN
          indices1(1:i-1)=indices(1:i-1)
          PCL1(1:i-1,0:3)=PCL_IN(1:i-1,0:3)
          M2L1(1:i-1)=M2L(1:i-1)
          indices1(i:NLOOPLINE-1)=indices(i+1:NLOOPLINE)
          PCL1(i:NLOOPLINE-1,0:3)=PCL_IN(i+1:NLOOPLINE,0:3)
          M2L1(i:NLOOPLINE-1)=M2L(i+1:NLOOPLINE)
          si=comp_scalar_integral_reduce(NLOOPLINE-1,idim,indices1,PCL1,M2L1)
          RETURN
       ENDIF
    ENDDO
    ! set the first propagator as q^2-m^2
    PMOM(0:3)=PCL_IN(1,0:3)
    PCL(1,0:3)=0d0
    DO i=2,NLOOPLINE
       DO j=0,3
          PCL(i,j)=PCL_IN(i,j)-PMOM(j)
       ENDDO
    ENDDO
    IF(RECYCLING)THEN
       PCL0(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL0(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%stable=.TRUE.
       item%indices(0)=idim
       item%indices(1:NLOOPLINE)=indices(1:NLOOPLINE)
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL0(1:NLOOPLINE,0:3)
       CALL cibppave_bt_search(item,cibp_save,find)
       IF(find)THEN
          si(1:4)=item%value(1:4)
          STABLE_IREGI=item%stable
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    ! NLOOPLINE<= 2 to avoid the mixing of UV and IR 
    IF(NLOOPLINE.EQ.1.AND.indices(1).EQ.1)THEN
       IF(ABS(M2L(1)).LT.EPS)THEN
          IF(idim.NE.-2)THEN
             si(1:4)=DCMPLX(0d0)
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ELSE
             ! aviod mixing of UV and IR poles
             si(1:4)=B0CC1(0d0,DCMPLX(0d0),DCMPLX(0d0))*(-prefactor)
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ENDIF
       ELSE
          IF(idim.EQ.0)THEN
             si(1:4)=A0CC1(M2L(1))
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ENDIF
       ENDIF
    ENDIF
    ! NLOOPLINE<= 2 to avoid the mixing of UV and IR 
    IF(NLOOPLINE.EQ.2.AND.idim.EQ.0)THEN
       ! One should not but the two IF together
       ! Otherwise, it will not pass in ML5
       IF(indices(1).EQ.1.AND.indices(2).EQ.1)THEN
          r12(0:3)=PCL(1,0:3)-PCL(2,0:3)
          p2=scalarprod(r12,r12)
          si(1:4)=B0CC1(p2,M2L(1),M2L(2))
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
    ENDIF
    ! main loop
    CALL CRSMatrices(NLOOPLINE,PCL,M2L,RMATRIX,SMATRIX,detR,detS)
    indices_one=.TRUE.
    nindtot=0
    lind=0
    DO i=1,NLOOPLINE
       IF(indices(i).NE.1.AND.indices_one)THEN
          indices_one=.FALSE.
          lind=i
       ENDIF
       nindtot=nindtot+indices(i)
       indices0(i)=indices(i)
    ENDDO
    IF(ABS(detR).GE.EPS.AND.ABS(detS).GE.EPS)THEN
       ! detR!=0 & detS!=0
       ! the end of the recursion
       IF(idim.EQ.0.AND.indices_one)THEN
          IF(NLOOPLINE.GE.5)THEN
             ! Eq.(4.2) in 1009.4436 and generalization of Eq.(D3) in arXiv:0509141
             indices1(1:NLOOPLINE-1)=indices(2:NLOOPLINE)
             PCL1(1:NLOOPLINE-1,0:3)=PCL(2:NLOOPLINE,0:3)
             M2L1(1:NLOOPLINE-1)=M2L(2:NLOOPLINE)
             si(1:4)=DCMPLX(0d0)
             DO i=1,NLOOPLINE
                IF(i.GT.1)THEN
                   indices1(i-1)=indices(i-1)
                   PCL1(i-1,0:3)=PCL(i-1,0:3)
                   M2L1(i-1)=M2L(i-1)
                ENDIF
                temp=SIGNED_CMINOR11(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),1,i+1)
                IF(ABS(temp).LT.EPS)CYCLE
                si(1:4)=si(1:4)-temp*&
                     comp_scalar_integral_reduce(NLOOPLINE-1,idim,indices1,PCL1,M2L1)
             ENDDO
             si(1:4)=si(1:4)/detR
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ELSE
             si(1:4)=I0CC1(NLOOPLINE,PCL,M2L)
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ENDIF
       ENDIF
       ! if idim <0
       ! D<4, Eq .15 in hep-ph/0303184v3.
       IF(idim.LT.0)THEN
          idim0=idim+2
          z0(0)=DCMPLX(1d0)
          z0(1:NLOOPLINE)=DCMPLX(0d0)
          CALL CMNXNINV(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),ok_flag)
          IF(.NOT.ok_flag)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          z0(0:NLOOPLINE)=ccrdot(NLOOPLINE+1,NLOOPLINE+1,&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),z0(0:NLOOPLINE))
          c=-z0(0)
          z(1:NLOOPLINE)=z0(1:NLOOPLINE)
          maxtemp=MAXVAL(ABS(z0(0:NLOOPLINE)))
          maxtemp=MAX(maxtemp,1d-99)
          si(1:4)=comp_scalar_integral_reduce(NLOOPLINE,idim0,indices,PCL,M2L)
          temp=DCMPLX(3+idim0-nindtot)
          si(4)=temp*si(4)-2d0*(si(1)+si(2))
          si(1)=temp*si(1)
          si(2)=temp*si(2)-2d0*si(3)
          si(3)=temp*si(3)
          si(1:4)=prefactor*si(1:4)
          DO i=1,NLOOPLINE
             IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
             indices0(i)=indices0(i)-1
             si(1:4)=si(1:4)+z(i)*&
                  comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
             indices0(i)=indices0(i)+1
          ENDDO
          si(1:4)=si(1:4)/c
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
       ! Eq .18 in hep - ph/0303184 v3
       IF(indices_one)THEN
          ! usually appears in pentagon
          ! now, I just take the simplest way,i.e., GramDet!=0
          ! improve it like as done in arXiv:1009.4436
          IF(3+idim-nindtot.EQ.0)THEN
             CALL IBP2PAVE(NLOOPLINE,idim,indices,nindtot,factor,paveindices)
             si(1:4)=comp_pavefun_reduce(NLOOPLINE,paveindices,PCL,M2L)*factor
             IF(RECYCLING)item%value(1:4)=si(1:4)
             RETURN
          ENDIF
          z0(0)=DCMPLX(1d0)
          z0(1:NLOOPLINE)=DCMPLX(0d0)
          CALL CMNXNINV(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),ok_flag)
          IF(.NOT.ok_flag)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          z0(0:NLOOPLINE)=ccrdot(NLOOPLINE+1,NLOOPLINE+1,&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),z0(0:NLOOPLINE))
          c=-z0(0)
          z(1:NLOOPLINE)=z0(1:NLOOPLINE)
          maxtemp=MAXVAL(ABS(z0(0:NLOOPLINE)))
          maxtemp=MAX(maxtemp,1d-99)
          idim0=idim-2
          IF(ABS(c)/maxtemp.LT.EPS)THEN
             si(1:4)=DCMPLX(0d0)
          ELSE
             si(1:4)=c*comp_scalar_integral_reduce(NLOOPLINE,idim0,indices,PCL,M2L)
          ENDIF
          DO i=1,NLOOPLINE
             IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
             indices0(i)=indices0(i)-1
             si(1:4)=si(1:4)-z(i)*&
                  comp_scalar_integral_reduce(NLOOPLINE,idim0,indices0,PCL,M2L)
             indices0(i)=indices0(i)+1
          ENDDO
          si(1:4)=si(1:4)/prefactor
          temp=3+idim-nindtot
          temp=1d0/temp
          si(4)=si(4)*temp+2d0*temp**2*(si(1)+si(2))+4d0*temp**3*si(3)
          si(1)=si(1)*temp
          si(2)=si(2)*temp+2d0*temp**2*si(3)
          si(3)=si(3)*temp
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
       ! Eq .19 in hep - ph/0303184 v3
       IF(idim.EQ.0)THEN
          CALL CMNXNINV(NLOOPLINE,RMATRIX(1:NLOOPLINE,1:NLOOPLINE),&
               INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE),ok_flag)
          IF(.NOT.ok_flag)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          maxtemp=MAXVAL(ABS(INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE)))
          maxtemp=MAX(maxtemp,1d-99)
          DO i=1,NLOOPLINE
             IF(ABS(INVRMATRIX(lind,i))/maxtemp.LT.EPS)THEN
                lnonzero(i)=.FALSE.
             ELSE
                lnonzero(i)=.TRUE.
             ENDIF
          ENDDO
          indices0(lind)=indices0(lind)-1
          indices00(1:NLOOPLINE)=indices0(1:NLOOPLINE)
          nindtot=nindtot-1
          temp=4+idim-nindtot
          sitemp(1:4)=(-1d0)*comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
          DO i=1,NLOOPLINE
             IF(.NOT.lnonzero(i))THEN
                siz0(i,1:4)=DCMPLX(0d0)
                CYCLE
             ENDIF
             indices00(i)=indices00(i)-1
!             siz0(i,1:4)=(-1d0)*scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
             siz0(i,1:4)=sitemp(1:4)
             siz0(i,4)=siz0(i,4)*temp-2d0*(siz0(i,1)+siz0(i,2))
             siz0(i,1)=siz0(i,1)*temp
             siz0(i,2)=siz0(i,2)*temp-2d0*siz0(i,3)
             siz0(i,3)=siz0(i,3)*temp
             DO j=1,NLOOPLINE
                indices00(j)=indices00(j)+1
                siz0(i,1:4)=siz0(i,1:4)+&
                     indices0(j)*comp_scalar_integral_reduce(NLOOPLINE,idim,indices00,PCL,M2L)
                indices00(j)=indices00(j)-1
             ENDDO
             indices00(i)=indices00(i)+1
          ENDDO
          si(1:4)=ccldot(NLOOPLINE,4,siz0(1:NLOOPLINE,1:4),&
               INVRMATRIX(lind,1:NLOOPLINE))
          si(1:4)=si(1:4)/DCMPLX(indices0(lind))
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
       ! Eq .21 in hep - ph/0303184 v3
       IF(idim.GT.0)THEN
          CALL CMNXNINV(NLOOPLINE,RMATRIX(1:NLOOPLINE,1:NLOOPLINE),&
               INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE),ok_flag)
          IF(.NOT.ok_flag)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          maxtemp=MAXVAL(ABS(INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE)))
          maxtemp=MAX(maxtemp,1d-99)
          DO i=1,NLOOPLINE
             IF(ABS(INVRMATRIX(lind,i))/maxtemp.LT.EPS)THEN
                lnonzero(i)=.FALSE.
             ELSE
                lnonzero(i)=.TRUE.
             ENDIF
          ENDDO
          z0(0)=1d0
          z0(1:NLOOPLINE)=0d0
          CALL CMNXNINV(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),ok_flag)
          IF(.NOT.ok_flag)THEN
             STABLE_IREGI=.FALSE.
             IF(RECYCLING)THEN
                item%value(1:4)=DCMPLX(0d0)
                item%stable=.FALSE.
             ENDIF
             RETURN
          ENDIF
          z0(0:NLOOPLINE)=ccrdot(NLOOPLINE+1,NLOOPLINE+1,&
               INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),z0(0:NLOOPLINE))
          c=-z0(0)
          z(1:NLOOPLINE)=z0(1:NLOOPLINE)
          maxtemp=1d0
          idim0=idim-2
          indices0(lind)=indices0(lind)-1
          indices00(1:NLOOPLINE)=indices0(1:NLOOPLINE)
          nindtot=nindtot-1
          IF(ABS(c).GE.EPS)THEN
             sitemp(1:4)=(-1d0*c)*comp_scalar_integral_reduce(NLOOPLINE,idim0,indices0,PCL,M2L)
           ELSE
             sitemp(1:4)=DCMPLX(0d0)
          ENDIF
          DO i=1,NLOOPLINE
             IF(.NOT.lnonzero(i))THEN
                siz0(i,1:4)=DCMPLX(0d0)
                CYCLE
             ENDIF
!             siz0(i,1:4)=(-1d0*c)*scalar_integral_reduce(NLOOPLINE,idim0,indices0,PCL,M2L)
             siz0(i,1:4)=sitemp(1:4)
             DO j=1,NLOOPLINE
                temp=z(j)
                IF(j.EQ.i)temp=temp-DCMPLX(1d0)
                IF(ABS(temp)/maxtemp.LT.EPS)CYCLE
                indices00(j)=indices00(j)-1
                siz0(i,1:4)=siz0(i,1:4)+&
                     temp*comp_scalar_integral_reduce(NLOOPLINE,idim0,indices00,PCL,M2L)
                indices00(j)=indices00(j)+1
             ENDDO
             siz0(i,1:4)=siz0(i,1:4)*prefactor**(-1)
          ENDDO
          si(1:4)=ccldot(NLOOPLINE,4,siz0(1:NLOOPLINE,1:4),&
               INVRMATRIX(lind,1:NLOOPLINE))
          si(1:4)=si(1:4)/DCMPLX(indices0(lind))
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
    ENDIF

    ! Eq .10 and delete the sum in i, cancel the common z(i) && Eq .14 in hep - ph/0303184 v3.
    IF(ABS(detR).LT.EPS.AND.ABS(detS).GE.EPS.AND.3+idim-nindtot.EQ.0)THEN
       ! to avoid the mixing of UV and IR poles
       IF(idim.EQ.0.AND.NLOOPLINE.EQ.3.AND.indices_one)THEN
          si(1:4)=I0CC1(NLOOPLINE,PCL,M2L)
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
       CALL CMNXNINV(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
            INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),ok_flag)
       IF(.NOT.ok_flag)THEN
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
       maxtemp=MAXVAL(ABS(INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE)))
       maxtemp=MAX(maxtemp,1d-99)
       DO i=0,NLOOPLINE
          IF(ABS(INVSMATRIX(lind,i))/maxtemp.LT.EPS)THEN
             lnonzero2(i)=.FALSE.
          ELSE
             lnonzero2(i)=.TRUE.
          ENDIF
       ENDDO
       indices0(lind)=indices0(lind)-1
       indices00(1:NLOOPLINE)=indices0(1:NLOOPLINE)
       nindtot=nindtot-1
       idim0=idim-2
       IF(lnonzero2(0))THEN
          siz1(0,1:4)=-1d0/prefactor*&
               comp_scalar_integral_reduce(NLOOPLINE,idim0,indices0,PCL,M2L)
       ELSE
          siz1(0,1:4)=DCMPLX(0d0)
       ENDIF
       temp=4+idim-nindtot
       DO i=1,NLOOPLINE
          IF(.NOT.lnonzero2(i))THEN
             siz1(i,1:4)=DCMPLX(0d0)
             CYCLE
          ENDIF
          siz1(i,1:4)=(-1d0)*comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
          siz1(i,4)=temp*siz1(i,4)-2d0*(siz1(i,1)+siz1(i,2))
          siz1(i,1)=temp*siz1(i,1)
          siz1(i,2)=temp*siz1(i,2)-2d0*siz1(i,3)
          siz1(i,3)=temp*siz1(i,3)
          indices00(i)=indices00(i)-1
          DO j=1,NLOOPLINE
             indices00(j)=indices00(j)+1
             siz1(i,1:4)=siz1(i,1:4)+&
                  DCMPLX(indices0(j))*comp_scalar_integral_reduce(NLOOPLINE,idim,indices00,PCL,M2L)
             indices00(j)=indices00(j)-1
          ENDDO
          indices00(i)=indices00(i)+1
       ENDDO
       si(1:4)=ccldot(NLOOPLINE+1,4,siz1(0:NLOOPLINE,1:4),&
            INVSMATRIX(lind,0:NLOOPLINE))
       si(1:4)=si(1:4)/DCMPLX(indices0(lind))
       IF(RECYCLING)item%value(1:4)=si(1:4)
       RETURN
    ENDIF

    ! Eq .22 in hep - ph/0303184 v3
    IF(ABS(detR).LT.EPS.AND.ABS(detS).GE.EPS)THEN
       z0(0)=DCMPLX(1d0)
       z0(1:NLOOPLINE)=DCMPLX(0d0)
       CALL CMNXNINV(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
            INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),ok_flag)
       IF(.NOT.ok_flag)THEN
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
       z0(0:NLOOPLINE)=ccrdot(NLOOPLINE+1,NLOOPLINE+1,&
            INVSMATRIX(0:NLOOPLINE,0:NLOOPLINE),z0(0:NLOOPLINE))
       c=-z0(0)
       z(1:NLOOPLINE)=z0(1:NLOOPLINE)
       maxtemp=MAXVAL(ABS(z0(0:NLOOPLINE)))
       maxtemp=MAX(maxtemp,1d-99)
       temp=DBLE(3+idim-nindtot)
       temp=1d0/temp
       idim0=idim-2
       si(1:4)=DCMPLX(0d0)
       DO i=1,NLOOPLINE
          IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
          indices0(i)=indices0(i)-1
          si(1:4)=si(1:4)-z(i)*&
               comp_scalar_integral_reduce(NLOOPLINE,idim0,indices0,PCL,M2L)
          indices0(i)=indices0(i)+1
       ENDDO
       si(4)=si(4)*temp+2d0*temp**2*(si(1)+si(2))+4d0*temp**3*si(3)
       si(1)=si(1)*temp
       si(2)=si(2)*temp+2d0*temp**2*si(3)
       si(3)=si(3)*temp
       si(1:4)=si(1:4)/prefactor
       IF(RECYCLING)item%value(1:4)=si(1:4)
       RETURN
    ENDIF
    
    ! Eq .23 in hep - ph/0303184 v3
    IF(ABS(detR).GE.EPS.AND.ABS(detS).LT.EPS)THEN
       CALL CMNXNINV(NLOOPLINE,RMATRIX(1:NLOOPLINE,1:NLOOPLINE),&
            INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE),ok_flag)
       IF(.NOT.ok_flag)THEN
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
       c=DCMPLX(1d0)
       z(1:NLOOPLINE)=DCMPLX(1d0)
       z(1:NLOOPLINE)=ccrdot(NLOOPLINE,NLOOPLINE,&
            INVRMATRIX(1:NLOOPLINE,1:NLOOPLINE),z(1:NLOOPLINE))
       maxtemp=MAXVAL(ABS(z(1:NLOOPLINE)))
       maxtemp=MAX(maxtemp,1d-99)
       si(1:4)=DCMPLX(0d0)
       DO i=1,NLOOPLINE
          IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
          indices0(i)=indices0(i)-1
          si(1:4)=si(1:4)+z(i)*comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
          indices0(i)=indices0(i)+1
       ENDDO
       IF(RECYCLING)item%value(1:4)=si(1:4)
       RETURN
    ENDIF

    IF(ABS(detR).LT.EPS.AND.ABS(detS).LT.EPS)THEN
       CALL solve_cleqs(NLOOPLINE+1,SMATRIX(0:NLOOPLINE,0:NLOOPLINE),&
            z0(0:NLOOPLINE),triv_flag)
       IF(triv_flag)THEN
          STABLE_IREGI=.FALSE.
          IF(RECYCLING)THEN
             item%value(1:4)=DCMPLX(0d0)
             item%stable=.FALSE.
          ENDIF
          RETURN
       ENDIF
       c=-z0(0)
       z(1:NLOOPLINE)=z0(1:NLOOPLINE)
       maxtemp=MAXVAL(ABS(z0(0:NLOOPLINE)))
       maxtemp=MAX(maxtemp,1d-99)
       IF(ABS(c)/maxtemp.GE.EPS)THEN
          ! Eq .26 in hep - ph/0303184 v3
          z(1:NLOOPLINE)=z(1:NLOOPLINE)/c
          maxtemp=MAXVAL(ABS(z(1:NLOOPLINE)))
          maxtemp=MAX(maxtemp,1d-99)
          si(1:4)=DCMPLX(0d0)
          DO i=1,NLOOPLINE
             IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
             indices0(i)=indices0(i)-1
             si(1:4)=si(1:4)+z(i)*&
                  comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
             indices0(i)=indices0(i)+1
          ENDDO
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ELSE
          ! Eq .28 in hep - ph/0303184 v3
          j=1
          DO i=2,NLOOPLINE
             IF(ABS(z(j))/maxtemp.LT.EPS.OR.(ABS(z(i))/maxtemp.GE.EPS&
                  .AND.indices0(i).GT.indices0(j)))THEN
                j=i
             ENDIF
          ENDDO
          indices0(j)=indices0(j)+1
          si(1:4)=DCMPLX(0d0)
          DO i=1,NLOOPLINE
             IF(i.EQ.j)CYCLE
             IF(ABS(z(i))/maxtemp.LT.EPS)CYCLE
             indices0(i)=indices0(i)-1
             si(1:4)=si(1:4)-z(i)*&
                  comp_scalar_integral_reduce(NLOOPLINE,idim,indices0,PCL,M2L)
             indices0(i)=indices0(i)+1
          ENDDO
          si(1:4)=si(1:4)/z(j)
          IF(RECYCLING)item%value(1:4)=si(1:4)
          RETURN
       ENDIF
    ENDIF

    WRITE(*,*)"ERROR in comp_scalar_integral_reduce"
    STOP
  END FUNCTION comp_scalar_integral_reduce


  FUNCTION comp_pave_opt_reduce(NLOOPLINE,paveindices,PCL,M2L) RESULT(res)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::paveindices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(4)::res
    REAL(KIND(1d0))::factor
    INTEGER::idim
    INTEGER,DIMENSION(NLOOPLINE)::indices
    res(1:4)=comp_pavefun_reduce(NLOOPLINE,paveindices,PCL,M2L)
    IF(.NOT.STABLE_IREGI)THEN
       STABLE_IREGI=.TRUE.
       CALL PAVE2IBP(NLOOPLINE,paveindices,factor,idim,indices)
       res(1:4)=factor*comp_scalar_integral_reduce(NLOOPLINE,idim,indices,PCL,M2L)
    ENDIF
    RETURN
  END FUNCTION comp_pave_opt_reduce

END MODULE csi_reduce
