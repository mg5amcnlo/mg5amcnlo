MODULE matrices
USE global
USE funlib
USE kinematics
USE matrix_base
USE cmatrix_base
USE binary_tree
IMPLICIT NONE
CONTAINS
  SUBROUTINE CaleyMatrix(NLOOPLINE,PCL,M2L,matrix,det)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(0:3)::rij
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::matrix
    REAL(KIND(1d0)),INTENT(OUT)::det
    INTEGER::i,j
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          matrix(i,j)=scalarprod(rij(0:3),rij(0:3))-M2L(i)-M2L(j)
          IF(i.LT.j)THEN
             matrix(j,i)=matrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    det=MNXNDET(NLOOPLINE,matrix)
    RETURN
  END SUBROUTINE CaleyMatrix

  SUBROUTINE Complex_CaleyMatrix(NLOOPLINE,PCL,M2L,matrix,det)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(0:3)::rij
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::matrix
    COMPLEX(KIND(1d0)),INTENT(OUT)::det
    INTEGER::i,j
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          matrix(i,j)=scalarprod(rij(0:3),rij(0:3))-M2L(i)-M2L(j)
          IF(i.LT.j)THEN
             matrix(j,i)=matrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    det=CMNXNDET(NLOOPLINE,matrix)
    RETURN
  END SUBROUTINE Complex_CaleyMatrix

  SUBROUTINE CaleyMatrix2(NLOOPLINE,PijMatrix,M2L,matrix,det)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix ! PijMatrix(i,j)=Pi.Pj
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::matrix
    REAL(KIND(1d0)),INTENT(OUT)::det
    INTEGER::i,j
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          matrix(i,j)=PijMatrix(i,i)+PijMatrix(j,j)-2d0*PijMatrix(i,j)-M2L(i)-M2L(j)
          IF(i.LT.j)THEN
             matrix(j,i)=matrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    det=MNXNDET(NLOOPLINE,matrix)
    RETURN
  END SUBROUTINE CaleyMatrix2

  SUBROUTINE GramMatrix(NLOOPLINE,PCL,M2L,matrix,det)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(0:3)::rij
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::matrix
    REAL(KIND(1d0)),INTENT(OUT)::det
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3)::PE
    INTEGER::i,j
    CALL PCL2PE(NLOOPLINE,PCL,PE)
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          matrix(i,j)=scalarprod(PE(i,0:3),PE(j,0:3))
          IF(i.LT.j)THEN
             matrix(j,i)=matrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    det=MNXNDET(NLOOPLINE,matrix)
    RETURN
  END SUBROUTINE GramMatrix
  ! hep-ph/0303184
  SUBROUTINE RSMatrices(NLOOPLINE,PCL,M2L,rmatrix,smatrix,rdet,sdet)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
!    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(0:3)::rij
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::rmatrix
    REAL(KIND(1d0)),DIMENSION(0:NLOOPLINE,0:NLOOPLINE),INTENT(OUT)::smatrix
    REAL(KIND(1d0)),INTENT(OUT)::rdet,sdet
    INTEGER::i,j
    LOGICAL::find
    TYPE(rsmatrices_node),POINTER::item
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL1
    IF(RECYCLING)THEN
       PCL1(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL1(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL1(1:NLOOPLINE,0:3)
       CALL rsmatrices_bt_search(item,rsmatrices_save,find)
       IF(find)THEN
          rdet=item%detR
          sdet=item%detS
          RMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%RMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          SMATRIX(0:NLOOPLINE,0:NLOOPLINE)=&
               item%SMATRIX(0:NLOOPLINE,0:NLOOPLINE)
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    smatrix(0,0)=0d0
    DO i=1,NLOOPLINE
       smatrix(0,i)=1d0
       smatrix(i,0)=1d0
    ENDDO
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          rmatrix(i,j)=scalarprod(rij(0:3),rij(0:3))-M2L(i)-M2L(j)
          smatrix(i,j)=rmatrix(i,j)
          IF(i.LT.j)THEN
             rmatrix(j,i)=rmatrix(i,j)
             smatrix(j,i)=smatrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    rdet=MNXNDET(NLOOPLINE,rmatrix)
    sdet=MNXNDET(NLOOPLINE+1,smatrix(0:NLOOPLINE,0:NLOOPLINE))
    IF(RECYCLING)THEN
       item%detS=sdet
       item%detR=rdet
       item%SMATRIX(0:NLOOPLINE,0:NLOOPLINE)=SMATRIX(0:NLOOPLINE,0:NLOOPLINE)
       item%RMATRIX(1:NLOOPLINE,1:NLOOPLINE)=RMATRIX(1:NLOOPLINE,1:NLOOPLINE)
    ENDIF
    RETURN
  END SUBROUTINE RSMATRICES

  SUBROUTINE CRSMatrices(NLOOPLINE,PCL,M2L,rmatrix,smatrix,rdet,sdet)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(0:3)::rij
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::rmatrix
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPLINE,0:NLOOPLINE),INTENT(OUT)::smatrix
    COMPLEX(KIND(1d0)),INTENT(OUT)::rdet,sdet
    INTEGER::i,j
    LOGICAL::find
    TYPE(crsmatrices_node),POINTER::item
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL1
    IF(RECYCLING)THEN
       PCL1(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL1(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL1(1:NLOOPLINE,0:3)
       CALL crsmatrices_bt_search(item,crsmatrices_save,find)
       IF(find)THEN
          rdet=item%detR
          sdet=item%detS
          RMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%RMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          SMATRIX(0:NLOOPLINE,0:NLOOPLINE)=&
               item%SMATRIX(0:NLOOPLINE,0:NLOOPLINE)
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    smatrix(0,0)=DCMPLX(0d0)
    DO i=1,NLOOPLINE
       smatrix(0,i)=DCMPLX(1d0)
       smatrix(i,0)=DCMPLX(1d0)
    ENDDO
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          rmatrix(i,j)=scalarprod(rij(0:3),rij(0:3))-M2L(i)-M2L(j)
          smatrix(i,j)=rmatrix(i,j)
          IF(i.LT.j)THEN
             rmatrix(j,i)=rmatrix(i,j)
             smatrix(j,i)=smatrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    rdet=CMNXNDET(NLOOPLINE,rmatrix)
    sdet=CMNXNDET(NLOOPLINE+1,smatrix(0:NLOOPLINE,0:NLOOPLINE))
    IF(RECYCLING)THEN
       item%detS=sdet
       item%detR=rdet
       item%SMATRIX(0:NLOOPLINE,0:NLOOPLINE)=SMATRIX(0:NLOOPLINE,0:NLOOPLINE)
       item%RMATRIX(1:NLOOPLINE,1:NLOOPLINE)=RMATRIX(1:NLOOPLINE,1:NLOOPLINE)
    ENDIF
    RETURN
  END SUBROUTINE CRSMATRICES

  SUBROUTINE RSMatrices2(NLOOPLINE,PijMatrix,M2L,rmatrix,smatrix,rdet,sdet)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE,1:NLOOPLINE),INTENT(IN)::PijMatrix ! PijMatrix(i,j)=Pi.Pj 
    REAL(KIND(1d0)),DIMENSION(1:NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::rmatrix
    REAL(KIND(1d0)),DIMENSION(0:NLOOPLINE,0:NLOOPLINE),INTENT(OUT)::smatrix
    REAL(KIND(1d0)),INTENT(OUT)::rdet,sdet
    INTEGER::i,j
    smatrix(0,0)=0d0
    DO i=1,NLOOPLINE
       smatrix(0,i)=1d0
       smatrix(i,0)=1d0
    ENDDO
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rmatrix(i,j)=PijMatrix(i,i)+PijMatrix(j,j)-2d0*PijMatrix(i,j)-M2L(i)-M2L(j)
          smatrix(i,j)=rmatrix(i,j)
          IF(i.LT.j)THEN
             rmatrix(j,i)=rmatrix(i,j)
             smatrix(j,i)=smatrix(i,j)
          ENDIF
       ENDDO
    ENDDO
    rdet=MNXNDET(NLOOPLINE,rmatrix)
    sdet=MNXNDET(NLOOPLINE+1,smatrix(0:NLOOPLINE,0:NLOOPLINE))
    RETURN
  END SUBROUTINE RSMATRICES2
  ! in hep - ph/0509141
  SUBROUTINE XYZMATRICES(NLOOPLINE,PCL,M2L,XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(2:NLOOPLINE,2:NLOOPLINE),INTENT(OUT)::ZMATRIX
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::YMATRIX,XMATRIX
    REAL(KIND(1d0)),INTENT(OUT)::detY,detZ
    INTEGER::i,j
    REAL(KIND(1d0)),DIMENSION(0:3)::ri1,rj1,rij
    LOGICAL::find
    TYPE(xyzmatrices_node),POINTER::item
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL1
    IF(RECYCLING)THEN
       PCL1(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL1(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL1(1:NLOOPLINE,0:3)
       CALL xyzmatrices_bt_search(item,xyzmatrices_save,find)
       IF(find)THEN
          detY=item%detY
          detZ=item%detZ
          YMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%YMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          XMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%XMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)=&
               item%ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          YMATRIX(i,j)=-scalarprod(rij(0:3),rij(0:3))+M2L(i)+M2L(j)
          IF(i.GE.2)THEN
             ri1(0:3)=PCL(i,0:3)-PCL(1,0:3)
             rj1(0:3)=PCL(j,0:3)-PCL(1,0:3)
             ZMATRIX(i,j)=2d0*scalarprod(ri1(0:3),rj1(0:3))
             XMATRIX(i,j)=ZMATRIX(i,j)
          ELSE
             IF(j.EQ.1)THEN
                XMATRIX(i,j)=2*M2L(1)
             ELSE
                ri1(0:3)=PCL(j,0:3)-PCL(1,0:3)
                XMATRIX(i,j)=scalarprod(ri1(0:3),ri1(0:3))-M2L(j)+M2L(1)
             ENDIF
          ENDIF
          IF(i.LT.j)THEN
             YMATRIX(j,i)=YMATRIX(i,j)
             XMATRIX(j,i)=XMATRIX(i,j)
             IF(i.GE.2)THEN
                ZMATRIX(j,i)=ZMATRIX(i,j)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    detZ=MNXNDET(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE))
    detY=MNXNDET(NLOOPLINE,YMATRIX(1:NLOOPLINE,1:NLOOPLINE))
    IF(RECYCLING)THEN
       item%detY=detY
       item%detZ=detZ
       item%XMATRIX(1:NLOOPLINE,1:NLOOPLINE)=XMATRIX(1:NLOOPLINE,1:NLOOPLINE)
       item%YMATRIX(1:NLOOPLINE,1:NLOOPLINE)=YMATRIX(1:NLOOPLINE,1:NLOOPLINE)
       item%ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)=ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)
    ENDIF
    RETURN
  END SUBROUTINE XYZMATRICES

  SUBROUTINE CXYZMATRICES(NLOOPLINE,PCL,M2L,XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    COMPLEX(KIND(1d0)),DIMENSION(2:NLOOPLINE,2:NLOOPLINE),INTENT(OUT)::ZMATRIX
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::YMATRIX,XMATRIX
    COMPLEX(KIND(1d0)),INTENT(OUT)::detY,detZ
    INTEGER::i,j
    REAL(KIND(1d0)),DIMENSION(0:3)::ri1,rj1,rij
    LOGICAL::find
    TYPE(cxyzmatrices_node),POINTER::item
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3)::PCL1
    IF(RECYCLING)THEN
       PCL1(1,0:3)=0d0
       DO i=2,NLOOPLINE
          PCL1(i,0:3)=PCL(i,0:3)-PCL(1,0:3)
       ENDDO
       ALLOCATE(item)
       item%NLOOPLINE=NLOOPLINE
       item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
       item%PCL(1:NLOOPLINE,0:3)=PCL1(1:NLOOPLINE,0:3)
       CALL cxyzmatrices_bt_search(item,cxyzmatrices_save,find)
       IF(find)THEN
          detY=item%detY
          detZ=item%detZ
          YMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%YMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          XMATRIX(1:NLOOPLINE,1:NLOOPLINE)=&
               item%XMATRIX(1:NLOOPLINE,1:NLOOPLINE)
          ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)=&
               item%ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)
          DEALLOCATE(item)
          RETURN
       ENDIF
    ENDIF
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij(0:3)=PCL(i,0:3)-PCL(j,0:3)
          YMATRIX(i,j)=-scalarprod(rij(0:3),rij(0:3))+M2L(i)+M2L(j)
          IF(i.GE.2)THEN
             ri1(0:3)=PCL(i,0:3)-PCL(1,0:3)
             rj1(0:3)=PCL(j,0:3)-PCL(1,0:3)
             ZMATRIX(i,j)=2d0*scalarprod(ri1(0:3),rj1(0:3))
             XMATRIX(i,j)=ZMATRIX(i,j)
          ELSE
             IF(j.EQ.1)THEN
                XMATRIX(i,j)=2*M2L(1)
             ELSE
                ri1(0:3)=PCL(j,0:3)-PCL(1,0:3)
                XMATRIX(i,j)=scalarprod(ri1(0:3),ri1(0:3))-M2L(j)+M2L(1)
             ENDIF
          ENDIF
          IF(i.LT.j)THEN
             YMATRIX(j,i)=YMATRIX(i,j)
             XMATRIX(j,i)=XMATRIX(i,j)
             IF(i.GE.2)THEN
                ZMATRIX(j,i)=ZMATRIX(i,j)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    detZ=CMNXNDET(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE))
    detY=CMNXNDET(NLOOPLINE,YMATRIX(1:NLOOPLINE,1:NLOOPLINE))
    IF(RECYCLING)THEN
       item%detY=detY
       item%detZ=detZ
       item%XMATRIX(1:NLOOPLINE,1:NLOOPLINE)=XMATRIX(1:NLOOPLINE,1:NLOOPLINE)
       item%YMATRIX(1:NLOOPLINE,1:NLOOPLINE)=YMATRIX(1:NLOOPLINE,1:NLOOPLINE)
       item%ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)=ZMATRIX(2:NLOOPLINE,2:NLOOPLINE)
    ENDIF
    RETURN
  END SUBROUTINE CXYZMATRICES

  SUBROUTINE XYZMATRICES2(NLOOPLINE,PijMatrix,M2L,XMATRIX,YMATRIX,ZMATRIX,detY,detZ)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix ! PijMatrix(i,j)=pi.pj
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    REAL(KIND(1d0)),DIMENSION(2:NLOOPLINE,2:NLOOPLINE),INTENT(OUT)::ZMATRIX
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(OUT)::YMATRIX,XMATRIX
    REAL(KIND(1d0)),INTENT(OUT)::detY,detZ
    INTEGER::i,j
    REAL(KIND(1d0))::ri1,rj1,rij
    DO i=1,NLOOPLINE
       DO j=i,NLOOPLINE
          rij=PijMatrix(i,i)+PijMatrix(j,j)-2d0*PijMatrix(i,j)
          YMATRIX(i,j)=-rij+M2L(i)+M2L(j)
          IF(i.GE.2)THEN
             ZMATRIX(i,j)=2d0*(PijMatrix(i,j)+PijMatrix(1,1)-PijMatrix(i,1)-PijMatrix(j,1))
             XMATRIX(i,j)=ZMATRIX(i,j)
          ELSE
             IF(j.EQ.1)THEN
                XMATRIX(i,j)=2*M2L(1)
             ELSE
                ri1=PijMatrix(i,i)+PijMatrix(1,1)-2d0*PijMatrix(i,1)
                XMATRIX(i,j)=ri1-M2L(j)+M2L(1)
             ENDIF
          ENDIF
          IF(i.LT.j)THEN
             YMATRIX(j,i)=YMATRIX(i,j)
             XMATRIX(j,i)=XMATRIX(i,j)
             IF(i.GE.2)THEN
                ZMATRIX(j,i)=ZMATRIX(i,j)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    detZ=MNXNDET(NLOOPLINE-1,ZMATRIX(2:NLOOPLINE,2:NLOOPLINE))
    detY=MNXNDET(NLOOPLINE,YMATRIX(1:NLOOPLINE,1:NLOOPLINE))
    RETURN
  END SUBROUTINE XYZMATRICES2

END MODULE matrices
