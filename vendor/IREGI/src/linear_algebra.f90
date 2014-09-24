MODULE linear_algebra
CONTAINS
  SUBROUTINE solve_leqs(N,A,sol,triv)
    ! solve the linear equations Ax=0
    ! return a minimal solution,i.e.,in x the number of nonzero element is samllest but not 0
    IMPLICIT NONE
    INTEGER,INTENT(IN)::N
    REAL(KIND(1d0)),DIMENSION(N,N),INTENT(IN)::A
    REAL(KIND(1d0)),DIMENSION(N,N)::B
    INTEGER,DIMENSION(N)::iexchange
    INTEGER::i,j,inow,k,iswap,izero
    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    REAL(KIND(1d0)),DIMENSION(N),INTENT(OUT)::sol
    REAL(KIND(1d0)),DIMENSION(N)::row,coloum
    REAL(KIND(1d0))::maxv,temp
    LOGICAL,INTENT(OUT)::triv
    LOGICAL::allzero
    ! make matrix A to be upper triangular
    B(1:N,1:N)=A(1:N,1:N)
    maxv=MAXVAL(ABS(B(1:N,1:N)))
    IF(maxv.LT.EPS)THEN
       ! a trivial solution
       sol(1)=1d0
       sol(2:N)=0d0
       triv=.FALSE.
       RETURN
    ENDIF
    DO i=1,N
       iexchange(i)=i
    ENDDO
    inow=1
    izero=0
    DO i=1,N
       allzero=.TRUE.
       DO j=inow,N
          IF(ABS(B(j,inow))/maxv.GT.EPS)THEN
             allzero=.FALSE.
             IF(j.NE.inow)THEN
                row(1:N)=B(inow,1:N)
                B(inow,1:N)=B(j,1:N)
                B(j,1:N)=row(1:N)
             ENDIF
             EXIT
          ENDIF
       ENDDO
       IF(.NOT.allzero)THEN
          DO j=inow+1,N
             IF(inow+1.LE.N)THEN
                B(j,inow+1:N)=B(j,inow+1:N)-B(inow,inow+1:N)*B(j,inow)/B(inow,inow)
             ENDIF
             B(j,inow)=0d0
          ENDDO
          inow=inow+1
       ELSE
          ! the coloum is zero, put it to the end
          coloum(1:N)=B(1:N,inow)
          izero=izero+1
          iswap=iexchange(N)
          iexchange(N)=iexchange(inow)
          DO j=inow+1,N
             B(1:N,j-1)=B(1:N,j)
             IF(j.NE.N)THEN
                iexchange(j-1)=iexchange(j)
             ELSE
                iexchange(j-1)=iswap
             ENDIF
          ENDDO
          B(1:N,N)=coloum(1:N)
       ENDIF
!       PRINT *,i
!       DO j=1,N
!          WRITE(*,*)B(j,1:N)
!       ENDDO
    ENDDO
!    DO i=1,N
!       WRITE(*,*)B(i,1:N)
!    ENDDO
!    WRITE(*,*)"================"
    IF(izero.GT.0)THEN
       sol(iexchange(N))=1d0
       DO inow=1,N-1
          i=N-inow
          IF(ABS(B(i,i))/maxv.LT.EPS)THEN
             sol(iexchange(i))=0d0
          ELSE
             temp=0d0
             DO j=i+1,N
                temp=temp-B(i,j)*sol(iexchange(j))
             ENDDO
             sol(iexchange(i))=temp/B(i,i)
          ENDIF
       ENDDO
       triv=.FALSE.
    ELSE
       sol(1:N)=0d0
       triv=.TRUE.
    ENDIF
    RETURN
  END SUBROUTINE SOLVE_LEQS

  SUBROUTINE solve_cleqs(N,A,sol,triv)
    ! solve the complex linear equations Ax=0
    ! return a minimal solution,i.e.,in x the number of nonzero element is samllest but not 0
    IMPLICIT NONE
    INTEGER,INTENT(IN)::N
    COMPLEX(KIND(1d0)),DIMENSION(N,N),INTENT(IN)::A
    COMPLEX(KIND(1d0)),DIMENSION(N,N)::B
    INTEGER,DIMENSION(N)::iexchange
    INTEGER::i,j,inow,k,iswap,izero
    REAL(KIND(1d0)),PARAMETER::EPS=1d-10
    COMPLEX(KIND(1d0)),DIMENSION(N),INTENT(OUT)::sol
    COMPLEX(KIND(1d0)),DIMENSION(N)::row,coloum
    REAL(KIND(1d0))::maxv
    COMPLEX(KIND(1d0))::temp
    LOGICAL,INTENT(OUT)::triv
    LOGICAL::allzero
    ! make matrix A to be upper triangular
    B(1:N,1:N)=A(1:N,1:N)
    maxv=MAXVAL(ABS(B(1:N,1:N)))
    IF(maxv.LT.EPS)THEN
       ! a trivial solution
       sol(1)=DCMPLX(1d0)
       sol(2:N)=DCMPLX(0d0)
       triv=.FALSE.
       RETURN
    ENDIF
    DO i=1,N
       iexchange(i)=i
    ENDDO
    inow=1
    izero=0
    DO i=1,N
       allzero=.TRUE.
       DO j=inow,N
          IF(ABS(B(j,inow))/maxv.GT.EPS)THEN
             allzero=.FALSE.
             IF(j.NE.inow)THEN
                row(1:N)=B(inow,1:N)
                B(inow,1:N)=B(j,1:N)
                B(j,1:N)=row(1:N)
             ENDIF
             EXIT
          ENDIF
       ENDDO
       IF(.NOT.allzero)THEN
          DO j=inow+1,N
             IF(inow+1.LE.N)THEN
                B(j,inow+1:N)=B(j,inow+1:N)-B(inow,inow+1:N)*B(j,inow)/B(inow,inow)
             ENDIF
             B(j,inow)=0d0
          ENDDO
          inow=inow+1
       ELSE
          ! the coloum is zero, put it to the end
          coloum(1:N)=B(1:N,inow)
          izero=izero+1
          iswap=iexchange(N)
          iexchange(N)=iexchange(inow)
          DO j=inow+1,N
             B(1:N,j-1)=B(1:N,j)
             IF(j.NE.N)THEN
                iexchange(j-1)=iexchange(j)
             ELSE
                iexchange(j-1)=iswap
             ENDIF
          ENDDO
          B(1:N,N)=coloum(1:N)
       ENDIF
    ENDDO
    IF(izero.GT.0)THEN
       sol(iexchange(N))=DCMPLX(1d0)
       DO inow=1,N-1
          i=N-inow
          IF(ABS(B(i,i))/maxv.LT.EPS)THEN
             sol(iexchange(i))=DCMPLX(0d0)
          ELSE
             temp=DCMPLX(0d0)
             DO j=i+1,N
                temp=temp-B(i,j)*sol(iexchange(j))
             ENDDO
             sol(iexchange(i))=temp/B(i,i)
          ENDIF
       ENDDO
       triv=.FALSE.
    ELSE
       sol(1:N)=DCMPLX(0d0)
       triv=.TRUE.
    ENDIF
    RETURN
  END SUBROUTINE SOLVE_CLEQS

END MODULE linear_algebra
