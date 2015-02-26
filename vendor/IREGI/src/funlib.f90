MODULE funlib
  USE global
  IMPLICIT NONE
CONTAINS
  RECURSIVE FUNCTION factorial(i) RESULT(fac)
    IMPLICIT NONE
    INTEGER::fac
    INTEGER,INTENT(IN)::i
    INTEGER::init=0,j
    LOGICAL,DIMENSION(12)::lfacsave
    INTEGER,DIMENSION(12)::facsave
    SAVE facsave,init,lfacsave
    IF(init.EQ.0)THEN
       lfacsave(1:12)=.FALSE.
       init=1
    ENDIF
    IF(i.LT.0)THEN
       WRITE(*,*)"ERROR: i<0 in factorial with i=",i
       STOP
    ENDIF
    IF(i.GT.12)THEN
       WRITE(*,*)"ERROR: i > 12, please take long type (KIND=LINT) integer"
       STOP
    ENDIF
    IF(i.EQ.0)THEN
       fac=1
    ELSE
       IF(lfacsave(i))THEN
          fac=facsave(i)
       ELSE
          fac=i*factorial(i-1)
          facsave(i)=fac
          lfacsave(i)=.TRUE.
       ENDIF
    ENDIF
  END FUNCTION factorial

  SUBROUTINE PCL2PE(NLOOPLINE,PCL,PE)
    ! transfer the loop momenta q,q+p1,q+p1+p2,q+p1+p2+p3,... 
    ! to p1,p2,p3,p4,... 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::NLOOPLINE
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(OUT)::PE
    INTEGER::i
    DO i=1,NLOOPLINE-1
       PE(i,0:3)=PCL(i+1,0:3)-PCL(i,0:3)
    ENDDO
    PE(NLOOPLINE,0:3)=PCL(1,0:3)-PCL(NLOOPLINE,0:3)
    RETURN
  END SUBROUTINE PCL2PE

  RECURSIVE SUBROUTINE calc_all_integers(n,ntot,i,sol)
    ! finds all the non-negative solutions to x1,...,xn
    ! that x1+x2+...+xn==i
    ! the number of solutions should be ntot=C(i+n-1)^(n-1)=(i+n-1)!/(n-1)!/i!
    ! it can be recycled for all phase space point
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n,ntot,i
    INTEGER,DIMENSION(ntot,n),INTENT(OUT)::sol
    INTEGER::j,k,k0
    IF(i.EQ.0)THEN
       sol(1,1:n)=0
       RETURN
    ENDIF
    IF(n.EQ.1)THEN
       sol(1,1)=i
       RETURN
    ENDIF
    k0=0
    DO j=0,i
       k=xiarray_arg1(i-j,n-1)
       sol(k0+1:k0+k,1)=j
       CALL calc_all_integers(n-1,k,i-j,sol(k0+1:k0+k,2:n))
       k0=k0+k
    ENDDO
    RETURN
  END SUBROUTINE calc_all_integers

  SUBROUTINE all_Integers(n,ntot,i,sol,factor)
    ! finds all the non-negative solutions to x1,...,xn 
    ! that x1+x2+...+xn==i 
    ! the number of solutions should be ntot=C(i+n-1)^(n-1)=(i+n-1)!/(n-1)!/i!
    ! it can be recycled for all phase space point 
    IMPLICIT NONE
    INTEGER,INTENT(IN)::n,ntot,i
    INTEGER,DIMENSION(ntot,n),INTENT(OUT)::sol
    REAL(KIND(1d0)),DIMENSION(ntot),INTENT(OUT)::factor
    INTEGER::ifirst=0,j,jk,k,ntemptot
    INTEGER::maxxiarray,inum,nnum,maxxiarray1,maxxiarray2
    INTEGER::maxfactor_xiarray
    SAVE ifirst
    ! calculate xiarray_i_n first
    IF(ifirst.EQ.0)THEN
       maxxiarray=(MAXRANK_IREGI+1)*(MAXNLOOP_IREGI-1)
       IF(.NOT.ALLOCATED(xiarray))THEN
          ALLOCATE(xiarray(maxxiarray))
       ENDIF
       maxfactor_xiarray=(MAXRANK_IREGI+1)*(MAXNLOOP_IREGI-2)
       IF(.NOT.ALLOCATED(factor_xiarray))THEN
          ALLOCATE(factor_xiarray(MAXRANK_IREGI+2:maxfactor_xiarray+MAXRANK_IREGI+1))
       ENDIF
       ! x1+x2+...+xn==i
       ! C(i+n-1)^(n-1)=(i+n-1)!/(n-1)!/i!
       DO j=1,maxxiarray
          inum=MOD(j-1,MAXRANK_IREGI+1) ! i
          nnum=(j-1)/(MAXRANK_IREGI+1)+1 ! n
          maxxiarray1=xiarray_arg1(inum,nnum)
          maxxiarray2=nnum
          IF(.NOT.ALLOCATED(xiarray(j)%xiarray_i_n))THEN
             ALLOCATE(xiarray(j)%xiarray_i_n(maxxiarray1,maxxiarray2))
          ENDIF
          IF(nnum.EQ.1)THEN
             xiarray(j)%xiarray_i_n(1,1)=j-1
             CYCLE
          ENDIF
          CALL calc_all_integers(nnum,maxxiarray1,inum,&
               xiarray(j)%xiarray_i_n(1:maxxiarray1,1:maxxiarray2))
          IF(.NOT.ALLOCATED(factor_xiarray(j)%factor_xiarray_i_n))THEN
             ALLOCATE(factor_xiarray(j)%factor_xiarray_i_n(maxxiarray1))
          ENDIF
          DO jk=1,maxxiarray1
             factor_xiarray(j)%factor_xiarray_i_n(jk)=DBLE(factorial(inum))
             DO k=1,nnum
                factor_xiarray(j)%factor_xiarray_i_n(jk)=factor_xiarray(j)%factor_xiarray_i_n(jk)/&
                     DBLE(factorial(xiarray(j)%xiarray_i_n(jk,k)))
             ENDDO
          ENDDO
          ntot_xiarray(inum,nnum)=maxxiarray1
       ENDDO
       ifirst=1
    ENDIF
    IF(n.GT.MAXNLOOP_IREGI-1.OR.n.LT.1)THEN
       WRITE(*,100)"ERROR: n is out of range 1<=n<=",MAXNLOOP_IREGI-1," in all_integers"
       STOP
    ENDIF
    IF(i.GT.MAXRANK_IREGI.OR.i.LT.0)THEN
       WRITE(*,100)"ERROR: r is out of range 0<=r<=",MAXRANK_IREGI," in all_integers"
       STOP
    ENDIF
    IF(n.EQ.1.AND.ntot.NE.1)THEN
       WRITE(*,*)"ERROR:ntot should be 1 when n=1 in all_integers"
       STOP
    ENDIF
    IF(n.GE.2.AND.n.LE.MAXNLOOP_IREGI-1)THEN
       ! Make it work in MadLoop, otherwise it is wrong
       IF(ntot.NE.ntot_xiarray(i,n))THEN
          WRITE(*,*)"ERROR: ntot is not correct in all_integers"
          STOP
       ENDIF
    ENDIF
    j=(n-1)*(MAXRANK_IREGI+1)+i+1
    SELECT CASE(n)
       CASE(1)
          sol(1:ntot,1:n)=xiarray(j)%xiarray_i_n(1:1,1:n)
          factor(1)=1d0
       CASE DEFAULT
          sol(1:ntot,1:n)=xiarray(j)%xiarray_i_n(1:ntot,1:n)
          factor(1:ntot)=factor_xiarray(j)%factor_xiarray_i_n(1:ntot)
    END SELECT
    RETURN
100 FORMAT(2X,A31,I2,A16)
  END SUBROUTINE all_Integers

  SUBROUTINE calc_factorial_pair
    IMPLICIT NONE
    INTEGER::i,j
    factorial_pair(1:MAXINDICES_IREGI,0)=1d0
    DO i=1,MAXINDICES_IREGI
       DO j=1,MAXRANK_IREGI
          factorial_pair(i,j)=DBLE(factorial(i+j-1))&
               /DBLE(factorial(i-1))/DBLE(factorial(j))
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE calc_factorial_pair

  FUNCTION number_coefs_for_rank(rank) RESULT(res)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::rank
    INTEGER::res,i
    IF(rank.LT.0)res=0
    res=0
    DO i=0,rank
       res=res+((3+i)*(2+i)*(1+i))/6
    ENDDO
    RETURN
  END FUNCTION number_coefs_for_rank

  SUBROUTINE timestamp
    IMPLICIT NONE    
    CHARACTER(len=8)::ampm
    INTEGER::d
    INTEGER::h
    INTEGER::m,n
    INTEGER::mm
    CHARACTER( len = 9 ),PARAMETER,DIMENSION(12) :: month = (/ &
         'January  ', 'February ', 'March    ', 'April    ', &
         'May      ', 'June     ', 'July     ', 'August   ', &
         'September', 'October  ', 'November ', 'December ' /)
    INTEGER::dn
    INTEGER::s
    INTEGER,DIMENSION(8)::values
    INTEGER::y
    CALL date_and_time(values=values)
    y = values(1)
    m = values(2)
    d = values(3)
    h = values(5)
    n = values(6)
    s = values(7)
    mm = values(8)
    IF( h.LT.12 )THEN
       ampm = 'AM'
    ELSEIF( h.EQ.12 )THEN
       IF( n.EQ.0 .AND. s.EQ.0 )THEN
          ampm = 'Noon'
       ELSE
          ampm = 'PM'
       ENDIF
    ELSE
       h = h - 12
       IF( h.LT.12 )THEN
          ampm = 'PM'
       ELSEIF( h.EQ.12 )THEN
          IF(n.EQ.0.AND.s.EQ.0)THEN
             ampm = 'Midnight'
          ELSE
             ampm = 'AM'
          ENDIF
       ENDIF
    ENDIF
    WRITE( *, '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
         d, TRIM( month(m) ), y, h, ':', n, ':', s, '.', mm, TRIM( ampm )  
    RETURN
  END SUBROUTINE timestamp
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!                                                                               !
! FOLLOWING SUBROUTINES/FUNCTIONS ARE ONLY FOR THE REORDERING THE TIR COEFS     !
!                                                                               !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  SUBROUTINE SORT_IREGICOEFS(RANK,NLOOPCOEFS,OLDCOEFS,NEWCOEFS)
    !
    ! CONVERT THE OUTPUT OF IREGI TO THAT OF (NEW) MADLOOP 
    !
    ! THE NEW OUTPUT OF COEFS FROM MADLOOP IS
    ! RANK=0: (,)
    ! RANK=1: (0,),(1,),(2,),(3,)
    ! RANK=2: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)
    ! ...
    ! THE OLD OUTPUT OF COEFS FROM MADLOOP IS
    ! RANK=0: (,)
    ! RANK=1: (0,),(1,),(2,),(3,)
    ! RANK=2: (0,0),(0,1),(0,2),(0,3),(1,1),(2,1),(3,1),(2,2),(2,3),(3,3)
    ! ...
    !
    ! ARGUMENTS
    IMPLICIT NONE
    INTEGER,INTENT(IN)::RANK,NLOOPCOEFS
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPCOEFS-1,3),INTENT(IN)::OLDCOEFS
    COMPLEX(KIND(1d0)),DIMENSION(0:NLOOPCOEFS-1,3),INTENT(OUT)::NEWCOEFS
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I
    ! MAX RANK SET AS 10
    ! Sum((3+r)*(2+r)*(1+r)/6,{r,0,10})=1001
    INTEGER,PARAMETER::LOOPMAXCOEFS_IREGI=1001
    INTEGER,DIMENSION(0:LOOPMAXCOEFS_IREGI-1)::POS
    SAVE POS
    LOGICAL::INIT=.TRUE.
    SAVE INIT
    ! ----------
    ! BEGIN CODE
    ! ----------
            
    IF(INIT)THEN
       IF(NLOOPCOEFS.GT.LOOPMAXCOEFS_IREGI)&
            STOP "ERROR:LOOPMAXCOEFS_IREGI IS TOO SMALL!!!"
       INIT=.FALSE.
       ! ASSIGN THE POSITION OF POS FOR SWAP
       CALL ASSIGN_PJPOS(POS)
    ENDIF
    
    DO I=0,NLOOPCOEFS-1
    !   NEWCOEFS(I,1:3)=OLDCOEFS(POS(I),1:3)
       NEWCOEFS(POS(I),1:3)=OLDCOEFS(I,1:3)
    ENDDO
    
  END SUBROUTINE SORT_IREGICOEFS

  SUBROUTINE ASSIGN_PJPOS(POS)
    !
    ! ASSIGN THE POSITION OF POS FOR SWAP
    !
    !
    IMPLICIT NONE
    !
    ! CONSTANS
    !
    ! MAX RANK SET AS 10
    ! Sum((3+r)*(2+r)*(1+r)/6,{r,0,10})=1001 
    INTEGER,PARAMETER::LOOPMAXCOEFS_IREGI=1001,MAXRANK=10
    ! 
    ! ARGUMENTS
    ! 
    INTEGER,DIMENSION(0:LOOPMAXCOEFS_IREGI-1),INTENT(OUT)::POS
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I,J,K,SHIFT,DN
    INTEGER,DIMENSION(MAXRANK)::POSINDEX,PJPOSINDEX
    ! ----------
    ! BEGIN CODE
    ! ----------
    POS(0)=0
    DO I=1,4
       POS(I)=I
    ENDDO
    SHIFT=4
    DO J=2,MAXRANK
       DN=(J+3)*(J+2)*(J+1)/6
       POSINDEX(1:MAXRANK)=0
       PJPOSINDEX(1:MAXRANK)=0
       DO I=1,DN
          IF(I.GT.1)CALL NEXTINDEX(J,POSINDEX)
          CALL CONVERT_PJPOSINDEX(J,POSINDEX,PJPOSINDEX)
          K=DN-QPOLYPOS(J,PJPOSINDEX)+1+SHIFT
          !POS(K)=I+SHIFT
          POS(I+SHIFT)=K
       ENDDO
       SHIFT=SHIFT+DN
    ENDDO
    
  END SUBROUTINE ASSIGN_PJPOS

  SUBROUTINE NEXTINDEX(RANK,POSINDEX)
    !
    ! CALL FOR THE NEXT INDEX
    !
    IMPLICIT NONE
    !
    ! CONSTANTS
    !
    INTEGER,PARAMETER::MAXRANK=10
    !
    ! ARGUMENTS
    !
    INTEGER,INTENT(IN)::RANK
    INTEGER,DIMENSION(MAXRANK),INTENT(INOUT)::POSINDEX
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I
    ! ----------
    ! BEGIN CODE
    ! ----------
    DO I=1,RANK
       POSINDEX(I)=POSINDEX(I)+1
       IF(POSINDEX(I).GT.3)THEN
          POSINDEX(I)=0
          IF(I.EQ.RANK)THEN
             RETURN
          ENDIF
       ELSE
          IF(I.GT.1)THEN
             POSINDEX(1:I-1)=POSINDEX(I)
          ENDIF
          RETURN
       ENDIF
    ENDDO
    
  END SUBROUTINE NEXTINDEX

  SUBROUTINE CONVERT_PJPOSINDEX(RANK,POSINDEX,PJPOSINDEX)
    !
    ! CONVERT POSINDEX TO PJPOSINDEX
    !
    IMPLICIT NONE
    !
    ! CONSTANTS
    !
    INTEGER,PARAMETER::MAXRANK=10
    !
    ! ARGUMENTS
    !
    INTEGER,INTENT(IN)::RANK
    INTEGER,DIMENSION(MAXRANK),INTENT(IN)::POSINDEX
    INTEGER,DIMENSION(MAXRANK),INTENT(OUT)::PJPOSINDEX
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I
    ! ----------
    ! BEGIN CODE
    ! ----------
    DO I=1,RANK
       PJPOSINDEX(RANK+1-I)=3-POSINDEX(I)
    ENDDO
    RETURN
  END SUBROUTINE CONVERT_PJPOSINDEX

  FUNCTION QPOLYPOS(RANK,POSINDEX)
    !
    ! COMPUTATION THE RELATIVE POSITION OF INDEX WITH RANK
    ! IN THE *OLD* MADLOOP CONVENTION
    !
    IMPLICIT NONE
    !
    ! CONSTANTS
    !
    INTEGER,PARAMETER::MAXRANK=10
    !
    ! ARGUMENTS
    !
    INTEGER,INTENT(IN)::RANK
    INTEGER,DIMENSION(MAXRANK),INTENT(IN)::POSINDEX
    INTEGER::QPOLYPOS
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I,J,IMIN
    ! ----------
    ! BEGIN CODE
    ! ----------

    IF(RANK.EQ.0)THEN
       QPOLYPOS=1
       RETURN
    ENDIF
    
    IF(RANK.EQ.1)THEN
       QPOLYPOS=POSINDEX(1)+1
       RETURN
    ENDIF
    
    QPOLYPOS=POSINDEX(1)-POSINDEX(2)+1
    DO I=2,RANK
       IF(I.EQ.RANK)THEN
          IMIN=0
       ELSE
          IMIN=POSINDEX(I+1)
       ENDIF
       DO J=IMIN,POSINDEX(I)-1
          QPOLYPOS=QPOLYPOS+QPOLYNUMBER(J,I-1)
       ENDDO
    ENDDO
    RETURN
  END FUNCTION QPOLYPOS

  FUNCTION NEWQPOLYPOS(RANK,POSINDEX)
    !
    ! COMPUTATION THE RELATIVE POSITION OF INDEX WITH RANK
    ! IN THE *NEW* MADLOOP CONVENTION
    !
    IMPLICIT NONE
    !
    ! CONSTANTS
    !
    INTEGER,PARAMETER::MAXRANK=10
    !
    ! ARGUMENTS
    !
    INTEGER,INTENT(IN)::RANK
    INTEGER,DIMENSION(MAXRANK),INTENT(IN)::POSINDEX
    INTEGER::NEWQPOLYPOS
    !
    ! LOCAL VARIABLES
    !
    INTEGER::I,J,IMIN
    ! ----------
    ! BEGIN CODE
    ! ----------                                                                                                                                                                                            

    IF(RANK.EQ.0)THEN
       NEWQPOLYPOS=1
       RETURN
    ENDIF

    IF(RANK.EQ.1)THEN
       NEWQPOLYPOS=POSINDEX(1)+1
       RETURN
    ENDIF

    NEWQPOLYPOS=1
    DO I=1,RANK
       IF(POSINDEX(RANK-I+1).EQ.0)THEN
          IMIN=0
       ELSE
          ! Eq.(3.4.6) in Valentin's PhD Thesis
          IMIN=factorial(POSINDEX(RANK-I+1)+I-1)/factorial(I)&
               /factorial(POSINDEX(RANK-I+1)-1)
       ENDIF
       NEWQPOLYPOS=NEWQPOLYPOS+IMIN
    ENDDO
    RETURN
  END FUNCTION NEWQPOLYPOS

  FUNCTION QPOLYNUMBER(I,RANK)
    !
    ! THE INDEPENDENT NUMBER OF Q POLY WITH \MU=I,...,3 AND RANK
    !
    IMPLICIT NONE
    !
    ! CONSTANTS
    !
    !
    ! ARGUMENTS
    !
    INTEGER,INTENT(IN)::I,RANK
    INTEGER::QPOLYNUMBER
    !
    ! LOCAL VARIABLES
    !
    ! ----------
    ! BEGIN CODE
    ! ----------
    SELECT CASE(I)
    CASE(0)
       QPOLYNUMBER=(3+RANK)*(2+RANK)*(1+RANK)/6
    CASE(1)
       QPOLYNUMBER=(2+RANK)*(1+RANK)/2
    CASE(2)
       QPOLYNUMBER=(1+RANK)
    CASE(3)
       QPOLYNUMBER=1
    CASE DEFAULT
       STOP 'I must be >= 0 and <=3 in QPOLYNUMBER.'
    END SELECT
    RETURN
  END FUNCTION QPOLYNUMBER

  FUNCTION POS2RANK(ipos)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::ipos
    INTEGER::POS2RANK
    LOGICAL::INIT=.TRUE.
    ! NOW ITS MAX RANK IS 10
    INTEGER,PARAMETER::MAXRANK=10
    INTEGER,DIMENSION(0:MAXRANK)::IRANGE
    INTEGER::I
    SAVE IRANGE,INIT
    IF(INIT)THEN
       IRANGE(0)=0
       DO I=1,MAXRANK
          IRANGE(I)=IRANGE(I-1)+(I+3)*(I+2)*(I+1)/6
       ENDDO
       INIT=.FALSE.
    ENDIF
    IF(ipos.EQ.0)THEN
       POS2RANK=0
       RETURN
    ENDIF
    DO I=1,MAXRANK
       IF(ipos.LE.IRANGE(I))THEN
          POS2RANK=I
          RETURN
       ENDIF
    ENDDO
    WRITE(*,*)"ERROR in POS2RANK,ipos=",ipos
    STOP
  END FUNCTION POS2RANK

  FUNCTION RANK_NUM(MAXRANK)
    IMPLICIT NONE
    INTEGER::RANK_NUM
    INTEGER,INTENT(IN)::MAXRANK
    INTEGER::I
    RANK_NUM=0
    IF(MAXRANK.LT.0)RETURN
    DO I=0,MAXRANK
       RANK_NUM=RANK_NUM+(I+3)*(I+2)*(I+1)/6
    ENDDO
    RETURN
  END FUNCTION RANK_NUM

  SUBROUTINE SHIFT_MOM(MOM,NCOEFS,TICOEFS_SAVE,TICOEFS)
    ! SHIFT MOMENTUM MOM
    IMPLICIT NONE
    !
    ! CONSTANS
    !
    ! MAX RANK SET AS 10
    ! Sum((3+r)*(2+r)*(1+r)/6,{r,0,10})=1001 --> LOOPMAXCOEFS_IREGI
    ! MAX OF (i1+1)*(i2+1)*(i3+1)*(i4+1) with i1+i2+i3+i4<=MAXRANK = 144 --> MAXSPLIT
    INTEGER,PARAMETER::LOOPMAXCOEFS_IREGI=1001,MAXRANK=10,MAXSPLIT=144
    REAL(KIND(1d0)),DIMENSION(0:3),INTENT(IN)::MOM
    INTEGER,INTENT(IN)::NCOEFS
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(IN)::TICOEFS_SAVE
    COMPLEX(KIND(1d0)),DIMENSION(0:NCOEFS-1,1:4),INTENT(OUT)::TICOEFS
    LOGICAL::INIT=.TRUE.
    INTEGER,DIMENSION(0:LOOPMAXCOEFS_IREGI-1)::NTERM
    REAL(KIND(1d0)),DIMENSION(0:LOOPMAXCOEFS_IREGI-1,MAXSPLIT)::SPLIT_FACTOR
    ! i=0-3 -> number of lorentz indices =i
    ! i=4 -> the position of TICOEFS that should be multiplied
    ! e.g. (q+MOM)^mu1(q+MOM)^mu2...(q+MOM)^muk
    ! it can be splitted into q^mu1...q^muk+q^mu1...q^mui Mom^mui+1 ...q^muk ...
    ! for some specific term configurateion mu1=0,mu2=0,...muk=3
    ! NTERM stores the total number of terms
    ! SPLIT_FACTOR is the factor for each term
    ! SPLIT_INFO(J,K,0-3) is the number of 0,1,2,3 with MOM^mu
    ! SPLIT_INFO(J,K,4) is the corresponding q^mu (remaining lorentz indices)
    INTEGER,DIMENSION(0:LOOPMAXCOEFS_IREGI-1,MAXSPLIT,0:4)::SPLIT_INFO  
    SAVE INIT,NTERM,SPLIT_FACTOR,SPLIT_INFO
    INTEGER::I,J,K,L,NINDEX,r,incr
    INTEGER::i1,i2,i3,i4,nsplit,itit,rte,ND
    REAL(KIND(1d0))::temp
    INTEGER,DIMENSION(MAXRANK)::POSINDEX
    IF(INIT)THEN
       incr=0
       ND=0
       DO r=0,MAXRANK
          DO I=r,0,-1
             DO J=r-I,0,-1
                DO K=r-I-J,0,-1
                   L=r-I-J-K
                   ! exhaust all of the subsets
                   nsplit=0
                   NTERM(incr)=(I+1)*(J+1)*(K+1)*(L+1)
                   DO i1=0,I
                      !DO itit=1,i1
                      !   POSINDEX(itit)=0
                      !ENDDO
                      DO i2=0,J
                         !DO itit=i1+1,i1+i2
                         !   POSINDEX(itit)=1
                         !ENDDO
                         DO i3=0,K
                            !DO itit=i1+i2+1,i1+i2+i3
                            !   POSINDEX(itit)=2
                            !ENDDO
                            DO i4=0,L
                               !DO itit=i1+i2+i3+1,i1+i2+i3+i4
                               !   POSINDEX(itit)=3
                               !ENDDO
                               nsplit=nsplit+1
                               rte=i1+i2+i3+i4
                               DO itit=1,i1
                                  POSINDEX(rte-itit+1)=0
                               ENDDO
                               DO itit=i1+1,i1+i2
                                  POSINDEX(rte-itit+1)=1
                               ENDDO
                               DO itit=i1+i2+1,i1+i2+i3
                                  POSINDEX(rte-itit+1)=2
                               ENDDO
                               DO itit=i1+i2+i3+1,i1+i2+i3+i4
                                  POSINDEX(rte-itit+1)=3
                               ENDDO
                               SPLIT_INFO(incr,nsplit,4)=QPOLYPOS(rte,POSINDEX)+RANK_NUM(rte-1)-1
                               SPLIT_INFO(incr,nsplit,0)=I-i1
                               SPLIT_INFO(incr,nsplit,1)=J-i2
                               SPLIT_INFO(incr,nsplit,2)=K-i3
                               SPLIT_INFO(incr,nsplit,3)=L-i4
                               SPLIT_FACTOR(incr,nsplit)=DBLE(factorial(I))&
                                    /DBLE(factorial(i1)*factorial(I-i1))
                               SPLIT_FACTOR(incr,nsplit)=SPLIT_FACTOR(incr,nsplit)*&
                                    DBLE(factorial(J))/DBLE(factorial(i2)*factorial(J-i2))
                               SPLIT_FACTOR(incr,nsplit)=SPLIT_FACTOR(incr,nsplit)*&
                                    DBLE(factorial(K))/DBLE(factorial(i3)*factorial(K-i3))
                               SPLIT_FACTOR(incr,nsplit)=SPLIT_FACTOR(incr,nsplit)*&
                                    DBLE(factorial(L))/DBLE(factorial(i4)*factorial(L-i4))
                               IF(.NOT.ML5_CONVENTION)THEN
                                  SPLIT_FACTOR(incr,nsplit)=SPLIT_FACTOR(incr,nsplit)*&
                                       DBLE(factorial(r))/DBLE(factorial(I)*factorial(J))&
                                       /DBLE(factorial(K)*factorial(L))
                                  SPLIT_FACTOR(incr,nsplit)=SPLIT_FACTOR(incr,nsplit)/&
                                       DBLE(factorial(rte))*DBLE(factorial(i1))*DBLE(factorial(i2))*&
                                       DBLE(factorial(i3))*DBLE(factorial(i4))
                               ENDIF
                            ENDDO
                         ENDDO
                      ENDDO
                   ENDDO
                   incr=incr+1
                ENDDO
             ENDDO
          ENDDO
          ND=ND+(3+r)*(2+r)*(1+r)/6
       ENDDO
       INIT=.FALSE.
    ENDIF
    DO I=0,NCOEFS-1
       TICOEFS(I,1:4)=DCMPLX(0d0)
       DO J=1,NTERM(I)
          temp=SPLIT_FACTOR(I,J)
          DO K=0,3
             NINDEX=SPLIT_INFO(I,J,K)
             IF(NINDEX.GT.0)THEN
                temp=temp*MOM(K)**NINDEX
             ENDIF
          ENDDO
          TICOEFS(I,1:4)=TICOEFS(I,1:4)&
               +TICOEFS_SAVE(SPLIT_INFO(I,J,4),1:4)*temp
       ENDDO
    ENDDO
    RETURN
  END SUBROUTINE SHIFT_MOM

  FUNCTION xiarray_arg1(i,n)
    IMPLICIT NONE
    INTEGER,INTENT(IN)::i,n
    INTEGER::xiarray_arg1
    INTEGER::imax,imin,j
    !INTEGER(KIND=LINT)::itemp
    IF(i+n-1.GT.12)THEN
       imax=MAX(n-1,i)
       imin=MIN(n-1,i)
       xiarray_arg1=1
       DO j=imax+1,i+n-1
          xiarray_arg1=xiarray_arg1*j
       ENDDO
       IF(imin.LE.12)THEN
          xiarray_arg1=xiarray_arg1/factorial(imin)
       ELSE
          DO j=1,imin
             xiarray_arg1=xiarray_arg1/j
          ENDDO
       ENDIF
    ELSE
       xiarray_arg1=factorial(i+n-1)/factorial(n-1)/factorial(i)
    ENDIF
    RETURN
  END FUNCTION xiarray_arg1
END MODULE FUNLIB
