MODULE binary_tree
  USE global
  IMPLICIT NONE
CONTAINS
  SUBROUTINE ibppave_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( ibppave_node ),POINTER::item1
    !TYPE ( ibppave_node ), POINTER::item2
    INTEGER::i,icomp
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       !.OR.head%ITERATION.NE.CURRENT_PS)THEN
       !CALL free_ibppave_bt(head) ! free the memory of the binary tree
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=ibppave_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             !.OR.item1%left%ITERATION&.NE.CURRENT_PS)THEN
             !CALL free_ibppave_bt(item1%left)
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             ! .OR.item1%right%ITERATION.NE.CURRENT_PS)THEN
             !CALL free_ibppave_bt(item1%right)
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          item%value(1:4)=item1%value(1:4)
          item%stable=item1%stable
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ibppave_bt_search

  SUBROUTINE cibppave_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( cibppave_node ),POINTER::item1
    INTEGER::i,icomp
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=cibppave_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          item%value(1:4)=item1%value(1:4)
          item%stable=item1%stable
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE cibppave_bt_search

  SUBROUTINE xyzmatrices_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(xyzmatrices_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( xyzmatrices_node ),POINTER::item1
    INTEGER::i,icomp,nn
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=xyzmatrices_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          nn=item%NLOOPLINE
          !PRINT *, item1%xmatrix(1,1) ! Debug
          item%xmatrix(1:nn,1:nn)=item1%xmatrix(1:nn,1:nn)
          item%ymatrix(1:nn,1:nn)=item1%ymatrix(1:nn,1:nn)
          item%zmatrix(2:nn,2:nn)=item1%zmatrix(2:nn,2:nn)
          item%detY=item1%detY
          item%detZ=item1%detZ
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE xyzmatrices_bt_search

  SUBROUTINE cxyzmatrices_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(cxyzmatrices_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( cxyzmatrices_node ),POINTER::item1
    INTEGER::i,icomp,nn
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=cxyzmatrices_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          nn=item%NLOOPLINE
          item%xmatrix(1:nn,1:nn)=item1%xmatrix(1:nn,1:nn)
          item%ymatrix(1:nn,1:nn)=item1%ymatrix(1:nn,1:nn)
          item%zmatrix(2:nn,2:nn)=item1%zmatrix(2:nn,2:nn)
          item%detY=item1%detY
          item%detZ=item1%detZ
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE cxyzmatrices_bt_search

  SUBROUTINE rsmatrices_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(rsmatrices_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( rsmatrices_node ),POINTER::item1
    INTEGER::i,icomp,nn
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=rsmatrices_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          nn=item%NLOOPLINE
          item%rmatrix(1:nn,1:nn)=item1%rmatrix(1:nn,1:nn)
          item%smatrix(0:nn,0:nn)=item1%smatrix(0:nn,0:nn)
          item%detS=item1%detS
          item%detR=item1%detR
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE rsmatrices_bt_search

  SUBROUTINE crsmatrices_bt_search (item,head,find)
    IMPLICIT NONE
    TYPE(crsmatrices_node),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( crsmatrices_node ),POINTER::item1
    INTEGER::i,icomp,nn
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=crsmatrices_node_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          nn=item%NLOOPLINE
          item%rmatrix(1:nn,1:nn)=item1%rmatrix(1:nn,1:nn)
          item%smatrix(0:nn,0:nn)=item1%smatrix(0:nn,0:nn)
          item%detS=item1%detS
          item%detR=item1%detR
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE crsmatrices_bt_search

  SUBROUTINE ibppave_bt_opt_search (head,NLOOPLINE,&
       indices,PCL,M2L,item,find)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(INOUT)::head
    TYPE(ibppave_node),POINTER,INTENT(OUT)::item
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::indices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    LOGICAL,INTENT(OUT)::find
    TYPE ( ibppave_node ),POINTER::item1
    !TYPE ( ibppave_node ), POINTER::item2
    INTEGER::i,icomp
    find=.FALSE.
    IF(.NOT.ASSOCIATED(head))THEN
       CALL ibppave_node_alloc(NLOOPLINE,indices,PCL,M2L,item)
       head => item
       IF(.NOT.ASSOCIATED(ibppave_save_array(NLOOPLINE)%ptr))THEN
          ibppave_save_array(NLOOPLINE)%ptr => item
       ENDIF
       RETURN
    ENDIF
    IF(.NOT.ASSOCIATED(ibppave_save_array(NLOOPLINE)%ptr))THEN
       item1 => head
    ELSE
       item1 => ibppave_save_array(NLOOPLINE)%ptr
    ENDIF
    DO
       icomp=ibppave_node_opt_compare(NLOOPLINE,indices,PCL,M2L,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             CALL ibppave_node_alloc(NLOOPLINE,indices,PCL,M2L,item)
             item1%left => item
             item%parent => item1
             IF(.NOT.ASSOCIATED(ibppave_save_array(NLOOPLINE)%ptr))THEN
                ibppave_save_array(NLOOPLINE)%ptr => item
             ENDIF
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             CALL ibppave_node_alloc(NLOOPLINE,indices,PCL,M2L,item)
             item1%right => item
             item%parent => item1
             IF(.NOT.ASSOCIATED(ibppave_save_array(NLOOPLINE)%ptr))THEN
                ibppave_save_array(NLOOPLINE)%ptr => item
             ENDIF
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          item => item1
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ibppave_bt_opt_search

  SUBROUTINE ibppave_node_alloc(NLOOPLINE,indices,PCL,M2L,item)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(OUT)::item
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::indices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    ALLOCATE(item)
    !item%ITERATION=CURRENT_PS
    item%NLOOPLINE=NLOOPLINE
    item%stable=.TRUE.
    item%indices(0:NLOOPLINE)=indices(0:NLOOPLINE)
    item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
    item%PCL(1:NLOOPLINE,0:3)=PCL(1:NLOOPLINE,0:3)
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    RETURN
  END SUBROUTINE ibppave_node_alloc

  SUBROUTINE cibppave_node_alloc(NLOOPLINE,indices,PCL,M2L,item)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(OUT)::item
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::indices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    COMPLEX(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    ALLOCATE(item)
    item%NLOOPLINE=NLOOPLINE
    item%stable=.TRUE.
    item%indices(0:NLOOPLINE)=indices(0:NLOOPLINE)
    item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
    item%PCL(1:NLOOPLINE,0:3)=PCL(1:NLOOPLINE,0:3)
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    RETURN
  END SUBROUTINE cibppave_node_alloc

  SUBROUTINE ibppave_node2_alloc(NLOOPLINE,indices,PijMatrix,M2L,item)
    IMPLICIT NONE
    TYPE(ibppave_node2),POINTER,INTENT(OUT)::item
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::indices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,NLOOPLINE),INTENT(IN)::PijMatrix
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    ALLOCATE(item)
    !item%ITERATION=CURRENT_PS
    item%NLOOPLINE=NLOOPLINE
    item%stable=.TRUE.
    item%indices(0:NLOOPLINE)=indices(0:NLOOPLINE)
    item%M2L(1:NLOOPLINE)=M2L(1:NLOOPLINE)
    item%PijMatrix(1:NLOOPLINE,1:NLOOPLINE)=&
         PijMatrix(1:NLOOPLINE,1:NLOOPLINE)
    RETURN
  END SUBROUTINE ibppave_node2_alloc

 SUBROUTINE ibppave_bt_search2(item,head,find)
    IMPLICIT NONE
    TYPE(ibppave_node2),POINTER,INTENT(INOUT)::head,item
    LOGICAL,INTENT(OUT)::find
    TYPE ( ibppave_node2 ),POINTER::item1
    !TYPE ( ibppave_node2 ), POINTER::item2
    INTEGER::i,icomp
    find=.FALSE.
    NULLIFY(item%parent)
    NULLIFY(item%left)
    NULLIFY(item%right)
    IF(.NOT.ASSOCIATED(head))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       icomp=ibppave_node2_compare(item,item1)
       IF(icomp.LT.0)THEN
          IF(.NOT.ASSOCIATED(item1%left))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSEIF(icomp.GT.0)THEN
          IF(.NOT.ASSOCIATED(item1%right))THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ELSE
          find=.TRUE.
          item%value(1:4)=item1%value(1:4)
          item%stable=item1%stable
          EXIT
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ibppave_bt_search2

  SUBROUTINE ibppave_bt_insert ( item, head )
    IMPLICIT NONE
    TYPE ( ibppave_node ),POINTER,INTENT(INOUT)::head
    INTEGER::i
    TYPE ( ibppave_node ),POINTER,INTENT(IN)::item
    TYPE ( ibppave_node ),POINTER::item1
    TYPE ( ibppave_node ), POINTER::item2

    NULLIFY( item%parent )
    NULLIFY( item%left )
    NULLIFY( item%right )
!
!  In the case of an empty tree.  
!
    IF(.NOT.ASSOCIATED( head ))THEN
       head => item
       RETURN
    ENDIF
    item1 => head
    DO
       IF(ibppave_node_smaller(item,item1))THEN
          IF(.NOT.ASSOCIATED(item1%left ))THEN
             item1%left => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%left
          ENDIF
       ELSE
          IF(.NOT.ASSOCIATED ( item1%right ) )THEN
             item1%right => item
             item%parent => item1
             EXIT
          ELSE
             item1 => item1%right
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE ibppave_bt_insert

  RECURSIVE SUBROUTINE ibppave_bt_print ( head )
    IMPLICIT NONE
    TYPE ( ibppave_node ),POINTER,INTENT(IN)::head
    INTEGER::n,i
    IF( ASSOCIATED( head ) )THEN
       CALL ibppave_bt_print( head%left )
       WRITE(*,*)"======================================================================================"
       n=head%NLOOPLINE
       WRITE(*,*)n,head%indices(0:n)
       WRITE(*,*)head%M2L(1:n)
       DO i=1,n
          WRITE(*,*)head%PCL(i,0:3)
       ENDDO
       WRITE(*,*)"value=",head%value(1:4)
       WRITE(*,*)"stable=",head%stable
       WRITE(*,*)"======================================================================================"
       CALL ibppave_bt_print( head%right )
    ENDIF
    RETURN
  END SUBROUTINE ibppave_bt_print

  RECURSIVE SUBROUTINE cibppave_bt_print ( head )
    IMPLICIT NONE
    TYPE ( cibppave_node ),POINTER,INTENT(IN)::head
    INTEGER::n,i
    IF( ASSOCIATED( head ) )THEN
       CALL cibppave_bt_print( head%left )
       WRITE(*,*)"======================================================================================"
       n=head%NLOOPLINE
       WRITE(*,*)n,head%indices(0:n)
       WRITE(*,*)head%M2L(1:n)
       DO i=1,n
          WRITE(*,*)head%PCL(i,0:3)
       ENDDO
       WRITE(*,*)"value=",head%value(1:4)
       WRITE(*,*)"stable=",head%stable
       WRITE(*,*)"======================================================================================"
       CALL cibppave_bt_print( head%right )
    ENDIF
    RETURN
  END SUBROUTINE cibppave_bt_print

  RECURSIVE FUNCTION ibppave_node_count (head) RESULT(res)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(IN)::head
    INTEGER::res
    IF( .NOT.ASSOCIATED(head))THEN
       res=0
       RETURN
    ENDIF
    res=1+ibppave_node_count(head%left)&
         +ibppave_node_count(head%right)
    RETURN
  END FUNCTION ibppave_node_count

  RECURSIVE FUNCTION cibppave_node_count (head) RESULT(res)
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(IN)::head
    INTEGER::res
    IF( .NOT.ASSOCIATED(head))THEN
       res=0
       RETURN
    ENDIF
    res=1+cibppave_node_count(head%left)&
         +cibppave_node_count(head%right)
    RETURN
  END FUNCTION cibppave_node_count

  RECURSIVE SUBROUTINE ibppave_bt_print2 ( head )
    IMPLICIT NONE
    TYPE ( ibppave_node2 ),POINTER,INTENT(IN)::head
    INTEGER::n,i
    IF( ASSOCIATED( head ) )THEN
       CALL ibppave_bt_print2( head%left )
       WRITE(*,*)"======================================================================================"
       n=head%NLOOPLINE
       WRITE(*,*)n,head%indices(0:n)
       WRITE(*,*)head%M2L(1:n)
       DO i=1,n          
          WRITE(*,*)head%PijMatrix(i,1:n)
       ENDDO
       WRITE(*,*)"value=",head%value(1:4)
       WRITE(*,*)"stable=",head%stable
       WRITE(*,*)"======================================================================================"
       CALL ibppave_bt_print2( head%right )
    ENDIF
    RETURN
  END SUBROUTINE ibppave_bt_print2

  FUNCTION ibppave_node_smaller(item1,item2) RESULT(res)
    ! .LE.
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(IN)::item1,item2
    LOGICAL::res
    INTEGER::icomp
    icomp=ibppave_node_compare(item1,item2)
    IF(icomp.LE.0)THEN
       res=.TRUE.
       RETURN
    ELSE
       res=.FALSE.
       RETURN
    ENDIF
  END FUNCTION ibppave_node_smaller

  FUNCTION cibppave_node_smaller(item1,item2) RESULT(res)
    ! .LE.
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(IN)::item1,item2
    LOGICAL::res
    INTEGER::icomp
    icomp=cibppave_node_compare(item1,item2)
    IF(icomp.LE.0)THEN
       res=.TRUE.
       RETURN
    ELSE
       res=.FALSE.
       RETURN
    ENDIF
  END FUNCTION cibppave_node_smaller

  FUNCTION ibppave_node_opt_compare(NLOOPLINE,indices,PCL,M2L,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(IN)::item2
    INTEGER,INTENT(IN)::NLOOPLINE
    INTEGER,DIMENSION(0:NLOOPLINE),INTENT(IN)::indices
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE,0:3),INTENT(IN)::PCL
    REAL(KIND(1d0)),DIMENSION(NLOOPLINE),INTENT(IN)::M2L
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    DO i=0,NLOOPLINE
       k=indices(i)-item2%indices(i)
       IF(k.LT.0)THEN
          res=-1
          RETURN
       ELSEIF(k.GT.0)THEN
          res=1
          RETURN
       ENDIF
    ENDDO
    DO i=1,NLOOPLINE
       icomp=real_compare(M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,NLOOPLINE
       DO k=0,3
          icomp=real_compare(PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION ibppave_node_opt_compare

  FUNCTION ibppave_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=0,n
       k=item1%indices(i)-item2%indices(i)
       IF(k.LT.0)THEN
          res=-1
          RETURN
       ELSEIF(k.GT.0)THEN
          res=1
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       icomp=real_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION ibppave_node_compare

  FUNCTION cibppave_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=0,n
       k=item1%indices(i)-item2%indices(i)
       IF(k.LT.0)THEN
          res=-1
          RETURN
       ELSEIF(k.GT.0)THEN
          res=1
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       icomp=complex_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION cibppave_node_compare

  FUNCTION xyzmatrices_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(xyzmatrices_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=1,n
       icomp=real_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION xyzmatrices_node_compare

  FUNCTION cxyzmatrices_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(cxyzmatrices_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=1,n
       icomp=complex_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION cxyzmatrices_node_compare

  FUNCTION rsmatrices_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(rsmatrices_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=1,n
       icomp=real_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION rsmatrices_node_compare

  FUNCTION crsmatrices_node_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(crsmatrices_node),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=1,n
       icomp=complex_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=0,3
          icomp=real_compare(item1%PCL(i,k),item2%PCL(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION crsmatrices_node_compare

  FUNCTION ibppave_node2_compare(item1,item2) RESULT(res)
    IMPLICIT NONE
    TYPE(ibppave_node2),POINTER,INTENT(IN)::item1,item2
    INTEGER::res,n,i,k
    REAL(KIND(1d0)),PARAMETER::eps1=1d-6,zthr=1d-4
    REAL(KIND(1d0))::temp
    INTEGER::icomp
    n=item1%NLOOPLINE-item2%NLOOPLINE
    IF(n.LT.0)THEN
       res=-1
       RETURN
    ELSEIF(n.GT.0)THEN
       res=1
       RETURN
    ENDIF
    n=item1%NLOOPLINE
    DO i=0,n
       k=item1%indices(i)-item2%indices(i)
       IF(k.LT.0)THEN
          res=-1
          RETURN
       ELSEIF(k.GT.0)THEN
          res=1
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       icomp=real_compare(item1%M2L(i),item2%M2L(i),eps,zthr)
       IF(icomp.NE.0)THEN
          res=icomp
          RETURN
       ENDIF
    ENDDO
    DO i=1,n
       DO k=i,n
          icomp=real_compare(item1%PijMatrix(i,k),item2%PijMatrix(i,k),eps,zthr)
          IF(icomp.NE.0)THEN
             res=icomp
             RETURN
          ENDIF
       ENDDO
    ENDDO
    res=0
    RETURN
  END FUNCTION ibppave_node2_compare

  FUNCTION real_compare(r1,r2,eps,zthr) RESULT(res)
    IMPLICIT NONE
    REAL(KIND(1d0)),INTENT(IN)::r1,r2,eps,zthr
    REAL(KIND(1d0))::maxr,diff
    INTEGER::res
    maxr=MAX(ABS(r1),ABS(r2))
    IF(maxr.LT.zthr)THEN
       res=0
       RETURN
    ENDIF
    diff=r1-r2
    IF(ABS(diff)/maxr.LT.eps)THEN
       res=0
       RETURN
    ENDIF
    IF(diff.GT.0d0)THEN
       res=1
       RETURN
    ELSE
       res=-1
       RETURN
    ENDIF
  END FUNCTION real_compare

  FUNCTION complex_compare(c1,c2,eps,zthr) RESULT(res)
    IMPLICIT NONE
    COMPLEX(KIND(1d0)),INTENT(IN)::c1,c2
    REAL(KIND(1d0)),INTENT(IN)::eps,zthr
    REAL(KIND(1d0))::r1,r2
    REAL(KIND(1d0))::maxr,diff
    INTEGER::res
    r1=DREAL(c1)
    r2=DREAL(c2)
    res=real_compare(r1,r2,eps,zthr)
    IF(res.NE.0)RETURN
    r1=DIMAG(c1)
    r2=DIMAG(c2)
    res=real_compare(r1,r2,eps,zthr)
    RETURN
  END FUNCTION complex_compare

  RECURSIVE SUBROUTINE free_ibppave_bt(head)
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_ibppave_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_ibppave_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_ibppave_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_ibppave_bt

  RECURSIVE SUBROUTINE free_cibppave_bt(head)
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_cibppave_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_cibppave_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_cibppave_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_cibppave_bt

  RECURSIVE SUBROUTINE free_xyzmatrices_bt(head)
    IMPLICIT NONE
    TYPE(xyzmatrices_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_xyzmatrices_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_xyzmatrices_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_xyzmatrices_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_xyzmatrices_bt

  RECURSIVE SUBROUTINE free_cxyzmatrices_bt(head)
    IMPLICIT NONE
    TYPE(cxyzmatrices_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_cxyzmatrices_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_cxyzmatrices_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_cxyzmatrices_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_cxyzmatrices_bt

  RECURSIVE SUBROUTINE free_rsmatrices_bt(head)
    IMPLICIT NONE
    TYPE(rsmatrices_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_rsmatrices_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_rsmatrices_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_rsmatrices_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_rsmatrices_bt

  RECURSIVE SUBROUTINE free_crsmatrices_bt(head)
    IMPLICIT NONE
    TYPE(crsmatrices_node),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_crsmatrices_node(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf
    IF(ASSOCIATED(head%left))THEN
       CALL free_crsmatrices_bt(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_crsmatrices_bt(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_crsmatrices_bt

  RECURSIVE SUBROUTINE free_ibppave_bt2(head)
    IMPLICIT NONE
    TYPE(ibppave_node2),POINTER,INTENT(INOUT)::head
    INTEGER::leaf
    leaf=leaf_ibppave_node2(head)
    IF(leaf.EQ.0)RETURN
    IF(leaf.EQ.1)THEN
       ! it is a leaf 
       DEALLOCATE(head)
       RETURN
    ENDIF
    ! it is not a leaf 
    IF(ASSOCIATED(head%left))THEN
       CALL free_ibppave_bt2(head%left)
    ENDIF
    IF(ASSOCIATED(head%right))THEN
       CALL free_ibppave_bt2(head%right)
    ENDIF
    DEALLOCATE(head)
    RETURN
  END SUBROUTINE free_ibppave_bt2

  FUNCTION leaf_ibppave_node(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(ibppave_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_ibppave_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_ibppave_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_ibppave_node=1
    ELSE
       leaf_ibppave_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_ibppave_node

  FUNCTION leaf_cibppave_node(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(cibppave_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_cibppave_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_cibppave_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_cibppave_node=1
    ELSE
       leaf_cibppave_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_cibppave_node

  FUNCTION leaf_xyzmatrices_node(head)
    ! 0: it is not associated 
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(xyzmatrices_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_xyzmatrices_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_xyzmatrices_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_xyzmatrices_node=1
    ELSE
       leaf_xyzmatrices_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_xyzmatrices_node

  FUNCTION leaf_cxyzmatrices_node(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(cxyzmatrices_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_cxyzmatrices_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_cxyzmatrices_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_cxyzmatrices_node=1
    ELSE
       leaf_cxyzmatrices_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_cxyzmatrices_node

  FUNCTION leaf_rsmatrices_node(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(rsmatrices_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_rsmatrices_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_rsmatrices_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_rsmatrices_node=1
    ELSE
       leaf_rsmatrices_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_rsmatrices_node

  FUNCTION leaf_crsmatrices_node(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(crsmatrices_node),POINTER,INTENT(IN)::head
    INTEGER::leaf_crsmatrices_node
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_crsmatrices_node=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_crsmatrices_node=1
    ELSE
       leaf_crsmatrices_node=-1
    ENDIF
    RETURN
  END FUNCTION leaf_crsmatrices_node

  FUNCTION leaf_ibppave_node2(head)
    ! 0: it is not associated
    ! 1: it is associated and a leaf
    ! -1: it is associated and not a leaf
    IMPLICIT NONE
    TYPE(ibppave_node2),POINTER,INTENT(IN)::head
    INTEGER::leaf_ibppave_node2
    IF(.NOT.ASSOCIATED(head))THEN
       leaf_ibppave_node2=0
    ELSEIF(.NOT.ASSOCIATED(head%left).AND.&
         .NOT.ASSOCIATED(head%right))THEN
       leaf_ibppave_node2=1
    ELSE
       leaf_ibppave_node2=-1
    ENDIF
    RETURN
  END FUNCTION leaf_ibppave_node2

  SUBROUTINE free_ibppave_save
    IMPLICIT NONE
    CALL free_ibppave_bt(ibp_save)
    CALL free_ibppave_bt(pave_save)
    CALL free_ibppave_bt(shiftpaveden_save)
    CALL free_cibppave_bt(cibp_save)
    CALL free_cibppave_bt(cpave_save)
    CALL free_cibppave_bt(cshiftpaveden_save)
  END SUBROUTINE free_ibppave_save

  SUBROUTINE free_ibppave_save2
    IMPLICIT NONE
    CALL free_ibppave_bt2(ibp_save2)
    CALL free_ibppave_bt2(pave_save2)
    CALL free_ibppave_bt2(shiftpaveden_save2)
  END SUBROUTINE free_ibppave_save2
END MODULE binary_tree
