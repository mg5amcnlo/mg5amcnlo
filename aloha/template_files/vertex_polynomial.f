      MODULE VERTEX_POLYNOMIAL

          implicit none
      
          include 'coef_specs.inc'

          TYPE MP_V_COEF
              sequence
              COMPLEX*32 COEFS(0:VERTEXMAXCOEFS-1)
              LOGICAL    IS_ZERO(0:VERTEXMAXCOEFS-1)
              INTEGER    NON_ZERO_IDS(VERTEXMAXCOEFS)
              INTEGER    N_NON_ZERO_IDS
          END TYPE MP_V_COEF

          TYPE V_COEF
              sequence
              COMPLEX*16 COEFS(0:VERTEXMAXCOEFS-1)
              LOGICAL    IS_ZERO(0:VERTEXMAXCOEFS-1)
              INTEGER    NON_ZERO_IDS(VERTEXMAXCOEFS)
              INTEGER    N_NON_ZERO_IDS
          END TYPE V_COEF

          interface initialize_v_coefs
            module procedure dp_initialize_v_coefs,
     &          mp_initialize_v_coefs, dp_initialize_all_v_coefs, 
     &          mp_initialize_all_v_coefs
          end interface initialize_v_coefs

          interface filter_all_v_coefs
            module procedure dp_filter_all_v_coefs, 
     &                                      mp_filter_all_v_coefs
          end interface filter_all_v_coefs

          interface print_v_coefs
            module procedure dp_print_one_v_coefs, mp_print_one_v_coefs,
     &          dp_print_all_v_coefs, mp_print_all_v_coefs
          end interface print_v_coefs

      CONTAINS

          SUBROUTINE dp_initialize_v_coefs(this)
            TYPE(V_COEF), INTENT(inout) :: this
            this%COEFS(:)        = DCMPLX(0d0,0d0)
            this%IS_ZERO(:)      = .TRUE.
            this%NON_ZERO_IDS(:) = -1
            this%N_NON_ZERO_IDS  = 0
          END SUBROUTINE dp_initialize_v_coefs

          SUBROUTINE mp_initialize_v_coefs(this)
            TYPE(MP_V_COEF), INTENT(inout) :: this
            this%COEFS(:)        = CMPLX(0e0_16,0e0_16,KIND=16)
            this%IS_ZERO(:)      = .TRUE.
            this%NON_ZERO_IDS(:) = -1
            this%N_NON_ZERO_IDS  = 0
          END SUBROUTINE mp_initialize_v_coefs

          SUBROUTINE dp_initialize_all_v_coefs(this)
            TYPE(V_COEF), INTENT(inout) :: this(MAXLWFSIZE,MAXLWFSIZE)
            INTEGER I,J            
            DO I=1,MAXLWFSIZE
              DO J=1,MAXLWFSIZE
                CALL dp_initialize_v_coefs(this(I,J))
              ENDDO
            ENDDO
          END SUBROUTINE dp_initialize_all_v_coefs

          SUBROUTINE mp_initialize_all_v_coefs(this)
            TYPE(MP_V_COEF), INTENT(inout) :: 
     &                                    this(MAXLWFSIZE,MAXLWFSIZE)
            INTEGER I,J
            DO I=1,MAXLWFSIZE
              DO J=1,MAXLWFSIZE
                CALL mp_initialize_v_coefs(this(I,J))
              ENDDO
            ENDDO
          END SUBROUTINE mp_initialize_all_v_coefs


          SUBROUTINE dp_filter_all_v_coefs(raw_coefs,filtered_coefs)
            COMPLEX*16     ,INTENT(in)  :: 
     &               raw_coefs(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
            TYPE(V_COEF)   ,INTENT(out) :: 
     &                             filtered_coefs(MAXLWFSIZE,MAXLWFSIZE)
            INTEGER                     :: i,j,k
            do j=1,MAXLWFSIZE
              do k=1,MAXLWFSIZE
                call initialize_v_coefs(filtered_coefs(j,k))
                DO I=0,VERTEXMAXCOEFS-1
                  if ((raw_coefs(j,I,k).ne.DCMPLX(0d0,0d0))
     &                         .and.filtered_coefs(j,k)%is_zero(I)) then
                    filtered_coefs(j,k)%is_zero(I) = .False.
                    filtered_coefs(j,k)%coefs(I)   = raw_coefs(j,I,k)
                    filtered_coefs(j,k)%N_NON_ZERO_IDS = 
     &                              filtered_coefs(j,k)%N_NON_ZERO_IDS+1
                    filtered_coefs(j,k)%NON_ZERO_IDS(
     &                           filtered_coefs(j,k)%N_NON_ZERO_IDS) = I
                  endif
                ENDDO
              ENDDO
            ENDDO
          END SUBROUTINE dp_filter_all_v_coefs

          SUBROUTINE mp_filter_all_v_coefs(raw_coefs, filtered_coefs)
            COMPLEX*32     ,INTENT(in)  :: 
     &               raw_coefs(MAXLWFSIZE,0:VERTEXMAXCOEFS-1,MAXLWFSIZE)
            TYPE(MP_V_COEF),INTENT(out) :: 
     &                             filtered_coefs(MAXLWFSIZE,MAXLWFSIZE)
            INTEGER                     :: i,j,k
            do j=1,MAXLWFSIZE
              do k=1,MAXLWFSIZE
                call initialize_v_coefs(filtered_coefs(j,k))
                DO I=0,VERTEXMAXCOEFS-1
                  if ((raw_coefs(j,I,k).ne.CMPLX(0e0_16,0e0_16,KIND=16))
     &                         .and.filtered_coefs(j,k)%is_zero(I)) then
                    filtered_coefs(j,k)%is_zero(I) = .False.
                    filtered_coefs(j,k)%coefs(I)   = raw_coefs(j,I,k)
                    filtered_coefs(j,k)%N_NON_ZERO_IDS = 
     &                              filtered_coefs(j,k)%N_NON_ZERO_IDS+1
                    filtered_coefs(j,k)%NON_ZERO_IDS(
     &                           filtered_coefs(j,k)%N_NON_ZERO_IDS) = I
                  endif
                ENDDO
              ENDDO
            ENDDO
          END SUBROUTINE mp_filter_all_v_coefs

          SUBROUTINE FROM_DP_TO_QP_V_COEFS(DP_COEFS, QP_COEFS)
              TYPE(V_COEF),INTENT(in)    :: DP_COEFS
              TYPE(MP_V_COEF),INTENT(out) :: QP_COEFS
              INTEGER I
              QP_COEFS%N_NON_ZERO_IDS    = DP_COEFS%N_NON_ZERO_IDS
              DO I=1,VERTEXMAXCOEFS
                QP_COEFS%IS_ZERO(I)      = DP_COEFS%IS_ZERO(I)
                QP_COEFS%NON_ZERO_IDS(I) = DP_COEFS%NON_ZERO_IDS(I)
                QP_COEFS%COEFS(I)        = 
     &                                  CMPLX(DP_COEFS%COEFS(I),KIND=16)
              ENDDO
          END SUBROUTINE FROM_DP_TO_QP_V_COEFS

          SUBROUTINE FROM_QP_TO_DP_V_COEFS(QP_COEFS, DP_COEFS)
              TYPE(MP_V_COEF),INTENT(in)  :: QP_COEFS
              TYPE(V_COEF),INTENT(out)    :: DP_COEFS
              INTEGER I              
              DP_COEFS%N_NON_ZERO_IDS    = QP_COEFS%N_NON_ZERO_IDS
              DO I=1,VERTEXMAXCOEFS
                DP_COEFS%IS_ZERO(I)      = QP_COEFS%IS_ZERO(I)
                DP_COEFS%NON_ZERO_IDS(I) = QP_COEFS%NON_ZERO_IDS(I)
                DP_COEFS%COEFS(I)        = DCMPLX(QP_COEFS%COEFS(I))
              ENDDO
          END SUBROUTINE FROM_QP_TO_DP_V_COEFS

          SUBROUTINE DP_PRINT_ONE_V_COEFS(COEFS)
              TYPE(V_COEF),INTENT(in)    :: COEFS
              INTEGER I
              WRITE(*,*) 'n_non_zero_IDs =',COEFS%N_NON_ZERO_IDS
              DO I=1,COEFS%N_NON_ZERO_IDS
                WRITE(*,*) 'Vertex COEFS(',COEFS%NON_ZERO_IDS(I),')=',
     &                                COEFS%COEFS(COEFS%NON_ZERO_IDS(I))
              ENDDO
          END SUBROUTINE DP_PRINT_ONE_V_COEFS

          SUBROUTINE MP_PRINT_ONE_V_COEFS(QP_COEFS)
              TYPE(MP_V_COEF),INTENT(in) :: QP_COEFS
              TYPE(V_COEF)               :: DP_COEFS
              CALL FROM_QP_TO_DP_V_COEFS(QP_COEFS,DP_COEFS)
              CALL DP_PRINT_ONE_V_COEFS(DP_COEFS)
          END SUBROUTINE MP_PRINT_ONE_V_COEFS

          SUBROUTINE DP_PRINT_ALL_V_COEFS(COEFS)
              TYPE(V_COEF),INTENT(in)    :: COEFS(MAXLWFSIZE,MAXLWFSIZE)
              INTEGER I,J,K
              LOGICAL FOUNDSOME
              FOUNDSOME = .FALSE.
              DO J=1,MAXLWFSIZE
                DO K=1,MAXLWFSIZE
                  IF (COEFS(J,K)%N_NON_ZERO_IDS.NE.0) THEN
                    FOUNDSOME = .TRUE.
                    WRITE(*,*) 'For (',J,',',K,'), n_non_zero_IDs =',
     &                                         COEFS(J,K)%N_NON_ZERO_IDS
                    DO I=1,COEFS(J,K)%N_NON_ZERO_IDS
                      WRITE(*,*) ' >> Vertex COEFS(',
     &                   COEFS(J,K)%NON_ZERO_IDS(I),')=',
     &                      COEFS(J,K)%COEFS(COEFS(J,K)%NON_ZERO_IDS(I))
                    ENDDO
                  ENDIF
                ENDDO
              ENDDO
              IF(.NOT.FOUNDSOME) THEN
                WRITE(*,*) 'Empty list of vertex coefs.'
              ENDIF 
          END SUBROUTINE DP_PRINT_ALL_V_COEFS

          SUBROUTINE MP_PRINT_ALL_V_COEFS(COEFS)
              TYPE(MP_V_COEF),INTENT(in) :: COEFS(MAXLWFSIZE,MAXLWFSIZE)
              DOUBLE COMPLEX TMP
              INTEGER I,J,K
              LOGICAL FOUNDSOME
              FOUNDSOME = .FALSE.
              DO J=1,MAXLWFSIZE
                DO K=1,MAXLWFSIZE
                  IF (COEFS(J,K)%N_NON_ZERO_IDS.NE.0) THEN
                    FOUNDSOME = .TRUE.
                    WRITE(*,*) 'For (',J,',',K,'), n_non_zero_IDs =',
     &                                         COEFS(J,K)%N_NON_ZERO_IDS
                    DO I=1,COEFS(J,K)%N_NON_ZERO_IDS
C                     Cast in double precision to avoid problems with the
C                     known gfortran bug on quadprec writeout
                      TMP = DCMPLX(COEFS(J,K)%COEFS(
     &                                      COEFS(J,K)%NON_ZERO_IDS(I)))
                      WRITE(*,*) ' >> Vertex COEFS(',
     &                               COEFS(J,K)%NON_ZERO_IDS(I),')=',TMP
                    ENDDO
                  ENDIF
                ENDDO
              ENDDO
              IF(.NOT.FOUNDSOME) THEN
                WRITE(*,*) 'Empty list of vertex coefs.'
              ENDIF 
          END SUBROUTINE MP_PRINT_ALL_V_COEFS

      END MODULE VERTEX_POLYNOMIAL
