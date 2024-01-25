!ZW module to handle dynamic allocation of all the vectorisation
module vectorize
   implicit none
   include 'orders.inc'

   ! Declare allocatable arrays
   double precision, allocatable :: AMP_SPLIT_STORE_R(:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_B(:,:)
   double complex, allocatable :: AMP_SPLIT_STORE_CNT(:,:,:,:)
   double precision, allocatable :: AMP_SPLIT_STORE_BSF(:,:,:,:)

contains
   ! Procedure to allocate arrays dynamically based on vec_size
   subroutine allocate_storage(vector_size)
       integer, intent(in) :: vector_size

       ! Allocate arrays with runtime size
       allocate(AMP_SPLIT_STORE_R(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_B(AMP_SPLIT_SIZE, vector_size))
       allocate(AMP_SPLIT_STORE_CNT(AMP_SPLIT_SIZE, 2, NSPLITORDERS, vector_size))
       allocate(AMP_SPLIT_STORE_BSF(AMP_SPLIT_SIZE, 5, 5, vector_size))
   end subroutine allocate_storage
   ! Add other module procedures here if necessary
end module vectorize
!ZW