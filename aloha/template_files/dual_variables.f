      module dual_variables
      implicit none
      private
      public :: Dual
      public :: assignment(=)
      public :: operator(+), operator(-)
      public :: operator(*), operator(/)
      public :: operator(**), sqrt, log
      public :: CONJG, DCONJG
      public :: DBLE, DIMAG
      public :: size


      !=================================================================
      ! Class initialization
      !=================================================================
      include "dual_opts.inc"
      integer, parameter :: bn_saved_hc(10)=(/1,2,5,15,52,203,877,4140,
     &21147,115975/)
      type :: Dual
         complex(kind(1d0)),dimension(0:(1+der_order)**npwave-1) :: 
     &   comp = (0d0,0d0) ! npwave taken from onia.inc
   
      ! Printing routine used for constitency checks
      contains
         procedure :: initZERO
      end type Dual

      !=================================================================
      ! Operators Interfaces
      !=================================================================
      interface assignment(=)
         module  procedure assign_Double
         module  procedure assign_Complex
         module procedure assign_Dual
      end interface assignment(=)

      interface operator(+)
         module procedure add_DualVariable
         module procedure add_CN_DualVariable
         module procedure add_DualVariable_CN
         module procedure add_RN_DualVariable
         module procedure add_DualVariable_RN
         module procedure add_NON_DualVariable
      end interface operator(+)

      interface operator(-)
         module procedure minus_DualVariable
         module procedure minus_CN_DualVariable
         module procedure minus_DualVariable_CN
         module procedure minus_RN_DualVariable
         module procedure minus_DualVariable_RN
         module procedure minus_NON_DualVariable
      end interface operator(-)

      interface operator(/)
         module procedure division_DualVariable
         module procedure division_CN_DualVariable
         module procedure division_DualVariable_CN
         module procedure division_RN_DualVariable
         module procedure division_DualVariable_RN
         module procedure division_IN_DualVariable
         module procedure division_DualVariable_IN
      end interface operator(/)

      interface operator(*)
         module procedure multiply_DualVariable
         module procedure multiply_CN_DualVariable
         module procedure multiply_DualVariable_CN
         module procedure multiply_RN_DualVariable
         module procedure multiply_DualVariable_RN
         module procedure multiply_IN_DualVariable
         module procedure multiply_DualVariable_IN
      end interface operator(*)

      interface operator(**)
         module procedure power_DualVariable_int
         module procedure power_DualVariable_real
      end interface operator(**)

      interface sqrt
         module procedure sqrt_DualVariable
      end interface sqrt

      interface log
         module procedure log_DualVariable
      end interface log

      interface DBLE
            module procedure DBLE_DualVariable
      end interface DBLE

      interface DIMAG
            module procedure DIMAG_DualVariable
      end interface DIMAG

      interface CONJG
            module procedure Imaginary_Conjugation
      end interface CONJG

      interface DCONJG
            module procedure Imaginary_Conjugation
      end interface DCONJG

      interface size
            module procedure dual_length
      end interface size
      
      
      contains
      !=================================================================
      ! Modules
      !=================================================================

      !=================================================================
      ! Assignment rules
      !=================================================================
      subroutine assign_Double(self,dn)
         type(Dual),intent(out)::self
         real(kind(1d0)),intent(in)::dn
         integer::i

         self%comp(0) = dn
         self%comp(1:i) = (0d0,0d0)
      end subroutine assign_Double

      subroutine assign_Complex(self,cn)
         type(Dual),intent(inout)::self
         complex*16,intent(in)::cn
         integer::i

         self%comp(0) = cn
         self%comp(1:) = (0d0,0d0)
      end subroutine assign_Complex

      subroutine assign_Dual(self,other)
         type(Dual),intent(inout)::self
         type(Dual),intent(in)::other
         integer::i

         do i = 0, size(self)
            self%comp(i) = other%comp(i)
         enddo
      end subroutine assign_Dual

      !=================================================================
      ! Proprieties
      !=================================================================
      ! Array size
      pure function Dual_Length(a) result(res)
         type(Dual),intent(in)::a
         Integer::res

         res = size(a%comp) - 1
      end function Dual_Length


      ! Initialization & Resetting of dual components
      subroutine initZERO(self)
         class(Dual)::self
         Integer::i

         do i = 0, size(self)
            self%comp(i) = (0d0,0d0)
         enddo
      end subroutine initZERO

      !=================================================================
      ! Addition rules
      !=================================================================
      function add_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual) :: res
         integer::i
         do i=0,2**npwave-1
         res%comp(i) = a%comp(i) + b%comp(i)
         enddo
      end function add_DualVariable

      function add_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i
         res%comp(0) = cn+a%comp(0)
         do i=1,2**npwave-1
            res%comp(i) = a%comp(i)
         enddo
      end function add_CN_DualVariable

      function add_DualVariable_CN(a,cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res

         res=add_CN_DualVariable(cn,a)
      end function add_DualVariable_CN

      function add_RN_DualVariable(rn,a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i
         res%comp(0) = dcmplx(rn,0d0)+a%comp(0)
         do i=1,2**npwave-1
            res%comp(i) = a%comp(i)
         enddo
      end function add_RN_DualVariable

      function add_DualVariable_RN(a,rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res

         res=add_RN_DualVariable(rn,a)
      end function add_DualVariable_RN

      function add_NON_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         do i=0,2**npwave-1
            res%comp(i) = a%comp(i)
         enddo
      end function add_NON_DualVariable
      

      !=================================================================
      ! Subtraction rules
      !=================================================================
      function minus_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual) :: res
         integer::i

         do i=0,2**npwave-1
            res%comp(i) = a%comp(i) - b%comp(i)
         enddo
      end function minus_DualVariable

      function minus_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         res%comp(0)=cn-a%comp(0)
         do i=1,2**npwave-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_CN_DualVariable

      function minus_DualVariable_CN(a,cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         res%comp(0)=a%comp(0)-cn
         do i=1,2**npwave-1
            res%comp(i) = a%comp(i)
         enddo
      end function minus_DualVariable_CN

      function minus_RN_DualVariable(rn,a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         res%comp(0)=dcmplx(rn,0d0)-a%comp(0)
         do i=1,2**npwave-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_RN_DualVariable
      
      function minus_DualVariable_RN(a,rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         res%comp(0)=a%comp(0)-dcmplx(rn,0d0)
         do i=1,2**npwave-1
            res%comp(i) = a%comp(i)
         enddo
      end function minus_DualVariable_RN

      function minus_NON_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i
         do i=0,2**npwave-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_NON_DualVariable


      !=================================================================
      ! Multiplication rules
      !=================================================================
      function multiply_DualVariable(a, b) result(res)
         type(Dual),intent(in)::a,b
         type(Dual)::res
         integer::i,j

         do i=0,2**npwave-1
            j=i
            do 
               res%comp(i) = res%comp(i) + a%comp(j)*b%comp(i - j)
            if (j.eq.0) exit
            j = iand(j - 1, i)
            enddo
         enddo
      end function multiply_DualVariable

      function multiply_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i

         do i=0,2**npwave-1
            res%comp(i)=cn*a%comp(i)
         enddo
      end function multiply_CN_DualVariable

      function multiply_DualVariable_CN(a,cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=multiply_CN_DualVariable(cn,a)
      end function multiply_DualVariable_CN

      function multiply_RN_DualVariable(rn,a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i
         do i=0,2**npwave-1
            res%comp(i)=dcmplx(rn,0d0)*a%comp(i)
         enddo
      end function multiply_RN_DualVariable

      function multiply_DualVariable_RN(a,rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=multiply_RN_DualVariable(rn,a)
      end function multiply_DualVariable_RN

      function multiply_IN_DualVariable(jn,a) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i
         do i=0,2**npwave-1
         res%comp(i)=dcmplx(jn,0d0)*a%comp(i)
         enddo
      end function multiply_IN_DualVariable

      function multiply_DualVariable_IN(a,jn) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res = multiply_IN_DualVariable(jn,a)
      end function multiply_DualVariable_IN

      
      !=================================================================
      ! Power rules
      !=================================================================
      function power_DualVariable_int(a,np) result(res)
         type(Dual),intent(in)::a
         integer,intent(in)::np
         type(Dual)::res
         integer::i,j,k,bn
         integer:: nd
         complex(kind(1d0))::cterm,prefact

         if (np.gt.1) then       ! positive powers (identiy excluded)
            res%comp(0) = (a%comp(0))**np
            do i=1,2**npwave-1
               res%comp(i) = dcmplx(0d0,0d0)
               bn = 1
               ! Different groups are computed independently
               ! Each group corresponds to a number of derivative nd
               do nd=1,min(nd_max(i),np)
                  if (nd.eq.np) then
                     prefact=1.d0
                  else
                     prefact=(a%comp(0))**(np-nd)
                  endif
                  ! Determing the prefactor (same for all nd entries)
                  if (np.le.4) then
                     prefact = prefact*fallfact_int(np,nd)
                  else
                     prefact = prefact*falling_factorial(np,nd)
                  endif
                  ! The number of entries of each group is given by the
                  ! Stirling number of the second kind (hardcoded)
                  do j = 1, sn_array(nd_max(i),nd)
                     cterm = prefact
                     ! Each entry involves products of nd derivatives
                     ! The index of the dual component is determined by 
                     ! a binary splitting (hardcoded)
                     do k = 1, nd
                        cterm = cterm*(a%comp(split(i,bn,k)))
                     enddo
                     bn = bn+1
                     res%comp(i)=res%comp(i)+cterm
                  enddo
               enddo
            enddo
            return

         elseif (np.lt.0) then   ! negative powers
            if ((a%comp(0).eq.dcmplx(0d0,0d0)))then
               print*, "Error in power_DualVariable_int: 0 cannot be "//
     1               "raised to a non-positive power"
               stop
            endif
            res%comp(0) = (a%comp(0))**np
            do i=1,2**npwave-1
               res%comp(i) = dcmplx(0d0,0d0)
               bn = 1
               ! Different groups are computed independently
               ! Each group corresponds to a number of derivative nd
               do nd=1,nd_max(i)
                  prefact=(a%comp(0))**(np-nd)
                  ! Determing the prefactor (same for all nd entries)
                  if (np.ge.-4) then
                     prefact = prefact*fallfact_int(np,nd)
                  else
                     prefact = prefact*falling_factorial(np,nd)
                  endif
                  ! The number of entries of each group is given by the
                  ! Stirling number of the second kind (hardcoded)
                  do j = 1, sn_array(nd_max(i),nd)
                     cterm = prefact
                     ! Each entry involves products of nd derivatives
                     ! The index of the dual component is determined by 
                     ! a binary splitting (hardcoded)
                     do k = 1, nd
                        cterm = cterm*(a%comp(split(i,bn,k)))
                     enddo
                     bn = bn+1
                     res%comp(i)=res%comp(i)+cterm
                  enddo
               enddo
            enddo
            return

         elseif (np.eq.0) then
           res%comp(0)=dcmplx(1d0,0d0)
           do i=1,2**npwave-1
              res%comp(i)=dcmplx(0d0,0d0)
           enddo
           return
         else  ! np.eq.1
            res = a
            return
         endif
      end function power_DualVariable_int

      function power_DualVariable_real(a,np) result(res)
         type(Dual),intent(in)::a
         real(kind(1d0)),intent(in)::np
         type(Dual)::res
         integer::i,j,k,bn
         integer:: nd
         complex(kind(1d0))::cterm,prefact

         if (abs(dble(nint(np))-np).lt.1d-12) then
            ! If np is an integer, power_DualVariable_int is used
            res=power_DualVariable_int(a,nint(np))
            return
         endif

         if (np.gt.0) then ! positive (non-integer) powers
            res%comp(0) = (a%comp(0))**np
            do i=1,2**npwave-1
               res%comp(i) = dcmplx(0d0,0d0)
               bn = 1
               ! Different groups are computed independently
               ! Each group corresponds to a number of derivative nd
               do nd=1,nd_max(i)
                  prefact=(a%comp(0))**(np-nd)
                  ! Determing the prefactor (same for all nd entries)
                  if (np.eq.0.5d0 .or. np.eq.1.5d0) then
                     prefact = prefact*fallfact_real(int(np+0.5),nd)
                  else
                     prefact = prefact*falling_factorial_r(np,nd)
                  endif
                  ! The number of entries of each group is given by the
                  ! Stirling number of the second kind (hardcoded)
                  do j = 1, sn_array(nd_max(i),nd)
                     cterm = prefact
                     ! Each entry involves products of nd derivatives
                     ! The index of the dual component is determined by 
                     ! a binary splitting (hardcoded)
                     do k = 1, nd
                        cterm = cterm*(a%comp(split(i,bn,k)))
                     enddo
                     bn = bn+1
                     res%comp(i)=res%comp(i)+cterm
                  enddo
               enddo
            enddo
            return

         else              ! negative (non-integer) powers
            if ((a%comp(0).eq.dcmplx(0d0,0d0)))then
               print*, "Error in power_DualVariable_int: 0 cannot be "//
     1               "raised to a non-positive power"
               stop
            endif
            res%comp(0) = (a%comp(0))**np
            do i=1,2**npwave-1
               res%comp(i) = dcmplx(0d0,0d0)
               bn = 1
               ! Different groups are computed independently
               ! Each group corresponds to a number of derivative nd
               do nd=1,nd_max(i)
                  prefact=(a%comp(0))**(np-nd)
                  ! Determing the prefactor (same for all nd entries)
                  if (np.eq.-0.5d0 .or. np.eq.-1.5d0) then
                     prefact = prefact*fallfact_real(int(np+0.5),nd)
                  else
                     prefact = prefact*falling_factorial_r(np,nd)
                  endif
                  ! The number of entries of each group is given by the
                  ! Stirling number of the second kind (hardcoded)
                  do j = 1, sn_array(nd_max(i),nd)
                     cterm = prefact
                     ! Each entry involves products of nd derivatives
                     ! The index of the dual component is determined by 
                     ! a binary splitting (hardcoded)
                     do k = 1, nd
                        cterm = cterm*(a%comp(split(i,bn,k)))
                     enddo
                     bn = bn+1
                     res%comp(i)=res%comp(i)+cterm
                  enddo
               enddo
            enddo
            return

         endif
      end function power_DualVariable_real


      !=================================================================
      ! Division rules
      !=================================================================
      function division_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual)::oneoverb
         type(Dual) :: res
         oneoverb=power_DualVariable_int(b,-1)
         res=multiply_DualVariable(a,oneoverb)
         return
      end function division_DualVariable

      function division_CN_DualVariable(cn, a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_CN_DualVariable(cn,oneovera)
         return
      end function division_CN_DualVariable

      function division_DualVariable_CN(a, cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=multiply_CN_DualVariable(dcmplx(1d0,0d0)/cn,a)
         return
      end function division_DualVariable_CN

      function division_RN_DualVariable(rn, a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_RN_DualVariable(rn,oneovera)
         return
      end function division_RN_DualVariable

      function division_DualVariable_RN(a, rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=multiply_RN_DualVariable(1d0/rn,a)
         return
      end function division_DualVariable_RN

      function division_IN_DualVariable(jn, a) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_IN_DualVariable(jn,oneovera)
         return
      end function division_IN_DualVariable

      function division_DualVariable_IN(a, jn) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=multiply_RN_DualVariable(1d0/jn,a)
         return
      end function division_DualVariable_IN


      !=================================================================
      ! Sqrt-root rule
      !=================================================================
      function sqrt_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res=power_DualVariable_real(a,0.5d0)
         return
      end function sqrt_DualVariable


      !=================================================================
      ! Natural-Logarithm rule
      !=================================================================
      function log_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,j,k,bn,ng,n_ones
         integer::pref
         ! bnmax means we can have at maximum size(a)=nmax
         ! otherwise, please increase the dimension
         integer,dimension(bn_saved_hc(size(a)),0:size(a))::c_split
         complex(kind(1d0))::cterm

         do i=0,2**npwave-1
            if(i.eq.0)then
               res%comp(0) = LOG(a%comp(0))
               cycle
            endif
            call count_binary_ones(i,n_ones)
            bn=bell(n_ones)
            call generate_binary_partitions(i,n_ones,bn,c_split(1:bn,
     &      0:n_ones))
            res%comp(i) = dcmplx(0d0,0d0)
            do j=1, bn
               ng=c_split(j,0)
               pref=factorial(ng-1)
               if(mod(ng,2).eq.0)pref=pref*(-1)
               cterm=pref/(a%comp(0))**ng
               do k=1,ng
                  cterm=cterm*a%comp(c_split(j,k))
               enddo
               res%comp(i)=res%comp(i)+cterm
            enddo
         enddo
         return
      end function log_DualVariable


      !=================================================================
      ! Double-real conversion rules
      !=================================================================
      function DBLE_DualVariable(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i

         do i=0,2**npwave-1
            res%comp(i) = dcmplx(a%comp(i)%re, 0d0)
         enddo
         return
      end function DBLE_DualVariable

      !=================================================================
      ! Double-imaginary conversion rules
      !=================================================================
      function DIMAG_DualVariable(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i

         do i=0,2**npwave-1
            res%comp(i) = dcmplx(a%comp(i)%im, 0d0)
         enddo
         return
      end function DIMAG_DualVariable

      ! Conjugation rules
      function Imaginary_Conjugation(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i

         do i=0,2**npwave-1
            res%comp(i) = DCONJG(a%comp(i))
         enddo
         return
      end function Imaginary_Conjugation


      !=================================================================
      ! Additional functions
      !=================================================================
      ! Truncated factorial (n*(n-1)*...*(n-a))
      ! Used for on-the-fly computations
      integer(kind=8) function falling_factorial(a, n) result(res)
         integer, intent(in) :: a, n
         integer :: i
         res = 1_8
         if (n.lt.0) then
         print *, "Error: n must be >= 0"
         stop
         end if
         do i = 0, n-1
         res = res*(a-i)
         end do
      end function falling_factorial

      real(kind(1d0)) function falling_factorial_r(a, n) result(res)
         real(kind(1d0)),intent(in)::a
         integer, intent(in) :: n
         integer :: i
         res = 1d0
         if (n.lt.0) then
         print *, "Error: n must be >= 0"
         stop
         end if
         do i = 0, n-1
         res = res*(a-dble(i))
         end do
      end function falling_factorial_r

      
      !=================================================================
      ! Additional functions (not used)
      !=================================================================
      ! These functions are included for self-consistency but unused in
      ! the MadGraph implementation
      SUBROUTINE convert_to_binary(n1,x,y)
         ! converts to a binary number
         !     y -result
         !     x -input number
         !     n -levels
         INTEGER,INTENT(IN)::x,n1
         INTEGER,DIMENSION(n1),INTENT(OUT)::y
         INTEGER::X1,J
         y(1:n1)=0
         IF(x.EQ.0)RETURN
         X1=X
         DO J=n1,1,-1
         y(J)=X1-INT(X1/2)*2
         IF((X1/2).LT.1.) EXIT
         X1=INT(X1/2)
         ENDDO
         RETURN
      END SUBROUTINE convert_to_binary

      SUBROUTINE count_binary_ones(c,n)
         IMPLICIT NONE
         INTEGER,INTENT(IN)::c
         INTEGER,INTENT(OUT)::n
         INTEGER::nbits,i
         ! Determine number of bits and positions of 1s in c
         nbits=bit_size(c)
         n=0
         ! Count 1 bits
         do i=0,nbits-1
         if (btest(c, i))n=n+1
         enddo
         return
      end SUBROUTINE count_binary_ones

      SUBROUTINE split_binary(c,n,c12)
         IMPLICIT NONE
         INTEGER,INTENT(IN)::c
         INTEGER,INTENT(IN)::n ! number of 1s in the binary number c
         INTEGER,DIMENSION(2**n,2),INTENT(OUT)::c12
         INTEGER,DIMENSION(n)::one_positions
         INTEGER::i,j,n_ones,combs,a,b
         INTEGER::nbits
         ! Store positions of 1 bits
         nbits = bit_size(c)
         n_ones=0
         DO i=0, nbits-1
         IF(btest(c, i))then
            n_ones=n_ones+1
            one_positions(n_ones)=i
         endif
         ENDDO
         if(n_ones.ne.n)then
         WRITE(*,*)"ERROR: input does not match in split_binary"
         stop
         ENDIF
         combs = 2**n_ones
         do j=0, combs-1
            a=0
            b=0
            do i=1,n_ones
               if(btest(j,i-1))then
                  b=ibset(b,one_positions(i))
               else
                  a=ibset(a,one_positions(i))
               endif
            enddo
            c12(j+1,1)=a
            c12(j+1,2)=b
         enddo
         return
      end SUBROUTINE split_binary

      pure function to_binary_in_str(x, nbits) result(str)
         integer, intent(in) :: x, nbits
         character(len=nbits) :: str
         integer :: i
         do i = 1, nbits
            if (btest(x, nbits - i)) then
               str(i:i) = '1'
            else
               str(i:i) = '0'
            end if
         end do
      end function to_binary_in_str

      ! it generates all possible partitions of c
      ! split a binary number c into all possible one or more binary numbers.
      ! In each possible case, 1 in c appears only either in ak.
      ! The order of {a1,a2,...,an} does not matter.
      subroutine generate_binary_partitions(c,n,bn,c_split)
         IMPLICIT NONE
         INTEGER,INTENT(IN)::c
         INTEGER,INTENT(IN)::n ! number of 1s in the binary number c
         ! it has Bell(n) splittings
         INTEGER,INTENT(IN)::bn
         INTEGER,DIMENSION(bn,0:n),INTENT(OUT)::c_split
         integer::ngroups
         integer,DIMENSION(n)::group
         INTEGER,DIMENSION(n)::one_positions
         INTEGER::i,n_ones,nbits,ic
         nbits=bit_size(c)
         ! Extract bit positions of 1s of c
         n_ones=0
         DO i=0, nbits-1
         IF(btest(c, i))then
            n_ones=n_ones+1
            one_positions(n_ones)=i
         endif
         ENDDO
         if(n_ones.ne.n)then
         WRITE(*,*)"ERROR: input does not match in generate_binary"//
     1                "_partitions"
         stop
         ENDIF
         ngroups = 0
         ic = 0
         call genrate_binary_partitions_rec(1,n,group,ngroups,
     &   one_positions,bn,ic,c_split)
         return
      end subroutine generate_binary_partitions

      recursive subroutine genrate_binary_partitions_rec(i,m,group,
     &ngroups,bits,bn,ic,c_split)
         IMPLICIT NONE
         integer,intent(in)::i,m
         integer,dimension(:),intent(in)::bits
         integer,intent(inout)::ngroups
         integer,dimension(:),intent(inout)::group
         ! it has Bell(n) splittings
         INTEGER,INTENT(IN)::bn
         INTEGER,INTENT(INOUT)::ic
         INTEGER,DIMENSION(bn,0:m),INTENT(INOUT)::c_split
         integer::g,a,j

         if(i.gt.m)then
            ic=ic+1
            c_split(ic,0)=ngroups
            do g=1,ngroups
               a=0
               do j=1,m
                  if(group(j).eq.g)a=ibset(a,bits(j))
               enddo
               c_split(ic,g)=a
            enddo
            return
         endif

         ! Try adding element i to each existing group
         do g = 1, ngroups
         group(i) = g
         call genrate_binary_partitions_rec(i+1,m,group,ngroups,bits,bn,
     &   ic,c_split)
         end do

         ! Create a new group
         ngroups = ngroups + 1
         group(i) = ngroups
         call genrate_binary_partitions_rec(i+1,m,group,ngroups,bits,bn,
     &   ic,c_split)
         ngroups = ngroups - 1
         return
      end subroutine genrate_binary_partitions_rec

      !===================================================================
      ! Compute Bell(n) using the Bell triangle (Dobinski’s table)
      ! Works for n ≤ 14 with 64-bit integers.
      !===================================================================
      integer(kind=8) function bell(n) result(B)
         implicit none
         integer, intent(in) :: n
         integer(kind=8), allocatable :: T(:,:)
         integer :: i, j

         if (n.lt.0) then
         B = 0
         return
         end if

         allocate(T(0:n, 0:n))
         T = 0_8

         ! Bell triangle initialization
         T(0,0) = 1_8

         do i = 1, n
         T(i,0) = T(i-1,i-1)     ! first element of each row
            do j = 1, i
               T(i,j) = T(i-1,j-1) + T(i,j-1)
            end do
         end do

         B = T(n,0)

         deallocate(T)
      end function bell

      !-------------------------------------------------------
      ! Function to compute factorial 1*2*...*n
      ! Works for integer n >= 0
      !-------------------------------------------------------
      integer(kind=8) function factorial(n) result(res)
         integer, intent(in) :: n
         integer :: i
         res = 1_8
         if (n.lt.0) then
         print *, "Error: n must be >= 0"
         stop
         end if
         if(n.le.1)return
            
         do i = 2, n
         res = res*i
         end do
      end function factorial
      
      end module dual_variables