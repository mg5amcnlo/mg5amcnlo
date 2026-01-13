      module dual_variables
      implicit none
      private
      public :: Dual
      public :: operator(+), operator(-)
      public :: operator(*), operator(/)
      public :: operator(**), sqrt, log
      public :: CONJG, DCONJG, DUALCONJG
      public :: DBLE, DIMAG
      public :: size
   
      include "onia.inc"
   
      type :: Dual
         complex(kind(1d0)),dimension(0:(1+der_order)**npwave-1) :: comp = (0d0,0d0) ! npwave taken from onia.inc
   
      ! Printing routine used for constitency checks
      contains
         procedure :: write_formatted
         generic :: write(formatted) => write_formatted
         procedure :: pmath
      end type Dual

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

      interface DUALCONJG
            module procedure Dual_Conjugation
      end interface DUALCONJG

      interface size
            module procedure dual_length
      end interface size

      integer,parameter::nmax=10
      integer,parameter::bnarray(nmax)=(/1,2,5,15,52,203,877,4140,21147,
     &115975/)
   
      contains

      !subroutine get_lenght(DV)
      !  class(Dual),intent(in)::DV
      !  integer,intent(out)::n

      !  n = size(DV%compoennts)
      !  return
      !end subroutine get_lenght

      ! logical function compare(self, other) result(res)
      !   class(Dual), intent(in) :: self
      !   class(Dual), intent(in) :: other   ! accepts any type
      !   res = .false.
      !      
      !   ! res = (self%length() == other_var%length())
      !   res = (size(self%comp) .eq. size(other%comp))
      ! end function compare

      ! Array size
      pure function Dual_Length(a) result(res)
         type(Dual), intent(in) :: a
         Integer :: res

         res = size(a%comp) - 1
      end function Dual_Length


      ! Printing routine
      subroutine write_formatted(dtv, unit, iotype, v_list, iostat, 
     &iomsg)
         class(Dual), intent(in) :: dtv
         integer, intent(in) :: unit
         character(len=*), intent(in) :: iotype
         integer, intent(in) :: v_list(:)
         integer, intent(out) :: iostat
         character(len=*), intent(inout) :: iomsg

         integer :: i

         write(unit,'(A)', advance='no') '{'
         do i = 0, size(dtv)
            write(unit,'("(",G0,",",G0,")")', advance='no')
     &           real(dtv%comp(i)), aimag(dtv%comp(i))
            if (i < size(dtv)) write(unit,'(A)', advance='no') ', '
         end do
         write(unit,'(A)') '}'

         iostat = 0
      end subroutine write_formatted

      subroutine pmath(self, text)
         class(Dual), intent(in) :: self
         integer :: i
         character(len=*), intent(in), optional :: text
         real(kind(1d0)) :: re, im

         if (present(text)) write(*,'(1X,A)', advance='no') text

         write(*,'(A)', advance='no') '{'
         do i = 0, size(self)
            re = real(self%comp(i))
            im = aimag(self%comp(i))

            if (im >= 0d0) then
               write(*,'(G0," + ",G0,"*I")', advance='no') re, im
            else
               write(*,'(G0," - ",G0,"*I")', advance='no') re, -im
            end if
            if (i < size(self)) write(*,'(A)', advance='no') ', '
         end do
         write(*,'(A)') '}'
      end subroutine pmath


      ! Addition rules
      function add_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual) :: res
         integer::i,nn

         if(size(a).ne.size(b))then
         WRITE(*,*)"Error: cannot add two different type of dual"//
     1            " variables"
         stop
         endif
         nn=size(a)
         do i=0,2**nn-1
         res%comp(i) = a%comp(i) + b%comp(i)
         enddo
      end function add_DualVariable

      function add_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1            " in add"
            stop
         endif

         res%comp(0) = cn+a%comp(0)
         do i=1,2**nn-1
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
         integer::i,nn
         nn=size(a)

         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1            " in add"
            stop
         endif

         res%comp(0) = dcmplx(rn,0d0)+a%comp(0)
         do i=1,2**nn-1
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
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1            " in add"
            stop
         endif

         do i=0,2**nn-1
            res%comp(i) = a%comp(i)
         enddo
      end function add_NON_DualVariable
      

      ! Subtraction rules
      function minus_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual) :: res
         integer::i,nn

         if(size(a).ne.size(b))then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1            " in add"
            WRITE(*,*)"Error: cannot minus two different type of dual"//
     1            " variables"
            stop
         endif

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1               " in minus"
            stop
         endif

         do i=0,2**nn-1
            res%comp(i) = a%comp(i) - b%comp(i)
         enddo
      end function minus_DualVariable

      function minus_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1              " in minus"
            stop
         endif

         res%comp(0)=cn-a%comp(0)
         do i=1,2**nn-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_CN_DualVariable

      function minus_DualVariable_CN(a,cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1              " in minus"
            stop
         endif

         res%comp(0)=a%comp(0)-cn
         do i=1,2**nn-1
            res%comp(i) = a%comp(i)
         enddo
      end function minus_DualVariable_CN

      function minus_RN_DualVariable(rn,a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                 " in minus"
            stop
         endif

         res%comp(0)=dcmplx(rn,0d0)-a%comp(0)
         do i=1,2**nn-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_RN_DualVariable
      
      function minus_DualVariable_RN(a,rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in minus"
            stop
         endif

         res%comp(0)=a%comp(0)-dcmplx(rn,0d0)
         do i=1,2**nn-1
            res%comp(i) = a%comp(i)
         enddo
      end function minus_DualVariable_RN

      function minus_NON_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                   " in minus"
            stop
         endif
         do i=0,2**nn-1
            res%comp(i) = -a%comp(i)
         enddo
      end function minus_NON_DualVariable


      ! Multiplication rules
      function multiply_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual) :: res
         integer::i,j,nn,n_ones,ii
         integer,dimension(2**size(a),2)::c12

         if(size(a).ne.size(b))then
            WRITE(*,*)"Error: cannot multiply two different type of "//
     1                "dual variables"
            stop
         endif
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                   " in multiply"
            stop
         endif    
         do i=0,2**nn-1
            if(i.eq.0)then
               res%comp(0) = a%comp(0)*b%comp(0)
               cycle
            endif
            call count_binary_ones(i,n_ones)
            call split_binary(i,n_ones,c12(1:2**n_ones,1:2))
               res%comp(i) = dcmplx(0d0,0d0)
            do j=1,2**n_ones
                  res%comp(i)=res%comp(i)+a%comp(c12(j,1))*
     &            b%comp(c12(j,2))
            enddo
         enddo
      end function multiply_DualVariable

      function multiply_CN_DualVariable(cn,a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in multiply"
            stop
         endif

         do i=0,2**nn-1
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
         integer::i,nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in multiply"
            stop
         endif
         do i=0,2**nn-1
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
         integer::i,nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in multiply"
            stop
         endif
         do i=0,2**nn-1
         res%comp(i)=dcmplx(jn,0d0)*a%comp(i)
         enddo
      end function multiply_IN_DualVariable

      function multiply_DualVariable_IN(a,jn) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         res = multiply_IN_DualVariable(jn,a)
      end function multiply_DualVariable_IN


      ! Power rules
      function power_DualVariable_int(a,np) result(res)
         type(Dual),intent(in)::a
         integer,intent(in)::np
         type(Dual)::res
         integer::i,j,k,nn,bn,ng,n_ones
         integer::pref
         ! bnmax means we can have at maximum size(a)=nmax
         ! otherwise, please increase the dimension
         integer,dimension(bnarray(size(a)),0:size(a))::c_split
         complex(kind(1d0))::cterm

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not"//
     1                " match in power"
            stop
         endif

         if(np.eq.0)then
         res%comp(0)=dcmplx(1d0,0d0)
         do i=1,2**nn-1
            res%comp(i)=dcmplx(0d0,0d0)
         enddo
         return
         endif

         if(a%comp(0).eq.0d0)then
         do i=0,2**nn-1
            res%comp(i)=dcmplx(0d0,0d0)
         enddo
         return
         endif

         if(nn.gt.nmax)then
         write(*,*)"Error: please increase nmax and bnarray in"//
     1                " power_DualVariable_int"
         stop
         endif

         do i=0,2**nn-1
         if(i.eq.0)then
            res%comp(0) = (a%comp(0))**np
            cycle
         endif
         call count_binary_ones(i,n_ones)
         bn=bell(n_ones)
         call generate_binary_partitions(i,n_ones,bn,c_split(1:bn,
     &   0:n_ones))
         res%comp(i) = dcmplx(0d0,0d0)
         do j=1,bn
            ng=c_split(j,0)
            pref=falling_factorial(np,ng)
            cterm=pref*(a%comp(0))**(np-ng)
            do k=1,ng
            cterm=cterm*a%comp(c_split(j,k))
            enddo
            res%comp(i)=res%comp(i)+cterm
         enddo
         enddo
         return
      end function power_DualVariable_int

      function power_DualVariable_real(a,np) result(res)
         type(Dual),intent(in)::a
         real(kind(1d0)),intent(in)::np
         type(Dual)::res
         integer::i,j,k,nn,bn,ng,n_ones
         real(kind(1d0))::pref
         ! bnmax means we can have at maximum size(a)=nmax
         ! otherwise, please increase the dimension
         integer,dimension(bnarray(size(a)),0:size(a))::c_split
         complex(kind(1d0))::cterm

         nn=size(a)
         if(size(res).ne.nn)then
         WRITE(*,*)"Error: the output dual variable does not"//
     1                " match in power"
         stop
         endif

         if(np.eq.0d0)then
         res%comp(0)=dcmplx(1d0,0d0)
         do i=1,2**nn-1
               res%comp(i)=dcmplx(0d0,0d0)
         enddo
         return
         endif

         if(a%comp(0).eq.0d0)then
         do i=0,2**nn-1
            res%comp(i)=dcmplx(0d0,0d0)
         enddo
         return
         endif

         if(nn.gt.nmax)then
         write(*,*)"Error: please increase nmax and bnarray in "//
     1            "power_DualVariable_real"
         stop
         endif

         do i=0,2**nn-1
            if(i.eq.0)then
               res%comp(0) = (a%comp(0))**np
               cycle
            endif
            call count_binary_ones(i,n_ones)
            bn=bell(n_ones)
            call generate_binary_partitions(i,n_ones,bn,c_split(1:bn,
     &      0:n_ones))
            res%comp(i) = dcmplx(0d0,0d0)
            do j=1,bn
               ng=c_split(j,0)
               pref=falling_factorial_r(np,ng)
               cterm=pref*(a%comp(0))**(np-dble(ng))
               do k=1,ng
                  cterm=cterm*a%comp(c_split(j,k))
               enddo
               res%comp(i)=res%comp(i)+cterm
            enddo
         enddo
         return
      end function power_DualVariable_real


      ! Division rules
      function division_DualVariable(a, b) result(res)
         type(Dual), intent(in) :: a, b
         type(Dual)::oneoverb
         type(Dual) :: res
         integer::nn
         if(size(a).ne.size(b))then
            WRITE(*,*)"Error: cannot division two different type of"//
     1                   " dual variables"
            stop
         endif
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                   " in division"
            stop
         endif
         oneoverb=power_DualVariable_int(b,-1)
         res=multiply_DualVariable(a,oneoverb)
         return
      end function division_DualVariable

      function division_CN_DualVariable(cn, a) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
            stop
         endif
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_CN_DualVariable(cn,oneovera)
         return
      end function division_CN_DualVariable

      function division_DualVariable_CN(a, cn) result(res)
         complex(kind(1d0)),intent(in)::cn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
            stop
         endif
         res=multiply_CN_DualVariable(dcmplx(1d0,0d0)/cn,a)
         return
      end function division_DualVariable_CN

      function division_RN_DualVariable(rn, a) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
            stop
         endif
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_RN_DualVariable(rn,oneovera)
         return
      end function division_RN_DualVariable

      function division_DualVariable_RN(a, rn) result(res)
         real(kind(1d0)),intent(in)::rn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
            stop
         endif
         res=multiply_RN_DualVariable(1d0/rn,a)
         return
      end function division_DualVariable_RN

      function division_IN_DualVariable(jn, a) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual)::oneovera
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
            stop
         endif
         oneovera=power_DualVariable_int(a,-1)
         res=multiply_IN_DualVariable(jn,oneovera)
         return
      end function division_IN_DualVariable

      function division_DualVariable_IN(a, jn) result(res)
         integer,intent(in)::jn
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
           WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in division"
           stop
         endif
         res=multiply_RN_DualVariable(1d0/jn,a)
         return
      end function division_DualVariable_IN


      ! Square-root rule
      function sqrt_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::nn
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in sqrt"
            stop
         endif
         res=power_DualVariable_real(a,0.5d0)
         return
      end function sqrt_DualVariable

      ! Natural-logarithmic rule
      function log_DualVariable(a) result(res)
         type(Dual), intent(in) :: a
         type(Dual) :: res
         integer::i,j,k,nn,bn,ng,n_ones
         integer::pref
         ! bnmax means we can have at maximum size(a)=nmax
         ! otherwise, please increase the dimension
         integer,dimension(bnarray(size(a)),0:size(a))::c_split
         complex(kind(1d0))::cterm
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not match"//
     1                " in log"
            stop
         endif

         if(nn.gt.nmax)then
            write(*,*)"Error: please increase nmax and bnarray in"//
     1                " log_DualVariable"
            stop
         endif

         do i=0,2**nn-1
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


      ! Double-real rule
      function DBLE_DualVariable(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i,nn

         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not"//
     1                " match the original one"
            stop
         endif

         do i=0,2**nn-1
            res%comp(i) = dcmplx(a%comp(i)%re, 0d0)
         enddo
         return
      end function DBLE_DualVariable

      ! Double-image rule
      function DIMAG_DualVariable(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i,nn
            
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not"//
     1                " match the original one"
            stop
         endif

         do i=0,2**nn-1
            res%comp(i) = dcmplx(a%comp(i)%im, 0d0)
         enddo
         return
      end function DIMAG_DualVariable

      ! Conjugation rules
      function Imaginary_Conjugation(a) result(res)
         type(Dual),intent(in)::a
         type(Dual)::res
         integer::i,nn
            
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not"//
     1                " match the original one"
            stop
         endif

         do i=0,2**nn-1
            res%comp(i) = DCONJG(a%comp(i))
         enddo
         return
      end function Imaginary_Conjugation

      function Dual_Conjugation(a, j) result(res)
         type(Dual),intent(in)::a
         integer,intent(in),dimension(:),optional::j
         type(Dual)::res
         integer,allocatable::indices(:)
         integer::i,nn,idx
         integer::min_idx, max_idx
         
         nn=size(a)
         if(size(res).ne.nn)then
            WRITE(*,*)"Error: the output dual variable does not"//
     1                " match the original one"
            stop
         endif
            
         if (present(j)) then
            allocate(indices(size(j)))
            indices = j
         else
            allocate(indices(size(a)))
            indices = [(i, i = 1, size(a))]
         end if

         min_idx = indices(1)
         max_idx = indices(1)
         if (size(indices).GT.1) Then
            do i = 2, size(indices)
               if (indices(i) < min_idx) min_idx = indices(i)
               if (indices(i) > max_idx) max_idx = indices(i)
            enddo
         endif

         if (min_idx.LE.0) then
            write(*,*) "Error: index of dual conjugation cannot be "//
     1                "less or equal than 0"
            stop
         end if

         if (max_idx.GT.nn) then
            write(*,'(A,I0,A,I0,A)') ""//
     1            "Error: index of dual conjugation (", j, ") "//    
     2            "exceed dual variable dimension (", nn, ")"
            stop
         end if
            

         do i=1, size(indices)
            idx = indices(i)
            res%comp(idx) = -a%comp(idx)
         enddo
         return
      end function Dual_Conjugation



      ! Additional functions
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
      ! Same as BellB[n] in Mathematica
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

      !-------------------------------------------------------
      ! Function to compute a*(a-1)*(a-2)*...*(a-n+1)
      ! Works for integer n >= 0
      !-------------------------------------------------------
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
      
      end module dual_variables
