!---------------------------------------------------------------------
!  EXAMPLE OF MAIN PROGRAM
!---------------------------------------------------------------------
      program example1
      implicit none
!                     !----------------------------------!
      external test   ! Name of the subroutine computing !
!                     ! the numerator function.          ! 
!                     ! Location numerators.f.           ! 
!                     !----------------------------------! 
!
!                     !--------------------------------------!
      external mptest ! Name of the mpsubroutine (if present)!        
!                     ! computing the numerator function.    !
!                     ! Location mpnumerators.f90.           ! 
!                     ! If absent put 'external dummy'       !
!                     !--------------------------------------!
!---------------------------------------------------------------------
      integer maxden
      parameter (maxden= 8)
      complex*16 m2(0:maxden-1)
      real*8 pp(0:3,0:maxden-1)           
!---------------------------------------------------------------------
! Auxiliary variables:
!---------------------------------------------------------------------
      common/rango/rango  ! only used by the toy numerator 
      real*8 xm(1:maxden-2),p(4,maxden-2),k(0:3,maxden)           
      real*8 rootsvalue,limitvalue,roots,muscale
      complex*16 amp(0:2),ampcc,ampr1
      integer number_propagators
      integer rnk,i,j,l,iter,rango
      integer scaloop,imode
      integer npoints
      logical stable,discarded,ext_num_for_r1
!---------------------------------------------------------------------
! Read number of MC points, the number of propagators and the rank:
!--------------------------------------------------------------------- 
      print*,'enter npoints,number_propagators,rank,scaloop,muscale'
      print*,'    '
      print*,'scaloop= 1 -> looptools 1-loop '
      print*,'scaloop= 2 -> avh 1-loop (massive with complex masses)'
      print*,'scaloop= 3 -> qcdloop   1-loop (Ellis and Zanderighi)'
      print*,'muscale (dimension of energy) is the scale' 
      print*,'for the 1-loop integrals' 
      print*,'    '
      read*,npoints,number_propagators,rnk,scaloop,muscale
      rango= rnk                ! only used by the toy numerators 
                                ! located in numerators.f and  
                                ! mpnumerators.f90
!                               !
      if (number_propagators.gt.maxden) then
         stop 'increase maxden in example1.f90'
      endif
      roots     = 50.d0         ! value of sqrt(s)
      rootsvalue= roots
      limitvalue= 1.d-2 
      ext_num_for_r1=.true.
      imode     = 0

!---------------------------------------------------------------------
! Input momenta and masses in each denominator:
!---------------------------------------------------------------------
      k(0,1)= roots/2.d0
      k(1,1)= 0.d0
      k(2,1)= 0.d0
      k(3,1)= roots/2.d0
      k(0,2)= roots/2.d0
      k(1,2)= 0.d0
      k(2,2)= 0.d0
      k(3,2)=-roots/2.d0
!
      do i= 1,number_propagators-2
         xm(i) = 1.d0
      enddo
!---------------------------------------------------------------------
! To initialize CutTools call ctsinit(limitvalue,scaloop,ext_num_for_r1)
!
! INPUT:
!
!         real*8  limitvalue -> limit of precision below which
!                               the PS point is considered stable
!                               by tha A=A test.      
!
!         integer scaloop    -> library used to compute the scalar 
!                               1-loop functions:  
!                               scaloop= 1 -> looptools (not implemented) 
!                               scaloop= 2 -> avh (complex masses)   
!                               scaloop= 3 -> qcdloop.  
!
!      logical ext_num_for_r1 -> numfunc or numfuncrec is used in the
!                                computation of R1 if put to .true. 
!                                or .false
! OUTPUT: none
!---------------------------------------------------------------------
      call ctsinit(limitvalue,scaloop,ext_num_for_r1)
      do iter= 1,npoints   ! do-loop over events 
        if     (number_propagators.ge.4) then
           call rambo(0,number_propagators-2,roots,xm,p)
        elseif (number_propagators.eq.3) then
           p(4,1)= k(0,1)+k(0,2) 
           p(1,1)= k(1,1)+k(1,2)
           p(2,1)= k(2,1)+k(2,2)
           p(3,1)= k(3,1)+k(3,2)
        elseif (number_propagators.eq.2) then
        elseif (number_propagators.eq.1) then
        else
          print*,'number_propagators=',number_propagators,'not allowed!'
          stop
        endif 
          do j= 0,3
           if (j.ne.0) then
              do l= 1,number_propagators-2
                 k(j,l+2)= p(j,l) 
              enddo
           else
            do l= 1,number_propagators-2
               k(j,l+2)= p(4,l) 
            enddo
         endif
        enddo
!
        do l= 0,3 
          pp(l,0)= 0.d0
        enddo
        do i= 1,number_propagators-1
           do l= 0,3 
             pp(l,i)=  k(l,2)
           enddo
           do j= 3,i+1 
              do l= 0,3 
                pp(l,i)= pp(l,i)-k(l,j)
              enddo 
           enddo
        enddo
!
!        do i= 0,number_propagators-1
!           m2(i)= 0.d0            
!        enddo
!
        m2(0)= 1.d0 
        m2(1)= 2.d0 
        m2(2)= 3.d0 
        m2(3)= 4.d0 
        m2(4)= 0.d0 
        m2(5)= 0.d0 
!
!comment
!        do i= 1,number_propagators-1
!           do l= 0,3
!             print*,'l,i,pp(l,i)=',l,i,pp(l,i)
!           enddo
!        enddo
!comment
!---------------------------------------------------------------------
! To compute the 1-loop amplitude
!
!      call ctsxcut(imode,rootsvalue,muscale,number_propagators,test,
!     & mpfortest,rnk,pp,m2,amp,ampcc,ampr1,stable)
!
!
! INPUT: integer imode              -> the running mode of CutTools according
!                                      to the following scheme:  
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                            !
!! imode:|  actions performed by ctsxcut:                                     !
!!       |                                                                    !
!!   0   | (dp_dir,dp_inv)-> dp_Atest -> stable -> (only if stable=.false.) ->!
!!       | (mp_dir,mp_inv)-> mp_Atest -> stable                               ! 
!!   1   | (dp_dir)       -> dp_Ntest -> stable                               !
!!   2   | (dp_inv)       -> dp_Ntest -> stable                               !
!!   3   | (dp_dir,dp_inv)-> dp_Atest -> stable                               !
!!   4   | (mp_dir)       -> mp_Ntest -> stable                               ! 
!!   5   | (mp_inv)       -> mp_Ntest -> stable                               ! 
!!   6   | (mp_dir,mp_inv)-> mp_Atest -> stable                               !
!!                                                                            !
!! Legenda:                                                                   !
!!                                                                            !
!! dp_dir    = compute amp in double precision with normal   propagator order !
!! dp_inv    = compute amp in double precision with reversed propagator order !
!! mp_dir    = compute amp in multi  precision with normal   propagator order !
!! mp_inv    = compute amp in multi  precision with reversed propagator order !
!! dp_Atest  = perform the A=A test in double precision                       !
!! mp_Atest  = perform the A=A test in multi  precision                       !
!! dp_Ntest  = perform the N=N test in double precision                       !
!! mp_Ntest  = perform the N=N test in multi  precision                       !
!! -> stable = set stable=.true. or stable=.false.                            !
!!             according to the outcome of the test                           !
!!                                                                            !
!! Tests:                                                                     !
!!                                                                            !
!! -The N=N test is a test on the reconstructed OPP integrand performed       !
!!  by comparing original and reconstacted integrands at an arbirtary value   !
!!  of the integration momentum.                                              !
!!                                                                            ! 
!! -The A=A test checks the 2 amplitudes obtained with dir and inv orders     !
!!               of the propagators given in the input                        !
!! Notes:                                                                     !
!!                                                                            ! 
!! a) imode= 0 is recommended, unless you really know what you are doing.     !
!!                                                                            !
!! b) When two determinations of amp are available, that one with more        !
!!    accurate recounstructed numerator (coming from the N=N test) is used.   !
!!                                                                            !
!! c) When running in multi precision with scaloop= 3 (qcdloop), the loop     !
!!    functions are computed in double precision only. A full multi           !
!!    precision result can only be obtained with scaloop= 2 (OneLoop).        !
!!                                                                            ! 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!        real*8  rootsvalue         -> the arbitrary OPP scale 
!                                      set event by event.
!
!        real*8  muscale            -> the scale for the 1-loop integrals
!                                      set event by event.
!                                      It has dimension of an energy. 
!
!        integer number_propagators -> number of propagators.
!
!        external test              -> name of the subroutine
!                                      computing N(q). 
!
!        external mpfortest         -> name of the subroutine
!                                      computing N(q) in  
!                                      multi-precision (if absent
!                                      put dummy).
!
!        integer rnk                -> the maximum rank of N(q) (if 
!                                      unknown put number_propagators).
!
!        real*8 pp(0:3,0:number_propagators-1)           
!                                   -> momenta flowing in the internal 
!                                      propagators.
!
!        complex*16 m2(0:number_propagators-1)           
!                                   -> masses squared of the internal 
!                                      propagators. When scaloop supports it,
!                                      they can be complex. When scaloop does
!                                      not support complex masses, only 
!                                      the real part of m2 is used.  
!   
!               
! OUTPUT:  complex*16 amp(0:2)      -> Amplitude (without r2):     
!                                      amp(0) is the total finite part
!                                      (INCLUDING R1!!!)   
!                                      amp(1) is the coeff. of 1/eps   pole
!                                      amp(2) is the coeff. of 1/eps^2 pole.
!
!          complex*16 ampcc         -> the Cut Constructible contribution
!          complex*16 ampr1         -> the R1 contribution
!                                      (NOTE that ampcc+ampr1 = amp(0))
!
!          logical stable           -> .false. if CutTools detects
!                                      numerical instabilities.         
!---------------------------------------------------------------------
        call ctsxcut(imode,rootsvalue,muscale,number_propagators,test,
     &   mptest,rnk,pp,m2,amp,ampcc,ampr1,stable)
        write(*,*)'               '
        write(*,*)'  iter= ',iter
        write(*,*)'               '
        write(*,*)'               '
        write(*,*)' Complete Amplitude (without r2):     '
        write(*,*)'               '
        write(*,*)'               '
        write(*,*)' finite part           amp(0)=',amp(0)
        write(*,*)' coeff of 1/eps   pole amp(1)=',amp(1)
        write(*,*)' coeff of 1/eps^2 pole amp(2)=',amp(2)
        write(*,*)'                        ampcc=',ampcc
        write(*,*)'                           R1=',ampr1
        write(*,*)'                       stable=',stable  
        write(*,*)'               '
      enddo
!---------------------------------------------------------------------
! To know the statistics of the run call ctsstatistics(discarded)
!
! INPUT :  none
!
! OUTPUT:  logical discarded  -> .true. if there are discarded PS points
!
! Print out of the statistics of the run:
!
!          n_mp   ->  n.of points evaluated in multi-precision.
!          n_disc ->  n.of discarded points.               
!          n_tot  ->  n.of calls to ctsxcut               
!---------------------------------------------------------------------
      call ctsstatistics(discarded)
      end program example1


 





