!---------------------------------------------------------------------
!  EXAMPLE OF MAIN PROGRAM
!
!  Pourpose of the program:  
!  -----------------------
! 
!
!                     /                 N(q)
!  Computing    amp=  | d^n q   ------------------------  
!                     /         D_0 D_1 D_2 D_3 D_4 D_5
!
!
!  General Structure of the program:
!  --------------------------------
!
!  To perform the computation, three subroutines should be called
!
!       call ctsinit       (to initialize cuttools)  
!       call ctsxcut       (to get the amplitude amp and R1)
!       call ctsstatistics (to get statistical information on the run)  
!             
!  as detailed below.
!---------------------------------------------------------------------
!
      program example0
      implicit none
!                     !----------------------------------!
      external test   ! Name of the subroutine computing !
!                     ! the numerator function N(q).     ! 
!                     ! Location: numerators.f.          ! 
!                     !----------------------------------! 
!
!                     !---------------------------------------!
      external mptest ! Name of the mpsubroutine (if present) !        
!                     ! computing the numerator function N(q).!
!                     ! in multiprecision.                    !
!                     ! Location: mpnumerators.f90.           ! 
!                     ! If absent put 'external dummy'        !
!                     !---------------------------------------!
!
!---------------------------------------------------------------------

      common/rango/rango ! only used by the toy numerators test and mptest
      complex*16 amp(0:2),ampcc,ampr1
      real*8 rootsvalue,limitvalue,muscale
      real*8 pp(0:3,0:5)           
      complex*16 m2(0:5)            
      integer number_propagators
      integer rnk,rango,imode
      integer scaloop
      logical stable,discarded,ext_num_for_r1
!
      rootsvalue= 50.d0
      limitvalue= 1.d-2
      imode  = 0     
      scaloop= 2 
      muscale= 1.d0 
      ext_num_for_r1=.true.
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

      number_propagators= 6 
      rango= 6
      rnk= rango 
!
!     momenta flowing in the 6 internal propagators: 
!
!     0 is the energy component
!     1 is the x component
!     2 is the y component
!     3 is the z component
!
      pp(0,0)=    0.d0
      pp(1,0)=    0.d0
      pp(2,0)=    0.d0
      pp(3,0)=    0.d0
!
      pp(0,1)=    25.d0
      pp(1,1)=    0.d0
      pp(2,1)=    0.d0
      pp(3,1)=   -25.d0
!
      pp(0,2)=    12.2944131682730d0
      pp(1,2)=    1.03940959319740d0
      pp(2,2)=   -6.52053527152409d0
      pp(3,2)=   -35.8551455176305d0
!
      pp(0,3)=   -3.11940819171487d0
      pp(1,3)=    16.2625830020165d0
      pp(2,3)=   -7.82868841887933d0
      pp(3,3)=   -33.8229999456535d0
!
      pp(0,4)=   -4.47137880909124d0
      pp(1,4)=    15.9241558606968d0
      pp(2,4)=   -8.11309412871339d0
      pp(3,4)=   -35.1006560075423d0
!
      pp(0,5)=   -25.d0
      pp(1,5)=     0.d0
      pp(2,5)=     0.d0
      pp(3,5)=   -25.d0
!
!     masses of the 6 internal propagators: 
!
      m2(0)= 0.d0          
      m2(1)= 0.d0            
      m2(2)= 0.d0            
      m2(3)= 0.d0            
      m2(4)= 0.d0            
      m2(5)= 0.d0            
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
     & mptest,rnk,pp,m2,amp,ampcc,ampr1,stable)
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
      end program example0


 





