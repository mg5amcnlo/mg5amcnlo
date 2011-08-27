!---------------------------------------------------------------------
!
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
!       call ctsxcut       (to get the amplitude amp)
!       call ctsstatistics (to get statistical information on the run)  
!             
!  as detailed below.
!   
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
      complex*16 amp(0:2)
      complex*16 ar1
      real*8 rootsvalue,limitvalue,muscale,thrs
      real*8 pp(0:3,0:5)           
      complex*16 m2(0:5)            
      integer number_propagators
      integer rnk,idig,rango
      integer scaloop
      integer n_mp,n_disc
      logical stable,forcemp
!
      rootsvalue= 50.d0
      limitvalue= 1.d-2
      idig      = 64 
      scaloop= 2 
      muscale= 1.d0 
      thrs= 1.d-6
!                
!---------------------------------------------------------------------
!
! subroutine ctsinit(rootsvalue,limitvalue,idig,scaloop,muscale,thrs)
!
! Initialization of CutTools:
!
!
! INPUT:  real*8  rootsvalue -> used as an internal arbitrary scale of the OPP 
!                               algorithm. It should be of the same order 
!                               of sqrt(s_hat). The result should not depend 
!                               on it.
!
!         real*8  limitvalue -> limit of precision below which
!                               the mp routines activate.      
!
!         integer idig       -> idig sets the max number of digits of 
!                               the multi-precision routines. 
!                               If idig = 0 the events below limitvalue 
!                               are simply discarded.
!                               idig= 0 should to be used when no multi 
!                               precision version of N(q) is available.
!
!         integer scaloop    -> library used to compute the scalar 
!                               1-loop functions:  
!                               scaloop= 1 -> looptools 
!                               scaloop= 2 -> avh (complex masses)   
!                               scaloop= 3 -> qcdloop.  
!
!         real*8  muscale    -> it is the scale for the 1-loop integrals.
!                               It has dimension of an energy. 
!
!         real*8  thrs       -> it is the numerical threshold for the 
!                               collinear/soft divergences in 
!                               avh_olo routines (scaloop= 2)
!                               
! OUTPUT: none
!
!---------------------------------------------------------------------
!
      call ctsinit(rootsvalue,limitvalue,idig,scaloop,muscale,thrs)

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
      pp(0,1)=    0.d0
      pp(1,1)=    0.d0
      pp(2,1)=    0.d0
      pp(3,1)=    0.d0
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
!
!---------------------------------------------------------------------
!
!      call ctsxcut(rootsvalue,muscale,number_propagators,test,mptest,
!     &             rnk,pp,m2,amp,ar1,stable)
!
! The total amplitude amp is computed:
!
!
! INPUT: real*8  rootsvalue         -> the arbitrary OPP scale 
!                                      set event by event.
!
!        real*8  muscale            -> the scale for the 1-loop integrals.
!                                      set event by event.
!
!        integer number_propagators -> number of propagators.
!
!        external test              -> name of the subroutine
!                                      computing N(q). 
!
!        external mptest            -> name of the subroutine
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
!        logical forceamp           -> if .true. forces cuttools to run
!                                      in multiprecision    
!               
! OUTPUT:  complex*16 amp(0:2)      -> Amplitude (without r2):     
!                                      amp(0) is the finite part   
!                                      amp(1) is the coeff. of 1/eps   pole
!                                      amp(2) is the coeff. of 1/eps^2 pole.
!
!          complex*16 ar1           -> the R_1 contribution.
!
!          logical stable           -> .false. if CutTools detects
!                                      numerical instabilities.         
!
!---------------------------------------------------------------------

      forcemp=.false. 
      call ctsxcut(rootsvalue,muscale,number_propagators,test,mptest,
     &             rnk,pp,m2,amp,ar1,stable,forcemp)
      write(*,*)'               '
      write(*,*)' Complete Amplitude (without r2):     '
      write(*,*)'               '
      write(*,*)'               '
      write(*,*)' finite part           amp(0)=',amp(0)
      write(*,*)' coeff of 1/eps   pole amp(1)=',amp(1)
      write(*,*)' coeff of 1/eps^2 pole amp(2)=',amp(2)
      write(*,*)'                          R_1=',ar1
      write(*,*)'                   amp(0)+R_1=',amp(0)+ar1
      write(*,*)'                       stable=',stable  
      write(*,*)'               '
!
!---------------------------------------------------------------------
!
! subroutine ctsstatistics(n_mp,n_disc) 
!
! Print out of the statistics of the run:
!
!
! INPUT :  none
!
! OUTPUT:  integer n_mp   ->  n.of points evaluated in multi-precision.
!
!          integer n_disc ->  n.of discarded points.               
!
!---------------------------------------------------------------------
      call ctsstatistics(n_mp,n_disc)
      write(*,*) 'n_mp  =',n_mp  
      write(*,*) 'n_disc=',n_disc
      end program example0


 





