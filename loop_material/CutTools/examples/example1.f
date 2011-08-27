!---------------------------------------------------------------------
!
!  EXAMPLE OF MAIN PROGRAM
!
!---------------------------------------------------------------------
!
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
!
!---------------------------------------------------------------------
      integer maxden
      parameter (maxden= 10)
      complex*16 m2(0:maxden-1)
      real*8 pp(0:3,0:maxden-1)           
!
!---------------------------------------------------------------------
!
! Auxiliary variables:
!
!---------------------------------------------------------------------
!
      common/rango/rango  ! only used by the toy numerator 
      real*8 xm(1:maxden-2),p(4,maxden-2),k(0:3,maxden)           
      real*8 rootsvalue,limitvalue,roots,muscale,thrs
      complex*16 amp(0:2),ar1
      integer number_propagators
      integer rnk,i,j,l,iter,idig,rango
      integer scaloop
      integer npoints
      logical stable,forcemp
      integer n_mp,n_disc
!
!---------------------------------------------------------------------
!
! Read number of MC points, the number of propagators and the rank:
!
!---------------------------------------------------------------------
! 
      print*,'enter npoints,number_propagators,rank,scaloop,muscale,
     &thrs'
      print*,'    '
      print*,'scaloop= 1 -> looptools 1-loop '
      print*,'scaloop= 2 -> avh 1-loop (massive with complex masses)'
      print*,'scaloop= 3 -> qcdloop   1-loop (Ellis and Zanderighi)'
      print*,'muscale (dimension of energy) is the scale' 
      print*,'for the 1-loop integrals' 
      print*,'    '
      read*,npoints,number_propagators,rnk,scaloop,muscale,thrs
      rango= rnk                ! only used by the toy numerators 
                                ! located in numerators.f and  
                                ! mpnumerators.f90
!                               !
      if (number_propagators.gt.maxden) then
         stop 'increase maxden in example1.f90'
      endif
      roots     = 50.d0         ! value of sqrt(s)
!
!---------------------------------------------------------------------
!
! Input momenta and masses in each denominator:
!
!---------------------------------------------------------------------
!
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
         xm(i) = 0.d0
      enddo
!
!---------------------------------------------------------------------
!
! Input variables of ctsinit (to initialize of CutTools)
!
!---------------------------------------------------------------------
!
      rootsvalue= roots  ! used as an internal scale of the OPP algorithm
      limitvalue= 1.d-2  ! limit of precision below which 
!                        ! the mp routines activate
      idig      = 64     ! idig      = 64             
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
!
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
        do i= 0,number_propagators-1
           m2(i)= 0.d0            
        enddo
!
!         m2(0)= 10.1d0 
!         m2(1)= 10.2d0 
!         m2(2)= 10.3d0 
!         m2(3)= 10.4d0 
!         m2(4)= 20.5d0 
!         m2(5)= 200.d0 
!
!        do i= 1,number_propagators-1
!           do l= 0,3
!             print*,'l,i,pp(l,i)=',l,i,pp(l,i)
!           enddo
!        enddo
!
!
!---------------------------------------------------------------------
!
!      call ctsxcut(rootsvalue,muscale,number_propagators,test,mptest,
!     &             rnk,pp,m2,amp,ar1,stable,forceamp)
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
!
! HERE the total amplitude is computed by calling CutTools
!
!---------------------------------------------------------------------
!
        forcemp=.false. 
        call ctsxcut(rootsvalue,muscale,number_propagators,test,mptest,
     &               rnk,pp,m2,amp,ar1,stable,forcemp)
        print*,'               '
        print*,'  iter= ',iter
        print*,'               '
        print*,'               '
        print*,' Complete Amplitude (without r2):     '
        print*,'               '
        print*,'               '
        print*,' finite part           amp(0)=',amp(0)
        print*,' coeff of 1/eps   pole amp(1)=',amp(1)
        print*,' coeff of 1/eps^2 pole amp(2)=',amp(2)
        print*,'                          R_1=',ar1
        print*,'                   amp(0)+R_1=',amp(0)+ar1
        print*,'                       stable=',stable  
        print*,'               '
      enddo
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
      print*,'n_mp  =',n_mp   ! n.of points evaluated in mult. prec.
      print*,'n_disc=',n_disc ! n.of discarded points
      end program example1


 





