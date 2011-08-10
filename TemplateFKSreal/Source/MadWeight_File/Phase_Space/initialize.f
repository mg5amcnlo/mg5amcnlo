      subroutine initialize
c**********************************************************************************
c     This subroutine initializes the phase space integration
c     It determines
c     c_point(J,I,K) : * central variables of integration + width 
c                    J -> number of final particle (madgraph notation ) start from 3 !!!
c                    I -> component : I=1 : y
c                                     I=2 : phi
c                                     I=3 : PT
c                    K ->     K = 1 : variable
c                             K = 2 : associated width   (0 if transfert function : delta function)
c                                                        (-1 if the variable is "fixed" by tranverse momentum conservation)
c
c***********************************************************************************
      implicit none
c
c     parameter
c
      double precision zero
      parameter (zero=0d0)
      include '../../nexternal.inc'
c
c      local
c
      integer I,J,temp
c
c      global
c
      double precision c_point(1:nexternal,3,2)
      common/ph_sp_init/c_point
c
      logical pass_event
      common /to_pass_event/pass_event
c
      integer Ndimens
      common /to_dimension/ Ndimens
c
      integer matching(3:nexternal) 
      integer inv_matching(3:nexternal)
      common/madgraph_order_info/matching,inv_matching
c------
c Begin Code
c------
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      s = 4d0*ebeam(1)*ebeam(2)
c
c      define the central point
c
       call get_central_point
c
c      counting the number of dimensions
c
      temp=-2 ! conservation of PT
      do i=3,nexternal
        do j=1,3
          if (c_point(i,j,2).ne.zero) temp=temp+1
          write(*,*) 'temp c_point ' , i,j,c_point(i,j,2)
        enddo
      enddo

      if (c_point(1,1,2).eq.zero) temp=temp-1
      if (c_point(2,1,2).eq.zero) temp=temp-1
      
      write(*,*) 'the number of dimension is ', temp

      Ndimens=temp

c      if (temp.ne.(Ndimens)) then
c      write(*,*) 'error : Ndimens is not # widths + 1 '
c      write(*,*) 'Ndimens=',Ndimens
c      pause
c      endif

      write(*,*) 'end initialize'
      return
      end


c      subroutine set_pmass
c*********************************************************
c  this subroutine set the masses of the external particles
c**********************************************************
c
c     parameter 
c
c      include '../../nexternal.inc'
c      include '../../coupl.inc'
c      double precision ZERO
c      parameter (ZERO=0d0)
c
c     local
c
c      integer i
c
c     global
c
c      double precision pmass(1:nexternal)
c      common / to_pmass/pmass
c
c      include '../../pmass.inc'
c
c      write(*,*) 'pmass ', pmass
c
c      return
c      end
