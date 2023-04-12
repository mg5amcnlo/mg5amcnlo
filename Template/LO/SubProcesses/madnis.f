************************************************************************
*	API to call madgraph functionality with python
*
*       Init for wrapper by     subroutine configure_code_cc
*       Initialization by 	  subroutine configure_code
*       Full event weight by 	  subroutine madnis_api
*

#ifdef TF
************************************************************************
      subroutine configure_code_cc(multi_channel_in,helicity_sum,dconfig)
************************************************************************
*	initialize multi-channel and phase-space generation for wrapper
*     just calls the normal configure_code but properly handels bools
*     from C/C++ side
************************************************************************
      use iso_c_binding
      implicit none
      logical(c_bool), intent(in), value :: multi_channel_in, helicity_sum
      logical :: multi_channel, hel_sum
      double precision, intent(inout) :: dconfig

      multi_channel = multi_channel_in
      hel_sum = helicity_sum
      call configure_code(multi_channel, hel_sum, dconfig)

      end subroutine
#endif

************************************************************************
      subroutine configure_code(multi_channel_in, helicity_sum, dconfig)
************************************************************************
*	initialize multi-channel and phase-space generation
************************************************************************

      IMPLICIT NONE

      logical, intent(in) :: multi_channel_in, helicity_sum
      double precision, intent(inout) :: dconfig
CF2PY logical, intent(in) :: multi_channel
CF2PY logical, intent(in) :: helicity_sum
CF2PY double precision, intent(in) :: dconfig
      

c      include 'maxparticles.inc'
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'genps.inc'
c      integer NCOMB
c TODO: need templating      
c      parameter (NCOMB=16)

      integer ncode
      integer iconfig
      integer nconfigs
      integer ninvar
      integer nb_tchannel
      integer j
      integer jconfig
      integer ndim
      logical fopened
c     
c     GLOBAL VARIABLE
c
      character*30 param_card_name
      common/to_param_card_name/param_card_name
cc
      include 'run.inc'
      
      integer diag_number
      common/to_diag_number/diag_number

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      
      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

c TODO: check if needed?      
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar

c     certainly not needed but does not hurt      
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig

      double precision zero
      parameter       (ZERO = 0d0)

      double precision pmass(nexternal)
      common/to_mass/  pmass

      double precision twgt, maxwgt,swgt(maxevents)
      integer                             lun, nw
      common/to_unwgt/twgt, maxwgt, swgt, lun, nw
      
      include 'coupl.inc'

      
      multi_channel = multi_channel_in
      write(*,*) "multi_channel = ", multi_channel, multi_channel_in
      write(*,*) "helicity_sum = ", helicity_sum
c     helicity setting
c======================
      if(helicity_sum)then
         isum_hel =0
c         call DS_register_dimension('Helicity',NCOMB)
c        Also set the minimum number of points for which each helicity
c        should be probed before the grid is used for sampling.
C     Typically 10 * n_matrix<i>
C TODO: need templating         
c         call DS_set_min_points(20,'Helicity') ! to check how the 20 is set
      else
         isum_hel = 1
      endif

c     channel setting
c ===================
c     ncode is number of digits needed for the BW code
      ncode=int(dlog10(3d0)*(max_particles-3))+1
      iconfig = int(dconfig*(1+10**(-ncode)))
c       write(*,*) 'Running Configuration Number: ',iconfig
c       diag_number = iconfig

c
c     Read process number
c
c      call open_file(lun+1, 'dname.mg', fopened)
c      if (.not.fopened)then
c         goto 11
c      endif
      
c
c     Here I want to set up with B.W. we map and which we don't
c
      dconfig = dconfig-iconfig
      if (dconfig .eq. 0) then
         write(*,*) 'Not subdividing B.W.'
         lbw(0)=0
      else
         lbw(0)=1
         jconfig=dconfig*(10**ncode + 0.1)
         write(*,*) 'Using dconfig=',jconfig
         call DeCode(jconfig,lbw(1),3,nexternal)
         write(*,*) 'BW Setting ', (lbw(j),j=1,nexternal-2)
      endif

c     driver setup
c     ================
c      lun = 27
c      open(unit=lun,status='scratch')
      

      param_card_name = 'param_card.dat'
      call setrun                !Sets up run parameters
      call setpara(param_card_name )   !Sets up couplings and masses
      include 'pmass.inc'        !Sets up particle masses
      call setcuts               !Sets up cuts
c      call printout              !Prints out a summary of paramaters
c      call run_printout          !Prints out a summary of the run settings
      nconfigs = 1
      call init_good_hel()
      call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,
     &     nexternal,nincoming,nb_tchannel)

      write(*,*) "Completed mapping",nexternal
      ndim = 3*(nexternal-nincoming)-4
      if (nincoming.gt.1.and.abs(lpp(1)) .ge. 1) ndim=ndim+1
      if (nincoming.gt.1.and.abs(lpp(2)) .ge. 1) ndim=ndim+1
      ninvar = ndim
      do j=mincfig,maxcfig
         if (abs(lpp(1)) .ge. 1 .and. abs(lpp(1)) .ge. 1) then
            if(ndim.gt.1) minvar(ndim-1,j)=ninvar-1
            minvar(ndim,j) = ninvar
         elseif (abs(lpp(1)) .ge. 1 .or. abs(lpp(1)) .ge. 1) then
            minvar(ndim,j) = ninvar
         endif
      enddo

      call sample_init(ninvar,1000,5,ninvar,1)
      
      return
      end

************************************************************************
      subroutine madnis_api(R, ndim, channel, apply_cut, wgt)
************************************************************************
*     This is a subroutine that returns the full event weight
*
*     INPUTS:  R           == random numbers
*              ndims       == number of dimensions
*              channel     == choose channel config
*              apply_cut   == apply ps cuts (bool)           
*     OUTPUTS: wgt         == updated weight after choosing points
************************************************************************
c TODO determine dimension here + template     
Cf2py double precision, intent(in), dimension(ndim) :: R
Cf2py integer, intent(in) :: ndim
Cf2py integer, intent(inout) :: channel
Cf2py logical, intent(in) :: apply_cut      
Cf2py double precision, intent(out) :: wgt      
      implicit none
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'genps.inc'
      
      integer, intent(in) :: ndim
      double precision, intent(in), dimension(ndim) ::  R
      integer, intent(inout) :: channel
      logical, intent(in) :: apply_cut
      double precision, intent(out) ::  wgt
      double precision p(0:3,nexternal)
      double precision dsig
      external dsig

      double precision fx
      
      logical pass_point
      integer diag_number
      common/to_diag_number/diag_number

      include 'run.inc'

      logical pass_points, PASSCUTS
c     TODO determine dimension here + template
      double precision x ! for cross-check
      
      integer ninvar
      save ninvar
      logical first
      save first
      data first /.true./
c     certainly not needed but does not hurt
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig

      integer        lbw(0:nexternal)  !Use of B.W.
      common /to_BW/ lbw

      LOGICAL CUTSDONE,CUTSPASSED
      COMMON/TO_CUTSDONE/CUTSDONE,CUTSPASSED
      integer ncode
      integer j
      integer iconfig
      integer jconfig
      integer           Minvar(maxdim,lmaxconfigs)
      common /to_invar/ Minvar
      integer nconfigs
      integer nb_tchannel

      cutsdone=.false.
      call store_random_number(R)

      if(first)then
         ninvar= 3*(nexternal-nincoming)-4
         if (nincoming.gt.1.and.abs(lpp(1)) .ge. 1) ninvar=ninvar+1
         if (nincoming.gt.1.and.abs(lpp(2)) .ge. 1) ninvar=ninvar+1
         first = .false.
      endif
      wgt = 1d0

c     channel setting
c ===================

      ncode=int(dlog10(3d0)*(max_particles-3))+1
      iconfig = int(channel*(1+10**(-ncode)))
c      channel = int(channel*(1+10**(-ncode)))

      channel = channel-iconfig
      if (channel .eq. 0) then
c         write(*,*) 'Not subdividing B.W.'
         lbw(0)=0
      else
         lbw(0)=1
         jconfig=channel*(10**ncode + 0.1)
c         write(*,*) 'Using dconfig=',jconfig
         call DeCode(jconfig,lbw(1),3,nexternal)
c         write(*,*) 'BW Setting ', (lbw(j),j=1,nexternal-2)
      endif

      nconfigs = 1
      call map_invarients(minvar,nconfigs,ninvar,mincfig,maxcfig,
     &     nexternal,nincoming,nb_tchannel)


c     Call weight
c ===================

C      write(*, *) "channel = ", iconfig
      call x_to_f_arg(ninvar,iconfig,mincfig,maxcfig,ninvar,wgt,x,p)
C      write(*, *) "outputs = ", wgt, p
      if (apply_cut) then
         if (PASSCUTS(p)) then
c            write(*,*) "wgt (in)", wgt
            fx = dsig(p,wgt,0)  !Evaluate function
c            write(*,*) "fx", fx
            wgt = wgt*fx
         else
            wgt = 0d0
         endif
      else
         cutsdone = .true.
         cutspassed = .true.
         fx = dsig(p,wgt,0)     !Evaluate function
         wgt = wgt*fx         
      endif

c     Store momenta
c ===================

      call store_momenta(p)
      return
      end

************************************************************************
      subroutine get_number_of_random_used(answer)
************************************************************************
CF2PY   integer, intent(out) :: answer
      implicit none
      integer answer

c     CAREFUL we use the max of nexternal for different process here
c     since this is a common block with the Source directory      
      include 'nexternal.inc'
      double precision Rstore(4*nexternal)
      integer r_used
      logical use_external_random_number
      integer mode(4*nexternal)
      common/external_random/ Rstore, use_external_random_number, r_used, mode

      answer = r_used
      return
      end

************************************************************************
      subroutine get_random_used_utility(utility)
************************************************************************
c     0 means number used by genps.f (i.e. kinematic)
c     1 means number taken by ranmar (i.e. auto_dsig and matrix.f) [not clear status yet]
c     2 means number associated to dsample.f (related to discrete dimension for sure)
c     3 means for the selection of which helicity is written on disk
c     4 means the selection of flavor written on disk
c     random number for color selection and unweighting are using ran1 and are not catched      
CF2PY   integer, intent(out), dimension(20) :: utility
      implicit none

c     CAREFUL we use the max of nexternal for different process here
c     since this is a common block with the Source directory            
      include 'nexternal.inc' 
      integer utility(4*nexternal)
      double precision Rstore(4*nexternal)
      integer r_used
      logical use_external_random_number
      integer mode(4*nexternal)
      common/external_random/ Rstore, use_external_random_number, r_used, mode

      utility(:) = mode(:)
      return
      end
      
************************************************************************     
      subroutine store_random_number(R)
************************************************************************
c     CAREFUL we use the max of nexternal for different process here
c     since this is a common block with the Source directory            
      include 'nexternal.inc'
      double precision R(4*nexternal), Rstore(4*nexternal)
      integer r_used
      logical use_external_random_number
      integer mode(4*nexternal)
      common/external_random/ Rstore, use_external_random_number, r_used, mode

      use_external_random_number = .true.
      Rstore(:) = R(:)
C      write(*,*) "stored rans = ", Rstore
      r_used = 0
      mode(:) = 0
      return
      end
************************************************************************
      subroutine store_momenta(pin)
************************************************************************
c     CAREFUL we use the max of nexternal for different process here
c     since this is a common block with the Source directory                  
      include 'nexternal.inc'
      double precision p(0:3,nexternal), pin(0:3,nexternal)
      common/madnis_api_p/p
      p(:,:) = pin(:,:)
      return
      end
************************************************************************
      subroutine get_momenta(pout)
************************************************************************
CF2PY double precision, intent(out), dimension(0:3,5) :: pout
c     CAREFUL we use the max of nexternal for different process here
c     since this is a common block with the Source directory                  
      include 'nexternal.inc'
      double precision p(0:3,nexternal)
      double precision, intent(inout) :: pout(4,nexternal)
      common/madnis_api_p/p
      pout(:,:) = p(:,:)
      return
      end

************************************************************************
      subroutine get_multichannel(alphaout, used_channel)
************************************************************************
CF2PY double precision, intent(out), dimension(8) :: alphaout
CF2PY integer, intent(out) :: used_channel
C     since symfact.dat implements a Monte-Carlo over some channel of integration
C     the channel under consideration is not always the input channel
C     used_channel returns the channel selected for the current event
C     and therefore which of the alphaout needs to be associated to the event.
      implicit none

      include 'ngraphs.inc'
      double precision, intent(out) :: alphaout(n_max_cg)
      integer used_channel

      include 'maxamps.inc'
      include 'maxconfigs.inc'

      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      COMMON/TO_AMPS/  AMP2,       JAMP2

      integer mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      
      integer i
      double precision total

      total = 0d0
      do i=1,MAXAMPS
         total = total + amp2(i)
      enddo

      do i=1,MAXAMPS
         alphaout(i) = amp2(i) / total
      enddo

      used_channel = this_config
      
      return
      end      
