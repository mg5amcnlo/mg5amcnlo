      subroutine BinothLHA(p,born_wgt,virt_wgt)
c
c Given the Born momenta, this is the Binoth-Les Houches interface file
c that calls the OLP and returns the virtual weights. For convenience
c also the born_wgt is passed to this subroutine.
c
      use FKSParams
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include 'born_nhel.inc'
      double precision pi, zero,mone
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0d0)
      double precision p(0:3,nexternal-1)
      double precision virt_wgt,born_wgt,double,single
     $     ,born_wgt_recomputed,born_wgt_recomp_direct
      double precision, allocatable :: virt_wgts(:,:)
      double precision, allocatable :: virt_wgts_hel(:,:)
      double precision mu,ao2pi,conversion,alpha_S
      save conversion
      logical firsttime,firsttime_conversion
      data firsttime,firsttime_conversion /.true.,.true./
      logical firsttime_run
      data firsttime_run /.true./
      double precision qes2
      common /coupl_es/ qes2
      logical fksprefact
      parameter (fksprefact=.true.)
      integer ret_code
      double precision madfks_single, madfks_double
      double precision tolerance
      double precision, allocatable :: accuracies(:)
      integer i,j,IOErr, IOErrCounter
      integer dt(8)
      integer nbad, nbadmax
      double precision target,ran2,accum
      external ran2
      double precision hel_fact
CCC      double precision wgt_hel(max_bhel)
CCC      common/c_born_hel/wgt_hel
      integer nsqso, MLResArrayDim
c statistics for MadLoop
      double precision avgPoleRes(2),PoleDiff(2)
      integer ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1(0:9)
      common/ups_stats/ntot,nsun,nsps,nups,neps,n100,nddp,nqdp,nini,n10,n1
      parameter (nbadmax = 5)
      double precision pmass(nexternal)
      character*1 include_hel(max_bhel)
      integer goodhel(max_bhel),hel(0:max_bhel)
      save hel,goodhel
      logical fillh
      integer mc_hel,ihel
      double precision volh
      common/mc_int2/volh,mc_hel,ihel,fillh
      logical cpol
      integer getordpowfromindex_ml5
      integer orders_to_amp_split_pos
      logical, allocatable, save :: keep_order(:)
      include 'orders.inc'
      integer amp_orders(nsplitorders)
      integer split_amp_orders(nsplitorders), iamp
      double precision amp_split_finite_ML(amp_split_size)
      common /to_amp_split_finite/amp_split_finite_ML
      double precision prec_found(amp_split_size)
      double precision amp_split_poles_ML(amp_split_size,2),
     $ amp_split_poles_FKS(amp_split_size,2)
      common /to_amp_split_poles_FKS/amp_split_poles_FKS
      logical force_polecheck, polecheck_passed
      common /to_polecheck/force_polecheck, polecheck_passed
      integer ret_code_common
      common /to_ret_code/ret_code_common
      double precision born_hel_from_virt

      logical updateloop
      common /to_updateloop/updateloop
c masses
      include 'pmass.inc'
      data nbad / 0 /

      IOErrCounter = 0
c update the ren_scale for MadLoop and the couplings (should be the
c Ellis-Sexton scale)
      mu_r = sqrt(QES2)
      ! force to update also loop-related parameters
      updateloop=.true.
      call update_as_param()
      updateloop=.false.

      alpha_S=g**2/(4d0*PI)
      ao2pi= alpha_S/(2d0*PI)
      virt_wgt= 0d0
      single  = 0d0
      double  = 0d0
      born_hel_from_virt = 0d0
C     reset the amp_split array
      amp_split(1:amp_split_size) = 0d0
      amp_split_finite_ML(1:amp_split_size) = 0d0
      amp_split_poles_ML(1:amp_split_size,1) = 0d0
      amp_split_poles_ML(1:amp_split_size,2) = 0d0
      prec_found(1:amp_split_size) = 0d0
c This is no longer needed, because now the Born has the correct symmetry factor:
      if (firsttime_run) then
c The helicity double check should have been performed during the
c pole check, so we skip it here. It also makes sure that there is
c no conflict on the write access to HelFilter.dat when running
c locally in multicore without a cluster_tmp_path
         if (.not. force_polecheck)
     &        call set_forbid_hel_doublecheck(.True.)
         call get_nsqso_loop(nsqso)
         call get_answer_dimension(MLResArrayDim)
         allocate(accuracies(0:nsqso))
         allocate(virt_wgts(0:3,0:MLResArrayDim))
         allocate(virt_wgts_hel(0:3,0:MLResArrayDim))
         allocate(keep_order(nsqso))
c Make sure that whenever in the initialisation phase, MadLoop calls
c itself again to perform stability check to make sure no unstable EPS
c splips unnoticed.
         CALL FORCE_STABILITY_CHECK(.TRUE.)
         IF (.not. force_polecheck) THEN ! still have the pole for the pole check
            CALL COLLIER_COMPUTE_UV_POLES(.FALSE.)
            CALL COLLIER_COMPUTE_IR_POLES(.FALSE.)
         else
            CALL COLLIER_COMPUTE_UV_POLES(.TRUE.)
            CALL COLLIER_COMPUTE_IR_POLES(.TRUE.)
         endif
         firsttime_run = .false.
      endif
      firsttime=firsttime.or.force_polecheck
      if (firsttime) then
         write(*,*) "alpha_s value used for the virtuals"/
     &        /" is (for the first PS point): ", alpha_S
         tolerance=IRPoleCheckThreshold/10d0 ! for the pole check below
         call sloopmatrix_thres(p, virt_wgts, tolerance, accuracies,
     $        ret_code)
C        look for orders which match the nlo order constraint 
         do i = 1, nsqso
           keep_order(i) = .true.
           do j = 1, nsplitorders
             if(getordpowfromindex_ML5(j, i) .gt. nlo_orders(j)) then
               keep_order(i) = .false.
               exit
             endif
           enddo
           if (keep_order(i)) then
             write(*,*) 'VIRT: keeping split order ', i
           else
             write(*,*) 'VIRT: not keeping split order ', i
           endif
         enddo
         do i = 1, nsqso
           if (keep_order(i)) then
             virt_wgt= virt_wgt + virt_wgts(1,i)
             single  = single + virt_wgts(2,i)
             double  = double + virt_wgts(3,i)
C         keep track of the separate pieces correspoinding to
C          different coupling combinations
             do j = 1, nsplitorders
              amp_orders(j) = getordpowfromindex_ML5(j, i)
             enddo
             amp_split_finite_ML(orders_to_amp_split_pos(amp_orders)) =
     $            virt_wgts(1,i)
             amp_split_poles_ML(orders_to_amp_split_pos(amp_orders),1) =
     $            virt_wgts(2,i)
             amp_split_poles_ML(orders_to_amp_split_pos(amp_orders),2) =
     $            virt_wgts(3,i)
             prec_found(orders_to_amp_split_pos(amp_orders))
     $            =accuracies(i)
           endif
        enddo
      else
         tolerance=PrecisionVirtualAtRunTime
c Just set the accuracy found to a positive value as it is not specified
c once the initial pole check is performed.
         if (mc_hel.eq.0) then

            call sloopmatrix_thres(p,virt_wgts,tolerance,accuracies
     $           ,ret_code)
            do i = 1, nsqso
              if (keep_order(i)) then
                virt_wgt= virt_wgt + virt_wgts(1,i)
                single  = single + virt_wgts(2,i)
                double  = double + virt_wgts(3,i)
C         keep track of the separate pieces correspoinding to
C          different coupling combinations
                do j = 1, nsplitorders
                 amp_orders(j) = getordpowfromindex_ML5(j, i)
                enddo
                amp_split_finite_ML(orders_to_amp_split_pos(amp_orders))
     $               = virt_wgts(1,i)
                amp_split_poles_ML(orders_to_amp_split_pos(amp_orders)
     $               ,1) = virt_wgts(2,i)
                amp_split_poles_ML(orders_to_amp_split_pos(amp_orders)
     $               ,2) = virt_wgts(3,i)
                prec_found(orders_to_amp_split_pos(amp_orders))
     $               =accuracies(i)
              endif
            enddo
         elseif (mc_hel.eq.1) then
c Use the Born helicity amplitudes to sample the helicities of the
c virtual as flat as possible
            call PickHelicityMC(p,goodhel,hel,ihel,volh)
            !
            fillh=.false.
            call sloopmatrixhel_thres(p,hel(ihel),virt_wgts_hel
     $           ,tolerance,accuracies,ret_code)
            hel_fact = dble(goodhel(ihel))/volh/4d0
            do i = 1, nsqso
              if (keep_order(i)) then
                born_hel_from_virt=born_hel_from_virt+virt_wgts_hel(0,i)
                virt_wgt= virt_wgt + virt_wgts_hel(1,i) * hel_fact
                single  = single + virt_wgts_hel(2,i) * hel_fact
                double  = double + virt_wgts_hel(3,i) * hel_fact
C         keep track of the separate pieces correspoinding to
C          different coupling combinations
                do j = 1, nsplitorders
                 amp_orders(j) = getordpowfromindex_ML5(j, i)
                enddo
                amp_split_finite_ML(orders_to_amp_split_pos(amp_orders))
     $               = virt_wgts_hel(1,i) * hel_fact
                amp_split_poles_ML(orders_to_amp_split_pos(amp_orders)
     $               ,1) = virt_wgts_hel(2,i) * hel_fact
                amp_split_poles_ML(orders_to_amp_split_pos(amp_orders)
     $               ,2) = virt_wgts_hel(3,i) * hel_fact
                prec_found(orders_to_amp_split_pos(amp_orders))
     $               =accuracies(i)
              endif
            enddo

CCC            if (abs((wgt_hel(hel(ihel))-born_hel_from_virt/4d0)
CCC     $           /wgt_hel(hel(ihel))).gt.1e-5) then
CCC               write(*,*) 'ERROR HEL', wgt_hel(hel(ihel))
CCC     $              ,born_hel_from_virt/4d0,wgt_hel(hel(ihel))
CCC     $              /(born_hel_from_virt/4d0)
CCC                stop
CCC            endif
c Average over initial state helicities 
            if (nincoming.ne.2) then
               write (*,*)
     &              'Cannot do MC over helicities for 1->N processes'
               stop
            endif
         else
            write (*,*) 'Can only do sum over helicities,'/
     $           /' or pure MC over helicities',mc_hel
            stop
         endif
      endif
c======================================================================
c If the Virtuals are in the Dimensional Reduction scheme, convert them
c to the CDR scheme with the following factor (not needed for MadLoop,
c because they are already in the CDR scheme format)
c      if (firsttime_conversion) then
c         call DRtoCDR(conversion)
c         firsttime_conversion=.false.
c      endif
c      virt_wgt=virt_wgt+conversion*born_wgt*ao2pi
c======================================================================
c
c Check poles for the first PS points when doing MC over helicities, and
c for all phase-space points when not doing MC over helicities. Skip
c MadLoop initialization PS points.
      cpol=.false.
      ret_code_common=ret_code
      if ((firsttime .or. mc_hel.eq.0) .and. mod(ret_code,100)/10.ne.3
     $     .and. mod(ret_code,100)/10.ne.4) then
         call getpoles(p,QES2,madfks_double,madfks_single,fksprefact)
         polecheck_passed = .true.
         ! loop over the full result and each of the amp_split
         ! contribution
         do iamp=0,amp_split_size
          ! skip 0 contributions in the amp_split array
            if (iamp.ne.0) then
               if (amp_split_poles_FKS(iamp,1).eq.0d0.and.
     $              amp_split_poles_FKS(iamp,1).eq.0d0) cycle
            endif
            if (iamp.eq.0) then
               if (firsttime) then
                  write(*,*) ''
                  write(*,*) 'Sum of all split-orders'
               endif
            else
               if (firsttime) then
                  write(*,*) ''
                  write(*,*) 'Splitorders', iamp
                  call amp_split_pos_to_orders(iamp,split_amp_orders)
                  do i = 1, nsplitorders
                     write(*,*) '      ',ordernames(i), ':',
     $                    split_amp_orders(i)
                  enddo
               endif
               single=amp_split_poles_ML(iamp,1)
               double=amp_split_poles_ML(iamp,2)
               madfks_single=amp_split_poles_FKS(iamp,1)
               madfks_double=amp_split_poles_FKS(iamp,2)
            endif
            avgPoleRes(1)=(single+madfks_single)/2.0d0
            avgPoleRes(2)=(double+madfks_double)/2.0d0
            PoleDiff(1)=dabs(single - madfks_single)
            PoleDiff(2)=dabs(double - madfks_double)
            if ((dabs(avgPoleRes(1))+dabs(avgPoleRes(2))).ne.0d0) then
               cpol = .not.((((PoleDiff(1)+PoleDiff(2))/
     $              (dabs(avgPoleRes(1))+dabs(avgPoleRes(2)))) .lt.
     $              tolerance*10d0).or.(mod(ret_code,10).eq.7.and..not.force_polecheck))
            else
               cpol = .not.((PoleDiff(1)+PoleDiff(2).lt.tolerance*10d0)
     $              .or.(mod(ret_code,10).eq.7))
            endif
            if (tolerance.lt.0.0d0) then
               cpol = .false.
            endif
            if (.not. cpol .and. firsttime) then
               write(*,*) "---- POLES CANCELLED ----"
               write(*,*) " COEFFICIENT DOUBLE POLE:"
               write(*,*) "       MadFKS: ", madfks_double,
     &              "          OLP: ", double
               write(*,*) " COEFFICIENT SINGLE POLE:"
               write(*,*) "       MadFKS: ",madfks_single,
     &              "          OLP: ",single
               if (iamp.eq.0) then
                  write(*,*) " FINITE:"
                  write(*,*) "          OLP: ",virt_wgt
                  write(*,*) "          BORN: ",born_wgt
                  write(*,*) " MOMENTA (Exyzm): "
                  do i = 1, nexternal-1
                     write(*,*) i,p(0,i),p(1,i),p(2,i),p(3,i),pmass(i)
                  enddo
               endif
               if (mc_hel.ne.0) then
 198              continue
c Set-up the MC over helicities. This assumes that the 'HelFilter.dat'
c exists, which should be the case when firsttime is false.
                  if (NHelForMCoverHels.lt.0) then
                     mc_hel=0
                     goto 203
                  endif
                  open (unit=67,
     $                 file='../MadLoop5_resources/HelFilter.dat',
     $                 status='old',action='read',iostat=IOErr, err=201)
                  hel(0)=0
                  j=0
c optimized loop output
                  do i=1,max_bhel
                     read(67,*,err=202) goodhel(i)
                     if (goodhel(i).gt.-10000 .and. goodhel(i).ne.0)
     $                    then
                        j=j+1
                        goodhel(j)=goodhel(i)
                        hel(0)=hel(0)+1
                        hel(j)=i
                     endif
                  enddo
                  goto 203
 201              continue
                  if (IOErr.eq.2.and.IOErrCounter.lt.10) then
                     IOErrCounter = IOErrCounter+1
                     write(*,*) "File HelFilter.dat busy, retrying for"
     $                    //" the ",IOErrCounter," time."
                     call date_and_time(values=dt)
                     call sleep(1+(dt(8)/200))
                     goto 198
                  endif
                  write (*,*) 'Cannot do MC over hel:'/
     &                 /' "HelFilter.dat" does not exist'/
     &                 /' or does not have the correct format.'/
     $                 /' Change NHelForMCoverHels in FKS_params.dat '/
     &                 /'to explicitly summ over them instead.'
                  stop
c                  mc_hel=0
c                  goto 203
 202              continue
c non optimized loop output
                  rewind(67)
                  read(67,*,err=201) (include_hel(i),i=1,max_bhel)
                  do i=1,max_bhel
                     if (include_hel(i).eq.'T') then
                        j=j+1
                        goodhel(j)=1
                        hel(0)=hel(0)+1
                        hel(j)=i
                     endif
                  enddo
 203              continue
c Only do MC over helicities if there are NHelForMCoverHels
c or more non-zero (independent) helicities
                  if (NHelForMCoverHels.eq.-1) then
                     write (*,*) 'Not doing MC over helicities: '/
     $                    /'HelForMCoverHels=-1'
                     mc_hel=0
                  elseif (hel(0).lt.NHelForMCoverHels) then
                     write (*,'(a,i3,a)') 'Only ',hel(0)
     $                    ,' independent helicities:'/
     $                    /' switching to explicitly summing over them'
                     mc_hel=0
                  endif
                  close(67)
               endif
            elseif(cpol .and. firsttime) then
               polecheck_passed = .false.
               write(*,*) "POLES MISCANCELLATION, DIFFERENCE > ",
     &              tolerance*10d0
               write(*,*) " COEFFICIENT DOUBLE POLE:"
               write(*,*) "       MadFKS: ", madfks_double,
     &              "          OLP: ", double
               write(*,*) " COEFFICIENT SINGLE POLE:"
               write(*,*) "       MadFKS: ",madfks_single,
     &              "          OLP: ",single
               if (iamp.eq.0) then
                  write(*,*) " FINITE:"
                  write(*,*) "          OLP: ",virt_wgt
                  write(*,*) "          BORN: ",born_wgt
                  write(*,*) " MOMENTA (Exyzm): "
                  do i = 1, nexternal-1
                     write(*,*) i,p(0,i),p(1,i),p(2,i),p(3,i),pmass(i)
                  enddo
               endif
               write(*,*) 
               write(*,*) " SCALE**2: ", QES2
               if (nbad .lt. nbadmax) then
                  nbad = nbad + 1
                  write(*,*) " Trying another PS point"
               elseif (.not.force_polecheck) then
                  write(*,*) " TOO MANY FAILURES, QUITTING"
                  stop
               endif
            endif
         enddo
         firsttime = .false.
      endif
c Update the statistics using the MadLoop return code (ret_code)
      ntot = ntot+1             ! total number of PS
      if (ret_code/100.eq.1) then
         nsun = nsun+1          ! stability unknown
      elseif (ret_code/100.eq.2) then
         nsps = nsps+1          ! stable PS point
      elseif (ret_code/100.eq.3) then
         nups = nups+1          ! unstable PS point, but rescued
      elseif (ret_code/100.eq.4) then
         neps = neps+1          ! exceptional PS point: unstable, and not possible to rescue
      else
         n100=n100+1            ! no known ret_code (100)
      endif
      if (mod(ret_code,100)/10.eq.1 .or. mod(ret_code,100)/10.eq.3) then
         nddp = nddp+1          ! only double precision was used
         if (mod(ret_code,100)/10.eq.3) nini=nini+1 ! MadLoop initialization phase
      elseif (mod(ret_code,100)/10.eq.2 .or. mod(ret_code,100)/10.eq.4)
     $        then
         nqdp = nqdp+1          ! quadruple precision was used
         if (mod(ret_code,100)/10.eq.4) nini=nini+1 ! MadLoop initialization phase
      else
         n10=n10+1              ! no known ret_code (10)
      endif
      n1(mod(ret_code,10))=n1(mod(ret_code,10))+1 ! unit ret code distribution

c Write out the unstable, non-rescued phase-space points (MadLoop return
c code is in the four hundreds) or the ones that are found by the pole
c check (only available when not doing MC over hels)
      do iamp=1,amp_split_size
         if (.not.firsttime .and. (ret_code/100.eq.4 .or. cpol .or.
     $        prec_found(iamp).gt.0.05d0 .or.
     $        isnan(amp_split_finite_ML(iamp)))) then
            if (neps.lt.10) then
               if (neps.eq.1) then
                  open(unit=78, file='UPS.log')
               else
                  open(unit=78, file='UPS.log', access='append')
               endif
               write(78,*) '===== EPS #',neps,' ====='
               write(78,*) 'mu_r    =',mu_r           
               write(78,*) 'alpha_S =',alpha_S
               write(78,*) 'MadLoop return code, pole check and'/
     $              /' accuracy reported',ret_code,cpol,prec_found
               if (mc_hel.ne.0) then
                  write (78,*)'helicity (MadLoop only)',hel(i),mc_hel
               endif
               write(78,*) '1/eps**2 expected from MadFKS='
     $              ,amp_split_poles_FKS(iamp,2)
               write(78,*) '1/eps**2 obtained in MadLoop ='
     $              ,amp_split_poles_ML(iamp,2)
               write(78,*) '1/eps    expected from MadFKS='
     $              ,amp_split_poles_FKS(iamp,1)
               write(78,*) '1/eps    obtained in MadLoop ='
     $              ,amp_split_poles_ML(iamp,1)
               write(78,*) 'finite   obtained in MadLoop ='
     $              ,amp_split_finite_ML(iamp)
               write(78,*) 'Accuracy estimated by MadLop ='
     $              ,prec_found(iamp)
               do i = 1, nexternal-1
                  write(78,'(i2,1x,5e25.15)') 
     &                 i, p(0,i), p(1,i), p(2,i), p(3,i), pmass(i)
               enddo
               close(78)
            endif
            if ( prec_found(iamp).gt.0.05d0 .or.
     $           isnan(amp_split_finite_ML(iamp)) ) then
               write (*,*) 'WARNING: unstable non-rescued phase-space'/
     $              /' found for which the accuracy reported by'/
     $              /' MadLoop is worse than 5%. Setting virtual to'/
     $              /' zero for this PS point.'
               amp_split_finite_ML(iamp) = 0d0
            endif
         endif
      enddo
c also set the central value to zero if it is very unstable
      if (.not.firsttime .and.
     $     (accuracies(0).gt.0.05d0 .or. isnan(virt_wgt))) then
         virt_wgt=0d0
      endif
c If a MadLoop initialisation PS point (and stability is unknown), we
c better set the virtual to zero to NOT include it in the
c result. Sometimes this can be an unstable point with a very large
c weight, screwing up the complete integration afterward.
      if ( ( mod(ret_code,100)/10.eq.4 .or. mod(ret_code,100)/10.eq.3 )
     $     .and. ret_code/100.eq.1) then
         do iamp=1,amp_split_size
            amp_split_finite_ML(iamp) = 0d0
         enddo
         virt_wgt=0d0
      endif
      return
      end


      subroutine BinothLHAInit(filename)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      integer status,procnum
      double precision s,mu,sumdot
      external sumdot
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      character*13 filename
      common /LH_procnum /procnum

c Rocket:
c      call get_procnum(filename,procnum)
c      call Init(filename,status)
c      if (status.ne.1) then
c         write (*,*) 'Something wrong with Rocket Les Houches '//
c     &        'initialization',status
c$$$         stop
c      endif
c BlackHat:
c      call get_procnum(filename,procnum)
c      if(procnum.ne.1) then
c         write (*,*) 'Error in BinothLHAInit', procnum
c         stop
c       endif
c      call OLE_Init(filename//Char(0))
      return
      end


      subroutine DRtoCDR(conversion)
c This subroutine computes the sum in Eq. B.3 of the MadFKS paper
c for the conversion from dimensional reduction to conventional
c dimension regularization.
      implicit none
      double precision conversion
      double precision CA,CF
      parameter (CA=3d0,CF=4d0/3d0)
      integer i,triplet,octet
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "coupl.inc"

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      include "pmass.inc"

      triplet=0
      octet=0
      conversion = 0d0
      do i=1,nexternal
         if (i.ne.i_fks .and. i.ne.j_fks) then
            if (pmass(i).eq.0d0) then
               if (abs(particle_type(i)).eq.3) then
                  conversion=conversion-CF/2d0
                  triplet=triplet+1
               elseif (particle_type(i).eq.8) then
                  conversion=conversion-CA/6d0
                  octet=octet+1
               endif
            endif
         elseif(i.eq.min(i_fks,j_fks)) then
            if (pmass(j_fks).eq.0d0 .and. pmass(i_fks).eq.0d0) then
               if (m_type.eq.8) then
                  conversion=conversion-CA/6d0
                  octet=octet+1
               elseif (abs(m_type).eq.3)then
                  conversion=conversion-CF/2d0
                  triplet=triplet+1
               else
                  write (*,*)'Error in DRtoCDR, fks_mother must be'//
     &                 'triplet or octet',i,m_type
                  stop
               endif
            endif
         endif
      enddo
      write (*,*) 'From DR to CDR conversion: ',octet,' octets and ',
     &     triplet,' triplets in Born (both massless), sum =',conversion
      return
      
      end


      subroutine get_procnum(filename,procnum)
      implicit none
      integer procnum,lookhere,procsize
      character*13 filename
      character*176 buff
      logical done

      open (unit=68,file=filename,status='old')
      done=.false.
      do while (.not.done)
         read (68,'(a)',end=889)buff
         if (index(buff,'->').ne.0) then
c Rocket
c            lookhere=index(buff,'process')+7
c BlackHat
c            lookhere=index(buff,'|')
            if (lookhere.ne.0 .and. lookhere.lt.170) then
c Rocket
c               read (buff(lookhere+1:176),*) procnum
c BlackHat
c               read (buff(lookhere+1:176),*) procsize,procnum
c               if (procsize.ne.1) then
c                  write (*,*)
c     &                 'Can only deal with 1 procnum per (sub)process',
c     &                 procsize
c               else
                  write (*,*)'Read process number from contract file',
     &                 procnum
                  close(68)
                  return
c               endif
               done=.true.
            else
               write (*,*) 'syntax contract file not understandable',
     &              lookhere
               stop
            endif
         endif
      enddo
      stop

      close(68)

      return

 889  write (*,*) 'Error in contract file'
      stop
      end
