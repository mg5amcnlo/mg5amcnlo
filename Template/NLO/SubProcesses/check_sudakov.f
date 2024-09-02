      Program DRIVER
c**************************************************************************
c     This is the driver for the whole calulation
c**************************************************************************
      use mint_module
      use FKSParams
      implicit none
C
C     CONSTANTS
C
      double precision pi, zero
      parameter (pi=3.1415926535897932385d0)
      parameter (zero = 0d0)
      integer npointsChecked
      integer i, j, k, l
      integer return_code
      double precision tolerance, tolerance_default
      double precision, allocatable :: accuracies(:)
      double precision ren_scale, energy
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      double precision p(0:3, nexternal), prambo(0:3,100)
      double precision p_born(0:3,nexternal-1),p_born_first(0:3,nexternal-1)
      common/pborn/p_born
      double precision pswgt
      double precision fks_double, fks_single
      double precision, allocatable :: virt_wgts(:,:)
      double precision double, single, finite
      double precision born, virt_wgt
      double precision totmass
      logical calculatedborn
      common/ccalculatedborn/calculatedborn
      logical fksprefact
      parameter (fksprefact=.true.)
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg,symfactvirt
      integer ngluons,nquarks(-6:6),nphotons
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                  fkssymmetryfactorDeg,ngluons,nquarks,nphotons
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
cc
      include 'run.inc'
      include 'coupl.inc'
      include 'q_es.inc'
      integer nsqso,MLResArrayDim
      double precision pmass(nexternal), pmass_first(nexternal), pmass_rambo(100)
      integer nfail
      logical first_time
      data first_time/.TRUE./
      double precision tiny
      parameter (tiny = 1d-12)
      integer getordpowfromindex_ml5
      logical, allocatable, save :: keep_order(:)
      include 'orders.inc'
      logical is_aorg(nexternal)
      common /c_is_aorg/is_aorg
      logical force_polecheck, polecheck_passed
      common /to_polecheck/force_polecheck, polecheck_passed
      integer ret_code_ml
      common /to_ret_code/ret_code_ml

      double complex amp_split_ewsud(amp_split_size)
      common /to_amp_split_ewsud/ amp_split_ewsud
      double complex born_from_sborn_onehel(amp_split_size)

      double complex amp_split_ewsud_lsc(amp_split_size)
      common /to_amp_ewsud_lsc/amp_split_ewsud_lsc
      double complex amp_split_ewsud_ssc(amp_split_size)
      common /to_amp_ewsud_ssc/amp_split_ewsud_ssc
      double complex amp_split_ewsud_xxc(amp_split_size)
      common /to_amp_ewsud_xxc/amp_split_ewsud_xxc
      double precision amp_split_born(amp_split_size)
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PAR(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PAR/AMP_SPLIT_EWSUD_PAR
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_QCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_QCD/AMP_SPLIT_EWSUD_QCD
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_PARQCD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_EWSUD_PARQCD/AMP_SPLIT_EWSUD_PARQCD
      integer iamp
      integer chosen_hel, total_hel
       double complex amp_split_born_onehel(amp_split_size)
       common /to_amp_born_onehel/amp_split_born_onehel
       integer ewsud_helselect
       common/to_ewsud_helselect/ewsud_helselect
       INTEGER  SDK_GET_NCOMB
       external SDK_GET_NCOMB
       double complex BORN_HEL_MAX(amp_split_size), BORN_HEL(amp_split_size) 
       logical debug,deepdebug
       common/ew_debug/debug 
       double precision s,t,u,invm2_04
       external invm2_04
       INTEGER HELS(NEXTERNAL-1)
       double precision invarianti((NEXTERNAL-1)*(NEXTERNAL-2)/2)
       double precision invariantifirst((NEXTERNAL-1)*(NEXTERNAL-2)/2)
       double precision invariantiprevious((NEXTERNAL-1)*(NEXTERNAL-2)/2)
       logical   printinewsdkf
       common /to_printinewsdkf/printinewsdkf
       integer   deb_settozero
       common /to_deb_settozero/deb_settozero
       Integer sud_mod
       COMMON /to_sud_mod/ sud_mod
c from MAdLoop
       integer USERHEL
       COMMON/USERCHOICE/USERHEL       

       integer cs_run
       COMMON/to_cs_run/cs_run

       double complex smallL,bigL
       external smallL,bigL

       double precision F1t, F2t

c       double precision PREC_FOUND
c       integer RET_CODE

      INCLUDE 'nsqso_born.inc'
      INCLUDE 'nsquaredSO.inc'
c      INTEGER    NSQUAREDSO
c      PARAMETER (NSQUAREDSO=1)

c      INTEGER USERHEL
c      COMMON/USERCHOICE/USERHEL

       double precision PREC_FOUND(0:NSQUAREDSO)
       integer RET_CODE


      INTEGER ANS_DIMENSION
      PARAMETER(ANS_DIMENSION=MAX(NSQSO_BORN,NSQUAREDSO))

       REAL*8        virthel(0:3,0:ANS_DIMENSION), born_leadhel(0:ANS_DIMENSION),
     .               virt_leadhel(0:ANS_DIMENSION),sud_leadhel(0:ANS_DIMENSION),
     .               born_allhel(0:ANS_DIMENSION),
     .               virt_allhel(0:ANS_DIMENSION),sud_allhel(0:ANS_DIMENSION)      

      integer maximumtries, tries
      logical first_time_momenta
      double precision trimomsquared, energy_increase_factor,min_inv_frac,
     .                 tolerance_next_point,frac_lead_hel

      integer orders(nsplitorders), iampvirt(amp_split_size_born)

      double precision QCDlogs
c, logfromLOip1
      double precision print_loop_over_born, print_sud_over_born,
     .                 print_loopminussud_over_born, print_born,
     ,                 previous_print_loopminussud_over_born(0:ANS_DIMENSION),
     .                 previous_print_loopminussud_over_born_hel(0:ANS_DIMENSION,3**8),          
     .                 previous_loopminussud(0:ANS_DIMENSION), 
     .                 previous_loopminussud_hel(0:ANS_DIMENSION,3**8)
      double precision QES2_value, ren_scale_value
      logical use_QES2_value, use_ren_scale_value

      double precision error_inv, sprevious

      logical sud_mc_hel
      COMMON /to_mc_hel/ sud_mc_hel

      external alphas
      double precision alphas

C-----
C  BEGIN CODE
C-----  

      cs_run=.true.
      sud_mc_hel=.false.

      if (nexternal-1.gt.8) then
        print*, "redefine range of previous_print_loopminussud_over_born_hel,
     .  too many particles"
        stop
      endif

      energy=1d4
      
      QES2_value=1d4
      ren_scale_value=1d2
      use_QES2_value=.false.
      use_ren_scale_value=.false.




      tries=0
      maximumtries=20000000
      energy_increase_factor=2.5d0
c Set a number that is possible. E.g. > 0.5d0 for a 2->2 
      min_inv_frac=1d0/8d0
      tolerance_next_point=1d-3
      frac_lead_hel=1d-3

c Do not change deb_settozero here
      deb_settozero=0
      printinewsdkf=.False.
      
      debug=.True.
c      debug=.False

      deepdebug=.False.
c      deepdebug=.False.
      if(deepdebug) debug=.True.      

      first_time_momenta=.True.

      force_polecheck = .true.
      if (first_time) then

!!!!!!!!!!!!!!!!!!!!!!  turn off for  LO_ONLY !!!!!!!!!!!!!!!!!!!!!!!!!!!!

          call get_nsqso_loop(nsqso)          
          call get_answer_dimension(MLResArrayDim)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

          allocate(virt_wgts(0:3,0:MLResArrayDim))
          allocate(accuracies(0:nsqso))
          allocate(keep_order(nsqso))
          first_time = .false.
      endif

      call setrun                !Sets up run parameters
      call setpara('param_card.dat')   !Sets up couplings and masses
      call setcuts               !Sets up cuts and particle masses
      call printout              !Prints out a summary of paramaters
      call run_printout          !Prints out a summary of the run settings
      include 'pmass.inc'
     
      call FKSParamReader('FKS_params.dat',.TRUE.,.FALSE.)
      tolerance_default = IRPoleCheckThreshold
      iconfig=1
      ichan=1
      iconfigs(1)=iconfig
c     Set the energy to be characteristic of the run
      totmass = 0.0d0

      if (energy.le.0d0) then
        do i=1,nexternal
          totmass = totmass + pmass(i)
        enddo
        energy = max((ebeam(1)+ebeam(2))/4.0d0,2.0d0*totmass)
      endif

c     In check_sa: Set the renormalization scale to be of the order of sqrt(s) but
c     not equal to it so as to be sensitive to all logs in the check.
c     Here: QES2=energy**2 is mandatory, ren_scale?

c      ren_scale = energy !/2.0d0

c      QES2=QES2_value
c      ren_scale=ren_scale_value
c      QES2=energy**2

c      call sdk_test_functions()

      write(*,*)' Insert the number of points to test'
      read(*,*) npoints
      write(*,*)'Insert the relative tolerance'
      write(*,*)' A negative number will mean use the default one: ',
     1 tolerance_default 
      read(*,*) tolerance
      if (tolerance .le. zero) then
          tolerance = tolerance_default
      else
          IRPoleCheckThreshold = tolerance
      endif

      mu_r = ren_scale
c      qes2 = energy**2 

      do i = nincoming+1, nexternal-1
          pmass_rambo(i-nincoming) = pmass(i)
      enddo

      iconfig=1
      ichan=1
      iconfigs(1)=iconfig
c Find the nFKSprocess for which we compute the Born-like contributions,
c ie. which is a Born+g real-emission process
      do nFKSprocess=1,fks_configs
         call fks_inc_chooser()
         if (is_aorg(i_fks)) exit
      enddo
      if (nFKSprocess.gt.fks_configs) then
c If there is no fks_configuration that has a gluon or photon as i_fks
c (this might happen in case of initial state leptons with
c include_lepton_initiated_processes=False) the Born and virtuals do not
c need to be included, and we can simply quit the process.
         return
      endif
      call fks_inc_chooser()
      call leshouche_inc_chooser()
      call setfksfactor(.false.)
      symfactvirt = 1d0

      nfail = 0
      npointsChecked = 0

c Make sure that stability checks are always used by MadLoop, even for
c initialization



!!!!!!!!!!!!!!!!!!!!!!  turn off for  LO_ONLY !!!!!!!!!!!!!!!!!!!!!!!!!!!!

      CALL FORCE_STABILITY_CHECK(.TRUE.)
      CALL COLLIER_COMPUTE_UV_POLES(.TRUE.)
      CALL COLLIER_COMPUTE_IR_POLES(.TRUE.)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


200   continue


       
          finite=0d0
          single=0d0
          double=0d0
          calculatedborn = .false.

          if (.not.first_time_momenta) goto 201

          if (nincoming.eq.1) then
              call rambo(0, nexternal-nincoming-1, pmass(1), 
     1         pmass_rambo, prambo)
              p_born(0,1) = pmass(1)
              p_born(1,1) = 0d0
              p_born(2,1) = 0d0
              p_born(3,1) = 0d0
          elseif (nincoming.eq.2) then
              if (nexternal - nincoming - 1 .eq.1) then
                  ! deal with the case of only one particle in the final
                  ! state
                  p_born(0,1) = pmass(3)/2d0
                  p_born(1,1) = 0d0
                  p_born(2,1) = 0d0
                  p_born(3,1) = pmass(3)/2d0
                  if (pmass(1) > 0d0) 
     1               p_born(3,1) = dsqrt(pmass(3)**2/4d0 - pmass(1)**2)
                  p_born(0,2) = pmass(3)/2d0
                  p_born(1,2) = 0d0
                  p_born(2,2) = 0d0
                  p_born(3,2) = -pmass(3)/2d0
                  if (pmass(2) > 0d0) 
     1               p_born(3,2) = -dsqrt(pmass(3)**2/4d0 - pmass(1)**2)

                  prambo(0,1) = pmass(3)
                  prambo(1,1) = 0d0
                  prambo(2,1) = 0d0
                  prambo(3,1) = 0d0

              else
                    
                  call rambo(0, nexternal-nincoming-1, energy, 
     1             pmass_rambo, prambo)
                  p_born(0,1) = energy/2d0
                  p_born(1,1) = 0d0
                  p_born(2,1) = 0d0
                  p_born(3,1) = energy/2d0
                  if (pmass(1) > 0d0) 
     1               p_born(3,1) = dsqrt(energy**2/4d0 - pmass(1)**2)
                  p_born(0,2) = energy/2
                  p_born(1,2) = 0d0
                  p_born(2,2) = 0d0
                  p_born(3,2) = -energy/2d0
                  if (pmass(2) > 0d0) 
     1               p_born(3,2) = -dsqrt(energy**2/4d0 - pmass(1)**2)
              endif
          else
              write(*,*) 'INVALID NUMBER OF INCOMING PARTICLES', 
     1          nincoming
              stop
          endif

          do j = 0, 3
            do k = nincoming+1, nexternal-1
              p_born(j,k) = prambo(j,k-nincoming)
            enddo
          enddo



c----------

201   continue



      if(first_time_momenta) then
          do j = 0, 3
            do k = 1, nexternal-1
              p_born_first(j,k) = p_born(j,k)
              pmass_first(k)=pmass(k)
            enddo
          enddo
       else
          energy=0d0
          do k = nincoming+1, nexternal-1 
            trimomsquared=0d0
            do j = 1, 3
              p_born(j,k) = p_born_first(j,k)*energy_increase_factor
              trimomsquared=trimomsquared+p_born(j,k)**2
            enddo
            p_born(0,k)=dsqrt(pmass_first(k)**2+trimomsquared)
            energy=energy+p_born(0,k)
          enddo
                            p_born(0,1) = energy/2d0
                  p_born(1,1) = 0d0
                  p_born(2,1) = 0d0
                  p_born(3,1) = energy/2d0
                  if (pmass(1) > 0d0)
     1               p_born(3,1) = dsqrt(energy**2/4d0 - pmass(1)**2)
                  p_born(0,2) = energy/2
                  p_born(1,2) = 0d0
                  p_born(2,2) = 0d0
                  p_born(3,2) = -energy/2d0
                  if (pmass(2) > 0d0)
     1               p_born(3,2) = -dsqrt(energy**2/4d0 - pmass(1)**2)
 
          do j = 0, 3
            do k = 1, nexternal-1
              p_born_first(j,k) = p_born(j,k)
              pmass_first(k)=pmass(k)
            enddo
          enddo




      endif 





      s=invm2_04(p_born(0,1),p_born(0,2),1d0)
      if (nexternal-1.eq.4) then
        t=invm2_04(p_born(0,1),p_born(0,3),-1d0)
        u=invm2_04(p_born(0,1),p_born(0,4),-1d0)
      endif
      k=0

      error_inv=0d0

      do i=1,nexternal-1
       do j=i+1, nexternal-1
         k=k+1         
         if(i.le.2.and.j.ge.3) then
          invarianti(k)=invm2_04(p_born(0,i),p_born(0,j),-1d0)
          else
          invarianti(k)=invm2_04(p_born(0,i),p_born(0,j),1d0)
         endif       



         if (first_time_momenta) then

           invariantifirst(k)=invarianti(k)
           if(k.eq.1) then
             sprevious=invarianti(1)
           else
             invariantiprevious(k)=invarianti(k)
           endif
           
           if (abs(invm2_04(p_born(0,i),p_born(0,j),1d0)).lt.s*min_inv_frac) then
            tries=tries+1
            if (tries.gt.maximumtries) then
              write(*,*), "after doing more than ", maximumtries, "tries, the good first PS point was not found
     ."
              stop
            else
              goto 200
            endif
           endif

         else




           if (abs(invarianti(k)/s-invariantifirst(k)/invariantifirst(1)).gt.tolerance_next_point) then
              write(*,*), "A good similar PS point was not found, try to increase tolerance_next_point
     . or min_inv_frac"

              print*, "invarianti=",invarianti
              print*, "invariantifirst=",invariantifirst
           
              stop
           else
           write(*,*), "delta inv/s for ", k, "= ", abs(invarianti(k)/s-invariantiprevious(k)/invariantiprevious(1))

           error_inv=error_inv+(abs(invarianti(k)/s-invariantiprevious(k)/invariantiprevious(1)))**2


           if(k.eq.1) then
             sprevious=invarianti(1)
           else
             invariantiprevious(k)=invarianti(k)
           endif
          


     
 
           endif

         endif        
       



       enddo
      enddo   

      invariantiprevious(1)=sprevious

      error_inv=dsqrt(error_inv)
      if(.not.first_time_momenta) WRITE(75,*),
     .        "error_invariants from previous step =", error_inv


     
      if(first_time_momenta) then


        write(*,*), "After doing", tries, "tries, the good PS point was found
     ."
        tries=0

        OPEN(90, FILE='PS.input', ACTION='WRITE')
      
 
        do l=1,nexternal-1
      
         write (90,*) P_born(0,l),P_born(1,l),P_born(2,l),P_born(3,l)

         do k=0,3
          if(debug) WRITE (*,*) "p(",k,",",l,")=",p_born(k,l)
         enddo
          if(debug) WRITE (*,*) " "
        enddo

      CLOSE(90)




      endif



c----------

         OPEN(73, FILE='Sud_Approx.dat', ACTION='WRITE')
         write(73,*), "energy    ", "helicity     ", "loop/born     ",
     .            "sud/born     ", "(loop-sud)/born     ", "born     "

         OPEN(74, FILE='allhel_Sud_Approx.dat', ACTION='WRITE')
         write(74,*), "energy    ", "helicity     ", "loop/born     ",
     .            "sud/born     ", "(loop-sud)/born     ", "born     "

         OPEN(75, FILE='Deltas_Sud_Approx.dat', ACTION='WRITE')
         if(.not.first_time_momenta) write(75,*), "energy    ", "helicity     "
     .    , "loop-sud     ",
     .             "(loop-sud)/born     "


           if(use_ren_scale_value) then
             ren_scale = ren_scale_value
           else
             ren_scale = energy
           endif


           if(use_QES2_value) then
             QES2=QES2_value
           else
             QES2=energy**2
           endif

           mu_r = ren_scale


          g=sqrt(4d0*pi*alphas(sqrt(QES2)))
          call update_as_param()


          print*,"now g=",g,"and alphas=", g**2/4d0/pi,"at scale=",sqrt(QES2)





          



          total_hel=SDK_GET_NCOMB()
          chosen_hel=0
          EWSUD_HELSELECT=chosen_hel

          call sborn(p_born, born)
          amp_split_born(:) = amp_split(:)
          call sudakov_wrapper(p_born)
          call BinothLHA(p_born, born, virt_wgt)
          USERHEL=-1
          call SLOOPMATRIX_THRES(p_born,virthel,1d-3,PREC_FOUND
     $ ,RET_CODE)
          
          do iamp= 1, amp_split_size_born
             iampvirt(iamp)=0
          enddo

          

          do iamp=1,amp_split_size_born
            call amp_split_pos_to_orders(iamp, orders)
            if(debug) print*, "NSQUAREDSO=",NSQUAREDSO
            do j= 1, NSQUAREDSO
              if(orders(1).eq.GETORDPOWFROMINDEX_ML5(1, j).and.
     .         orders(2).eq.(GETORDPOWFROMINDEX_ML5(2, j)-2) ) then
                 iampvirt(iamp)=j
              endif
            enddo
            





            if (iampvirt(iamp).gt.0.and.iamp+1.le.amp_split_size_born) then
              call amp_split_pos_to_orders(iamp+1, orders)
              if(debug) print*, "born2 orders =",orders(1),  orders(2)
              if(debug) print*, "born2 =",virthel(0,iampvirt(iamp))
            else
              if(debug) print*, "born2 orders are not present"
            endif








          enddo

          do iamp = 1, amp_split_size_born
            if (amp_split_born(iamp).eq.0) cycle
              if(debug) then
                write(*,*) 'SUMMED OVER HELICITIES'
                write(*,*) 'SPLITORDER', iamp
                write(*,*) 'BORN: ', amp_split_born(iamp)
                write(*,*) 'SUDAKOV/BORN: LSC', amp_split_ewsud_lsc(iamp)/amp_split_born(iamp)
                write(*,*) 'SUDAKOV/BORN: SSC', amp_split_ewsud_ssc(iamp)/amp_split_born(iamp)
                write(*,*) 'SUDAKOV/BORN: XXC', amp_split_ewsud_xxc(iamp)/amp_split_born(iamp)
                write(*,*) 'SUDAKOV/BORN: PAR', AMP_SPLIT_EWSUD_PAR(iamp)/AMP_SPLIT_BORN(iamp)
                write(*,*) 'SUDAKOV/BORN: QCD', (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))/AMP_SPLIT_BORN(iamp)

              endif


              print_loop_over_born = dble(virthel(1,iampvirt(iamp))/AMP_SPLIT_BORN(iamp)/2d0 /4d0*4d0)

              print_sud_over_born  = dble((amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)
     .        +amp_split_ewsud_xxc(iamp)+AMP_SPLIT_EWSUD_PAR(iamp)
     .        +(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))/AMP_SPLIT_BORN(iamp))

              print_loopminussud_over_born = dble((virthel(1,iampvirt(iamp))/2d0 /4d0*4d0
     .        -(amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)+amp_split_ewsud_xxc(iamp)
     .        +AMP_SPLIT_EWSUD_PAR(iamp)+(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))))/
     .        AMP_SPLIT_BORN(iamp))

              print_born = dble(AMP_SPLIT_BORN(iamp))
             
            
              write(73,*), energy, "summed", print_loop_over_born, print_sud_over_born,
     .        print_loopminussud_over_born, print_born  

              if(.not.first_time_momenta) write(75,*),  energy, "summed",  
     .        print_loopminussud_over_born*AMP_SPLIT_BORN(iamp) - previous_loopminussud(iamp),
     .        print_loopminussud_over_born - previous_print_loopminussud_over_born(iamp)

              previous_print_loopminussud_over_born(iamp)=print_loopminussud_over_born
              previous_loopminussud(iamp)= print_loopminussud_over_born*AMP_SPLIT_BORN(iamp)
          enddo



          write(*,*) 'NOW ALL THE HELICITIES'
          do iamp = 1, amp_split_size_born
            BORN_HEL_MAX(iamp)= (0D0,0D0)
          enddo

          do chosen_hel=1,total_hel
             if(debug) write(*,*) 'HELICITY CONFIGURATION NUMBER ', chosen_hel
             EWSUD_HELSELECT=chosen_hel




             call sdk_get_hels(chosen_hel, hels)
             do iamp = 1, amp_split_size_born

         

               CALL SBORN_ONEHEL(P_born,hels(1),chosen_hel,born_hel)
               born_from_sborn_onehel(:)=amp_split_ewsud(:)




               if(deepdebug) then
                 if (born_from_sborn_onehel(iamp).eq.0d0) cycle
                 call SLOOPMATRIXHEL_THRES(p_born,chosen_hel,virthel,1d-3,PREC_FOUND
     $ ,RET_CODE)

                 call sudakov_wrapper(p_born)

                 if (chosen_hel.eq.1) then
                     born_allhel(iamp)=(0d0,0d0)
                     virt_allhel(iamp)=(0d0,0d0)
                     sud_allhel(iamp)=(0d0,0d0)
                 endif



                 born_allhel(iamp)=born_allhel(iamp)+AMP_SPLIT_BORN_ONEHEL(iamp)
                 virt_allhel(iamp)=virt_allhel(iamp)+virthel(1,iampvirt(iamp))/2d0 /4d0
                 sud_allhel(iamp)=sud_allhel(iamp)+(amp_split_ewsud_lsc(iamp)+
     .                            amp_split_ewsud_ssc(iamp)+
     .                            amp_split_ewsud_xxc(iamp)+
     .                            AMP_SPLIT_EWSUD_PAR(iamp)+
     .                            (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))

                 print_loop_over_born = dble(virthel(1,iampvirt(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/2d0 /4d0)

                 print_sud_over_born  = dble((amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)
     .            +amp_split_ewsud_xxc(iamp)+AMP_SPLIT_EWSUD_PAR(iamp)
     .            +(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))/AMP_SPLIT_BORN_ONEHEL(iamp))

                 print_loopminussud_over_born = dble((virthel(1,iampvirt(iamp))/2d0 /4d0
     .           -(amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)+amp_split_ewsud_xxc(iamp)
     .            +AMP_SPLIT_EWSUD_PAR(iamp)+(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))))/
     .            AMP_SPLIT_BORN_ONEHEL(iamp))

                 print_born = dble(AMP_SPLIT_BORN_ONEHEL(iamp))

                 write(74,*), energy, chosen_hel, print_loop_over_born, print_sud_over_born,
     .           print_loopminussud_over_born, print_born


                 if (born_allhel(iamp).ne.0d0.and.chosen_hel.eq.total_hel) then
                    write(73,*), energy,
     .              "all hel summed",
     .              dble(virt_allhel(iamp)/born_allhel(iamp)),
     .              dble(sud_allhel(iamp)/born_allhel(iamp)),
     .              dble((virt_allhel(iamp)-sud_allhel(iamp))/born_allhel(iamp)), born_allhel(iamp)

                 endif

               endif


              if ( born_from_sborn_onehel(iamp).eq.0) cycle
                
                 if(abs(BORN_HEL_MAX(iamp)).lt.abs(born_from_sborn_onehel(iamp))) then
                    BORN_HEL_MAX(iamp)=born_from_sborn_onehel(iamp)
                 endif


                 if(debug) then 
                   write(*,*) 'SPLITORDER', iamp
                   write(*,*) 'BORN: ', AMP_SPLIT_BORN_ONEHEL(iamp)

                   write(*,*) 'SUDAKOV/BORN: LSC', amp_split_ewsud_lsc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: SSC', amp_split_ewsud_ssc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: XXC', amp_split_ewsud_xxc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: PAR', AMP_SPLIT_EWSUD_PAR(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: QCD', (AMP_SPLIT_EWSUD_QCD(iamp+1)+
     .             AMP_SPLIT_EWSUD_PARQCD(iamp+1))/AMP_SPLIT_BORN_ONEHEL(iamp)

                   write(*,*) ' '

                 endif

             enddo
          enddo







          if(debug) write(*,*) 'NOW ONLY DOMINANT HELICITIES'
          if(debug) write(*,*) ''
       

          OPEN(70, FILE='Lead_Hel.dat', ACTION='WRITE') 
          OPEN(71, FILE='Born_Sud.dat', ACTION='WRITE')
          OPEN(72, FILE='NonDiag_structure.dat', ACTION='WRITE')




          WRITE (70,*) , invarianti
          WRITE (72,*) , invarianti
          WRITE (71,*) , s

          WRITE (70,*) , nexternal-1
          WRITE (71,*) , nexternal-1
          WRITE (72,*) , nexternal-1


          WRITE (70,*) , pdg_type
          WRITE (71,*) , pdg_type
          WRITE (72,*) , pdg_type



          do iamp = 1, amp_split_size_born
            if(debug) write(*,*) 'DOMINANT HELICITIES FOR iamp=',iamp
            if(debug) write(*,*) ''
            WRITE (70,*) , iamp       
            WRITE (71,*) , iamp
            WRITE (72,*) , iamp


                 write(71,*) AMP_SPLIT_BORN_ONEHEL(iamp)
                 write(71,*) (amp_split_ewsud_lsc(iamp)+
     .                        amp_split_ewsud_ssc(iamp)+amp_split_ewsud_xxc(iamp)+
     .                        AMP_SPLIT_EWSUD_PAR(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)

            born_leadhel(iamp)=(0d0,0d0)
            virt_leadhel(iamp)=(0d0,0d0)
            sud_leadhel(iamp)=(0d0,0d0)

            do chosen_hel=1,total_hel

c Change deb_settozero here if you need
c deb_settozero: if it is 0 keeps everything, otherwise only the following NON_DIAGONAL contributions are kept for each deb_settozero value
ccc             1      ---> all the SSC non_diagonal
ccc             10     ---> LSC non_diagonal
ccc             100    ---> xxC non_diagonal
ccc             111    ---> all non_diagonal

              deb_settozero=0

              printinewsdkf=.False.
              EWSUD_HELSELECT=chosen_hel
              call sdk_get_hels(chosen_hel, hels)
              CALL SBORN_ONEHEL(P_born,hels(1),chosen_hel,born_hel)
              born_from_sborn_onehel(:)=amp_split_ewsud(:)








              if(debug) print*,"look into hel number",chosen_hel,
     .        "It is ",
     .         abs(born_from_sborn_onehel(iamp))/abs(BORN_HEL_MAX(iamp)),
     .        "of BORN_HEL_MAX"

              if (abs(BORN_HEL_MAX(iamp)).NE.0d0  
     .        .AND.    abs(born_from_sborn_onehel(iamp)).GT.frac_lead_hel*abs(BORN_HEL_MAX(iamp))) 
     .             then 




                    if (debug) printinewsdkf=.True. 
                    if(debug) write(*,*) 'HEL LEADCONF =',chosen_hel 
                    if(debug) write(*,*) '    '
                    call sudakov_wrapper(p_born) 


                    if(debug.and.nexternal.eq.5) 
     .              write(*,*),'t= ', t, "u =", u, "(t/u)=" , (t/u)

                    if(deb_settozero.eq.1.or.deb_settozero.eq.111.and.debug) then
                      write(*,*) 'ls SSC Non diag-->',
     .                (amp_split_ewsud_ssc(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/(smallL(s))

                    endif

                    
                    if(deb_settozero.eq.10.or.deb_settozero.eq.111.and.debug) then
                      write(*,*) 'LSC Non diag-->',
     .                (amp_split_ewsud_lsc(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/(bigL(s))
                    endif

                    if(deb_settozero.eq.100.or.deb_settozero.eq.111.and.debug) then
                      write(*,*) 'xxC Non diag-->',
     .                (amp_split_ewsud_xxc(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/(smallL(s))
                    endif



                 if(debug)  write(*,*) '    '
           
                    write(*,*) 'BORN for HEL LEADCONF ',chosen_hel,' = ', AMP_SPLIT_BORN_ONEHEL(iamp)

                 if(deb_settozero.ne.0.and.debug) write(*,*) ' !!!!!!SUDAKOV/BORN!!!! is !!!! wrong !!!.
     .                     You have to set deb_settozero to zero'

                 if(debug) then
                   write(*,*) 'SUDAKOV/BORN: LSC', amp_split_ewsud_lsc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: SSC', amp_split_ewsud_ssc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: XXC', amp_split_ewsud_xxc(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: PAR', AMP_SPLIT_EWSUD_PAR(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)
                   write(*,*) 'SUDAKOV/BORN: QCD', (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))/AMP_SPLIT_BORN(iamp)

                   write(*,*) 'PAR/ls', AMP_SPLIT_EWSUD_PAR(iamp)/AMP_SPLIT_BORN_ONEHEL(iamp)/smallL(s)


                   write(*,*) '     '
                   write(*,*) 'SUDAKOV/BORN for HEL CONF ',chosen_hel,
     .           ' = ',(amp_split_ewsud_lsc(iamp)+
     .                  amp_split_ewsud_ssc(iamp)+
     .                  amp_split_ewsud_xxc(iamp)+
     .                  AMP_SPLIT_EWSUD_PAR(iamp)+
     .                  (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))
                 endif


                 call sdk_get_hels(chosen_hel, hels)

                 WRITE (70,*) , hels,
     .            dble((amp_split_ewsud_lsc(iamp)+
     .                  amp_split_ewsud_ssc(iamp)+
     .                  amp_split_ewsud_xxc(iamp)+
     .                  AMP_SPLIT_EWSUD_PAR(iamp)+
     .                  (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))))/
     .            dble(AMP_SPLIT_BORN_ONEHEL(iamp))


                 call SLOOPMATRIXHEL_THRES(p_born,chosen_hel,virthel,1d-3,PREC_FOUND
     $ ,RET_CODE)


                 born_leadhel(iamp)=born_leadhel(iamp)+AMP_SPLIT_BORN_ONEHEL(iamp)
                 virt_leadhel(iamp)=virt_leadhel(iamp)+virthel(1,iampvirt(iamp))/2d0 /4d0   
                 sud_leadhel(iamp)=sud_leadhel(iamp)+(amp_split_ewsud_lsc(iamp)+
     .                  amp_split_ewsud_ssc(iamp)+
     .                  amp_split_ewsud_xxc(iamp)+
     .                  AMP_SPLIT_EWSUD_PAR(iamp)+
     .                  (AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))

                 if(debug) print*, 
     .            'virthel/born/2  and / 4  for HEL LEADCONF ',
     .            chosen_hel,' = ', virthel(1,iampvirt(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/2d0 /4d0
                 if(debug) write(*,*) '     '
                 if(debug) print*, 'PREC_FOUND=', PREC_FOUND, 
     .            'RET_CODE=', RET_CODE

                

                 if(debug) write(*,*) '    '


                  print_loop_over_born = dble(virthel(1,iampvirt(iamp))/AMP_SPLIT_BORN_ONEHEL(iamp)/2d0 /4d0)

                  print_sud_over_born  = dble((amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)
     .            +amp_split_ewsud_xxc(iamp)+AMP_SPLIT_EWSUD_PAR(iamp)
     .            +(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1)))/AMP_SPLIT_BORN_ONEHEL(iamp))

                  print_loopminussud_over_born = dble((virthel(1,iampvirt(iamp))/2d0 /4d0
     .            -(amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)+amp_split_ewsud_xxc(iamp)
     .            +AMP_SPLIT_EWSUD_PAR(iamp)+(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))))/
     .            AMP_SPLIT_BORN_ONEHEL(iamp))

                  print_born = dble(AMP_SPLIT_BORN_ONEHEL(iamp))
             
                  write(73,*), energy, chosen_hel, print_loop_over_born, print_sud_over_born,
     .            print_loopminussud_over_born, print_born  


                  if(deepdebug) write(73,*), energy, chosen_hel,
     .             "(loop-sud/born2)", 
     .            dble((virthel(1,iampvirt(iamp))/2d0 /4d0
     .            -(amp_split_ewsud_lsc(iamp)+amp_split_ewsud_ssc(iamp)+amp_split_ewsud_xxc(iamp)
     .            +AMP_SPLIT_EWSUD_PAR(iamp)+(AMP_SPLIT_EWSUD_QCD(iamp+1)+AMP_SPLIT_EWSUD_PARQCD(iamp+1))))/
     .            virthel(0,iampvirt(iamp))),
     .             dble(virthel(0,iampvirt(iamp)))


                  if(deb_settozero.ne.0) write(73,*), "Set deb_settozero to zero if you want sensible resutls here
     ."
                  if(sud_mod.ne.2) write(73,*), "Set sud_mod to 2 in ewsudakov_functions.f if you want sensible resutls here
     ."

                  if(.not.first_time_momenta) write(75,*),  energy, chosen_hel,
     .            print_loopminussud_over_born * AMP_SPLIT_BORN_ONEHEL(iamp) - 
     .            previous_loopminussud_hel(iamp,chosen_hel),
     .            print_loopminussud_over_born - previous_print_loopminussud_over_born_hel(chosen_hel,iamp)

                  previous_print_loopminussud_over_born_hel(chosen_hel,iamp) = print_loopminussud_over_born
                  previous_loopminussud_hel(iamp,chosen_hel)= print_loopminussud_over_born 
     .            * AMP_SPLIT_BORN_ONEHEL(iamp)

              endif
            enddo

            if (born_leadhel(iamp).ne.0d0) then 
              write(73,*), energy, 
     .        "lead-hel-summed",
     .         dble(virt_leadhel(iamp)/born_leadhel(iamp)),
     .         dble(sud_leadhel(iamp)/born_leadhel(iamp)),  
     .         dble((virt_leadhel(iamp)-sud_leadhel(iamp))/born_leadhel(iamp)), born_leadhel(iamp)

            endif



          enddo

          first_time_momenta=.False.

          CLOSE(70)
          CLOSE(71)
          CLOSE(72)










          ! extra initialisation calls: skip the first point
          ! as well as any other points which is used for initialization
          ! (according to the return code)






          if(debug) write(*,*) 'I go to the next point'
c          return    


          if (npointsChecked.eq.0) then
             if (mod(ret_code_ml,100)/10.eq.3 .or.
     &            mod(ret_code_ml,100)/10.eq.4) then
              ! this is to skip initialisation points
                write(*,*) 'INITIALIZATION POINT.'
                write(*,*)
     $               'RESULTS FROM INITIALIZATION POINTS WILL NOT '/
     $               /'BE USED FOR STATISTICS'
                goto 200
             endif
          endif
          write(*,*) 'MU_R    = ', ren_scale
          write(*,*) 'ALPHA_S = ', G**2/4d0/pi
C         Otherwise, perform the check
          npointsChecked = npointsChecked +1

          do j = 0, 3
            do k = 1, nexternal - 1
              p(j,k) = p_born(j,k)
            enddo
            p(j, nexternal) = 0d0
          enddo

          if ( tolerance.lt.0.0d0 ) then
               write(*,*) 'PASSED', tolerance
          else
              if (polecheck_passed) then
                write(*,*) 'PASSED', tolerance
              else
                write(*,*) 'FAILED', tolerance
                nfail=nfail+1
              endif
          endif
          write(*,*)

!modify here to increment energy next point
c          energy=energy
c          ren_scale = energy!/2.0d0
c          QES2=100d0!energy**2

      if (npointsChecked.lt.npoints) goto 200 

          write(*,*) 'NUMBER OF POINTS PASSING THE CHECK', 
     1     npoints - nfail
          write(*,*) 'NUMBER OF POINTS FAILING THE CHECK', 
     1     nfail
          write(*,*) 'TOLERANCE ', tolerance

      return
      end



      SUBROUTINE RAMBO(LFLAG,N,ET,XM,P)
c------------------------------------------------------
c
c                       RAMBO
c
c    RA(NDOM)  M(OMENTA)  B(EAUTIFULLY)  O(RGANIZED)
c
c    A DEMOCRATIC MULTI-PARTICLE PHASE SPACE GENERATOR
c    AUTHORS:  S.D. ELLIS,  R. KLEISS,  W.J. STIRLING
c    THIS IS VERSION 1.0 -  WRITTEN BY R. KLEISS
c    (MODIFIED BY R. PITTAU)
c
c                INPUT                 OUTPUT
c
c    LFLAG= 0:   N, ET, XM             P, (DJ)
c    LFLAG= 1:   N, ET, XM, P          (DJ)
c
c    N  = NUMBER OF PARTICLES (>1, IN THIS VERSION <101)
c    ET = TOTAL CENTRE-OF-MASS ENERGY
c    XM = PARTICLE MASSES ( DIM=100 )
c    P  = PARTICLE MOMENTA ( DIM=(4,100) )
c    DJ = 1/(WEIGHT OF THE EVENT)
c
c------------------------------------------------------
      IMPLICIT DOUBLE PRECISION(A-H,O-Z)
      DIMENSION XM(100),P(0:3,100),Q(4,100),Z(100),R(4),
     .   B(3),P2(100),XM2(100),E(100),V(100),IWARN(5)
      SAVE ACC,ITMAX,IBEGIN,IWARN,Z,TWOPI,PO2LOG
      DATA ACC/1.D-14/,ITMAX/10/,IBEGIN/0/,IWARN/5*0/
C
C INITIALIZATION STEP: FACTORIALS FOR THE PHASE SPACE WEIGHT
      IF(IBEGIN.NE.0) GOTO 103
      IBEGIN=1
      TWOPI=8.*DATAN(1.D0)
      PO2LOG=LOG(TWOPI/4.)
      Z(2)=PO2LOG
      DO 101 K=3,100
  101 Z(K)=Z(K-1)+PO2LOG-2.*LOG(DFLOAT(K-2))
      DO 102 K=3,100
  102 Z(K)=(Z(K)-LOG(DFLOAT(K-1)))
C
C CHECK ON THE NUMBER OF PARTICLES
  103 IF(N.GT.1.AND.N.LT.101) GOTO 104
      PRINT 1001,N
      STOP
C
C CHECK WHETHER TOTAL ENERGY IS SUFFICIENT; COUNT NONZERO MASSES
  104 XMT=0.
      NM=0
      DO 105 I=1,N
      IF(XM(I).NE.0.D0) NM=NM+1
  105 XMT=XMT+ABS(XM(I))
      IF(XMT.LE.ET) GOTO 201
      PRINT 1002,XMT,ET
      STOP

  201 CONTINUE 
      if (lflag.eq.1) then
        w0= exp((2.*N-4.)*LOG(ET)+Z(N))
        do j= 1,N
          v(j)= sqrt(p(1,j)**2+p(2,j)**2+p(3,j)**2)
        enddo

        a1= 0.d0
        a3= 0.d0
        a2= 1.d0
        do j= 1,N
          a1= a1+v(j)/ET
          a2= a2*v(j)/p(0,j)
          a3= a3+v(j)*v(j)/p(0,j)/ET
        enddo
        wm= a1**(2*N-3)*a2/a3
        dj= 1.d0/w0/wm
        return
      endif
C
C THE PARAMETER VALUES ARE NOW ACCEPTED
C
C GENERATE N MASSLESS MOMENTA IN INFINITE PHASE SPACE

      DO 202 I=1,N
      call rans(RAN1)
      call rans(RAN2)
      call rans(RAN3)
      call rans(RAN4)
      C=2.*RAN1-1.
      S=SQRT(1.-C*C)
      F=TWOPI*RAN2
      Q(4,I)=-LOG(RAN3*RAN4)
      Q(3,I)=Q(4,I)*C
      Q(2,I)=Q(4,I)*S*COS(F)
  202 Q(1,I)=Q(4,I)*S*SIN(F)
C
C CALCULATE THE PARAMETERS OF THE CONFORMAL TRANSFORMATION
      DO 203 I=1,4
  203 R(I)=0.
      DO 204 I=1,N
      DO 204 K=1,4
  204 R(K)=R(K)+Q(K,I)
      RMAS=SQRT(R(4)**2-R(3)**2-R(2)**2-R(1)**2)
      DO 205 K=1,3
  205 B(K)=-R(K)/RMAS
      G=R(4)/RMAS
      A=1./(1.+G)
      X=ET/RMAS
C
C TRANSFORM THE Q'S CONFORMALLY INTO THE P'S
      DO 207 I=1,N
      BQ=B(1)*Q(1,I)+B(2)*Q(2,I)+B(3)*Q(3,I)
      DO 206 K=1,3
  206 P(K,I)=X*(Q(K,I)+B(K)*(Q(4,I)+A*BQ))
  207 P(0,I)=X*(G*Q(4,I)+BQ)
C
C CALCULATE WEIGHT AND POSSIBLE WARNINGS
      WT=PO2LOG
      IF(N.NE.2) WT=(2.*N-4.)*LOG(ET)+Z(N)
      IF(WT.GE.-180.D0) GOTO 208
      IF(IWARN(1).LE.5) PRINT 1004,WT
      IWARN(1)=IWARN(1)+1
  208 IF(WT.LE. 174.D0) GOTO 209
      IF(IWARN(2).LE.5) PRINT 1005,WT
      IWARN(2)=IWARN(2)+1
C
C RETURN FOR WEIGHTED MASSLESS MOMENTA
  209 IF(NM.NE.0) GOTO 210
      WT=EXP(WT)
      DJ= 1.d0/WT
      RETURN
C
C MASSIVE PARTICLES: RESCALE THE MOMENTA BY A FACTOR X
  210 XMAX=SQRT(1.-(XMT/ET)**2)
      DO 301 I=1,N
      XM2(I)=XM(I)**2
  301 P2(I)=P(0,I)**2
      ITER=0
      X=XMAX
      ACCU=ET*ACC
  302 F0=-ET
      G0=0.
      X2=X*X
      DO 303 I=1,N
      E(I)=SQRT(XM2(I)+X2*P2(I))
      F0=F0+E(I)
  303 G0=G0+P2(I)/E(I)
      IF(ABS(F0).LE.ACCU) GOTO 305
      ITER=ITER+1
      IF(ITER.LE.ITMAX) GOTO 304
      PRINT 1006,ITMAX
      GOTO 305
  304 X=X-F0/(X*G0)
      GOTO 302
  305 DO 307 I=1,N
      V(I)=X*P(0,I)
      DO 306 K=1,3
  306 P(K,I)=X*P(K,I)
  307 P(0,I)=E(I)
C
C CALCULATE THE MASS-EFFECT WEIGHT FACTOR
      WT2=1.
      WT3=0.
      DO 308 I=1,N
      WT2=WT2*V(I)/E(I)
  308 WT3=WT3+V(I)**2/E(I)
      WTM=(2.*N-3.)*LOG(X)+LOG(WT2/WT3*ET)
C
C RETURN FOR  WEIGHTED MASSIVE MOMENTA
      WT=WT+WTM
      IF(WT.GE.-180.D0) GOTO 309
      IF(IWARN(3).LE.5) PRINT 1004,WT
      IWARN(3)=IWARN(3)+1
  309 IF(WT.LE. 174.D0) GOTO 310
      IF(IWARN(4).LE.5) PRINT 1005,WT
      IWARN(4)=IWARN(4)+1
  310 WT=EXP(WT)
      DJ= 1.d0/WT
      RETURN
C
 1001 FORMAT(' RAMBO FAILS: # OF PARTICLES =',I5,' IS NOT ALLOWED')
 1002 FORMAT(' RAMBO FAILS: TOTAL MASS =',D15.6,' IS NOT',
     . ' SMALLER THAN TOTAL ENERGY =',D15.6)
 1004 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY UNDERFLOW')
 1005 FORMAT(' RAMBO WARNS: WEIGHT = EXP(',F20.9,') MAY  OVERFLOW')
 1006 FORMAT(' RAMBO WARNS:',I3,' ITERATIONS DID NOT GIVE THE',
     . ' DESIRED ACCURACY =',D15.6)
      END


      subroutine rans(rand)
c     Just a wrapper to ran2      
      implicit none
      double precision rand, ran2
      rand = ran2()
      return 
      end

