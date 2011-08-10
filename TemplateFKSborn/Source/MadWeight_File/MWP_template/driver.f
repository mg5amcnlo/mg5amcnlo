      PROGRAM DRIVER
C**************************************************************************
C     THIS IS THE DRIVER FOR THE WHOLE CALCULATION
C
C**************************************************************************
C**************************************************************************
C
C    Structure of the driver
C
C       1) initialisation
C       2) first loop
C           2.1) loop on permutation
C           2.2) loop on configuration
C       3) second loop
C           3.1) loop on permutation
C           3.2) loop on configuration
C           3.3) write result
C          
C**************************************************************************
      IMPLICIT NONE
C
      include 'genps.inc'
      include 'nexternal.inc'
      include 'data.inc'
      include 'madweight_param.inc'
      include 'driver.inc'
C
C     PARAMETERS
C
C
C     LOCAL (Variable of integration)
C
      DOUBLE PRECISION SD,CHI,CROSS
      INTEGER I,convergence_status
      integer pos,channel_pos,ll
      double precision temp_err, temp_val
      double precision order_value(max_channel)
      double precision order_error(max_channel)
c
c     variable for permutation, channel
c
      integer perm_id(nexternal-2)     !permutation of 1,2,...,nexternal-2
      integer num_per                  !total number of permutations
      double precision weight,weight_error,final
c
c     variable for selecting perm in loop 1
c
      double precision minimal_value1,minimal_value2
      double precision best_precision, chan_val, chan_err
      double precision check_value
      integer loop_index,counter
c      integer best_config
c
c     logical
c
      logical debug

C
C     GLOBAL     (BLOCK IN COMMON WITH VEGAS)
C
      DOUBLE PRECISION Xl(20),XU(20),ACC
      INTEGER NDIM,NCALL,ITMX,NPRN
      COMMON/BVEG1/XL,XU,ACC, NDIM,NCALL,ITMX,NPRN
      Double precision calls,ti,tsi
      COMMON/BVEG4/calls,ti,tsi
      Double precision ALPH
      integer NDMX,MDS
      COMMON/BVEG3/ALPH,NDMX,MDS

C
C     Global  number of dimensions
C
      integer Ndimens
      common /to_dimension/ Ndimens
c
c      external
c
      double precision fct
      external  fct

      integer          iseed
      common /to_seed/ iseed

C**************************************************************************
C    step 1: initialisation     
C**************************************************************************
      call global_init

      if(use_perm) then
         call get_num_per(num_per) !calculate the number of permutations.
      else
         num_per=1
      endif

      if (num_per*nb_sol_config.gt.max_channel) then
        write(*,*) "warning: too many channels => increase max_channel"
        stop
      endif

c     VEGAS parameters for the first loop
      ACC = final_prec
      NPRN = 1
      NCALL = nevents
      ITMX = max_it_step1

c     initialization of the variables in the loops
      check_value=0d0
      loop_index=0
      counter=0
      do perm_pos=1,num_per
       do ll=1,nb_sol_config
       counter=counter+1
       order_value(counter)=1d5
       enddo
      enddo
C
C     initialization of the permutation
C
 1    call init_perm()
      do i=1,nexternal-2
         perm_id(i)=i
      enddo

      loop_index=loop_index+1
      counter=0
      temp_err=0d0
      temp_val=0d0
      do perm_pos=1,num_per
         call give_permut(perm_id)
         call assign_perm(perm_id)
         chan_val=0d0
         chan_err=0d0
         do ll=1,nb_sol_config
            counter=counter+1
            config_pos=ll
            write(*,*) "Now, channel=",config_pos
            write(*,*) "     permutation=",perm_pos

      iseed = iseed + 1 ! avoid to have the same seed
      call initialize
      NDIM=Ndimens
c
      write(*,*) NDIM,NCALL,ITMX
            write(*,*) order_value(counter), check_value
            if(order_value(counter).gt.check_value) then
            CALL VEGAS(fct,CROSS,SD,CHI)
              if (CROSS.lt.1d99) then
            temp_val=temp_val+cross
            chan_val=chan_val+cross
            temp_err=temp_err+SD
            chan_err=chan_err+SD
            order_value(counter) = cross
            order_error(counter) = sd
            if (histo) call histo_combine_iteration(counter)
               else
              order_value(counter)=0d0
              endif
            endif
         enddo
         write(32,*) perm_pos,chan_val, chan_err
      enddo

      temp_val=temp_val/dble(num_per)
      temp_err=temp_err/dble(num_per)
      check_value=temp_val/(nb_sol_config) * min_prec_cut1
      if (loop_index.eq.1) then
        NCALL = nevents
        ITMX = max_it_step2
        goto 1
      endif

      write(*,*) "the weight is",temp_val,"+/-",temp_err

      OPEN(UNIT=21,FILE='./weights.out',STATUS='UNKNOWN')
      write(21,*) temp_val,'  ',temp_err
      close(21)

      write(23,*) ' permutation channel   value      error'
      call init_perm()
      channel_pos=0
      do perm_pos=1,num_per
         call give_permut(perm_id)
         write(23,*) "======================================"
         write(23,*) perm_id
         do ll=1,nb_sol_config
            counter=counter+1
            write(23,*) perm_pos,' || ',ll,' || ',
     & order_value(counter),
     &           ' ||', order_error(counter)
         enddo
      enddo
      write(23,*) "======================================"
      write(23,*)'Weight: ',temp_val,'+-', temp_err
      write(23,*) "======================================"

C**************************************************************************                  |
C     write histogram file (not by default)                                           |
C**************************************************************************                  |C**************************************************************************************      
      if (histo)   call histo_final(num_per*nb_sol_config)  



      close(21)
      OPEN(UNIT=21,FILE='./stop',STATUS='UNKNOWN')
      close(21)



      END


C**************************************************************************************
        double precision function fct(x,wgt)
        implicit none

        include 'genps.inc'
        include 'nexternal.inc'
        include 'run.inc'
        include 'driver.inc'
c
c       this is the function which is called by the integrator
c
c       arguments
c
        double precision x(20),wgt
c
c       local
c
c        integer i,j ! debug mode
c
c       global
c
        double precision              S,X1,X2,PSWGT,JAC
        common /PHASESPACE/ S,X1,X2,PSWGT,JAC
        double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
        double precision mvir2(-max_branch:2*nexternal)                  ! records the sq invariant masses of intermediate particles (MG order)
        common /to_diagram_kin/ momenta, mvir2


        logical histo
        common /to_histo/histo
c
c       external
c
        double precision dsig
        external dsig
        include 'data.inc'

         if (x(1).eq.0) then
         write(*,*) "x1", x(1)
         stop
         endif
         call get_PS_point(x)
         if (jac.gt.0d0) then
c         write(*,*) "passed"
           
         fct=jac
         xbk(1)=X1
         xbk(2)=X2
         fct=fct*dsig(momenta(0,1),wgt)
         else
         fct=0d0
         return
         endif
         
         if (histo)  then
            call FILL_plot(fct,wgt,perm_pos*nb_sol_config+config_pos)
         endif
         end


C**************************************************************************************
      subroutine global_init
      implicit none
C
C     initialize all global variable
C
      include 'genps.inc'
      include 'nexternal.inc'
c
c     LHCO input
c
      integer tag_init(3:nexternal),type(nexternal),run_number,trigger
      double precision eta_init(nexternal),phi_init(nexternal),
     &pt_init(nexternal),j_mass(nexternal),ntrk(nexternal),
     &btag(nexternal),had_em(nexternal),dummy1(nexternal),
     &dummy2(nexternal)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger

      integer MG,j,k,i

      integer matching_type_part(3:nexternal) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:nexternal)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part

C     topol info
      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,
     +        num_ta,num_ata   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,
     & num_mu,num_amu,num_ta,num_ata !particle in the final state

      double precision pexp_init(0:3,nexternal)  !impulsion in original configuration
      common/to_pexp_init/pexp_init

      character*60 param_name

      double precision pmass(nexternal)
      common/to_mass/pmass
      double precision momenta(0:3,-max_branch:2*nexternal)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branch:2*nexternal)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2

      integer met_lhco,opt_lhco
      common/LHCO_met_tag/met_lhco,opt_lhco

      integer nevents
      common/number_integration_points/nevents
c
c     var to control histo
c
      logical histo
      common /to_histo/histo

      integer          isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel,multi_channel

      multi_channel=.false.

      open(unit=90,file="./start")
      write(90,*) 'start'
      close(90)
      open(UNIT=32,FILE='vegas_value.out',STATUS='unknown')
c
c     set parameters for the run
c
      CALL MW_SETRUN
      open(unit=89,file="./param.dat")
      read(89,*) param_name
c      read(89,*) nevents
      close(89)
      write(*,*) 'param name',param_name
      call setpara(param_name,.true.)
      write(*,*) 'number of points',nevents
c
c    set cuts
c
      CALL SETCUTS
c
c     deternime the final state content
c
      call info_final_part()                       !init variable concerning outut particle (num/type)+ give order
c
c     set parameters for the transfer functions
c
c      write(*,*) 'test 1'
      call setTF
c      write(*,*) 'test 2'
      call init_d_assignement()
      CALL PRINTOUT
      CALL RUN_PRINTOUT
c      call graph_init

      write(*,*) "read verif"
      OPEN(UNIT=24,file='./verif.lhco',status='old',err=48) ! input file

      read(24,*,err=48,end=48) k,run_number,trigger !line with tag 0
      do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         read(24,*,err=48,end=48) k,type(k),eta_init(k),
     &           phi_init(k),pt_init(k),j_mass(k),ntrk(k),btag(k),
     &           had_em(k) ,dummy1(k),dummy2(k)

         call four_momentum_set2(eta_init(k),phi_init(k),pt_init(k),
     &         j_mass(k),pexp_init(0,MG))
         tag_init(MG)=k
      enddo
      read(24,*,err=48,end=48) k,type(k),eta_init(k), !line with type 6
     &           phi_init(k),pt_init(k),j_mass(k),ntrk(k),btag(k),
     &           had_em(k) ,dummy1(k),dummy2(k)
      read(24,*,err=47,end=47) k,type(k),eta_init(k), !line with type 7 (optional)        
     &           phi_init(k),pt_init(k),j_mass(k),ntrk(k),btag(k),  
     &           had_em(k) ,dummy1(k),dummy2(k)
      opt_lhco=k
      close(24)
 47   write(*,*) "read experimental data:"
      write(*,*) "-----------------------"
      do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         write(*,*) (pexp_init(i,MG),i=0,3)
      enddo
c      call set_pmass ! done in setcuts
      do i=3,nexternal
         mvir2(i)=pmass(i)**2
      enddo
c      write(*,*) 'pmass ok'
      return
 48   stop
      end

C**************************************************************************************
C      subroutine save_grid(perm_pos,config)
C
C
C
C      integer perm_pos,config
C      character*11 buffer
C
C     GLOBAL     (BLOCK IN COMMON WITH VEGAS)
C
C      DOUBLE PRECISION Xl(20),XU(20),ACC
C      INTEGER NDIM,NCALL,ITMX,NPRN
C      COMMON/BVEG1/XL,XU,ACC, NDIM,NCALL,ITMX,NPRN
C      DOUBLE PRECISION              S,X1,X2,PSWGT,JAC
C      COMMON /PHASESPACE/ S,X1,X2,PSWGT,JAC
C      integer*4 it,ndo
C      double precision xi(50,20),si,si2,swgt,schi
C      common/bveg2/xi,si,si2,swgt,schi,ndo,it   

C      buffer='grid_00_000'
C      if (config.ge.10)then
C         write(buffer(6:7),'(I2)') config
C      else
C         write(buffer(7:7),'(I1)') config
C      endif
C      if (perm_pos.ge.100)then
C         write(buffer(9:11),'(i3)')perm_pos
C      elseif(perm_pos.ge.10)then
C         write(buffer(10:11),'(i2)')perm_pos
C      else
C         write(buffer(11:11),'(i1)')perm_pos
C      endif
C         write(*,*) buffer         

C      open(unit=88,file=buffer)
C      write(88,100)ndo
C      do i=1,ndo
C         write(88,101) (xi(i,j),j=1,20)
C      enddo
C      close(88)



C 100  format(5X,I5)
C 101  format(5x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5)
C 102  format(A5,I2,A1,I3)
c      end


C**************************************************************************************
C      subroutine read_grid(perm_pos,config)
C
C
C
C      integer perm_pos,config
C      character*20 buffer
C
C     GLOBAL     (BLOCK IN COMMON WITH VEGAS)
C
C      DOUBLE PRECISION Xl(20),XU(20),ACC
C      INTEGER NDIM,NCALL,ITMX,NPRN
C      COMMON/BVEG1/XL,XU,ACC, NDIM,NCALL,ITMX,NPRN
C      DOUBLE PRECISION              S,X1,X2,PSWGT,JAC
C      COMMON /PHASESPACE/ S,X1,X2,PSWGT,JAC
C      integer*4 it,ndo
C      double precision xi(50,20),si,si2,swgt,schi
C      common/bveg2/xi,si,si2,swgt,schi,ndo,it   

C      buffer='grid_00_000'
C      if (config.ge.10)then
C         write(buffer(6:7),'(I2)') config
C      else
C         write(buffer(7:7),'(I1)') config
C      endif
C      if (perm_pos.ge.100)then
C         write(buffer(9:11),'(i3)')perm_pos
C      elseif(perm_pos.ge.10)then
C         write(buffer(10:11),'(i2)')perm_pos
C      else
C         write(buffer(11:11),'(i1)')perm_pos
C      endif
C         write(*,*) buffer         
C      open(unit=88,file=buffer)
C      READ(88,100) ndo
C      do i=1,config
C         READ(88,101)(xi(i,j),j=1,20)
C      enddo
C      close(88) 


C 100  format(5X,I5)
C 101  format(5x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,
C     &       2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5,2x,E11.5)
C 102  format(A5,I2,A1,I3)
C
C      end
C**************************************************************************************
