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
      include 'phasespace.inc'
      include 'nexternal.inc'
      include 'data.inc'
      include 'permutation.inc'
      include 'madweight_param.inc'
      double precision perm_init_weight(NPERM)
      common/perm_init_weight/perm_init_weight
C
C     LOCAL (Variable of integration)
C
      DOUBLE PRECISION SD,CHI,CROSS
      INTEGER I,J,K,convergence_status
      integer pos,channel_pos
      double precision temp_err, temp_val
      double precision order_value(nb_channel)
      double precision order_error(nb_channel)
      double precision xi_by_channel(100, 20, nb_channel)
      integer ndo_by_channel(nb_channel)

      double precision normalize_perm
c
c     variable for permutation, channel
c
      integer perm_id(nexternal-2)     !permutation of 1,2,...,nexternal-2
      double precision weight,weight_error,final
      integer matching_type_part(3:max_particles) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part, inv_matching_type_part
c
c     variable for selecting perm in loop 1
c
      double precision minimal_value1,minimal_value2
      double precision best_precision
      double precision check_value
      integer loop_index
c      integer best_config
c
c     logical
c
      logical debug
      integer config_step

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
      double precision fct,fct_mint
      external  fct,fct_mint

      integer          iseed
      common /to_seed/ iseed
c
c     variables for mint-integrator
c
      integer ndimmax
      parameter (ndimmax=20)

      double precision xgrid(50,ndimmax),xint,ymax(50,ndimmax)
      double precision mint_ans,mint_err
      integer ncalls0,nitmax,imode

      double precision fun,random,fun_vegas
      external fun,random,fun_vegas

c      integer ifold(ndimmax)
c      common/cifold/ifold

      double precision store
      common /to_storage/store

      double precision permutation_weight
      common /to_permutation_weight/permutation_weight

      integer loc_perm(nexternal-2)
      double precision value, error

      integer*4 it,ndo
      double precision xi(100,20),si,si2,swgt,schi
      common/bveg2/xi,si,si2,swgt,schi,ndo,it

      integer integral_index, nb_sol_perm

c      double precision WEIGTHING_PERM
c      external WEIGTHING_PERM
C**************************************************************************
C    step 1: initialisation     
C**************************************************************************
      call global_init
      store=0d0
c      need to have everything initialize including perm at some point
c      call get_perm(1, perm_id)
c      call assign_perm(perm_id)


c     VEGAS parameters for the first loop
      ACC = final_prec
      NPRN = 1
      NCALL = nevents
      ITMX = max_it_step1
      imode=0

      if (montecarlo_perm) then
         nb_sol_perm = 1
      else
         nb_sol_perm = NPERM
      endif

c     initialization of the variables in the loops
      check_value=0d0
      loop_index=0
      integral_index = 0
      do perm_pos=1,nb_sol_perm ! nb_sol_perm=1 if MC over perm
         do config_pos=1,nb_sol_config
            integral_index = integral_index +1
            order_value(integral_index)=1d99
         enddo
      enddo

c     Remove the (apriori) pointless permuation!
      config_pos = 1
      call GET_PERM_WEIGHT()
      call get_min_perm() ! this also order the permutation
      do i =1, nb_sol_config
         min_perm(i) = min_perm(1)
         do j = 1, NPERM
            perm_order(j,i) = perm_order(j,1)
         enddo
      enddo
c     re-init permvalue
      do perm_pos = 1, NPERM
         perm_value(perm_pos,1) = 0d0
         nb_point_by_perm(perm_pos) = 0
      enddo

      OPEN(UNIT=23,FILE='./details.out',STATUS='UNKNOWN')
C
C     initialization of the permutation
C
 1    normalize_perm=0d0
      loop_index=loop_index+1
      temp_err=0d0
      temp_val=0d0
      permutation_weight=1d0
      call initialize
      normalize_perm = NPERM

      integral_index = 0
      do perm_pos=1,nb_sol_perm ! nb_sol_perm=1 if MC over perm
           if (.not.montecarlo_perm) then
               call get_perm(perm_pos, perm_id)
               call assign_perm(perm_id)
               curr_perm = perm_pos
           endif

      do config_step=1,nb_sol_config
          config_pos = config_ordering(config_step)
          integral_index = integral_index + 1
          if (config_pos.eq.0) goto 3
          if (loop_index.eq.1)then
              NCALL = nevents
          else
              NCALL = nevents_refine
          endif
          iseed = iseed + 1 ! avoid to have the same seed
          NDIM=Ndimens
          if(ISR.eq.3) NDIM=NDIM+2
          if(NPERM.ne.1) NDIM = NDIM + 1
          if (montecarlo_perm) NCALL = NCALL * (NPERM - min_perm(config_pos) + 1)
c          if (min_perm(config_pos).eq. NPERM) then
c             WRITE(*,*) 'ERROR IN MIN_PERM', min_perm(config_pos)
c             STOP
c             NCALL = nevents*NPERM
c             min_perm(config_pos) = 1
c          endif
c
          if(order_value(integral_index).gt.check_value) then
             write(*,*) "** Current channel of integration: ",integral_index,
     &                  "/", nb_sol_config*nb_sol_perm, ' **'
             if (loop_index.eq.1) then
                call MC_INIT_VAR()
                ndo=100
                do i =1, NDIM
                    do j =1, 100
                       xi(j,i) = j*1d0/100d0
                    enddo
                enddo
                do i =1, dim_phase_space
                  if (var2random(i,config_pos).ne.0.and.pretrained) then
                    call prepare_tf_grid(var2random(i,config_pos))
                  endif
                enddo

             else
                ndo = ndo_by_channel(integral_index)
                do i = 1, ndo
                    do j = 1, NDIM
                        xi(i,j) = xi_by_channel(i, j, integral_index)
                    enddo
                enddo
            endif
            if (loop_index.eq.1) then
                ITMX=2

                if ((.not.montecarlo_perm).and.(integrator.eq.1)) then
                   nitmax=ITMX
                   ncalls0=ncall
                   call mint(fct_mint,ndim,ncalls0,nitmax,imode,
     .             xgrid,xint,ymax,cross,SD,.false.,acc)
                else
                   CALL VEGAS1(fct,CROSS,SD,CHI)
                endif
c                See if this is require to continue to update this
                 if (sd/cross.le.final_prec) goto 2
                 if (config_pos.ne.1) then
                    if ((cross+3*SD).lt.(check_value)) then
                       write(*,*) 'Do not refine it too low', cross+3*SD,
     &                            '<', check_value
                       goto 2
                    endif
                endif
                if (montecarlo_perm) then
                    call sort_perm
                    call get_new_perm_grid(NDIM)
                endif
             endif
             if (loop_index.eq.1) then
                 ITMX = max_it_step1
                 acc = max(final_prec, 0.9* check_value/(cross+3*SD))
             else
                 ITMX = 2
                 acc = max(0.9*check_value/order_value(integral_index),
     &                                                       final_prec)
             endif
             call check_nan(cross)
c             write(*,*) 'cross', cross
             DO I =1,NPERM
                 perm_value(I,1) = 0d0
                 perm_error(I,1) = 0d0
             ENDDO
             if (cross.eq.0d0)then
                write(*,*) 'VEGAs no pre-training'
                NCALL = 2 * nevents
                if (montecarlo_perm) NCALL = 2 * nevents * NPERM
                do i = 1, NPERM
                    perm_order(i,config_pos) = i
                enddo
                if ((.not.montecarlo_perm).and.(integrator.eq.1)) then
                   nitmax=ITMX
                   ncalls0=ncall
                   call mint(fct_mint,ndim,ncalls0,nitmax,imode,
     .             xgrid,xint,ymax,cross,SD,.false.,acc)
                else
                   CALL VEGAS(fct,CROSS,SD,CHI)
                endif
             else
                if ((.not.montecarlo_perm).and.(integrator.eq.1)) then
                   nitmax=ITMX
                   ncalls0=ncall
                   call mint(fct_mint,ndim,ncalls0,nitmax,imode,
     .             xgrid,xint,ymax,cross,SD,.false.,acc)
                else
                  CALL VEGAS1(fct,CROSS,SD,CHI)
                endif
             endif
             call check_nan(cross)
c            See if this is require to continue to update this
             write(*,*) 'cross, sd', cross, sd
             if (cross.eq.0d0) goto 2
             if (sd/cross.le.final_prec) goto 2

             if (loop_index.eq.2) then
c                NCALL = NCALL * nb_sol_perm
                ITMX = max_it_step2
                write(*,*) check_value, cross,sd, check_value/(cross+SD), final_prec
                acc = max(0.9*check_value/(cross+SD), final_prec)
                if ((.not.montecarlo_perm).and.(integrator.eq.1)) then
                   nitmax=ITMX
                   ncalls0=ncall
                   call mint(fct_mint,ndim,ncalls0,nitmax,imode,
     .             xgrid,xint,ymax,cross,SD,.false.,acc)
                else
                   CALL VEGAS2(fct,CROSS,SD,CHI)
                endif
             endif

 2           if (CROSS.lt.1d99.and.cross.ne.0d0) then
                temp_val=temp_val+cross
                temp_err=temp_err+SD**2
                order_value(integral_index) = cross
                order_error(integral_index) = sd
                if (histo) call histo_combine_iteration(integral_index)
                check_value = max(
     &                  (cross + SD)* min_prec_cut1, check_value)
                do i = 1, 100
                    do j=1,NDIM
                        xi_by_channel(i, j, integral_index) = xi(i,j)
                    enddo
                enddo
                ndo_by_channel(integral_index) = ndo
             else
                order_value(integral_index)=0d0
             endif
             ! Compute the final value for each perm/tf choice!
             call MC_GET_INTEGRAL()
             DO J=1,NB_TF
                if (montecarlo_perm) then
                   value =  tf_value_it(j)
                   error =  tf_error_it(j)
                   write(23,*) 0,' ',config_pos,' ',J,' ',value,' ', error
                   DO I=1,NPERM
                      value = perm_value_it(i,j)
                      error = perm_error_it(i,j)
                      call get_perm(I, loc_perm)
                      write(23,*) I,' ',config_pos,' ',J,' ',value,' ', error,
     &              '1   2', (2+loc_perm(k-2), k=3,nexternal)
                      perm_value_it(i,j) = 0d0
                      perm_error_it(i,j) = 0d0
                      perm_value(i,j) = 0d0
                      perm_error(i,j) = 0d0
                      nb_point_by_perm(I) = 0
                   ENDDO
                else
                   I = perm_pos
                   value = perm_value_it(i,j)
                   error = perm_error_it(i,j)
                   call get_perm(I, loc_perm)
                   write(23,*) I,' ',config_pos,' ',J,' ',value,' ', error,                                                                                                                             
     &              '1   2', (2+loc_perm(k-2), k=3,nexternal)
                endif
                tf_value_it(j) = 0d0
                tf_error_it(j) = 0d0
             ENDDO
 3         endif
       enddo
       enddo
       check_value = temp_val * min_prec_cut1 
       if (loop_index.eq.1) then
          NCALL = nevents
          ITMX = max_it_step2
          if (temp_val.ne.0d0) goto 1 ! refine the integration
       endif
       temp_val = temp_val / nb_sol_perm
       temp_err = DSQRT(temp_err) / nb_sol_perm

       write(*,*) "the weight is",temp_val,"+/-",temp_err

      OPEN(UNIT=21,FILE='./weights.out',STATUS='UNKNOWN')
      write(21,*) temp_val,'  ',temp_err
      close(21)

c      write(23,*) ' permutation channel   value      error'

c      do perm_pos=1, NPERM
c         call get_perm(perm_pos, perm_id)
c      write(23,*) "======================================"
c
c      write(23,*) '1   2', (2+perm_id(i-2), i=3,nexternal)
c      enddo
c      write(23,*) "======================================"
c      write(23,*)'Weight: ',temp_val,'+-', temp_err
c      write(23,*) "======================================"

C**************************************************************************
C     write histogram file (not activated in the standard mode)           *
C************************************************************************** 
      if (histo)   call histo_final(NPERM*nb_sol_config)

      close(21)

      END
C**************************************************************************************
c     ==================================
      subroutine get_new_perm_grid(NDIM)
      implicit none
      integer i, j, step,NDIM,k
      double precision cross, total, prev_total, value
      include 'permutation.inc'
      include 'phasespace.inc'
      include 'madweight_param.inc'
      integer*4 it,ndo
      double precision xi(100,20),si,si2,swgt,schi
      common/bveg2/xi,si,si2,swgt,schi,ndo,it

      step = 1
      total = 0d0
      cross = 0d0
      DO I =1,NPERM
         cross = cross + perm_value_it(i,1)/(perm_error_it(i,1)+1d-99)
      ENDDO
c      write(*,*) '=>', cross
      prev_total = 0d0
      min_perm(config_pos) = 1
      do i=1,NPERM
           j = perm_order(i, config_pos)
           value = perm_value_it(j,1)/
     &                (perm_error_it(j,1)+1d-99)
           total = total + value
           do while (total.gt.((step)*cross/ndo))
              xi(step, NDIM) = ((i-1)*1d0/NPERM + (step*cross/(1d0*ndo)
     &          -prev_total)/(value*NPERM))
               step = step + 1
           enddo
           prev_total = total
          !check if it has to be consider or not
           if (value/cross.lt.min_prec_cut1) then
              min_perm(config_pos) = min_perm(config_pos) + 1
           endif
        enddo

c      do i=2,NPERM
c        j = perm_order(i-1, config_pos)
c        k = perm_order(i, config_pos)
c        if (perm_value(j,1)/(nb_point_by_perm(j)+1d-99)/total.gt.
c     &     perm_value(k,1)/(nb_point_by_perm(k)+1d-99)/total)then
c            write(*,*) 'FAIL',i, perm_value(j,1), perm_value(k,1)
c           stop
c        endif
c      enddo
c      WRITE(*,*) '***************************************', config_pos, min_perm(config_pos)

      return
      end
c     ==================================
      subroutine get_min_perm()
      implicit none
      double precision cross,tmp
      integer i,j      

      include 'permutation.inc'
      include 'madweight_param.inc'
      include 'phasespace.inc'
      double precision data(NPERM)
c     First sort them
      do i=1,NPERM
         perm_order(i, config_pos) = i
      enddo
      data(1) = perm_value(1,1)
      do i=2,NPERM
         data(i) = perm_value(i,1)
 2       do j=1, i-1
            if (data(i).ge.data(i-j)) then
                if (j.ne.1) then
                     call move_pos(i,i-j+1, data)
                endif
                goto 1
            endif
         enddo
         call move_pos(i,1,data)
 1    enddo

c     select them
      cross = 0d0
      DO I =1,NPERM
         cross = cross + perm_value(i,1)
      ENDDO
      min_perm(config_pos) = 1
      do i=1,NPERM
          j = perm_order(i, config_pos)
          tmp = perm_value(j,1)
          if (tmp/cross.lt.min_perm_cut) then
             min_perm(config_pos) = min_perm(config_pos) + 1
c             if ((i+1).ne.min_perm(config_pos)) stop
c          else
c             write(*,*) i, j, tmp, min_perm(config_pos), 'kept'
           endif


      enddo
      
      return
      end

C**************************************************************************************
c     ==================================
      subroutine sort_perm()
      implicit none
      include 'permutation.inc'
      include 'phasespace.inc'
      double precision data(NPERM)
      integer i,j
      do i=1,NPERM
         perm_order(i, config_pos) = i 
      enddo
      data(1) = perm_value_it(1,1)/(1d-99+perm_error_it(1,1))
      do i=2,NPERM
         data(i) = perm_value_it(i,1)/(perm_error_it(i,1)+1d-99)
 2       do j=1, i-1
            if (data(i).ge.data(i-j)) then
                if (j.ne.1) then
                     call move_pos(i,i-j+1, data)
                endif
                goto 1
            endif
         enddo
         call move_pos(i,1,data)
 1       enddo
c       write(*,*) perm_order
      end

c     ==================================
C**************************************************************************************
      subroutine move_pos(old,new, data)
      implicit none
      integer i, j, old, new, next, id_old
      include 'permutation.inc'
      include 'phasespace.inc'
      double precision data(NPERM), new_data(NPERM), data_old

      data_old = data(old)
      id_old = perm_order(old, config_pos)
      do j = old, new+1,-1
        data(j) = data(j-1)
        perm_order(j,config_pos) = perm_order(j-1,config_pos)
      enddo
      data(new) = data_old
      perm_order(new, config_pos) = id_old

c      write(*,*) (data(i), i=1,old)
c      write(*,*) (perm_order(i), i=1,old)
c      pause
      return
      end

c     ==================================
C**************************************************************************************
      subroutine prepare_tf_grid(var)

      integer var
      integer*4 it,ndo
      double precision xi(100,20),si,si2,swgt,schi
      common/bveg2/xi,si,si2,swgt,schi,ndo,it

        xi(1, var) = 0.26202870173391102
        xi(2, var) = 0.29627919469260666
        xi(3, var) = 0.31499038866628032
        xi(4, var) = 0.32765648935273894
        xi(5, var) = 0.33759486510935055
        xi(6, var) = 0.34577322735126004
        xi(7, var) = 0.35284416094523846
        xi(8, var) = 0.35924883571260102
        xi(9, var) = 0.36473559783805265
        xi(10, var) = 0.36967692843521249
        xi(11, var) = 0.37437312339719947
        xi(12, var) = 0.37882057530130681
        xi(13, var) = 0.38302435431944371
        xi(14, var) = 0.38692806832412335
        xi(15, var) = 0.39078365837111473
        xi(16, var) = 0.39443885419206620
        xi(17, var) = 0.39811533662198700
        xi(18, var) = 0.40156681122162508
        xi(19, var) = 0.40491010227563923
        xi(20, var) = 0.40818200369537122
        xi(21, var) = 0.41138356922904956
        xi(22, var) = 0.41453292219292942
        xi(23, var) = 0.41772005705536408
        xi(24, var) = 0.42088581400299130
        xi(25, var) = 0.42402420250952866
        xi(26, var) = 0.42713014443146680
        xi(27, var) = 0.43017871888363235
        xi(28, var) = 0.43313830428681271
        xi(29, var) = 0.43597872090369233
        xi(30, var) = 0.43881328519549234
        xi(31, var) = 0.44152742598930816
        xi(32, var) = 0.44423236714791764
        xi(33, var) = 0.44699571093981794
        xi(34, var) = 0.44974198364516216
        xi(35, var) = 0.45243196043904776
        xi(36, var) = 0.45511140219859592
        xi(37, var) = 0.45778427053265852
        xi(38, var) = 0.46044763794587357
        xi(39, var) = 0.46311055953600111
        xi(40, var) = 0.46580147058860472
        xi(41, var) = 0.46847589083720942
        xi(42, var) = 0.47118533852524436
        xi(43, var) = 0.47386033214596757
        xi(44, var) = 0.47650782585773122
        xi(45, var) = 0.47913937456217742
        xi(46, var) = 0.48172957159232704
        xi(47, var) = 0.48425707330124140
        xi(48, var) = 0.48680305366265453
        xi(49, var) = 0.48935741086756696
        xi(50, var) = 0.49190779017513964
        xi(51, var) = 0.49445075033624031
        xi(52, var) = 0.49698410897494982
        xi(53, var) = 0.49949247023900117
        xi(54, var) = 0.50203323690138346
        xi(55, var) = 0.50460212881414868
        xi(56, var) = 0.50717115079005592
        xi(57, var) = 0.50972312741975834
        xi(58, var) = 0.51239184585293929
        xi(59, var) = 0.51505220112364436
        xi(60, var) = 0.51768921296817161
        xi(61, var) = 0.52032324890300241
        xi(62, var) = 0.52306024406848717
        xi(63, var) = 0.52579002073412406
        xi(64, var) = 0.52854998647908269
        xi(65, var) = 0.53136335347286601
        xi(66, var) = 0.53425384859378400
        xi(67, var) = 0.53713629635349125
        xi(68, var) = 0.54010688964160236
        xi(69, var) = 0.54318794102497969
        xi(70, var) = 0.54639066313612672
        xi(71, var) = 0.54965439635960123
        xi(72, var) = 0.55303432289209986
        xi(73, var) = 0.55642840039224628
        xi(74, var) = 0.55988079309654537
        xi(75, var) = 0.56347832620429206
        xi(76, var) = 0.56719793056059142
        xi(77, var) = 0.57113488403650758
        xi(78, var) = 0.57520408498802678
        xi(79, var) = 0.57941659109901023
        xi(80, var) = 0.58380548335999183
        xi(81, var) = 0.58839287284612785
        xi(82, var) = 0.59318106149777727
        xi(83, var) = 0.59820224821080226
        xi(84, var) = 0.60374524721006018
        xi(85, var) = 0.60968724613431557
        xi(86, var) = 0.61618805855633685
        xi(87, var) = 0.62381426194961553
        xi(88, var) = 0.63245595088132800
        xi(89, var) = 0.64194814028290437
        xi(90, var) = 0.65255065423407199
        xi(91, var) = 0.66490189208356287
        xi(92, var) = 0.67967463147755103
        xi(93, var) = 0.69666374930629316
        xi(94, var) = 0.71697617102631228
        xi(95, var) = 0.74188445172370854
        xi(96, var) = 0.77167890678720286
        xi(97, var) = 0.80852455039619509
        xi(98, var) = 0.85502370883528123
        xi(99, var) = 0.91616841891459222
        xi(100, var) = 1.0000000000000000
        return
        end

      double precision function fct_mint(x,w,ifirst)

      implicit none
      double precision x(20),w
      integer ifirst

      double precision store
      common /to_storage/store

      double precision fct
      external fct
      double precision temp 

      temp = fct(x,w)*w
      store=store+temp
      if(ifirst.eq.2) then
      fct_mint=store
      else
      fct_mint=temp
      endif

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
C      double precision xi(100,20),si,si2,swgt,schi
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
C      double precision xi(100,20),si,si2,swgt,schi
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
**************************************************************************************

      SUBROUTINE MC_INIT_VAR()

      include 'nexternal.inc'
      include 'permutation.inc'

      integer i, j


      do i=1, NPERM
      do j=1, nb_tf
        perm_value(i,j) = 0d0 ! sum of fct
        perm_error(i,j) = 0d0 ! sum of fct**2
        perm_value_it(i,j) = 0d0   ! weigthed sum on iteration result
        perm_error_it(i,j) = 0d0   ! sum (1/sigma**2)
        tf_value_it(j) = 0d0       ! weigthed sum on iteration result for all perm
        tf_error_it(j) = 0d0       ! sum (1/sigma**2) all perm
      enddo
        nb_point_by_perm(i) = 0
      enddo

      return
      end

      SUBROUTINE MC_END_ITER()

      include 'nexternal.inc'
      include 'permutation.inc'

      integer i
      double precision sum_perm(nb_tf)
      double precision sum_error(nb_tf)
      double precision sum_nb_point

      double precision value, error

c     for debugging
c      double precision calls,ti,tsi
c      COMMON/BVEG4/calls,ti,tsi
c      double precision xi(100,20)
c      integer ndo,it
c      double precision si,si2,swgt,schi
c      common/bveg2/xi,si,si2,swgt,schi,ndo,it



      sum_nb_point = 0
      do j=1,nb_tf
        sum_perm(j) = 0d0
        sum_error(j) =0d0
      enddo
      do i=1, NPERM
          sum_nb_point = sum_nb_point + nb_point_by_perm(i)
          do j=1, nb_tf
            sum_perm(j) = sum_perm(j) + perm_value(i,j)
            sum_error(j) = sum_error(j) + perm_error(i,j)
            if (nb_point_by_perm(i).gt.0) then
                value = perm_value(i,j)
                error = perm_error(i,j) - perm_value(i,j)**2/(nb_point_by_perm(i)-1)
                if (error.gt.0d0)then
                    perm_value_it(i,j) = perm_value_it(i,j) + value**3/error
                    perm_error_it(i,j) = perm_error_it(i,j) + value**2/error
                endif
            endif
                perm_value(i,j) = 0d0
                perm_error(i,j) = 0d0
          enddo
          nb_point_by_perm(i) = 0
      enddo
      do j =1, nb_tf
          value = sum_perm(j) 
          if (sum_nb_point.gt.0) then
             error = sum_error(j) - value**2/(sum_nb_point-1)
          else
             error = 0d0
          endif
          if (error.gt.0d0)then
              tf_value_it(j) = tf_value_it(j) + value**3/error
              tf_error_it(j) = tf_error_it(j) + value**2/error
              write(*,*) '                       ',j, tf_value_it(j) / tf_error_it(j), '+-',
     &             tf_value_it(j) / tf_error_it(j)  / DSQRT(tf_error_it(j))
           endif
      enddo

c     SECURITY CHECK
c      if ((si/swgt - tf_value_it(1) / tf_error_it(1)).gt.(tf_value_it(1)
c     &    / tf_error_it(1)  / DSQRT(tf_error_it(1)))) then
c        write(*,*) 'ERROR ON ITERATION COMBINAISON'
c        write(*,*) 'INFO', SI, SWGT
c        write(*,*) 'VEGAS:', si/swgt
c        write(*,*) 'MINE', tf_value_it(1) / tf_error_it(1)
c        stop 1
c      endif

      return
      end

      SUBROUTINE MC_GET_INTEGRAL()

      include 'nexternal.inc'
      include 'permutation.inc'
      do j =1, nb_tf
          if (tf_error_it(j).gt.0)then
              tf_value_it(j) = tf_value_it(j) / tf_error_it(j)
              tf_error_it(j) = tf_value_it(j) / DSQRT(tf_error_it(j))
          endif
      enddo
      do i=1, NPERM
          do j=1, nb_tf
            if (perm_error_it(i,j).gt.0) then
                perm_value_it(i,j) = perm_value_it(i,j) / perm_error_it(i,j)
                perm_error_it(i,j) = perm_value_it(i,j) / DSQRT(perm_error_it(i,j))
            endif
          enddo
      enddo
      return
      end

