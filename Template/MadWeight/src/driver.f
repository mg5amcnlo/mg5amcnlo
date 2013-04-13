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
C
C     PARAMETERS
C
      character*1 integrator
      parameter (integrator='v')  ! v= vegas, m=mint
C
C     LOCAL (Variable of integration)
C
      DOUBLE PRECISION SD,CHI,CROSS
      INTEGER I,J,convergence_status
      integer pos,channel_pos,ll
      double precision temp_err, temp_val
      double precision order_value(nb_channel)
      double precision order_error(nb_channel)
      double precision xi_by_channel(50, 20, nb_channel)

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
      double precision best_precision, chan_val, chan_err
      double precision check_value
      integer loop_index
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

      integer ifold(ndimmax)
      common/cifold/ifold

      double precision store
      common /to_storage/store

      double precision permutation_weight
      common /to_permutation_weight/permutation_weight

      integer loc_perm(nexternal-2)
      double precision value, error

      integer*4 it,ndo
      double precision xi(50,20),si,si2,swgt,schi
      common/bveg2/xi,si,si2,swgt,schi,ndo,it

C**************************************************************************
C    step 1: initialisation     
C**************************************************************************
      call global_init
      store=0d0
c      need to have everything initialize including perm at some point
      call get_perm(1, perm_id)
      call assign_perm(perm_id)


c     VEGAS parameters for the first loop
      ACC = final_prec
      NPRN = 1
      NCALL = nevents
      ITMX = max_it_step1
      ncalls0=ncall  
      nitmax=ITMX  
      imode=0

c     initialization of the variables in the loops
      check_value=0d0
      loop_index=0
       do ll=1,nb_sol_config
       order_value(ll)=1d5
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

      chan_val=0d0
      chan_err=0d0

      do ll=1,nb_sol_config
          write(*,*) 'config', ll,'/',nb_sol_config


          config_pos=ll
          iseed = iseed + 1 ! avoid to have the same seed
          if (.not. NWA) then
              NDIM=Ndimens
          else
              NDIM=Ndimens-num_propa(config_pos)
          endif
          if(ISR.eq.3) NDIM=NDIM+2
          if(NPERM.ne.1) NDIM = NDIM + 1
c
          if(order_value(ll).gt.check_value) then
             write(*,*) "Current channel of integration: ",config_pos
             if (loop_index.eq.1) then
                do i = 1, NPERM
                    perm_order(i,ll) = i
                enddo
            else
                do i = 1, 50
                    do j = 1, 20
                        xi(i,j) = xi_by_channel(i, j, ll)
                    enddo
                enddo
            endif
            if (loop_index.eq.1) then
                ITMX=2
                CALL VEGAS(fct,CROSS,SD,CHI)
c                See if this is require to continue to update this
                 if (sd/cross.le.final_prec) goto 2
                 if (ll.ne.1) then
                    if ((cross+3*SD).lt.(check_value)) then
                       write(*,*) 'Do not refine it too low', cross+3*SD,
     &                            '<', check_value
                       goto 2
                    endif
                endif
             call sort_perm
             call get_new_perm_grid(NDIM)
             endif
             if (loop_index.eq.1) ITMX=2
             if (loop_index.eq.2) ITMX=4
             do i= 1, ndo
                xi(i,NDIM) = i*1d0/ndo
             enddo
             CALL VEGAS1(fct,CROSS,SD,CHI)
c            See if this is require to continue to update this
             if (sd/cross.le.final_prec) goto 2

             if (loop_index.eq.1) then
                ITMX=max_it_step1
             else
                ITMX = max_it_step2
                acc = max(0.25*check_value/order_value(ll), final_prec)
             endif
             CALL VEGAS2(fct,CROSS,SD,CHI)
 2           if (CROSS.lt.1d99) then
                temp_val=temp_val+cross
                chan_val=chan_val+cross
                temp_err=temp_err+SD
                chan_err=chan_err+SD
                order_value(ll) = cross
                order_error(ll) = sd
                if (histo) call histo_combine_iteration(ll)
                if (loop_index.eq.1) then
                  if (ll.eq.1)then
                     check_value = (cross + 3*SD) * min_prec_cut1 * 0.01
                  else
                     check_value = max(
     &                  (cross + SD)* min_prec_cut1 * 0.01, check_value)
                  endif
                endif
                do i = 1, 50
                    do j=1,20
                        xi_by_channel(i, j, ll) = xi(i,j)
                    enddo
                enddo
             else
                order_value(ll)=0d0
             endif
              if (nb_point_by_perm(1).ne.0)then
                  DO I =1,NPERM
                    value = perm_value(I) / (nb_point_by_perm(I))
                    error = sqrt(abs(perm_error(I) - nb_point_by_perm(I)
     &                             * value**2 )/(nb_point_by_perm(I)-1))
                    call get_perm(I, loc_perm)
                    write(23,*) I,' ',ll,' ',value,' ', error,
     &              '1   2', (2+loc_perm(j-2), j=3,nexternal)
                    nb_point_by_perm(I) = 0
                    perm_value(I) = 0
                    perm_error(I) = 0
                  ENDDO
              ENDIF
          endif
       enddo
       check_value = temp_val * min_prec_cut1 
       if (loop_index.eq.1) then
          NCALL = nevents
          ITMX = max_it_step2
          goto 1 ! refine the integration
       endif

       write(*,*) "the weight is",temp_val,"+/-",temp_err

      OPEN(UNIT=21,FILE='./weights.out',STATUS='UNKNOWN')
      write(21,*) temp_val,'  ',temp_err
      close(21)

c      write(23,*) ' permutation channel   value      error'

      do perm_pos=1, NPERM
         call get_perm(perm_pos, perm_id)
      write(23,*) "======================================"
      write(23,*) '1   2', (2+perm_id(i-2), i=3,8)

      enddo
      write(23,*) "======================================"
      write(23,*)'Weight: ',temp_val,'+-', temp_err
      write(23,*) "======================================"

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
      integer i, j, step,NDIM
      double precision cross, total, prev_total, value
      include 'permutation.inc'
      include 'phasespace.inc'
      integer*4 it,ndo
      double precision xi(50,20),si,si2,swgt,schi
      common/bveg2/xi,si,si2,swgt,schi,ndo,it

      step = 1
      total = 0d0
      cross = 0d0
      DO I =1,NPERM
         cross = cross + perm_value(i)/(nb_point_by_perm(i)+1e-99)
      ENDDO
      prev_total = 0d0
      do i=1,NPERM
           value = perm_value(perm_order(i, config_pos))/
     &                (nb_point_by_perm(perm_order(i,config_pos))+1e-99)
           total = total + value
c           write(*,*) i, value, total, step, ((step+1)*cross/ndo)
           do while (total.gt.((step)*cross/ndo))
              xi(step, NDIM) = ((i-1)*1d0/NPERM + (step*cross/(1d0*ndo)
     &          -prev_total)/(value*NPERM))
               step = step + 1
           enddo
           prev_total = total
      enddo
      write(*,*) (xi(i, NDIM), i=1,50)
      return
      end

C**************************************************************************************
c     ==================================
      subroutine sort_perm()
      implicit none
      include 'permutation.inc'
      double precision data(NPERM)
      integer i,j
      data(1) = perm_value(1)/nb_point_by_perm(1)
      do i=2,NPERM
         data(i) = perm_value(i)/nb_point_by_perm(i)
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
**************************************************************************************

