      subroutine histo_init(num_channel)
      implicit none
c
c     argument
c
      integer num_channel
c
c    local
c
      character*40 t1, t2, t3, t4, t0
      integer i
c
c     gobal
c
      double precision GeVbin, etabin, xbin
      common/to_bin/ GeVbin, etabin, xbin
c
c
      call inihist
c
      GeVbin=24d0
      etabin=0.1d0
c
      t0='histo for one vegas iteration (prov)'
      t1='final state invariant mass '
      t2='fct error'
      CALL MBOOK(1,200,t0,GeVbin,100d0,1300d0)
      CALL MBOOK(1,-200,t0,GeVbin,100d0,1300d0)
c
      do i=1,num_channel
         CALL MBOOK(1,i,t1,GeVbin,100d0,1300d0)
         CALL MBOOK(1,-1*i,t2,GeVbin,100d0,1300d0)
      enddo
      return
      
      end
ccccCcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      subroutine FILL_plot(fct,weight,nexternal)
      implicit none
      include '../../../SubProcesses/phasespace.inc'
c
c     arguments
c
      double precision weight,fct
      integer nexternal
c
c     local
c
      double precision Ptot(0:3), minv
      integer i,nu
c
c     common
c
      double precision momenta(0:3,-max_branches:2*max_particles)  ! records the momenta of external/intermediate legs     (MG order)
      double precision mvir2(-max_branches:2*max_particles)        ! records the sq invariant masses of intermediate particles (MG order)
      common /to_diagram_kin/ momenta, mvir2
c---
c Begin code for variable 1
c---
      do nu=0,3
        Ptot(nu)=0d0
        do i=3,nexternal
          Ptot(nu)=Ptot(nu)+momenta(nu,i)
        enddo 
      enddo 
      minv=dsqrt(Ptot(0)**2-Ptot(1)**2-Ptot(2)**2-Ptot(3)**2)
      
      CALL MFILL(1,200, minv, fct*weight)
      CALL MFILL(1,-200, minv, fct**2*weight)

       return
       end
cCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCcccccccccccccccccccccccccccc
      subroutine histo_final(num_channel)
c
c     argument
c   
      integer num_per
c
c     local
c      
      integer i,num_iter,j,k
      character*30 BUFFER
      character istring
c
c     gobal
c
      double precision GeVbin, etabin, xbin
      common/to_bin/ GeVbin, etabin, xbin
c     
c     include histo globlal
c      
      include 'dbook.inc'

      do k=1,NHistovar
         do i=1,num_channel
c     if (i.eq.1) istring='1'
c     if (i.eq.2) istring='2'

            call MFINAL(k, i)
            call MFINAL(k, -i)
            call MOPERA(k, -1*i, 'I',-1*i,-1*i,1d0,1d0) !contains final error square
            call MOPERA(k, i, '*',-i,i,1d0,1d0) !contains final estimation of integral

            if(i.ne.1) then
               call MOPERA(k,i, '+',1,1,1d0,1d0) 
               call MOPERA(k, -i, '+',-1,-1,1d0,1d0)
            endif

            call MFINAL(k, 1)
            call MOPERA(k, -1,'R',-1,-1,1d0,1d0) ! contains the error for the sum of all permutations
            call MFINAL(k, -1)
         enddo
         buffer='plot_00.dat'
         if (k.ge.10)then
            write(buffer(6:7),'(I2)') k
         else
            write(buffer(7:7),'(I1)') k
         endif

         OPEN(UNIT=98,file=buffer,STATUS='UNKNOWN')
         buffer=buffer(1:7)//'.top'
         OPEN(UNIT=99,file=buffer,STATUS='UNKNOWN')
         CALL MPRINT(k, 1)
         CALL MTOP(k, 1, 50,'  M_INV  (GeV)     '
     &        ,'d P/d M_inv   ','log')

         close(98)
         close(99)


         buffer='err_'//buffer(1:7)//'.dat'
         OPEN(UNIT=98,file=buffer,STATUS='UNKNOWN')
         buffer=buffer(1:11)//'.top'
         OPEN(UNIT=99,file=buffer,STATUS='UNKNOWN')
         CALL MPRINT(k, -1)
         CALL MTOP(k, -1,50,'  M_INV  (GeV)     '
     &        ,'d P/d M_inv   ','log')
c     
         close(98)
         close(99)
      enddo
      return
      end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      subroutine histo_combine_iteration(channel_pos)
      implicit none
c
c     argument
c   
      integer channel_pos
c
c     local
c      
      integer i,num_iter,j,k
      character*30 BUFFER
      character istring
c
c     gobal
c
      double precision GeVbin, etabin, xbin
      common/to_bin/ GeVbin, etabin, xbin
c     
c     include histo globlal
c      
      include 'dbook.inc'
c      call MFINAL(i)
c      call MFINAL(-1*i)
      
      do k=1,NHistoVar

      call MFINAL(k, 200)
      call MFINAL(k, -200)

      write(*,*) 'combine for histo', channel_pos
      call MOPERA(k, 200,'V',-200,-200,1d0,1d0)
      call MOPERA(k, -200,'S',0,-200,1d0,1d0)
      call MOPERA(k, 200,'/',-200,200,1d0,1d0)
      call MOPERA(k, channel_pos,'+',200,channel_pos,1d0,1d0)

      !define a unity histogram in 200 (after simply redifine to zero)
      call MOPERA(k, -200,'I',-200,-200,1d0,1d0)
      call MOPERA(k, -1*channel_pos,'+',-200,-1*channel_pos,1d0,1d0)

       call MOPERA(k, 200,'+',200,200,0d0,0d0)
       call MOPERA(k, -200,'+',-200,-200,0d0,0d0)

      enddo
      return
      end

