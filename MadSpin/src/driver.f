      program madspin
      implicit none

C---  integer    n_max_cg
      INCLUDE "ngraphs.inc"     !how many diagrams 
      double precision ZERO
      parameter (ZERO=0D0)
      !include 'genps.inc'
      include 'coupl.inc'
      include 'nexternal.inc'
      include 'nexternal_prod.inc'
      include 'helamp.inc'
      !include 'run.inc'

C     
C     LOCAL
C     
      INTEGER I,J,K
      INTEGER HELSET(NEXTERNAL)
      REAL*8 P(0:3,NEXTERNAL_PROD) 
      REAL*8 PFULL(0:3,NEXTERNAL), Ptrial(0:3,NEXTERNAL) 
      double precision x(36), Ecollider
      CHARACTER*120 BUFF(NEXTERNAL_PROD)
      integer iforest(2,-nexternal:-1,N_MAX_CG)
      integer itree(2,-nexternal:-1), iconfig
      integer  map_external2res(nexternal_prod) ! map (index in production) -> index in the full structure
c      integer mapconfig(0:lmaxconfigs)
      integer sprop(1,-nexternal:-1,N_MAX_CG) ! first col has one entry, since we should have group_processes=false
      integer tprid(-nexternal:-1,N_MAX_CG)
      integer            mapconfig(0:N_MAX_CG), this_config
      common/to_mconfigs/mapconfig, this_config
      double precision prmass(-nexternal:0,N_MAX_CG)
      double precision prwidth(-nexternal:0,N_MAX_CG)
      integer pow(-nexternal:0,N_MAX_CG)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),jac
      double precision M_PROD, M_FULL
      logical notpass
      integer counter,mode,nbpoints, counter2, counter3
      double precision mean, variance, maxweight,weight,std
      double precision temp
      double precision Pprod(0:3,nexternal_prod)
      integer nb_mc_masses, indices_mc_masses(nexternal)
      double precision values_mc_masses(nexternal)

      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

       double precision BWcut, maxBW
       common /to_BWcut/BWcut, maxBW

c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-nexternal:-1,-1:1),
     &     cBW_width(-nexternal:-1,-1:1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level


      DOUBLE PRECISION AMP2(n_max_cg)
      COMMON/TO_AMPS/  AMP2

       ! variables associate with the PS generation
       double precision totmassin, totmass
       double precision shat, sqrtshat, stot, y, m(-nexternal:nexternal)
       integer nbranch, ns_channel,nt_channel, pos_pz
       common /to_topo/
     & totmassin, totmass,shat, sqrtshat, stot,y, m,
     & nbranch, ns_channel,nt_channel, pos_pz

      integer*8       iseed, P_seed
      common /to_seed/iseed


      real*8 ranu(97),ranc,rancd,rancm
      integer iranmr,jranmr
      common/ raset1 / ranu,ranc,rancd,rancm
      common/ raset2 / iranmr,jranmr

c      call ntuple(x,0d0,1d0,1,2)  ! initialize the sequence of random
                                  ! numbers at the position reached 
                                  ! at the previous termination of the
                                  ! code

       open(unit=56,file='seeds.dat',status='old')
       read(56,*) iseed
       close(56)
       open(unit=56,file='offset.dat',status='old')
       read(56,*) P_seed
       close(56)
       iseed = iseed + P_seed

cccccccccccccccccccccccccccccccccccccccccccccccccccc
c   I. read momenta for the production events
c
cccccccccccccccccccccccccccccccccccccccccccccccccccc

      ! hard-code  for testing
c      buff(1)="1   0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  0.0000000E+00"
c      buff(2)="2   0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03  0.0000000E+00"
c      buff(3)="3   0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03  0.1730000E+03"
c      buff(4)="4   0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03  0.1730000E+03"
c      do i=1,nexternal_prod
c         read (buff(i),*) k, P(0,i),P(1,i),P(2,i),P(3,i)
c      enddo

 
1     continue
      maxBW=0d0
      read(*,*) mode,  BWcut, Ecollider, temp
 

      if (mode.eq.1) then    ! calculate the maximum weight
         nbpoints=int(temp)
      elseif (mode.eq.2) then
         maxweight=temp      ! unweight decay config   
      elseif (mode.eq.3) then
         continue      ! just retrun the value of M_full   
      else
           write(*,*) ranu,ranc,rancd,rancm,iranmr,jranmr
         call flush()
         goto 2                      ! and close the program  
      endif

      do i=1,nexternal_prod
         read (*,*) P(0,i),P(1,i),P(2,i),P(3,i) 
      enddo

      if (mode.eq.2) then   
        read(*,*) nb_mc_masses
        if (nb_mc_masses.gt.0) then
            read (*,*) (indices_mc_masses(k), k=1,nb_mc_masses)
            read (*,*) (values_mc_masses(k), k=1,nb_mc_masses)
        endif
      endif
c      write(*,*) sqrt(dot(P(0,3),P(0,3)))
c      write(*,*) sqrt(dot(P(0,4),P(0,4)))
c      write(*,*) dot(P(0,5),P(0,5))
c      write(*,*) 'shat', sqrt(2d0*dot(P(0,1),P(0,2)))
c      write(*,*) 'pt2g', sqrt(2d0*dot(P(0,4),P(0,5))+dot(P(0,4),P(0,4)))
cccccccccccccccccccccccccccccccccccccccccccccccccccc      
c    II. initialization of masses and widths        c
c       also load production topology information  c
cccccccccccccccccccccccccccccccccccccccccccccccccccc  

      include 'configs_production.inc'

      ! set masses 
      call set_parameters(p,Ecollider)

      include 'props_production.inc'


      ! do not consider the case of conflicting BW
      do i = -nexternal, -1
         cBW(i)=0
         cBW_level(i)=0
      enddo

cccccccccccccccccccccccccccccccccccccccccccccccccccc
c   III. compute production matrix element         c 
cccccccccccccccccccccccccccccccccccccccccccccccccccc

      do i=1,n_max_cg
      amp2(i)=0d0
      enddo
      call coup()
      CALL SMATRIX_PROD(P,M_PROD)
c      write(*,*) 'M_prod ', M_prod
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c   IV. select one topology                        c
cccccccccccccccccccccccccccccccccccccccccccccccccccc

      call get_config(iconfig)
      do i=-nexternal_prod+2,-1
         do j=1,2
            itree(j,i)=iforest(j,i,iconfig)
         enddo
      enddo

      do i=-nexternal_prod+3,-1
         qmass(i)=prmass(i,iconfig)
         qwidth(i)=prwidth(i,iconfig)
      enddo

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   V. load topology for the whole event select                c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       call  merge_itree(itree,qmass,qwidth, p,map_external2res)
       !write(*,*) keep_inv(-5)
       !write(*,*) 'm2_tchan ',m2_tchan(-5)
       !write(*,*) 'fixedinv', fixedinv(-5)
       !write(*,*)  'phi_tchan', phi_tchan(-5)
 
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   VIa. Calculate the max. weight                                      c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


      if (mode.eq.1) then

        mean=0d0
        variance=0d0
        maxweight=0d0
c        max_m=0d0
c        max_jac=0d0

        counter2=0
        counter3=0
        do i=1,nbpoints
           jac=1d0
           ivar=0
           no_gen=.false.
           do j = 1, 3*(nexternal-nexternal_prod)
              call  ntuple(x(j),0d0,1d0,j,1)
           enddo

c           do j=1,nexternal_prod
c              write (*,*) (p(k,j), k=0,3)  
c           enddo

           call generate_momenta_conf(jac,x,itree,qmass,qwidth,pfull,pprod,map_external2res) 
           if (jac.lt.0d0) then
             counter2=counter2+1 
             counter3=counter3+1
             if (counter3.gt.500) then
               write(*,*) "500_pts_failed_stop_executation"
               goto 1
             endif
             if (counter2.ge.8) then ! use another topology to generate PS points
               do k=1,n_max_cg
                 amp2(k)=0d0
               enddo
               CALL SMATRIX_PROD(P,M_PROD)
               call get_config(iconfig)
               do k=-nexternal_prod+2,-1
                do j=1,2
                  itree(j,k)=iforest(j,k,iconfig)
                enddo
               enddo
 
               do k=-nexternal_prod+3,-1
                 qmass(k)=prmass(k,iconfig)
                 qwidth(k)=prwidth(k,iconfig)
               enddo
               call  merge_itree(itree,qmass,qwidth, p,map_external2res)
               counter2=0
             endif

           cycle
           endif
           !do j=1,nexternal
           !   write (*,*) (pfull(k,j), k=0,3)  
           !enddo
           call SMATRIX(pfull,M_full)
           call SMATRIX_PROD(pprod,M_prod)
c           write(*,*) 'M_full ', M_full
c           write(*,*) 'jac',jac

           weight=M_full*jac/M_prod
           if (weight.gt.maxweight) then
            maxweight=weight
c            max_m=M_full
c            max_jac=jac
c            do k =1,nexternal
c            do j=0,3
c            max_mom(j,k)=pfull(j,k)
c            enddo
c            enddo
           endif
c           mean=mean+weight
c           variance=variance+weight**2
        enddo
c        mean=mean/real(nbpoints)   
c        variance=variance/real(nbpoints)-mean**2
c        std=sqrt(variance)
        write (*,*) maxweight   ! ,mean,std  
c        write (*,*) 'max_m',max_m 
c        write (*,*) 'max_jac', jac
c        write (*,*) 'Extrenal masses'
c        do k=1,nexternal
c        write(*,*) dot(max_mom(0,k), max_mom(0,k))
c        enddo
c        do j=0,3
c          pw1(j)=max_mom(j,4)+max_mom(j,5)
c          pt1(j)=pw1(j)+max_mom(j,3)
c          pw2(j)=max_mom(j,7)+max_mom(j,8)
c          pt2(j)=pw2(j)+max_mom(j,6)
c          pt2g(j)=pt2(j)+max_mom(j,9)
c        enddo
 
c        write (*,*) 'm45', sqrt(2D0*dot(max_mom(0,4),max_mom(0,5))) 
c        write (*,*) 'm78', sqrt(2d0*dot(max_mom(0,7),max_mom(0,8))) 
c        write (*,*) 'mt1', sqrt(dot(pt1,pt1)) 
c        write (*,*) 'mt2', sqrt(dot(pt2,pt2)) 
c        write (*,*) 'mt2g', sqrt(dot(pt2g,pt2g)) 
c        write (*,*) 'm9', sqrt(dot(max_mom(0,9),max_mom(0,9))) 
c        write (*,*) 'shat', sqrt(2D0*dot(max_mom(0,2),max_mom(0,1))) 
c        write(*,*)  (max_mom(j,1), j=0,3)
c        write(*,*)  (max_mom(j,2), j=0,3)
c        write(*,*)  (pt1(j), j=0,3)
c        write(*,*)  (pt2(j), j=0,3)
c        write(*,*)  (max_mom(j,9), j=0,3)
        call flush()
        goto 1
      endif


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   VIb. generate unweighted decay config                               c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       if (mode.eq.2) then
c        initialize the helicity amps
         do i=1,ncomb1
           helamp(i)=0d0
         enddo

         notpass=.true.
         counter=0
         counter2=0
         counter3=0
         do while (notpass) 
           maxBW=0d0
           counter=counter+1
           jac=1d0
           ivar=0
           no_gen=.false.
           do i = 1, 3*(nexternal-nexternal_prod)+1
              call  ntuple(x(i),0d0,1d0,i,1)
           enddo

           call generate_momenta_conf(jac,x,itree,qmass,qwidth,pfull,pprod,map_external2res) 
           if (jac.lt.0d0) then
             counter2=counter2+1 
             counter3=counter3+1 
c             if (counter3.gt.500) then
c               write(*,*) "500_pts_failed_stop_executation"
c               stop
c             endif
             if (counter2.ge.8) then ! use another topology to generate PS points
               do k=1,n_max_cg
                 amp2(k)=0d0
               enddo
               CALL SMATRIX_PROD(P,M_PROD)
               call get_config(iconfig)
               do i=-nexternal_prod+2,-1
                do j=1,2
                  itree(j,i)=iforest(j,i,iconfig)
                enddo
               enddo

               do i=-nexternal_prod+3,-1
                 qmass(i)=prmass(i,iconfig)
                 qwidth(i)=prwidth(i,iconfig)
               enddo
               call  merge_itree(itree,qmass,qwidth, p,map_external2res)
               counter2=0
             endif

             cycle
           endif
           call SMATRIX(pfull,M_full)
           call SMATRIX_PROD(pprod,M_prod)

           weight=M_full*jac/M_prod

           if (weight.gt.x(3*(nexternal-nexternal_prod)+1)*maxweight) notpass=.false.
        enddo

c       
        call get_helicity_ID(iconfig)
        call get_helicities(iconfig,helset) 

        if (nb_mc_masses.gt.0) then 

          no_gen=.True.
          do k=1,nb_mc_masses
             m(indices_mc_masses(k))=values_mc_masses(k)
          enddo
          call generate_momenta_conf(jac,x,itree,qmass,qwidth,ptrial,pprod,map_external2res) 
          
          if (jac.lt.0d0) then
            write(*,*) nexternal,  counter, maxBW, weight, counter3, 0
            do i=1,nexternal
               write (*,*) (pfull(j,i), j=0,3)  
            enddo
          else
            write(*,*) nexternal,  counter, maxBW, weight, counter2, 1
            do i=1,nexternal
              write (*,*) (ptrial(j,i), j=0,3)  
            enddo
          endif
        else
          write(*,*) nexternal,  counter, maxBW, weight, counter2, 0
          do i=1,nexternal
            write (*,*) (pfull(j,i), j=0,3)  
          enddo
        endif

        write(*,*) (helset(j),j=1,nexternal)

        call flush()
        goto 1
      endif

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c   VIc. return M_full^2                                                c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

       if (mode.eq.3) then

           jac=1d0
           ivar=0
           no_gen=.false.

         notpass=.true.
         do while (notpass)

           do i = 1, 3*(nexternal-nexternal_prod)+1
              call  ntuple(x(i),0d0,1d0,i,1)
           enddo

           call generate_momenta_conf(jac,x,itree,qmass,qwidth,pfull,pprod,map_external2res) 
           if (jac.lt.0d0) cycle
           notpass=.false.
           call SMATRIX(pfull,M_full)

           write(*,*) M_full
        enddo
        call flush()
        goto 1
      endif

2     continue
      end

      subroutine get_helicity_ID(iconfig)
      implicit none
      include 'helamp.inc'
c     argument
      integer iconfig

c     local
      integer i
      double precision cumulweight(0:ncomb1),random


      cumulweight(0)=0d0
      do i=1,ncomb1
         cumulweight(i)=helamp(i)+cumulweight(i-1)
      enddo

      call  ntuple(random,0d0,1d0,24,1)

      do i=1,ncomb1
         cumulweight(i)=cumulweight(i)/cumulweight(ncomb1)
         !write(*,*) random
         !write(*,*) cumulweight(i-1)
         !write(*,*) cumulweight(i)
         if (random.ge.cumulweight(i-1).and.random.le.cumulweight(i))then
           iconfig=i
           return
         endif
      enddo


      return
      end


      subroutine get_config(iconfig)
      implicit none

C---  integer    n_max_cg
      INCLUDE "ngraphs.inc"     !how many diagrams 

c     argument
      integer iconfig

c     local
      integer i
      double precision cumulweight(0:n_max_cg),random

c     common
      DOUBLE PRECISION AMP2(n_max_cg)
      COMMON/TO_AMPS/  AMP2

      integer            mapconfig(0:N_MAX_CG), this_config
      common/to_mconfigs/mapconfig, this_config

      if (mapconfig(0).eq.1) then
        iconfig=1
        return
      endif

      cumulweight(0)=0d0
      do i=1,mapconfig(0)
         cumulweight(i)=amp2(mapconfig(i))+cumulweight(i-1)
      enddo

      !do i=1,100
      call  ntuple(random,0d0,1d0,24,1)
      !enddo

      do i=1,mapconfig(0)
         cumulweight(i)=cumulweight(i)/cumulweight(mapconfig(0))
         !write(*,*) random
         !write(*,*) cumulweight(i-1)
         !write(*,*) cumulweight(i)
         if (random.le.cumulweight(i)) then 
           iconfig=i
           return
         endif 
      enddo

      write(*,*) 'Unable to generate iconfig ', random, mapconfig(0),
     -  amp2, cumulweight

      end


      subroutine set_parameters(p,Ecollider)
      implicit none

      double precision ZERO
      parameter (ZERO=0D0)
      include 'nexternal_prod.inc'
c     argument
      double precision p(0:3, nexternal_prod), Ecollider

c     local 
      integer i, j
      double precision ptot(0:3)

      include 'nexternal.inc'
      include 'coupl.inc'
      INCLUDE "input.inc"
      !include 'run.inc'
       ! variables associate with the PS generation
       double precision totmassin, totmass
       double precision shat, sqrtshat, stot, y, m(-nexternal:nexternal)
       integer nbranch, ns_channel,nt_channel, pos_pz
       common /to_topo/
     & totmassin, totmass,shat, sqrtshat, stot,y, m,
     & nbranch, ns_channel,nt_channel, pos_pz

c Masses of particles. Should be filled in setcuts.f
      double precision pmass(nexternal)
      common /to_mass/pmass

      double precision dot
      external dot

      include "../parameters.inc"
      stot=Ecollider**2

      call coup()

      include 'pmass.inc'

      ! m(i) pmass(i) already refer to the full kinematics
      do i=1,nexternal
            m(i)=pmass(i)
      enddo

      if (p(3,1).gt.p(3,2)) then
         pos_pz=1
      else 
         pos_pz=2
      endif
      do j = 0,3
        ptot(j)=p(j,1)+p(j,2)
      enddo
      shat=dot(ptot,ptot)
      sqrtshat=sqrt(shat)
      y = 0.5*log((ptot(0)+ptot(3))/(ptot(0)-ptot(3)))
c      write(*,*) shat
c      write(*,*) sqrtshat
c      write(*,*) 'y',y
c      write(*,*) (ptot(j), j=0,3)


c Make sure have enough mass for external particles
      totmassin=0d0
      do i=3-nincoming,2
         totmassin=totmassin+pmass(i)
      enddo
      totmass=0d0

      do i=3,nexternal 
         totmass=totmass+m(i)
      enddo
      if (stot .lt. max(totmass,totmassin)**2) then
         write (*,*) 'Fatal error #0 in one_tree:'/
     &         /'insufficient collider energy'
         stop
      endif

      end


      subroutine merge_itree(itree,qmass,qwidth,  p_ext,mapext2res)
      implicit none
      !include 'genps.inc'
      include 'coupl.inc'
      include 'nexternal_prod.inc'
      include 'nexternal.inc'
c
c     arguments
c
      integer itree(2,-nexternal:-1)   ! PS structure for the production
      double precision qmass(-nexternal:0),qwidth(-nexternal:0) 
      double precision p_ext(0:3,nexternal_prod)
      integer mapext2res(nexternal_prod)
c
c     local
c
      double precision normp
      ! info for the full process
      integer itree_full(2,-nexternal:-1),i,j
      double precision qmass_full(-nexternal:0),qwidth_full(-nexternal:0) 
      ! info for the decay part
      double precision idecay(2,-nexternal:-1), prmass(-nexternal:-1),prwidth(-nexternal:-1) 
      integer ns_channel_decay
      integer  map_external2res(nexternal_prod) ! map (index in production) -> index in the full structure
      double precision p(0:3,-nexternal:nexternal)
 
      integer idB, id1, index_p2, which_initial, last_branch, d1,d2
      double precision pa(0:3), pb(0:3), p1(0:3), p2(0:3),pboost(0:3)
      double precision pb_cms(0:3), p1_cms(0:3), p1_rot(0:3),temp

c     common
      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

       ! variables associate with the PS generation
       double precision totmassin, totmass
       double precision shat, sqrtshat, stot, y, m(-nexternal:nexternal)
       integer nbranch, ns_channel,nt_channel, pos_pz
       common /to_topo/
     & totmassin, totmass,shat, sqrtshat, stot,y, m,
     & nbranch, ns_channel,nt_channel, pos_pz

      double precision phi
      external phi
      double precision dot
      external dot

      !include 'run.inc'
      include  'configs_decay.inc'

      mapext2res=map_external2res
      which_initial=-1
c Determine number of s- and t-channel branches, at this point it
c includes the s-channel p1+p2
c      write(*,*) (itree(i,-1), i=1,2)
c      write(*,*) qmass(-1)
c      write(*,*) qwidth(-1)

         nbranch=nexternal_prod -2
         ns_channel=1
         do while(itree(1,-ns_channel).ne.1 .and.
     &        itree(1,-ns_channel).ne.2 .and. ns_channel.lt.nbranch)
         !   m(-ns_channel)=0d0
            ns_channel=ns_channel+1
         enddo
         ns_channel=ns_channel - 1
         nt_channel=nbranch-ns_channel-1
c If no t-channles, ns_channels is one less, because we want to exclude
c the s-channel p1+p2
         if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
            ns_channel=ns_channel-1
         endif
c Set one_body to true if it s a 2->1 process at the Born (i.e. 2->2 for the n+1-body)
         if((nexternal-nincoming).eq.1)then
            !one_body=.true.
            ns_channel=0
            nt_channel=0
         elseif((nexternal-nincoming).gt.1)then
            continue
            !one_body=.false.
         else
            write(*,*)'Error#1 in genps_madspin.f',nexternal,nincoming
            stop
         endif

c      write(*,*) 'ns_channel ',ns_channel 
c      write(*,*) 'nt_channel ',nt_channel 

      ! first fill the new itree for the the legs in the decay part
      do i=-(ns_channel_decay),-1       
         itree_full(1,i) = idecay(1,i)
         itree_full(2,i) = idecay(2,i)
         qmass_full(i)=prmass(i)
         qwidth_full(i)=prwidth(i)
         keep_inv(i)=.FALSE.
      enddo
 
c      write(*,*) (itree_full(i,-1), i=1,2)
c      write(*,*) (itree_full(i,-2), i=1,2)
c      write(*,*) (itree_full(i,-3), i=1,2)
c      write(*,*) (itree_full(i,-4), i=1,2)
c      write(*,*) qmass_full(-1)
c      write(*,*) qmass_full(-2)
c      write(*,*) qmass_full(-3)
c      write(*,*) qmass_full(-4)
c      write(*,*) qwidth_full(-1)
c      write(*,*) qwidth_full(-2)
c      write(*,*) qwidth_full(-3)
c      write(*,*) qwidth_full(-4)
  
      ! store the external momenta in the production event a
      ! new variable p that has the same labeling system as the new itree
      do i=1, nexternal_prod
          do j=0,3
              p(j,map_external2res(i)) = p_ext(j,i)
          enddo
      enddo

      ! fill the new itree with the kinematics associated with the production
      do i=-1,-(ns_channel+nt_channel)-1,-1  ! loop over invariants in the production
c         write(*,*) 'i prod',i
c         write(*,*) 'i full',i-ns_channel_decay
c         write(*,*) 'd1 prod',itree(1,i)
c         write(*,*) 'd2 prod',itree(2,i)
         if (itree(1,i).lt.0 ) then
c            write(*,*) 'd1 full',itree(1,i)-ns_channel_decay
            itree_full(1,i-ns_channel_decay) = itree(1,i)-ns_channel_decay
         else 
             if (itree(1,i).le.2) which_initial=itree(1,i)
             itree_full(1,i-ns_channel_decay) = map_external2res(itree(1,i))
         endif 
         if (itree(2,i).lt.0 ) then
c            write(*,*) 'd2 full',itree(2,i)-ns_channel_decay
             itree_full(2,i-ns_channel_decay) = itree(2,i)-ns_channel_decay
         else 
             if (itree(2,i).le.2) which_initial=itree(2,i)
             itree_full(2,i-ns_channel_decay) = map_external2res(itree(2,i))
         endif
         
         ! record the momentum of the intermediate leg         
         do j=0,3
             if (nt_channel.ne.0.and.i .lt.-ns_channel) then
             p(j,i-ns_channel_decay)=p(j,itree_full(1,i-ns_channel_decay))-p(j,itree_full(2,i-ns_channel_decay))
             else 
             p(j,i-ns_channel_decay)=p(j,itree_full(1,i-ns_channel_decay))+p(j,itree_full(2,i-ns_channel_decay))
             endif
         enddo
         
         keep_inv(i-ns_channel_decay)=.TRUE.
 
         if (i.ne.(-ns_channel-nt_channel-1)) then
            fixedinv(i-ns_channel_decay)=
     &           dot(p(0,i-ns_channel_decay),p(0,i-ns_channel_decay))
c MODIF May, 6, 2014 (R.F.)
c Due to numerical instabilities, sometimes the computation of the
c invariant gets the wrong sign. This happens only when the invariant is
c close to zero. If this is the case, the code might go into an infinite
c loop (because no momenta can be generated later). Simply fix it by
c forcing the correct sign for the invariants.
           if (nt_channel.ne.0.and.i .lt.-ns_channel) then
              if (fixedinv(i-ns_channel_decay).gt.stot/500d0) then
                 write(*,*)  'Error: t-channel invariant has a'/
     $                /' value larger than +stot/500 ' 
                 stop
              elseif (fixedinv(i-ns_channel_decay).gt.0d0) then
                 fixedinv(i-ns_channel_decay)=
     &                -fixedinv(i-ns_channel_decay)
              endif
           else
              if (fixedinv(i-ns_channel_decay).lt.-stot/500d0) then
                 write(*,*)  'Error: s-channel invariant has a'/
     $                /' value smaller than -stot/500 ' 
                 stop
              elseif (fixedinv(i-ns_channel_decay).lt.0d0) then
                 fixedinv(i-ns_channel_decay)=
     &                -fixedinv(i-ns_channel_decay)
              endif
           endif
c end MODIF R.F.
           qmass_full(i-ns_channel_decay)=qmass(i)
           qwidth_full(i-ns_channel_decay)=qwidth(i)
         endif
      enddo 
c
c        MODIF March 5, 2014 (P.A.) 
c        overwrite last t-channel invariant to avoid numerical unstabilities
c        This is crucial if the last t-channel propa is connected to two massless particles
c        (one inital + one final particles) which are colinear -> t ~ 0
c        Indeed, in that case the extraction of t in the previous loop may return a positive 
c        value: in that case, the code will always fail to generate momenta !!!

         if (nt_channel.ge.2) then
           last_branch=-ns_channel-nt_channel-1
           if (itree(1, last_branch).ge.0.or.itree(2, last_branch).ge.0) then
           ! only overwrite if last t-channel splitting is connected to initial particle + massless particle
           if (itree(1, last_branch).ge.0) d1=itree(1, last_branch)
           if (itree(2, last_branch).ge.0) d1=itree(2, last_branch)
           if (which_initial.eq.1) then 
              d2=2
           elseif (which_initial.eq.2) then
              d2=1
           else
              write(*,*) 'Problem with the last t-channel branching ',which_initial 
              stop
           endif 

           temp=-2d0*dot(p_ext(0,d1),p_ext(0,d2))
           if (map_external2res(d1).gt.0) then
           temp=temp+m(map_external2res(d1))**2
           else
           temp=temp+qmass_full(map_external2res(d1))**2
           endif
           temp=temp+m(map_external2res(d2))**2

           shat=2d0*dot(p_ext(0,1),p_ext(0,2))

c           write(*,*) temp, fixedinv(-ns_channel-nt_channel-ns_channel_decay)

c           pause
           if (abs(temp-fixedinv(-ns_channel-nt_channel-ns_channel_decay)).lt.(shat/1d2)) then 
           fixedinv(-ns_channel-nt_channel-ns_channel_decay)=temp
           else
c              write(*,*)  qmass_full(map_external2res(d1)), m(map_external2res(d2))
c              write(*,*) d1, d2, itree(1, last_branch), itree(2, last_branch)
              write(*,*) 'cannot extract last t-channel invariant', temp, fixedinv(-ns_channel-nt_channel-ns_channel_decay)
              stop
           endif
           endif
         endif



c     END MODIF MARCH 5, 2014
 
      !write(*,*) -ns_channel-nt_channel-1
      !write(*,*) map_external2res(itree(2,-ns_channel-nt_channel-1))
      !write(*,*) itree(1,-ns_channel-nt_channel-1)
      !write(*,*) itree(2,-ns_channel-nt_channel-1)
      !write(*,*) nbranch
      !write(*,*) nt_channel
 

      !write(*,*) (itree_full(i,-1), i=1,2)
c      write(*,*) (itree_full(i,-2), i=1,2)
c      write(*,*) (itree_full(i,-3), i=1,2)
c      write(*,*) (itree_full(i,-4), i=1,2)
c      write(*,*) (itree_full(i,-5), i=1,2)
c c     write(*,*) (itree_full(i,-6), i=1,2)
c c     write(*,*) qmass_full(-1)
c      write(*,*) qmass_full(-2)
c      write(*,*) qmass_full(-3)
c      write(*,*) qmass_full(-4)
c      write(*,*) qmass_full(-5)
c      write(*,*) qwidth_full(-1)
c      write(*,*) qwidth_full(-2)
c      write(*,*) qwidth_full(-3)
c      write(*,*) qwidth_full(-4)
c      write(*,*) qwidth_full(-5)

c       write(*,*) 'p -2', (p(j,-2), j=0,3)
c       write(*,*) 'p -4', (p(j,-4), j=0,3)
c       write(*,*) 'p  9', (p(j,9), j=0,3)
c      !overwrite the previous information
      nbranch=nexternal-2
      ns_channel= ns_channel+ns_channel_decay
c      write(*,*) 'ns_channel ',ns_channel 
c      write(*,*) 'nt_channel ',nt_channel 
      do i =-(ns_channel+nt_channel)-1,-1
         itree(1,i) =itree_full(1,i)
         itree(2,i) =itree_full(2,i)
         qmass(i)=qmass_full(i)
         qwidth(i)=qwidth_full(i)
      enddo

      ! extract the phi and m2 numbers for each t-channel branching
      do i=-(ns_channel+nt_channel),-ns_channel-1
          ! the t-branching process has the structure A+B > 1 + 2 
          ! with A and 1 the two daughters in itree
         idB=itree(1,i)
         id1=itree(2,i)      
          do j=0,3
             pa(j)=p(j,2)
             pb(j)=p(j,idB) 
             p1(j)=p(j,id1)
             p2(j)=pa(j)+ pb(j)-p1(j) 
          enddo    
             m2_tchan(i)=dot(p2,p2)
             if (m2_tchan(i).gt.0d0) then 
                 m2_tchan(i)=sqrt(m2_tchan(i))
             else
                ! might be negative because of numerical unstabilities
                index_p2=itree(2,i-1)
                if (index_p2.gt.0) then
                   m2_tchan(i)=m(index_p2)
                else
        write(*,*) 'Warning: m_2^2 is negative in t-channel branching ',m2_tchan(i) 
                endif
            endif
         ! extract phi
         do j=0,3
            pboost(j) = pa(j)+pb(j)
            if (j .gt. 0) then
               pboost(j) = -pboost(j)
            endif
         enddo
c
c        go to  pa+pb cms system
c
         call boostx(pb,pboost,pb_cms)
         call boostx(p1,pboost,p1_cms)
c        rotate such that pb is aligned with z axis
         call invrot(p1_cms, pb_cms,p1_rot)
         phi_tchan(i)=phi(p1_rot)
      enddo

      do i=-ns_channel-1, -1
          if (keep_inv(i)) then
            if (nt_channel.ne.0.and.i.eq.(ns_channel+1)) cycle
            pboost(0)=p(0,i) 
            do j=1,3
               pboost(j)=-p(j,i)
            enddo
            call boostx(p(0,itree(1,i)),pboost,p1)
            call boostx(p(0,itree(2,i)),pboost,p2)
c            write(*,*) 'p1 ', (p1(j), j=0,3)
c            write(*,*) 'p2 ', (p2(j), j=0,3)
            normp=sqrt(p1(1)**2+p1(2)**2+p1(3)**2)
            cosphi_schan(i)=p1(3)/normp
            phi_schan(i)=phi(p1) 
         endif
      enddo
      end


      subroutine generate_momenta_conf(jac,x,itree,qmass,qwidth,p,p_prod,mapext2res)
c
c
      implicit none
      !include 'genps.inc'
      include 'nexternal.inc'
      include 'nexternal_prod.inc'
      !include 'run.inc'
c arguments
      double precision jac,x(36),p(0:3,nexternal), p_prod(0:3,nexternal)
      integer itree(2,-nexternal:-1)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
c common

      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

c     
       ! variables associate with the PS generation
       double precision totmassin, totmass
       double precision shat, sqrtshat, stot, y, m(-nexternal:nexternal)
       integer nbranch, ns_channel,nt_channel,pos_pz
       common /to_topo/
     & totmassin, totmass,shat, sqrtshat, stot,y, m,
     & nbranch, ns_channel,nt_channel, pos_pz


c Masses of particles. Should be filled in setcuts.f
      double precision pmass(nexternal)
      common /to_mass/pmass

c local
      integer  mapext2res(nexternal_prod) ! map (index in production) -> index in the full structure
      integer i,j,
     &     imother
      double precision pb(0:3,-nexternal:nexternal)
     &     ,S(-nexternal:nexternal)
     &     ,xjac,xpswgt
     &     ,pwgt,p_born_CHECK(0:3,nexternal), ptot(0:3)
      logical pass

      double precision smax, smin, xm02,bwmdpl,bwmdmn,bwfmpl,bwfmmn
      double precision bwdelf, BWshift

c external
      double precision lambda
      external lambda
      double precision xbwmass3,bwfunc
      external xbwmass3,bwfunc
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      logical firsttime
      data firsttime/.true./
      double precision zero
      parameter (zero=0d0)
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-nexternal:-1,-1:1),
     &     cBW_width(-nexternal:-1,-1:1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level

       double precision BWcut, maxBW
       common /to_BWcut/BWcut, maxBW

      pass=.true.

     
c
      xjac=1d0
      xpswgt=1d0


c     STEP 1: generate the initial momenta

      if (nexternal_prod.gt.3) then
      s(-nbranch)  = shat
      m(-nbranch)  = sqrtshat
      pb(0,-nbranch)= m(-nbranch)

      else

c      write(*,*) 'nbranch' 
c      write(*,*) nbranch 
      ! P.A.: Shat needs to be generated accoding to a BW distribution
      smax=min(stot*0.99D0,(qmass(-nbranch+1)+BWcut*qwidth(-nbranch+1))**2 )
      smin=max(0.1d0,(qmass(-nbranch+1)-BWcut*qwidth(-nbranch+1))**2 )
      xm02=qmass(-nbranch+1)**2
      bwmdpl=smax-xm02
      bwmdmn=xm02-smin
      bwfmpl=atan(bwmdpl/(qmass(-nbranch+1)*qwidth(-nbranch+1)))
      bwfmmn=atan(bwmdmn/(qmass(-nbranch+1)*qwidth(-nbranch+1)))
      bwdelf=(bwfmpl+bwfmmn)/pi
      ivar=ivar+1
      s(-nbranch)=xbwmass3(x(ivar),xm02,qwidth(-nbranch+1),bwdelf
     &              ,bwfmmn)
 

      if (s(-nbranch).gt.smax.or.s(-nbranch).lt.smin) then
          xjac=-1d0
          pass=.false.
          return
      endif
       shat=s(-nbranch)
       sqrtshat=dsqrt(s(-nbranch))
       xjac=xjac*bwdelf/bwfunc(s(-nbranch),xm02,qwidth(-nbranch+1))
       m(-nbranch) = dsqrt(s(-nbranch))
       BWshift=abs(m(-nbranch)-qmass(-nbranch+1))/qwidth(-nbranch+1)
       if (BWshift.gt.maxBW) maxBW=BWshift
       pb(0,-nbranch)=m(-nbranch)
      endif
c      write(*,*) x(ivar)
c      write(*,*) smax
c      write(*,*) smin
c      write(*,*) nbranch
c      write(*,*) qmass(-nbranch+1)
c      write(*,*) qwidth(-nbranch+1)
c      write(*,*)  m(-nbranch)
     
      pb(1,-nbranch)= 0d0
      pb(2,-nbranch)= 0d0
      pb(3,-nbranch)= 0d0

      if(nincoming.eq.2) then
        if (pos_pz.eq.1)then
        call mom2cx(sqrtshat,m(1),m(2),1d0,0d0,pb(0,1),pb(0,2))
        else
        call mom2cx(sqrtshat,m(2),m(1),1d0,0d0,pb(0,2),pb(0,1))
        endif
      else
         pb(0,1)=sqrtshat
         do i=1,2
            pb(0,1)=0d0
         enddo
      endif

       ptot(0)=sqrtshat*cosh(y)
       ptot(1)=0d0
       ptot(2)=0d0
       ptot(3)=sqrtshat*sinh(y)
       call boostx(pb(0,1),ptot,pb(0,1))
       call boostx(pb(0,2),ptot,pb(0,2))
c

c    STEP 2:  generate all the invariant masses of the s-channels
      call generate_inv_mass_sch(ns_channel,itree,m,sqrtshat
     &     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac,pass)

      !write(*,*) 'jac s-chan ', xjac
      if (.not.pass) then
        jac=-1d0
        return
      endif


c If only s-channels, also set the p1+p2 s-channel
      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         s(-nbranch+1)=s(-nbranch) 
         m(-nbranch+1)=m(-nbranch)       !Basic s-channel has s_hat 
         pb(0,-nbranch+1) = m(-nbranch+1)*cosh(y)!and 0 momentum
         pb(1,-nbranch+1) = 0d0
         pb(2,-nbranch+1) = 0d0
         pb(3,-nbranch+1) = m(-nbranch+1)*sinh(y)
      endif

c
c     STEP 3:  do the T-channel branchings
c
      if (nt_channel.ne.0) then
         call generate_t_channel_branchings(ns_channel,nbranch,itree,m,s
     &        ,x,pb,xjac,xpswgt,pass)
         if (.not.pass) then
             jac=-1d0
             return 
         endif
      endif
      !write(*,*) 'jac t-chan ', xjac
c
c     STEP 4: generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
      call fill_momenta(nbranch,nt_channel
     &     ,x,itree,m,s,pb,xjac,xpswgt,pass)
      if (.not.pass) then
         jac=-1d0
         return
      endif 
      !write(*,*) 'jac fill ', xjac

      do i = 1, nexternal
       do j = 0, 3
         p(j,i)=pb(j,i)
       enddo
      enddo


      do i = 1, nexternal_prod
       do j = 0, 3
         p_prod(j,i)=pb(j,mapext2res(i))
       enddo
      enddo

      jac=xjac*xpswgt

      return
      end


      subroutine fill_momenta(nbranch,nt_channel
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      !include 'genps.inc'
      include 'nexternal.inc'
      integer nbranch,nt_channel,ionebody
      double precision M(-nexternal:nexternal),x(36)
      double precision s(-nexternal:nexternal)
      double precision pb(0:3,-nexternal:nexternal)
      integer itree(2,-nexternal:-1)
      double precision xjac0,xpswgt0
      logical pass,one_body
c
      double precision one
      parameter (one=1d0)
      double precision costh,phi,xa2,xb2
      integer i,ix,j
      double precision lambda,dot
      external lambda,dot
      double precision vtiny
      parameter (vtiny=1d-12)

      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

      pass=.true.

      do i = -nbranch+nt_channel+(nincoming-1),-1
         if (keep_inv(i).or.no_gen) then 
           costh=cosphi_schan(i)
           phi=phi_schan(i)
         else    
           ivar=ivar+1
           costh= 2d0*x(ivar)-1d0
           ivar=ivar+1
           phi  = 2d0*pi*x(ivar)
           phi_schan(i)=phi
           cosphi_schan(i)=costh
           xjac0 = xjac0 * 4d0*pi
         endif
         xa2 = m(itree(1,i))*m(itree(1,i))/s(i)
         xb2 = m(itree(2,i))*m(itree(2,i))/s(i)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            xjac0=-8
            pass=.false.
            return
         endif

c         write(*,*) i
c         write(*,*) sqrt(s(i))
c         write(*,*) m(itree(1,i))
c         write(*,*) m(itree(2,i))
c         write(*,*)  xa2, xb2

         if (.not.keep_inv(i).and..not.no_gen) then 
           xpswgt0 = xpswgt0*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI)
         endif
c         write(*,*)  xpswgt0

         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        pb(0,itree(1,i)),pb(0,itree(2,i)))
c          write(*,*) 'i ', i
c         write(*,*) 'costh, phi ', costh,phi
c         write(*,*) 'm init ', m(i)
c          write(*,*) (pb(j,itree(1,i)), j=0,3)
c          write(*,*) (pb(j,itree(2,i)), j=0,3)
c If there is an extremely large boost needed here, skip the phase-space point
c because of numerical stabilities.
         if (dsqrt(abs(dot(pb(0,i),pb(0,i))))/pb(0,i) 
     &        .lt.vtiny) then
            xjac0=-81
            pass=.false.
            return
         else
            call boostx(pb(0,itree(1,i)),pb(0,i),pb(0,itree(1,i)))
            call boostx(pb(0,itree(2,i)),pb(0,i),pb(0,itree(2,i)))
c          write(*,*) (pb(j,itree(1,i)), j=0,3)
c          write(*,*) (pb(j,itree(2,i)), j=0,3)
         endif
      enddo
c
c
c Special phase-space fix for the one_body
c      if (one_body) then
c Factor due to the delta function in dphi_1
c         xpswgt0=pi/m(ionebody)
c Kajantie s normalization of phase space (compensated below in flux)
c         xpswgt0=xpswgt0/(2*pi)
c         do i=0,3
c            pb(i,3) = pb(i,1)+pb(i,2)
c         enddo
c      endif
      return
      end


      subroutine generate_inv_mass_sch(ns_channel,itree,m,sqrtshat
     &     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      !include 'genps.inc'
      include 'nexternal.inc'

      double precision  totmass,BWshift
      double precision sqrtshat,  m(-nexternal:nexternal)
      integer  ns_channel
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision x(36)
      double precision s(-nexternal:nexternal)
      double precision xjac0
      integer itree(2,-nexternal:-1)
      integer i,j
      double precision smin,smax,xm02,bwmdpl,bwmdmn,bwfmpl,bwfmmn,bwdelf
     &     ,totalmass
      double precision xbwmass3,bwfunc
      external xbwmass3,bwfunc
      logical pass
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-nexternal:-1,-1:1),
     &     cBW_width(-nexternal:-1,-1:1)
      double precision b(-1:1),x0

      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

       double precision BWcut, maxBW
       common /to_BWcut/BWcut, maxBW

      pass=.true.
      totalmass=totmass
      do i = -1,-ns_channel,-1
c Generate invariant masses for all s-channel branchings of the Born
         if (keep_inv(i).or.no_gen) then 
            s(i)=fixedinv(i)
            goto 503
         endif
         smin = (m(itree(1,i))+m(itree(2,i)))**2
         smax = (sqrtshat-totalmass+sqrt(smin))**2
         if(smax.lt.smin.or.smax.lt.0.d0.or.smin.lt.0.d0)then
            write(*,*)'Error#13 in genps_madspin.f'
            write(*,*)smin,smax,i
            stop
         endif
         ivar=ivar+1
c Choose the appropriate s given our constraints smin,smax
         if(qwidth(i).ne.0.d0)then
c Breit Wigner
            if (cBW(i).eq.1 .and.
     &          cBW_width(i,1).gt.0d0 .and. cBW_width(i,-1).gt.0d0) then
c     conflicting BW on both sides
               do j=-1,1,2
                  b(j)=(cBW_mass(i,j)-qmass(i))/
     &                 (qwidth(i)+cBW_width(i,j))
                  b(j)=qmass(i)+b(j)*qwidth(i)
                  b(j)=b(j)**2
               enddo
               if (x(ivar).lt.1d0/3d0) then
                  x0=3d0*x(ivar)
                  s(i)=(b(-1)-smin)*x0+smin
                  xjac0=3d0*xjac0*(b(-1)-smin)
               elseif (x(ivar).gt.1d0/3d0 .and. x(ivar).lt.2d0/3d0) then
                  x0=3d0*x(ivar)-1d0
                  xm02=qmass(i)**2
                  bwmdpl=b(1)-xm02
                  bwmdmn=xm02-b(-1)
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=3d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               else
                  x0=3d0*x(ivar)-2d0
                  s(i)=(smax-b(1))*x0+b(1)
                  xjac0=3d0*xjac0*(smax-b(1))
               endif
            elseif (cBW(i).eq.1.and.cBW_width(i,1).gt.0d0) then
c     conflicting BW with alternative mass larger
               b(1)=(cBW_mass(i,1)-qmass(i))/
     &              (qwidth(i)+cBW_width(i,1))
               b(1)=qmass(i)+b(1)*qwidth(i)
               b(1)=b(1)**2
               if (x(ivar).lt.0.5d0) then
                  x0=2d0*x(ivar)
                  xm02=qmass(i)**2
                  bwmdpl=b(1)-xm02
                  bwmdmn=xm02-smin
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=2d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               else
                  x0=2d0*x(ivar)-1d0
                  s(i)=(smax-b(1))*x0+b(1)
                  xjac0=2d0*xjac0*(smax-b(1))
               endif
            elseif (cBW(i).eq.1.and.cBW_width(i,-1).gt.0d0) then
c     conflicting BW with alternative mass smaller
               b(-1)=(cBW_mass(i,-1)-qmass(i))/
     &              (qwidth(i)+cBW_width(i,-1)) ! b(-1) is negative here
               b(-1)=qmass(i)+b(-1)*qwidth(i)
               b(-1)=b(-1)**2
               if (x(ivar).lt.0.5d0) then
                  x0=2d0*x(ivar)
                  s(i)=(b(-1)-smin)*x0+smin
                  xjac0=2d0*xjac0*(b(-1)-smin)
               else
                  x0=2d0*x(ivar)-1d0
                  xm02=qmass(i)**2
                  bwmdpl=smax-xm02
                  bwmdmn=xm02-b(-1)
                  bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
                  bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
                  bwdelf=(bwfmpl+bwfmmn)/pi
                  s(i)=xbwmass3(x0,xm02,qwidth(i),bwdelf
     &                 ,bwfmmn)
                  xjac0=2d0*xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               endif
            else
c     normal BW
               ! P.A.: introduce the BWcutoff here
               smax=min(smax,(qmass(i)+BWcut*qwidth(i))**2 )
               smin=max(smin,(qmass(i)-BWcut*qwidth(i))**2 )

               xm02=qmass(i)**2
               bwmdpl=smax-xm02
               bwmdmn=xm02-smin
               bwfmpl=atan(bwmdpl/(qmass(i)*qwidth(i)))
               bwfmmn=atan(bwmdmn/(qmass(i)*qwidth(i)))
               bwdelf=(bwfmpl+bwfmmn)/pi
               s(i)=xbwmass3(x(ivar),xm02,qwidth(i),bwdelf
     &              ,bwfmmn)

               if (s(i).gt.smax.or.s(i).lt.smin) then
                  xjac0=-1d0
                  pass=.false.
                  return
               endif

               xjac0=xjac0*bwdelf/bwfunc(s(i),xm02,qwidth(i))
               !write(*,*) i , sqrt(s(i))
            endif
         else
c not a Breit Wigner
            s(i) = (smax-smin)*x(ivar)+smin
            xjac0 = xjac0*(smax-smin)
         endif

c If numerical inaccuracy, quit loop
         if (xjac0 .lt. 0d0) then
            xjac0 = -6
            pass=.false.
            return
         endif
         if (s(i) .lt. smin) then
            xjac0=-5
            pass=.false.
            return
         endif
         fixedinv(i)=s(i)
c
c     fill masses, update totalmass
c
503      continue
         m(i) = sqrt(s(i))
         if (.not.keep_inv(i).and..not.no_gen) then
           BWshift=abs(m(i)-qmass(i))/qwidth(i)
           if (BWshift.gt.maxBW) maxBW=BWshift
         endif
         totalmass=totalmass+m(i)-
     &        m(itree(1,i))-m(itree(2,i))
         if ( totalmass.gt.sqrtshat )then
            xjac0 = -4
            pass=.false.
            return
         endif
      enddo
      return
      end



      subroutine generate_t_channel_branchings(ns_channel,nbranch,itree
     &     ,m,s,x,pb,xjac0,xpswgt0,pass)
c First we need to determine the energy of the remaining particles this
c is essentially in place of the cos(theta) degree of freedom we have
c with the s channel decay sequence
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      !include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac0,xpswgt0
      double precision M(-nexternal:nexternal),x(36)
      double precision s(-nexternal:nexternal)
      double precision pb(0:3,-nexternal:nexternal)
      integer itree(2,-nexternal:-1)
      integer ns_channel,nbranch
      logical pass
c
      double precision totalmass,smin,smax,s1,ma2,mbq,m12,mnq,tmin,tmax
     &     ,t,tmax_temp,phi
      integer i,ibranch
      double precision lambda,dot
      external lambda,dot

      ! variables to keep track of the vegas numbers for the production part
      logical keep_inv(-nexternal:-1),no_gen
      integer ivar
      double precision fixedinv(-nexternal:0)
      double precision phi_tchan(-nexternal:0),m2_tchan(-nexternal:0)
      double precision cosphi_schan(-nexternal:0), phi_schan(-nexternal:0)
      common /to_fixed_kin/keep_inv,no_gen, ivar, fixedinv,
     & phi_tchan,m2_tchan,cosphi_schan, phi_schan 

c 
      pass=.true.
      totalmass=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totalmass=totalmass+m(itree(2,ibranch))
      enddo
      m(-ns_channel-1) = dsqrt(S(-nbranch))
c     
c Choose invariant masses of the pseudoparticles obtained by taking together
c all final-state particles or pseudoparticles found from the current 
c t-channel propagator down to the initial-state particle found at the end
c of the t-channel line.
      do ibranch = -ns_channel-1,-nbranch+2,-1
         totalmass=totalmass-m(itree(2,ibranch))  
         smin = totalmass**2                    
         smax = (m(ibranch) - m(itree(2,ibranch)))**2
         if (smin .gt. smax) then
            xjac0=-3d0
            pass=.false.
            return
         endif
         if (keep_inv(ibranch).or.no_gen) then
            m(ibranch-1)=m2_tchan(ibranch)
         else
            ivar=ivar+1
            m(ibranch-1)=dsqrt((smax-smin)*
     &        x(ivar))
            m2_tchan(ibranch)=m(ibranch-1)
            xjac0 = xjac0*(smax-smin)
         endif
         if (m(ibranch-1)**2.lt.smin.or.m(ibranch-1)**2.gt.smax
     &        .or.m(ibranch-1).ne.m(ibranch-1)) then
            xjac0=-1d0
            pass=.false.
            return
         endif
      enddo
c     
c Set m(-nbranch) equal to the mass of the particle or pseudoparticle P
c attached to the vertex (P,t,p2), with t being the last t-channel propagator
c in the t-channel line, and p2 the incoming particle opposite to that from
c which the t-channel line starts
      m(-nbranch) = m(itree(2,-nbranch))
c
c     Now perform the t-channel decay sequence. Most of this comes from: 
c     Particle Kinematics Chapter 6 section 3 page 166
c
c     From here, on we can just pretend this is a 2->2 scattering with
c     Pa                    + Pb     -> P1          + P2
c     p(0,itree(ibranch,1)) + p(0,2) -> p(0,ibranch)+ p(0,itree(ibranch,2))
c     M(ibranch) is the total mass available (Pa+Pb)^2
c     M(ibranch-1) is the mass of P2  (all the remaining particles)

      do ibranch=-ns_channel-1,-nbranch+1,-1
         s1  = m(ibranch)**2    !Total mass available
         ma2 = m(2)**2
         mbq = dot(pb(0,itree(1,ibranch)),pb(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mnq = m(ibranch-1)**2
         call yminmax(s1,t,m12,ma2,mbq,mnq,tmin,tmax)
         tmax_temp = tmax
         if (keep_inv(ibranch).or.no_gen) then
             t = fixedinv(ibranch) 
         else 
             ivar=ivar+1
             t=(tmax_temp-tmin)*x(ivar)+tmin
             fixedinv(ibranch)=t 
             xjac0=xjac0*(tmax_temp-tmin)
         endif

         if (t .lt. tmin .or. t .gt. tmax) then
            xjac0=-3d0
            pass=.false.
            return
         endif
         if (keep_inv(ibranch).or.no_gen) then
            phi=phi_tchan(ibranch)
         else
            ivar=ivar+1
            phi = 2d0*pi*x(ivar)
            phi_tchan(ibranch)=phi
            xjac0 = xjac0*2d0*pi
         endif

c Finally generate the momentum. The call is of the form
c pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
         
         call gentcms(pb(0,itree(1,ibranch)),pb(0,2),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),pb(0,itree(2,ibranch)),
     &        pb(0,ibranch),xjac0)
c
        if (xjac0 .lt. 0d0) then
            write(*,*) 'Failedgentcms',ibranch,xjac0
            pass=.false.
            return
         endif
         if (.not.keep_inv(ibranch).and..not.no_gen) then
         xpswgt0 = xpswgt0/(4d0*dsqrt(lambda(s1,ma2,mbq)))
         endif
      enddo
c We need to get the momentum of the last external particle.  This
c should just be the sum of p(0,2) and the remaining momentum from our
c last t channel 2->2
      do i=0,3
         pb(i,itree(2,-nbranch)) = pb(i,-nbranch+1)+pb(i,2)
      enddo
      return
      end

      subroutine gentcms(pa,pb,t,phi,m1,m2,p1,pr,jac)
c*************************************************************************
c     Generates 4 momentum for particle 1, and remainder pr
c     given the values t, and phi
c     Assuming incoming particles with momenta pa, pb
c     And outgoing particles with mass m1,m2
c     s = (pa+pb)^2  t=(pa-p1)^2
c*************************************************************************
      implicit none
c
c     Arguments
c
      double precision t,phi,m1,m2               !inputs
      double precision pa(0:3),pb(0:3),jac
      double precision p1(0:3),pr(0:3)           !outputs
c
c     local
c
      double precision ptot(0:3),E_acms,p_acms,pa_cms(0:3)
      double precision esum,ed,pp,md2,ma2,pt,ptotm(0:3)
      integer i
c
c     External
c
      double precision dot
      external dot
c-----
c  Begin Code
c-----
      do i=0,3
         ptot(i)  = pa(i)+pb(i)
         if (i .gt. 0) then
            ptotm(i) = -ptot(i)
         else
            ptotm(i) = ptot(i)
         endif
      enddo
      ma2 = dot(pa,pa)
c
c     determine magnitude of p1 in cms frame (from dhelas routine mom2cx)
c
      ESUM = sqrt(max(0d0,dot(ptot,ptot)))
      if (esum .eq. 0d0) then
         jac=-8d0             !Failed esum must be > 0
         return
      endif
      MD2=(M1-M2)*(M1+M2)
      ED=MD2/ESUM
      IF (M1*M2.EQ.0.) THEN
         PP=(ESUM-ABS(ED))*0.5d0
      ELSE
         PP=(MD2/ESUM)**2-2.0d0*(M1**2+M2**2)+ESUM**2
         if (pp .gt. 0) then
            PP=SQRT(pp)*0.5d0
         else
            write(*,*) 'Warning#12 in genps_madspin.f',pp
            jac=-1
            return
         endif
      ENDIF
c
c     Energy of pa in pa+pb cms system
c
      call boostx(pa,ptotm,pa_cms)
      E_acms = pa_cms(0)
      p_acms = dsqrt(pa_cms(1)**2+pa_cms(2)**2+pa_cms(3)**2)
c
      p1(0) = MAX((ESUM+ED)*0.5d0,0.d0)
      p1(3) = -(m1*m1+ma2-t-2d0*p1(0)*E_acms)/(2d0*p_acms)
      pt = dsqrt(max(pp*pp-p1(3)*p1(3),0d0))
      p1(1) = pt*cos(phi)
      p1(2) = pt*sin(phi)
c
      call rotxxx(p1,pa_cms,p1)          !Rotate back to pa_cms frame
      call boostx(p1,ptot,p1)            !boost back to lab fram
      do i=0,3
         pr(i)=pa(i)-p1(i)               !Return remainder of momentum
      enddo
      end




      function bwfunc(s,xm02,gah)
c Returns the Breit Wigner function, normalized in such a way that
c its integral in the range (-inf,inf) is one
      implicit none
      real*8 bwfunc,s,xm02,gah
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      bwfunc=xm0*gah/(pi*((s-xm02)**2+xm02*gah**2))
      return
      end


      SUBROUTINE YMINMAX(X,Y,Z,U,V,W,YMIN,YMAX)
C**************************************************************************
C     This is the G function from Particle Kinematics by
C     E. Byckling and K. Kajantie, Chapter 4 p. 91 eqs 5.28
C     It is used to determine physical limits for Y based on inputs
C**************************************************************************
      implicit none
c
c     Constant
c
      double precision tiny
      parameter       (tiny=1d-199)
c
c     Arguments
c
      Double precision x,y,z,u,v,w              !inputs  y is dummy
      Double precision ymin,ymax                !output
c
c     Local
c
      double precision y1,y2,yr,ysqr
c     
c     External
c
      double precision lambda
c-----
c  Begin Code
c-----
      ysqr = lambda(x,u,v)*lambda(x,w,z)
      if (ysqr .ge. 0d0) then
         yr = dsqrt(ysqr)
      else
         print*,'Error in yminymax sqrt(-x)',lambda(x,u,v),lambda(x,w,z)
         yr=0d0
      endif
      y1 = u+w -.5d0* ((x+u-v)*(x+w-z) - yr)/(x+tiny)
      y2 = u+w -.5d0* ((x+u-v)*(x+w-z) + yr)/(x+tiny)
      ymin = min(y1,y2)
      ymax = max(y1,y2)
      end


      function xbwmass3(t,xm02,ga,bwdelf,bwfmmn)
c Returns the boson mass squared, given 0<t<1, the nominal mass (xm0),
c and the mass range (implicit in bwdelf and bwfmmn). This function
c is the inverse of F(M^2), where
c   F(M^2)=\int_{xmlow2}^{M^2} ds BW(sqrt(s),M0,Ga)
c   BW(M,M0,Ga)=M0 Ga/pi 1/((M^2-M0^2)^2+M0^2 Ga^2
c and therefore eats up the Breit-Wigner when changing integration 
c variable M^2 --> t
      implicit none
      real*8 xbwmass3,t,xm02,ga,bwdelf,bwfmmn
      real*8 pi,xm0
      parameter (pi=3.1415926535897932d0)
c
      xm0=sqrt(xm02)
      xbwmass3=xm02+xm0*ga*tan(pi*bwdelf*t-bwfmmn)
      return
      end


      subroutine invrot(p,q, pp)
        ! inverse of the "rot" operation
        ! q is the four momentum that is aligned with z in the new frame 
        ! p is the four momentum to be rotated
        ! pp is rotated four momentum p 

        ! arguments
        double precision pp(0:3), p(0:3), q(0:3)
        ! local
        double precision qt2, qt, qq

        pp(0)=p(0)
        qt2 = (q(1))**2 + (q(2))**2

        if(qt2.eq.0.0d0) then
            if ( q(3).gt.0d0 ) then
                pp(1) = p(1)
                pp(2) = p(2)
                pp(3) = p(3)
            else
                pp(1) = -p(1)
                pp(2) = -p(2)
                pp(3) = -p(3)
            endif
        else
            qq = sqrt(qt2+q(3)**2)
            qt=sqrt(qt2)
            pp(2)=-q(2)/qt*p(1)+q(1)/qt*p(2)
            if (q(3).eq.0d0) then
                pp(1)=-qq/qt*p(3)
                if (q(2).ne.0d0) then
                    pp(3)=(p(2)-q(2)*q(3)/qq/qt-q(1)/qt*pp(2))*qq/q(2)
                else
                    pp(3)=(p(1)-q(1)*q(3)/qq/qt*pp(1)+q(2)/qt*pp(2))*qq/q(1)
                endif
            else
                if (q(2).ne.0d0) then
                    pp(3)=(qt**2*p(2)+q(2)*q(3)*p(3)-q(1)*qt*pp(2))/qq/q(2)
                else
                    pp(3)=(q(1)*p(1)+q(3)*p(3))/qq
                endif
                pp(1)=(-p(3)+q(3)/qq*pp(3))*qq/qt
            endif
        endif

        return
        end



      DOUBLE PRECISION  FUNCTION phi(p)
c************************************************************************
c     MODIF 16/11/06 : this subroutine defines phi angle
c                      phi is defined from 0 to 2 pi
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      double precision  p(0:3)
c
c     Parameter
c

      double precision pi,zero
      parameter (pi=3.141592654d0,zero=0d0)
c-----
c  Begin Code
c-----
c 
      if(p(1).gt.zero) then
      phi=datan(p(2)/p(1))
      else if(p(1).lt.zero) then
      phi=datan(p(2)/p(1))+pi
      else if(p(2).GE.zero) then !remind that p(1)=0
      phi=pi/2d0
      else if(p(2).lt.zero) then !remind that p(1)=0
      phi=-pi/2d0
      endif
      if(phi.lt.zero) phi=phi+2*pi
      return
      end

      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)

      if(dabs(dot).lt.1d-6)then ! solve numerical problem 
         dot=0d0
      endif

      end

      DOUBLE PRECISION FUNCTION LAMBDA(S,MA2,MB2)
      IMPLICIT NONE
C****************************************************************************
C     THIS IS THE LAMBDA FUNCTION FROM VERNONS BOOK COLLIDER PHYSICS P 662
C     MA2 AND MB2 ARE THE MASS SQUARED OF THE FINAL STATE PARTICLES
C     2-D PHASE SPACE = .5*PI*SQRT(1.,MA2/S^2,MB2/S^2)*(D(OMEGA)/4PI)
C****************************************************************************
      DOUBLE PRECISION MA2,MB2,S,tiny,tmp,rat
      parameter (tiny=1.d-8)
c
      tmp=S**2+MA2**2+MB2**2-2d0*S*MA2-2d0*MA2*MB2-2d0*S*MB2
      if(tmp.le.0.d0)then
        if(ma2.lt.0.d0.or.mb2.lt.0.d0)then
          write(6,*)'Error#1 in function Lambda:',s,ma2,mb2
          stop
        endif
        rat=1-(sqrt(ma2)+sqrt(mb2))/s
        if(rat.gt.-tiny)then
          tmp=0.d0
        else
          write(6,*)'Error#2 in function Lambda:',s,ma2,mb2
        endif
      endif
      LAMBDA=tmp
      RETURN
      END


