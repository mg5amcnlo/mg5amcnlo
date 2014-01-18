C**************************************************************************************
      subroutine global_init
      implicit none

      double precision zero,pi
      parameter (zero=0d0,pi=3.141592653589793d0)
C
      include 'nexternal.inc'
      include 'permutation.inc'
      include 'TF_param.inc'
      include 'coupl.inc'
      include 'run.inc'

c     particle index mapping
      integer MG,j,k,i
      integer matching_type_part(3:max_particles) !modif/link between our order by type for permutation
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part
 
      character*30 param_card_name
      common/to_param_card_name/param_card_name

C     info on final state particles
      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,
     +        num_ta,num_ata,num_photon   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,
     & num_mu,num_amu,num_ta,num_ata,num_photon !particle in the final state
      integer nparticles, num_invis
      COMMON/to_num_inv/nparticles,num_invis

c     experimental event
      double precision pexp_init(0:3,nexternal)  !impulsion in original configuration
      common/to_pexp_init/pexp_init

      double precision px_visible,py_visible
      common /to_pTrec_visible/px_visible,py_visible

      double precision  missPhi_EXP, missPT_EXP
      common /to_missEXP/  missPhi_EXP, missPT_EXP

      character*60 param_name

      double precision pmass(max_particles)
      common/to_mass/pmass

      integer ISR_mode
      common /to_correct_ISR/ISR_mode

      double precision alphas
      external alphas


c
c     set parameters of the run

      include 'madweight_param.inc'

      open(unit=89,file="./param.dat")
      read(89,*) param_name
      param_card_name=param_name

      call setrun

      include 'madweight_card.inc'
      ISR_mode=isr

c     specific case of nexternal=4, num_inv=0
      if (nexternal.eq.4.and.num_inv.eq.0) ISR_mode=3

      px_visible=0d0
      py_visible=0d0
c      open(unit=90,file="./start")
c      write(90,*) 'start'
c      close(90)
c      open(UNIT=32,FILE='vegas_value.out',STATUS='unknown')

      close(89)

c      call setpara(param_name)
       if(fixed_ren_scale) then
          G = SQRT(4d0*PI*ALPHAS(scale))
          call UPDATE_AS_PARAM()
       endif

c     set cuts
c      CALL SETCUTS

c     deternime the final state content
      call info_final_part()                

c     set parameters for the transfer functions
      call Init_MET_LHCO
      include 'transfer_card.inc'
      curr_tf = 1

c     set permutation info
      if (montecarlo_perm) then
        do i = 1, nb_channel2
           min_perm(i) = 1
        enddo
      endif

      curr_perm = 1
      do i = 1, NPERM
      do j =1, nb_tf
         perm_value(i,j) = 0d0
         perm_error(i,j) = 0d0
      enddo
      nb_point_by_perm(i) = 0
      enddo

      CALL PRINTOUT
      CALL RUN_PRINTOUT
c     call graph_init
      
      num_invis=num_inv
      nparticles=nexternal

      OPEN(UNIT=24,file='./verif.lhco',status='old',err=48) ! input file

      read(24,*,err=48,end=48) k,run_number,trigger !line with tag 0
      do j=3,nexternal-num_inv
         MG=inv_matching_type_part(j)
         read(24,*,err=48,end=48) k,type(k),eta_init(k),
     &           phi_init(k),pt_init(k),j_mass(k),ntrk(k),btag(k),
     &           had_em(k) ,dummy1(k),dummy2(k)

         call four_momentum_set2(eta_init(k),phi_init(k),pt_init(k),
     &         j_mass(k),pexp_init(0,MG))
         px_visible=px_visible+pexp_init(1,MG)
         py_visible=py_visible+pexp_init(2,MG)
         tag_init(MG)=k
      enddo
      read(24,*,err=48,end=48) k,type(k),eta_init(k), !line with type 6
     &           phi_init(k),pt_init(k),j_mass(k),ntrk(k),btag(k),
     &           had_em(k) ,dummy1(k),dummy2(k)
      if (type(k).eq.6.and.pt_init(k).gt.1d0.and.ISR.gt.0) then
         missPhi_EXP=phi_init(k)   ! apply boot correction based on reconstructed missing pT
         missPT_EXP=pt_init(k)
         write(*,*) "Using reconstructed ISR: "
         write(*,*) "px(ISR)= ", -px_visible-missPT_EXP*dcos(missPhi_EXP)
         write(*,*) "py(ISR)= ", -py_visible-missPT_EXP*dsin(missPhi_EXP)
      elseif(ISR.eq.0 .and. num_inv.eq.0) then ! apply boost correction based on visible particles
         missPhi_EXP=0d0
         missPT_EXP=1d-10
         write(*,*) "ISR deduced from pT(visible): "
         write(*,*) "px(ISR)= ", -px_visible
         write(*,*) "py(ISR)= ", -py_visible
      else
         write(*,*) "ISR effects are ignored "
         missPhi_EXP=-1d0
         missPT_EXP=-1d0
      endif
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
      include "pmass.inc" ! done in setcuts
      do i=3,nexternal
         mvir2(i)=pmass(i)**2
      enddo
      call get_central_point()


      call init_d_assignement()

c      write(*,*) 'pmass ok'
      return
 48   write(*,*) 'FAILS EVENTS READING!'
      stop
      end


       subroutine init_block_d_alignment(parg1,parg2,p1,p2)
       Implicit none
cc               input parg1/parg2: two MG number
cc               this functions determines which of the two TF s must be mapped onto a variable of integration in the block D.
cc               p1 is unmapped
cc               p2 is mapped

	  include 'nexternal.inc'
        include 'phasespace.inc'
        integer parg1,parg2 ! input variable
        integer p1,p2       ! output variable
cc
cc    COMMON
cc      
      include 'permutation.inc'
        double precision c_point(NPERM,1:max_particles,3,2)
        common/ph_sp_init/c_point
cc
cc    Begin of the code
cc
        if (parg1.gt.0.and.parg2.gt.0) then 
           if (c_point(1,parg1,3,2).gt.c_point(1,parg2,3,2)) then
              p1=parg1
              p2=parg2
           else
              p1=parg2
              p2=parg1
           endif
        elseif (parg1.gt.0.and.parg2.lt.0) then
           p1=parg1
           p2=parg2
        elseif (parg1.lt.0.and.parg2.gt.0) then
           p1=parg2
           p2=parg1 
        else 
           write(*,*) 'Warning: wrong phase space parametrization'
           stop
        endif
        
        return
        end

C-------------------------------------------------------
C     initialize tag value for LHCO MET                 
C-------------------------------------------------------
      SUBROUTINE INIT_MET_LHCO
      implicit none

      include 'nexternal.inc'
      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,mum_amu,num_ta,num_ata,num_photon   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,mum_amu,num_ta,num_ata,num_photon !particle in the final state

      integer met_lhco,opt_lhco
      common/LHCO_met_tag/met_lhco,opt_lhco

      met_lhco=nexternal-num_inv-1
      opt_lhco=0

      end

C-------------------------------------------------------
C     Check that x is real positive number              
C-------------------------------------------------------
      SUBROUTINE CHECK_NAN(x)
C                                                       
      IMPLICIT NONE
C                                                       
C     ARGUMENTS                                         
C                                                       
      double precision x
c                                                       
c     LOCAL                                             
c                                                       
      if(.not.(x.gt.0d0).and.x.ne.0d0) then
         x=0d0
      endif
      end



      subroutine initialize
c**********************************************************************************
c     This subroutine initializes the phase space integration
c     It determines
c     c_point(L,J,I,K) : * central variables of integration + width
c                    J -> number of final particle (madgraph notation ) start from 3 !!!
c                    I -> component : I=1 : y
c                                     I=2 : phi
c                                     I=3 : PT
c                    K ->     K = 1 : variable
c                             K = 2 : associated width   (0 if transfert function : delta function)
c                                                        (-1 if the variable is "fixed" by tranverse momentum conservation)
c                    L -> PERMUTATION NUMBER
c***********************************************************************************
      implicit none
c
c     parameter
c
      double precision zero
      parameter (zero=0d0)
      include 'nexternal.inc'
      include 'phasespace.inc'
      include 'permutation.inc'
c
c      local
c
      integer I,J,temp
      character*8 variable(3)
      integer num_inv
c
c      global
c
      double precision c_point(NPERM,1:max_particles,3,2)
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
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp
      double precision              S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

c------
c Begin Code
c------
      s = 4d0*ebeam(1)*ebeam(2)
      num_inv=0
c
c      define the central point
c
      variable(1)="theta"
      variable(2)="phi"
      variable(3)="|p|"
c
c      counting the number of dimensions
c
      temp=-2 ! conservation of PT
      do i=3,nexternal
          if (c_point(1,i,1,2).lt.-0.5)  num_inv=num_inv+1
        do j=1,3
          if (c_point(1,i,j,2).ne.zero) temp=temp+1
          write(*,*) 'exp. uncertainty on ',variable(j),'of particle',i,': ' ,
     &     c_point(1,i,j,2),'[',c_point(1,i,j,1),']'
        enddo
      enddo

      if (c_point(1,1,1,2).eq.zero) temp=temp-1
      if (c_point(1,2,1,2).eq.zero) temp=temp-1

c      write(*,*) 'the number of dimension is ', temp

      Ndimens=temp
c     very special case of two visible particles and nothing else
      if(nexternal.eq.4.and.num_inv.eq.0) Ndimens=Ndimens+2
      

c      if (temp.ne.(Ndimens)) then
c      write(*,*) 'error : Ndimens is not # widths + 1 '
c      write(*,*) 'Ndimens=',Ndimens
c      pause
c      endif

      return
      end


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine info_final_part()
c
c	this subroutine initializes the following common blocks concerning final state information 
c
c      common block: num_part
c      ----------------------
c           num_inv: number of missing particles
c           num_jet: number of jets
c           num_e:number of electrons
c           num_ae:number of positrons
c           num_mu:number of muons
c           num_amu:number of anti-muons
c           num_ta:number of taus
c           num_ata:number of anti-taus
c           num_photon: number of photon
c     local:
c     ------
c           jet(nexternal-2): containing madgraph number of the jets (last components set to 0)
c           bjet(nexternal-2): containing madgraph number of the bjets (if not consider like jet for permutations)
c           el(nexternal-2): containing madgraph number of the electrons (last components set to 0)
c           ae(nexternal-2): containing madgraph number of the positrons (last components set to 0)
c           mu(nexternal-2): containing madgraph number of the muons (last components set to 0)
c           amu(nexternal-2): containing madgraph number of the anti-muons (last components set to 0)
c           ta(nexternal-2): containing madgraph number of the muons (last components set to 0)
c           ata(nexternal-2): containing madgraph number of the anti-muons (last components set to 0)
c           neutrino(nexternal-2): containing madgraph number of the invisible particles (last components set to 0)
c           photon(nexternal-2): containing madgraph number of the photon particles (last components set to 0)
c
c
c     common block: madgraph_order_type
c     ---------------------------------
c           inv_matching_type_part(3:nexternal): function: MG order --> permutation order(classed by types of particles) 
c           matching_type_part(3:nexternal):     function:  permutation order  --> MG order
c

      implicit none
      
c      include './genps.inc'
      include 'phasespace.inc'
      include 'maxamps.inc'
      include 'nexternal.inc'

      logical use_perm,perm_with_b, montecarlo_perm
      common/global_run_perm/use_perm,perm_with_b, montecarlo_perm
      
      integer matching_type_part(3:max_particles) 
      integer inv_matching_type_part(3:max_particles)
      common/madgraph_order_type/matching_type_part,
     & inv_matching_type_part 
 
      integer i

      integer idup(nexternal,maxproc,maxsproc)
      integer mothup(2,nexternal)
      integer icolup(2,nexternal,maxflow,maxsproc)
      include 'leshouche.inc'
      
      logical pass     					!type agreement
      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,num_amu,
     + num_ta,num_ata,num_notfind, num_photon
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,
     + num_amu,num_ta,num_ata,num_photon
      
      
      integer jet(nexternal-2)
      integer bjet(nexternal-2)
      integer el(nexternal-2)
      integer ael(nexternal-2)
      integer mu(nexternal-2)
      integer amu(nexternal-2)
      integer ta(nexternal-2)
      integer ata(nexternal-2)
      integer neutrino(nexternal-2)
      integer photon(nexternal-2)
c
c     local
c         
      integer l,tag					!counter


c     init all variable:
      do l=1,nexternal-2
         jet(l)=0
         bjet(l)=0
         el(l)=0
         ael(l)=0
         mu(l)=0
         amu(l)=0
         ta(l)=0
         ata(l)=0
         neutrino(l)=0
         photon(l) = 0
      enddo
      num_inv=0
      num_jet=0
      num_bjet=0
      num_mu=0
      num_amu=0
      num_e=0
      num_ae=0
      num_ta=0
      num_ata=0
      num_photon=0
      num_notfind=0


cc***********************************************************************
cc
cc                    Identification of particles
cc
cc***********************************************************************


      do l=3,nexternal
c
c      invisible particule identification 
c
         pass=.false.
         if (abs(IDUP(l,1,1)).eq.12) then
            pass=.true.         !ve
         elseif (abs(IDUP(l,1,1)).eq.14)then
            pass=.true.         !vm 
         elseif (abs(IDUP(l,1,1)).eq.16)then
            pass=.true.         !vt
         elseif (abs(IDUP(l,1,1)).eq.18)then
            pass=.true.         !v'
         elseif (abs(IDUP(l,1,1)).eq.1000022)then
            pass=.true.         !chi0
         elseif (abs(IDUP(l,1,1)).eq.1000023)then
            pass=.true.         !chi0
         elseif (abs(IDUP(l,1,1)).eq.1000024)then
            pass=.true.         !chi0
         elseif (abs(IDUP(l,1,1)).eq.1000025)then
            pass=.true.         !chi0    
         elseif (abs(IDUP(l,1,1)).eq.1000035)then
            pass=.true.         !chi0
         elseif (abs(IDUP(l,1,1)).eq.1000012)then
            pass=.true.         !svm 
         elseif (abs(IDUP(l,1,1)).eq.1000014)then
            pass=.true.         !svm
         elseif (abs(IDUP(l,1,1)).eq.1000016)then
            pass=.true.         !chi0
         elseif (abs(IDUP(l,1,1)).eq.1000018)then
            pass=.true.         !chi0

         endif
	 
         if (pass)then
            num_inv=num_inv+1
            neutrino(num_inv) = l
	    goto 20 !continue (next identification)
         endif
c
c      jet identification 
c
         pass=.false.
         if (abs(IDUP(l,1,1)).le.4)then
            pass=.true.         !quark u,d,s,c
         elseif(perm_with_b.and.abs(IDUP(l,1,1)).eq.5)then
            pass=.true.         !quark b
         elseif (abs(IDUP(l,1,1)).eq.21)then
            pass=.true.         !gluon
         endif
	 
         if (pass)then
            num_jet=num_jet+1
            jet(num_jet)=l
	 goto 20 ! continue (next identification)
         endif
c
c      bjet identification 
c
         pass=.false.
         if(.not.perm_with_b.and.abs(IDUP(l,1,1)).eq.5)then
            pass=.true.
         endif
	 
         if (pass)then
            write(*,*) 'find bjet',perm_with_b
            num_bjet=num_bjet+1
            bjet(num_bjet)=l
	 goto 20 ! continue (next identification)
         endif
c
c      electron identification 
c
         pass=.false.
         if (IDUP(l,1,1).eq.11) pass=.true. !electron
         
  	 if (pass)then
            num_e=num_e+1
            el(num_e)=l
	 goto 20 !continue (next identification)
         endif

c
c      positron identification
c
         pass=.false.
         if (IDUP(l,1,1).eq.-11) pass=.true. !electron
         if (pass)then
            num_ae=num_ae+1
            ael(num_ae)=l
         goto 20 !continue (next identification)
         endif


c
c      muon identification 
c
         pass=.false.
         if (IDUP(l,1,1).eq.13) pass=.true. !muon

         if (pass)then
            num_mu=num_mu+1
            mu(num_mu)=l
	 goto 20 !continue (next identification)
         endif
c
c      anti muon identification
c
         pass=.false.
         if (IDUP(l,1,1).eq.-13) pass=.true. !muon

         if (pass)then
            num_amu=num_amu+1
            amu(num_amu)=l
         goto 20 !continue (next identification)
         endif


c
c      tau identification
c
         pass=.false.
         if (IDUP(l,1,1).eq.15) pass=.true. !muon

         if (pass)then
            num_ta=num_ta+1
            ta(num_ta)=l
         goto 20 !continue (next identification)
         endif
c
c      anti muon identification
c
         pass=.false.
         if (IDUP(l,1,1).eq.-15) pass=.true. !muon

         if (pass)then
            num_ata=num_ata+1
            ata(num_ata)=l
         goto 20 !continue (next identification)
         endif
c
c      photon identification
c
         pass=.false.
         if (IDUP(l,1,1).eq.22) pass=.true. !muon

         if (pass)then
            num_photon=num_photon+1
            photon(num_photon)=l
         goto 20 !continue (next identification)
         endif


c
c        count un-identified particle 
c
	 if(.not.pass)then
c	    num_notfind=num_notfind+1
            write(*,*)"Un-identified PID:",abs(IDUP(l,1,1))
            write(*,*) "will consider this particle as a missing particle"
            num_inv=num_inv+1
            neutrino(num_inv) = l

	 endif	 
 20    enddo
 

         write(*,*)'               '
         write(*,*)'* * *  topology identification  * * *'
         write(*,*)' '

 
         write(*,*)'Final state content:'
         write(*,*)'--------------------'
         write(*,*) num_inv,' missing particles'
c         write(*,*) "those are",(neutrino(i),i=1,num_inv)
         write(*,*) num_jet,' light jets'
c         write(*,*) "those are",(jet(i),i=1,num_jet)
         if(.not.perm_with_b) then
            write(*,*) num_bjet,' b jets'           
         endif
         write(*,*) num_e,' electrons'
         write(*,*) num_ae,' positronss'
         write(*,*) num_mu,' muons'
         write(*,*) num_amu,' anti muons'
         write(*,*) num_ta,' taus'
         write(*,*) num_ata,' anti taus'
         write(*,*) num_photon, 'photon'
	 if(num_notfind.ge.1)then
            write(*,*) "!!!!!WARNING!!!!!" 
	    write(*,*) num_notfind,'unknow particle detected'
            write(*,*) "!!!!!WARNING!!!!!" 
         endif

C
C     define the function inv_matching_type_part/matching_type_part
C             ( MG order <--> Permutation order )
C
         tag=3
c jet
         do l=1,nexternal-2
            if (jet(l).ne.0)then
               inv_matching_type_part(tag)=jet(l)
               tag=tag+1
            endif           
         enddo
c bjet
          do l=1,nexternal-2
            if (bjet(l).ne.0)then
               inv_matching_type_part(tag)=bjet(l)
               tag=tag+1
            endif           
         enddo
c elec
         do l=1,nexternal-2
            if (el(l).ne.0)then
               inv_matching_type_part(tag)=el(l)
               tag=tag+1
           endif           
         enddo
c posit
         do l=1,nexternal-2
            if (ael(l).ne.0)then
               inv_matching_type_part(tag)=ael(l)
               tag=tag+1
           endif
         enddo
c muon
         do l=1,nexternal-2
            if (mu(l).ne.0)then
               inv_matching_type_part(tag)=mu(l)
               tag=tag+1
           endif           
         enddo
c anti muon
         do l=1,nexternal-2
            if (amu(l).ne.0)then
               inv_matching_type_part(tag)=amu(l)
               tag=tag+1
           endif
         enddo
c tau
         do l=1,nexternal-2
            if (ta(l).ne.0)then
               inv_matching_type_part(tag)=ta(l)
               tag=tag+1
           endif
         enddo
c anti tau
         do l=1,nexternal-2
            if (ata(l).ne.0)then
               inv_matching_type_part(tag)=ata(l)
               tag=tag+1
           endif
         enddo
c photon
         do l=1,nexternal-2
            if (photon(l).ne.0)then
               inv_matching_type_part(tag)=photon(l)
               tag=tag+1
           endif
         enddo

c invisible
         do l=1,nexternal-2
            if (neutrino(l).ne.0)then
               inv_matching_type_part(tag)=neutrino(l)
               tag=tag+1
            endif           
         enddo
         do l=3,nexternal
            matching_type_part(inv_matching_type_part(l))=l
         enddo
c
c        write(*,*) "matching_type_part", (matching_type_part(l),l=3,nexternal)
        write(*,*) "inv_matching_type_part", 
     & (inv_matching_type_part(l),l=3,nexternal)



        return
	end



