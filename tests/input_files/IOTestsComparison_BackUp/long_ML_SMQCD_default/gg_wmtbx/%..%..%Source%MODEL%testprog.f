c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      program testprog
      
      call setpara('param_card.dat')


      
      call printout

      end

c$$$c
c$$$c     program testing the running. need to modify the makefile accordingly
c$$$c      
c$$$      program testprog
c$$$      implicit none
c$$$c     define the function that run alphas 
c$$$      DOUBLE PRECISION ALPHAS
c$$$      EXTERNAL ALPHAS
c$$$c     get the value of gs
c$$$      include '../coupl.inc'
c$$$c     for initialization of the running      
c$$$      include "../alfas.inc"
c$$$c     include parameter from the run_card (usefull for the running)      
c$$$      INCLUDE '../maxparticles.inc'
c$$$c      INCLUDE '../run.inc'
c$$$c     local 
c$$$      integer i
c$$$      double precision mu,as
c$$$
c$$$c
c$$$c     Scales
c$$$c
c$$$      real*8          scale,scalefact,alpsfact,mue_ref_fixed,mue_over_ref
c$$$      logical         fixed_ren_scale,fixed_fac_scale1, fixed_fac_scale2,fixed_couplings,hmult
c$$$      logical         fixed_extra_scale
c$$$      integer         ickkw,nhmult,asrwgtflavor, dynamical_scale_choice,ievo_eva
c$$$
c$$$      common/to_scale/scale,scalefact,alpsfact, mue_ref_fixed, mue_over_ref,
c$$$     $                fixed_ren_scale,fixed_fac_scale1, fixed_fac_scale2,
c$$$     $                fixed_couplings, fixed_extra_scale,ickkw,nhmult,hmult,asrwgtflavor,
c$$$     $                dynamical_scale_choice
c$$$
c$$$      
c$$$
c$$$c     read the param_card 
c$$$      call setpara('param_card.dat')
c$$$c     define your running for as...
c$$$      fixed_extra_scale = .false.
c$$$      asmz = G**2/(16d0*atan(1d0))
c$$$      nloop = 2
c$$$      MUE_OVER_REF = 1d0
c$$$
c$$$c     loop for the running
c$$$      do i=1,200
c$$$         scale = 10*i
c$$$         G = SQRT(4d0*PI*ALPHAS(scale))
c$$$         call UPDATE_AS_PARAM()
c$$$         call printout
c$$$      enddo
c$$$
c$$$
c$$$      end
c$$$
c$$$
