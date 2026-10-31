c In-process access to the MacNLOPS densities of this P* directory for
c the first-emission program (MCatNLO/srcPythia8/MacNLOPSMatch.cc).
c Momenta use the evaluator layout lab(0:3,leg) = (E,px,py,pz).

      subroutine macnlops_c_setup(nlegs,registry)
     $     bind(C,name='macnlops_c_setup')
c Initialise once and report the compiled multiplicity and the
c registry fingerprint that every source component packet must carry.
      use, intrinsic :: iso_c_binding, only: c_int,c_char
      implicit none
      include 'nexternal.inc'
      include 'mc_histories.inc'
      integer(c_int) nlegs
      character(kind=c_char) registry(64)
      integer i
      call macnlops_evaluator_setup()
      nlegs=nexternal
      do i=1,64
         registry(i)=MC_HIST_REGISTRY_SHA256(i:i)
      enddo
      end


      subroutine macnlops_c_sudakov_request(enabled,pt)
     $     bind(C,name='macnlops_c_sudakov_request')
      use, intrinsic :: iso_c_binding, only: c_int,c_double
      use weight_lines, only: macnlops_sudakov_enabled,
     $     macnlops_sudakov_requested_pt,macnlops_sudakov_ready
      implicit none
      integer(c_int), value :: enabled
      real(c_double), value :: pt
      macnlops_sudakov_enabled=enabled.ne.0
      macnlops_sudakov_requested_pt=pt
      macnlops_sudakov_ready=.false.
      end


      subroutine macnlops_c_sudakov_profile(ready,n,ids,x,mass,pt,
     $     source,cdf,q,w,fsr,nquarks,quarks)
     $     bind(C,name='macnlops_c_sudakov_profile')
      use, intrinsic :: iso_c_binding, only: c_int,c_double
      use weight_lines
      implicit none
      integer(c_int) ready,n,ids(2),nquarks
      real(c_double) x(2),mass,pt,source,cdf,
     $     q(macnlops_sudakov_max),w(macnlops_sudakov_max),
     $     fsr(macnlops_sudakov_max),quarks(4,64)
      ready=merge(1,0,macnlops_sudakov_ready)
      n=macnlops_sudakov_n
      ids=macnlops_sudakov_ids
      x=macnlops_sudakov_x
      mass=macnlops_sudakov_mass
      pt=macnlops_sudakov_pt
      source=macnlops_sudakov_source
      cdf=macnlops_sudakov_cdf
      q=macnlops_sudakov_q
      w=macnlops_sudakov_w
      fsr=macnlops_sudakov_fsr
      nquarks=macnlops_sudakov_nquarks
      quarks=macnlops_sudakov_quarks
      end


      subroutine macnlops_c_profile(cutoffs,xmax,masses,flavours)
     $     bind(C,name='macnlops_c_profile')
c The shower profile measured before generation and compiled into the
c source: ISR/FSR cutoffs, beam bounds, c/b mass floors, nQuarkIn and
c nGluonToQuark. The first-emission shower must reproduce all of them.
      use, intrinsic :: iso_c_binding, only: c_int,c_double
      implicit none
      include 'run.inc'
      real(c_double) cutoffs(2),xmax(2),masses(2)
      integer(c_int) flavours(2)
      cutoffs(1)=macnlops_isr_cutoff
      cutoffs(2)=macnlops_fsr_cutoff
      xmax=macnlops_beam_xmax
      masses(1)=macnlops_charm_mass
      masses(2)=macnlops_bottom_mass
      flavours(1)=macnlops_nquark_in
      flavours(2)=macnlops_ngluon_to_quark
      end


      subroutine macnlops_c_configure(xmax,isr,fsr)
     $     bind(C,name='macnlops_c_configure')
c Actual beam bounds and cutoffs of the running shower; the caller
c checks them against macnlops_c_profile first.
      use, intrinsic :: iso_c_binding, only: c_double
      implicit none
      real(c_double) xmax(2),isr,fsr
      double precision beam_xmax(2),ptmin_isr,ptmin_fsr
      common/macnlops_shower_config/beam_xmax,ptmin_isr,ptmin_fsr
      beam_xmax=xmax
      ptmin_isr=isr
      ptmin_fsr=fsr
      end


      subroutine macnlops_c_density(owner,ids,lab,config,graph,flow,
     $     damping,symmetry,has_born,born_ids,born_colours,born_lab,
     $     born_masses,r,d,k,alphas,q2fact_k,status)
     $     bind(C,name='macnlops_c_density')
c One local component request, as COMPONENT_LOCAL_V1 (and, with
c has_born, COMPONENT_BORN_V1) of the line protocol. Also returns K's
c emission coupling and squared factorisation scales (zero when no
c kernel was evaluated).
      use, intrinsic :: iso_c_binding, only: c_int,c_double
      use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
      use weight_lines, only: reset_delta_contexts,
     $     macnlops_kernel_alphas,macnlops_kernel_q2fact
      implicit none
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      integer(c_int), value :: owner,config,graph,flow,damping,
     $     has_born
      real(c_double), value :: symmetry
      integer(c_int) ids(nexternal),born_ids(nexternal-1),
     $     born_colours(2,nexternal-1),status
      real(c_double) lab(0:3,nexternal),born_lab(0:3,nexternal-1),
     $     born_masses(nexternal-1),r,d,k,alphas,q2fact_k(2)
      call reset_delta_contexts()
      status=1
      r=0d0
      d=0d0
      k=0d0
      macnlops_kernel_alphas=0d0
      macnlops_kernel_q2fact=0d0
      if(owner.ge.1.and.owner.le.fks_configs.and.
     $     all(ieee_is_finite(lab))) then
         call macnlops_component_density(owner,ids,lab,config,graph,
     $        flow,damping,symmetry,has_born.ne.0,born_ids,
     $        born_colours,born_lab,born_masses,r,d,k,status)
      endif
      alphas=macnlops_kernel_alphas
      q2fact_k=macnlops_kernel_q2fact
      call reset_delta_contexts()
      end
