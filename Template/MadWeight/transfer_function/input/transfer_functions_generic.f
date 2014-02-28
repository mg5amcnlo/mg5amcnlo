      subroutine get_central_point
c*******************************************************
c     this subroutine defines the central point
c*******************************************************
      implicit none
c
c     parameter   
c
      include '../../nexternal.inc'
      include '../../run.inc'
      include 'nb_tf.inc'
      include 'TF_param.inc'

c
c     global
c
      double precision c_point(1:nexternal,3,2)
      common/ph_sp_init/c_point
c
      double precision p_exp(0:3,nexternal)
      common /to_pexp/p_exp
      integer tag_lhco(3:nexternal)
      common/lhco_order/tag_lhco
c
c
c     local
c
      integer i
      integer n_lhco
c
c     external
c
      double precision sigma_e_j,sigma_phi_j,sigma_theta_j
      double precision sigma_e_b,sigma_phi_b,sigma_theta_b
      double precision sigma_e_g,sigma_phi_g,sigma_theta_g
      double precision sigma_e_e,sigma_phi_e,sigma_theta_e
      double precision sigma_e_m,sigma_phi_m,sigma_theta_m
      double precision sigma_e_t,sigma_phi_t,sigma_theta_t
      double precision c_point_x1,c_point_x2,sigma_x1,sigma_x2

      external sigma_e_j,sigma_phi_j,sigma_theta_j
      external sigma_e_b,sigma_phi_b,sigma_theta_b
      external sigma_e_g,sigma_phi_g,sigma_theta_g
      external sigma_e_e,sigma_phi_e,sigma_theta_e
      external sigma_e_m,sigma_phi_m,sigma_theta_m
      external sigma_e_t,sigma_phi_t,sigma_theta_t
      external c_point_x1,c_point_x2,sigma_x1,sigma_x2
c
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL)
      LOGICAL  IS_A_E(NEXTERNAL),IS_A_M(NEXTERNAL),IS_A_T(NEXTERNAL)
      LOGICAL  IS_A_G(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      COMMON/TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_E,IS_A_M,
     &IS_A_T,IS_A_G, IS_A_NU,IS_HEAVY
c--------
c     begin code
c--------
c
c    define c_point, var 
c-------------------------------------------------------------------
      c_point(1,1,1)=c_point_x1()    !cpoint for x1
      c_point(2,1,1)=c_point_x2()    !cpoint for x2
      c_point(1,1,2)=sigma_x1()      !width  for x1
      c_point(2,1,2)=sigma_x2()      !width  for x2

      do i=3,nexternal
         n_lhco=tag_lhco(i)
      if(IS_A_B(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_b(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_b(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_b(p_exp(0,i),n_lhco)/rho(p_exp(0,i))
      elseif(IS_A_G(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_g(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_g(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_g(p_exp(0,i),n_lhco)/rho(p_exp(0,i))  
      elseif(IS_A_J(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_j(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_j(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_j(p_exp(0,i),n_lhco)/rho(p_exp(0,i))  
      elseif(IS_A_E(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_e(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_e(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_e(p_exp(0,i),n_lhco)/rho(p_exp(0,i))     
      elseif(IS_A_M(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_m(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_m(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_m(p_exp(0,i),n_lhco)/rho(p_exp(0,i))       
      elseif(IS_A_T(i))then
         c_point(i,1,1)=theta(p_exp(0,i))
         c_point(i,2,1)=phi(p_exp(0,i))
         c_point(i,3,1)=rho(p_exp(0,i))
c
         c_point(i,1,2)=sigma_theta_t(p_exp(0,i),n_lhco)
         c_point(i,2,2)=sigma_phi_t(p_exp(0,i),n_lhco)
         c_point(i,3,2)=p_exp(0,i)*sigma_e_t(p_exp(0,i),n_lhco)/rho(p_exp(0,i))       
      else
         c_point(i,1,2)=-1d0
         c_point(i,2,2)=-1d0
         c_point(i,3,2)=-1d0         
      endif
      enddo
c
      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CC
CC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c
      subroutine transfer_fct(P,weight)
c----------------------------------------------------------------------
c     This is the transfer function 
c     scale on event-by-event basis.
c----------------------------------------------------------------------      
      implicit none
c
      integer    maxexternal
      parameter (maxexternal=15)
c
c     ARGUMENTS
c      
      DOUBLE PRECISION P(0:3,maxexternal)
      DOUBLE PRECISION weight
c
c     INCLUDE and COMMON
c
      include '../../genps.inc'
      include '../../nexternal.inc'
      include '../../run.inc'
c
c
      double precision pexp(0:3,nexternal)
      integer tag_lhco(3:nexternal)
      common/to_pexp/pexp
      common/lhco_order/tag_lhco
C
C     SPECIAL CUTS
C

c
      DOUBLE PRECISION R2,DOT,ET,ETA,DJ,SumDot,PT
c
c     LOCAL
c
      integer i,k
      integer n_lhco
c
c     Auxiliary functions
c
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_L(NEXTERNAL)
      LOGICAL  IS_A_B(NEXTERNAL),IS_A_A(NEXTERNAL)
      LOGICAL  IS_A_E(NEXTERNAL),IS_A_M(NEXTERNAL),IS_A_T(NEXTERNAL)
      LOGICAL  IS_A_G(NEXTERNAL)
      LOGICAL  IS_A_NU(NEXTERNAL),IS_HEAVY(NEXTERNAL)
      COMMON/TO_SPECISA/IS_A_J,IS_A_A,IS_A_L,IS_A_B,IS_A_E,IS_A_M,IS_A_T,
     &IS_A_G, IS_A_NU,IS_HEAVY


c----------
c     start
c----------

C
C     INITIALIZE WEIGHT
C
      weight=1d0

c
c      uncomment the following lines to print
c      reconstructed momenta (p_exp), and parton-level
c      momenta (p) 
c
c      do i=1,nexternal
c      write(*,*) 'p_exp',i, (pexp(k,i),k=0,3)
c      write(*,*) 'p',i, (p(k,i),k=0,3)
c      enddo
      call tf_init_x1(weight)
      call tf_init_x2(weight)
c
c     start loop over final particles
c
      do i = 3,nexternal  
         n_lhco=tag_lhco(i)
c
c     add a weight if particle != neut
c
         if(IS_A_B(i))then
            call tf_e_bjet(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_bjet(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_bjet(pexp(0,i),p(0,i),n_lhco,weight)
         elseif(IS_A_G(i)) then
            call tf_e_gluon(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_gluon(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_gluon(pexp(0,i),p(0,i),n_lhco,weight)
         elseif(IS_A_J(i)) then
            call tf_e_jet(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_jet(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_jet(pexp(0,i),p(0,i),n_lhco,weight)
         elseif(IS_A_E(i))then 
            call tf_e_elec(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_elec(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_elec(pexp(0,i),p(0,i),n_lhco,weight)           
         elseif(IS_A_M(i))then 
            call tf_e_muon(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_muon(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_muon(pexp(0,i),p(0,i),n_lhco,weight)
         elseif(IS_A_T(i))then 
            call tf_e_tau(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_phi_tau(pexp(0,i),p(0,i),n_lhco,weight)
            call tf_theta_tau(pexp(0,i),p(0,i),n_lhco,weight)
         endif! nothing for neutrino
         
         call check_nan(weight)

      enddo


      return
      end
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCC
CCCC                            SUBROUTINE FOR C-point
CCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C     Subroutine for X1
C
      DOUBLE PRECISION FUNCTION c_point_x1()
      
      include '../../nexternal.inc'

      integer tag_init(3:nexternal),type(nexternal),run_number,trigger
      double precision eta_init(nexternal),phi_init(nexternal),
     &pt_init(nexternal),j_mass(nexternal),ntrk(nexternal),
     &btag(nexternal),had_em(nexternal),dummy1(nexternal),
     &dummy2(nexternal)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger

      integer met_lhco,opt_lhco
      common/LHCO_met_tag/met_lhco,opt_lhco

      if (opt_lhco.gt.0)then
         c_point_x1=phi_init(opt_lhco)
      else
         c_point_x1=0d0
      endif
      return
      end
C
C     Subroutine for X2
C

      DOUBLE PRECISION FUNCTION c_point_x2()

      include '../../nexternal.inc'

      integer tag_init(3:nexternal),type(nexternal),run_number,trigger
      double precision eta_init(nexternal),phi_init(nexternal),
     &pt_init(nexternal),j_mass(nexternal),ntrk(nexternal),
     &btag(nexternal),had_em(nexternal),dummy1(nexternal),
     &dummy2(nexternal)
      common/LHCO_input/eta_init,phi_init,pt_init,
     &j_mass,ntrk,btag,had_em,dummy1,dummy2,tag_init,type,run_number,
     &trigger


      integer met_lhco,opt_lhco
      common/LHCO_met_tag/met_lhco,opt_lhco

      if (opt_lhco.gt.0)then
         c_point_x2=pt_init(opt_lhco)
      else
         c_point_x2=0d0
      endif
      return
      end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCC
CCCC                            SUBROUTINE FOR TRANSFER FUNCTION
CCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C     Subroutine for initial particle (X1)
C

      subroutine tf_init_x1(weight)
      implicit none

      double precision    S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      double precision tf_x1
      integer n_lhco
      double precision weight
      double precision pi
      double precision x1_exp,x2_exp
      parameter (pi=3.141592654d0)
      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     


      x1_exp=phi_init(opt_lhco)
      x2_exp=pt_init(opt_lhco)

$$tf_x1$$

      weight=weight*tf_x1
      
      return
      end
C
C     Subroutine for initial particle (X2)
C

      subroutine tf_init_x2(weight)
      implicit none

      double precision    S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      double precision tf_x2
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)
      double precision x1_exp,x2_exp

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     
      x1_exp=phi_init(opt_lhco)
      x2_exp=pt_init(opt_lhco)

$$tf_x2$$

      weight=weight*tf_x2
      
      return
      end
C
C     Subroutine for initial particle (PT)
C
c      subroutine tf_init_pt(pexp,p,n_lhco,weight)
c      implicit none
c
c      double precision tf_pt
c      double precision pexp(0:3)
c      double precision p(0:3)
c      integer n_lhco
c      double precision weight
c      double precision pi
c      parameter (pi=3.141592654d0)
c      include '../../nexternal.inc'
c            include 'nb_tf.inc'
c      include 'TF_param.inc'     
c
c$$tf_pt$$
c
c      weight=weight*tf_pt
c      
c      return
c      end
c
c
c
C
C     Subroutine for Jet
C

      subroutine tf_e_jet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_j
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)
      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_j$$

      weight=weight*tf_e_j
      
      return
      end

      subroutine tf_phi_jet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_j
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_j$$

      weight=weight*tf_phi_j
      
      return
      end

      subroutine tf_theta_jet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_j
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_j$$

      weight=weight*tf_theta_j
      
      return
      end

C
C     Subroutine for B-Jet
C

      subroutine tf_e_bjet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_b
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_b$$

      weight=weight*tf_e_b
      
      return
      end

      subroutine tf_phi_bjet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_b
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_b$$

      weight=weight*tf_phi_b
      
      return
      end

      subroutine tf_theta_bjet(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_b
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_b$$

      weight=weight*tf_theta_b
      
      return
      end
C
C     Subroutine for gluon-Jet
C

      subroutine tf_e_gluon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_g
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_g$$

      weight=weight*tf_e_g
      
      return
      end

      subroutine tf_phi_gluon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_g
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_g$$

      weight=weight*tf_phi_g
      
      return
      end

      subroutine tf_theta_gluon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_g
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_g$$

      weight=weight*tf_theta_g      
      return
      end

C
C     Subroutine for Electron
C

      subroutine tf_e_elec(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_e
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_e$$

      weight=weight*tf_e_e
      
      return
      end

      subroutine tf_phi_elec(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_e
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_e$$

      weight=weight*tf_phi_e
      
      return
      end

      subroutine tf_theta_elec(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_e
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_e$$

      weight=weight*tf_theta_e
      
      return
      end

C
C     Subroutine for Muon
C

      subroutine tf_e_muon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_m
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_m$$

      weight=weight*tf_e_m
      
      return
      end

      subroutine tf_phi_muon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_m
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_m$$

      weight=weight*tf_phi_m
      
      return
      end

      subroutine tf_theta_muon(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_m
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_m$$

      weight=weight*tf_theta_m
      
      return
      end

C
C     Subroutine for tau
C

      subroutine tf_e_tau(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_e_t
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_e_t$$

      weight=weight*tf_e_t
      
      return
      end

      subroutine tf_phi_tau(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_phi_t
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_phi_t$$

      weight=weight*tf_phi_t
      
      return
      end

      subroutine tf_theta_tau(pexp,p,n_lhco,weight)
      implicit none

      double precision tf_theta_t
      double precision pexp(0:3)
      double precision p(0:3)
      integer n_lhco
      double precision weight
      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$tf_theta_t$$

      weight=weight*tf_theta_t
      
      return
      end

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCC
CCCC                                       FUNCTION FOR SIGMA
CCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      DOUBLE PRECISION FUNCTION sigma_x1()
      implicit none

      double precision    S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_x1$$
      
      return
      end


      DOUBLE PRECISION FUNCTION sigma_x2()
      implicit none

      double precision    S,X1,X2,PSWGT,JAC
      common /PHASESPACE/ S,X1,X2,PSWGT,JAC

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_x2$$
      
      return
      end


c      DOUBLE PRECISION FUNCTION sigma_pt(pexp,n_lhco)
c      implicit none

c      double precision pexp(0:3)
c      integer n_lhco

c      double precision pi
c      parameter (pi=3.141592654d0)

c      include '../../nexternal.inc'
c            include 'nb_tf.inc'
      include 'TF_param.inc'

c$$sigma_pt$$
      
c      return
c      end

      DOUBLE PRECISION FUNCTION sigma_e_j(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_j$$
      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_j(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_j$$
      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_j(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_j$$

      
      return
      end

C
C      FUNCTION for B-Jet
C

      DOUBLE PRECISION FUNCTION sigma_e_b(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_b$$


      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_b(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_b$$


      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_b(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_b$$

 
      
      return
      end
C
C     Subroutine for gluon-Jet
C

      DOUBLE PRECISION FUNCTION sigma_e_g(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_g$$


      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_g(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_g$$

      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_g(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_g$$
     
      return
      end

C
C      FUNCTION for Electron
C

      DOUBLE PRECISION FUNCTION sigma_e_e(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_e$$

      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_e(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_e$$

      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_e(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_e$$

      
      return
      end

C
C      FUNCTION for Muon
C

      DOUBLE PRECISION FUNCTION sigma_e_m(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_m$$

      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_m(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_m$$
      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_m(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_m$$

      
      return
      end

C
C      FUNCTION for Tau
C

      DOUBLE PRECISION FUNCTION sigma_e_t(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_e_t$$

      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_phi_t(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_phi_t$$
      
      return
      end

      DOUBLE PRECISION FUNCTION sigma_theta_t(pexp,n_lhco)
      implicit none

      double precision pexp(0:3)
      integer n_lhco

      double precision pi
      parameter (pi=3.141592654d0)

      include '../../nexternal.inc'
            include 'nb_tf.inc'
      include 'TF_param.inc'     

$$sigma_theta_t$$

      
      return
      end

      SUBROUTINE INIT_MET_LHCO
C-------------------------------------------------------
C     initialize tag value for LHCO MET 
C-------------------------------------------------------
      include '../../nexternal.inc'
      integer num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,mum_amu,num_ta,num_ata,num_photon   !number of jet,elec,muon, undetectable
      COMMON/num_part/num_inv,num_jet,num_bjet,num_e,num_ae,num_mu,mum_amu,num_ta,num_ata,num_photon !particle in the final state

      integer met_lhco,opt_lhco
      common/LHCO_met_tag/met_lhco,opt_lhco

      met_lhco=nexternal-num_inv-1
      opt_lhco=0

      end

      SUBROUTINE CHECK_NAN(x)
C-------------------------------------------------------
C     Check that x is real positive number
C-------------------------------------------------------
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

      return
      end


