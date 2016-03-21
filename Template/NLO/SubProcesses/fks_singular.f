      subroutine compute_born
c This subroutine computes the Born matrix elements and adds its value
c to the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'reweight0.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      double complex wgt_c(2)
      double precision wgt1
      double precision p_born(0:3,nexternal-1)
      common /pborn/   p_born
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      call cpu_time(tBefore)
      if (f_b.eq.0d0) return
      if (xi_i_fks_ev .gt. xiBSVcut_used) return
      call sborn(p_born,wgt_c)
      wgt1=dble(wgt_c(1))*f_b/g**(nint(2*wgtbpower))
      call add_wgt(2,wgt1,0d0,0d0)
      call cpu_time(tAfter)
      tBorn=tBorn+(tAfter-tBefore)
      return
      end

      subroutine compute_nbody_noborn
c This subroutine computes the soft-virtual matrix elements and adds its
c value to the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'reweight.inc'
      include 'coupl.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      double precision wgt1,wgt2,wgt3,bsv_wgt,virt_wgt,born_wgt,pi,g2
     &     ,g22,wgt4
      parameter (pi=3.1415926535897932385d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      call cpu_time(tBefore)
      if (f_nb.eq.0d0) return
      if (xi_i_fks_ev .gt. xiBSVcut_used) return
      call bornsoftvirtual(p1_cnt(0,1,0),bsv_wgt,virt_wgt,born_wgt)
      g2=g**(nint(2*wgtbpower))
      g22=g**(nint(2*wgtbpower+2))
      wgt1=wgtnstmp*f_nb/g22
      wgt4=wgtnstmp_avgvirt*f_nb/g22
      if (ickkw.eq.3 .and. fxfx_exp_rewgt.ne.0d0) then
         wgt1=wgt1 - fxfx_exp_rewgt*born_wgt*f_nb/g2/(4d0*pi)
      elseif (ickkw.eq.-1) then
         if (wgtbpower.ne.0) then
            write (*,*) 'ERROR in VETO XSec: bpower should'/
     $           /' be zero (no QCD partons at the'/
     $           /' Born allowed)', wgtbpower
         endif
         H1_factor_virt=virt_wgt/(g22/(4d0*pi))/born_wgt
         born_wgt_veto=born_wgt/g2
         call compute_veto_compensating_factor(H1_factor_virt
     $        ,born_wgt_veto,1d0,1d0,veto_compensating_factor)
         call add_wgt(7,-veto_compensating_factor*f_nb,0d0,0d0)
      endif
      wgt2=wgtwnstmpmur*f_nb/g22
      wgt3=wgtwnstmpmuf*f_nb/g22
      call add_wgt(3,wgt1,wgt2,wgt3)
      call add_wgt(15,wgt4,0d0,0d0)
c Special for the soft-virtual needed for the virt-tricks. The
c *_wgt_mint variable should be directly passed to the mint-integrator
c and not be part of the plots nor computation of the cross section.
      virt_wgt_mint=virt_wgt*f_nb/g22
      born_wgt_mint=born_wgt*f_b/g2
      call add_wgt(14,virt_wgt_mint,0d0,0d0)
      call cpu_time(tAfter)
      tIS=tIS+(tAfter-tBefore)
      return
      end

      subroutine compute_real_emission(p,sudakov_damp)
c This subroutine computes the real-emission matrix elements and adds
c its value to the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'reweight0.inc'
      include 'timing_variables.inc'
      double precision x,dot,f_damp,ffact,s_ev,fks_Sij,p(0:3,nexternal)
     $     ,wgt1,fx_ev,sudakov_damp
      external dot,f_damp,fks_Sij
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      call cpu_time(tBefore)
      if (f_r.eq.0d0) return
      x = abs(2d0*dot(p(0,i_fks),p(0,j_fks))/shat)
      ffact = f_damp(x)
      if (ffact.le.0d0) return
      s_ev = fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
      if (s_ev.le.0.d0) return
      call sreal(p,xi_i_fks_ev,y_ij_fks_ev,fx_ev)
      wgt1=fx_ev*s_ev*f_r/g**(nint(2*wgtbpower+2))
      if (sudakov_damp.gt.0d0) then
         call add_wgt(1,wgt1*sudakov_damp,0d0,0d0)
      endif
      if (sudakov_damp.lt.1d0) then
         call add_wgt(11,wgt1*(1d0-sudakov_damp),0d0,0d0)
      endif
      call cpu_time(tAfter)
      tReal=tReal+(tAfter-tBefore)
      return
      end

      subroutine compute_soft_counter_term(replace_MC_subt)
c This subroutine computes the soft counter term and adds its value to
c the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'reweight0.inc'
      include 'timing_variables.inc'
      double precision wgt1,s_s,fks_Sij,fx_s,zero,replace_MC_subt,g22
      parameter (zero=0d0)
      external fks_Sij
      double precision     p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                     ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/ p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xi_i_fks_ev,y_ij_fks_ev
      double precision    p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      call cpu_time(tBefore)
      if (f_s.eq.0d0 .and. f_s_MC_S.eq.0d0 .and. f_s_MC_H.eq.0d0) return
      if (xi_i_fks_ev.gt.xiScut_used .and. replace_MC_subt.eq.0d0)
     $     return
      s_s = fks_Sij(p1_cnt(0,1,0),i_fks,j_fks,zero,y_ij_fks_ev)
      if (s_s.le.0d0) return
      call sreal(p1_cnt(0,1,0),0d0,y_ij_fks_ev,fx_s)
      g22=g**(nint(2*wgtbpower+2))
      if (replace_MC_subt.gt.0d0) then
         wgt1=fx_s*s_s/g22*replace_MC_subt
         call add_wgt(8,-wgt1*f_s_MC_H,0d0,0d0)
         wgt1=wgt1*f_s_MC_S
      else
         wgt1=0d0
      endif
      if (xi_i_fks_ev.le.xiScut_used) then
         wgt1=wgt1-fx_s*s_s*f_s/g22
      endif
      if (wgt1.ne.0d0) call add_wgt(4,wgt1,0d0,0d0)
      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_collinear_counter_term(replace_MC_subt)
c This subroutine computes the collinear counter term and adds its value
c to the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'fks_powers.inc'
      include 'reweight.inc'
      include 'timing_variables.inc'
      double precision zero,one,s_c,fks_Sij,fx_c,deg_xi_c,deg_lxi_c,wgt1
     &     ,wgt3,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision pmass(nexternal)
      call cpu_time(tBefore)
      include 'pmass.inc'
      if (f_c.eq.0d0 .and. f_dc.eq.0d0 .and. f_c_MC_S.eq.0d0 .and.
     $     f_c_MC_H.eq.0d0)return
      if ( (y_ij_fks_ev.le.1d0-deltaS .and. replace_MC_subt.eq.0d0) .or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_c = fks_Sij(p1_cnt(0,1,1),i_fks,j_fks,xi_i_fks_cnt(1),one)
      if (s_c.le.0d0) return
      g22=g**(nint(2*wgtbpower+2))
      call sreal(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,fx_c)
      if (replace_MC_subt.gt.0d0) then
         wgt1=fx_c*s_c/g22*replace_MC_subt
         call add_wgt(9,-wgt1*f_c_MC_H,0d0,0d0)
         wgt1=wgt1*f_c_MC_S
      else
         wgt1=0d0
      endif
      if (y_ij_fks_ev.gt.1d0-deltaS) then
         wgt1=wgt1-fx_c*s_c*f_c/g22
         call sreal_deg(p1_cnt(0,1,1),xi_i_fks_cnt(1),one,deg_xi_c
     $        ,deg_lxi_c)
         wgt1=wgt1+ ( wgtdegrem_xi+wgtdegrem_lxi*log(xi_i_fks_cnt(1)) )*
     $        f_dc/g22
         wgt3=wgtdegrem_muF*f_dc/g22
      else
         wgt3=0d0
      endif
      if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt(5,wgt1,0d0,wgt3)
      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_soft_collinear_counter_term(replace_MC_subt)
c This subroutine computes the soft-collinear counter term and adds its
c value to the list of weights using the add_wgt subroutine
      implicit none
      include 'nexternal.inc'
      include 'coupl.inc'
      include 'reweight.inc'
      include 'fks_powers.inc'
      include 'timing_variables.inc'
      double precision zero,one,s_sc,fks_Sij,fx_sc,wgt1,wgt3,deg_xi_sc
     $     ,deg_lxi_sc,g22,replace_MC_subt
      external fks_Sij
      parameter (zero=0d0,one=1d0)
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision pmass(nexternal)
      include 'pmass.inc'
      call cpu_time(tBefore)
      if (f_sc.eq.0d0 .and. f_dsc(1).eq.0d0 .and. f_dsc(2).eq.0d0 .and.
     $     f_dsc(3).eq.0d0 .and. f_dsc(4).eq.0d0 .and. f_sc_MC_S.eq.0d0
     $     .and. f_sc_MC_H.eq.0d0) return
      if ( ((xi_i_fks_cnt(1).ge.xiScut_used .or. y_ij_fks_ev.le.1d0
     $     -deltaS) .and. replace_MC_subt.eq.0d0).or.
     $     pmass(j_fks).ne.0.d0 ) return
      s_sc = fks_Sij(p1_cnt(0,1,2),i_fks,j_fks,zero,one)
      if (s_sc.le.0d0) return
      g22=g**(nint(2*wgtbpower+2))
      call sreal(p1_cnt(0,1,2),zero,one,fx_sc)
      if (replace_MC_subt.gt.0d0) then
         wgt1=-fx_sc*s_sc/g22*replace_MC_subt
         call add_wgt(10,-wgt1*f_sc_MC_H,0d0,0d0)
         wgt1=wgt1*f_sc_MC_S
      else
         wgt1=0d0
      endif
      if (xi_i_fks_cnt(1).lt.xiScut_used .and. 
     $     y_ij_fks_ev.gt.1d0-deltaS) then
         wgt1=wgt1+fx_sc*s_sc*f_sc/g22
         call sreal_deg(p1_cnt(0,1,2),zero,one, deg_xi_sc,deg_lxi_sc)
         wgt1=wgt1+(-(wgtdegrem_xi+wgtdegrem_lxi*log(xi_i_fks_cnt(1)))
     $        *f_dsc(1)-(wgtdegrem_xi*f_dsc(2)+wgtdegrem_lxi*f_dsc(3)))
     $        /g22
         wgt3=-wgtdegrem_muF*f_dsc(4)/g22
      else
         wgt3=0d0
      endif
      if (wgt1.ne.0d0 .or. wgt3.ne.0d0) call add_wgt(6,wgt1,0d0,wgt3)
      call cpu_time(tAfter)
      tCount=tCount+(tAfter-tBefore)
      return
      end

      subroutine compute_MC_subt_term(p,gfactsf,gfactcl,probne)
      implicit none
c This subroutine computes the MonteCarlo subtraction terms and adds
c their values to the list of weights using the add_wgt subroutine. It
c returns the values for the gfactsf, gfactcl and probne to check if we
c need to include the FKS subtraction terms as replacements in the soft
c and collinear limits and the Sudakov damping for the real-emission,
c respectively.
      include 'nexternal.inc'
      include 'madfks_mcatnlo.inc'
      include 'timing_variables.inc'
      include 'reweight.inc'
      include 'coupl.inc'
      integer nofpartners,i
      double precision p(0:3,nexternal),gfactsf,gfactcl,probne,x,dot
     $     ,fks_Sij,f_damp,ffact,sevmc,dummy,zhw(nexternal)
     $     ,xmcxsec(nexternal),g22,wgt1,xlum_mc_fact,fks_Hij
      external dot,fks_Sij,f_damp,fks_Hij
      logical lzone(nexternal),flagmc
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     $                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision    xm12
      integer                  ileg
      common/cscaleminmax/xm12,ileg
      integer           fks_j_from_i(nexternal,0:nexternal)
     &                  ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      logical              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      call cpu_time(tBefore)
      if (f_MC_S.eq.0d0 .and. f_MC_H.eq.0d0) return
      if(UseSfun)then
         x = abs(2d0*dot(p(0,i_fks),p(0,j_fks))/shat)
         ffact = f_damp(x)
         sevmc = fks_Sij(p,i_fks,j_fks,xi_i_fks_ev,y_ij_fks_ev)
         sevmc = sevmc*ffact
      else
         x = abs(2d0*dot(p(0,i_fks),p(0,j_fks))/shat)
         ffact = f_damp(x)
         sevmc = fks_Hij(p,i_fks,j_fks)
         sevmc = sevmc*ffact
      endif
      if (sevmc.eq.0d0) return
      call xmcsubt(p,xi_i_fks_ev,y_ij_fks_ev,gfactsf,gfactcl,probne,
     $             dummy,nofpartners,lzone,flagmc,zhw,xmcxsec)
      MCcntcalled=.true.
      if(ileg.gt.4 .or. ileg.lt.1)then
         write(*,*)'Error: unrecognized ileg in compute_MC_subt_term',
     $        ileg
         stop 1
      endif
      if (flagmc) then
         g22=g**(nint(2*wgtbpower+2))
         do i=1,nofpartners
            if(lzone(i))then
               call get_mc_lum(j_fks,zhw(i),xi_i_fks_ev,xlum_mc_fact)
               wgt1=sevmc*f_MC_S*xlum_mc_fact*xmcxsec(i)/g22
               call add_wgt(12,wgt1,0d0,0d0)
               wgt1=sevmc*f_MC_H*xlum_mc_fact*xmcxsec(i)/g22
               call add_wgt(13,-wgt1,0d0,0d0)
            endif
         enddo
      endif
      if( (.not.flagmc) .and. gfactsf.eq.1.d0 .and.
     $     xi_i_fks_ev.lt.0.02d0 .and. particle_type(i_fks).eq.8 )then
         write(*,*)'Error in compute_MC_subt_term: will diverge'
         stop
      endif
      call cpu_time(tAfter)
      t_MC_subt=t_MC_subt+(tAfter-tBefore)
      return
      end


      logical function pdg_equal(pdg1,pdg2)
c Returns .true. if the lists of PDG codes --'pdg1' and 'pdg2'-- are
c equal.
      implicit none
      include 'nexternal.inc'
      integer i,pdg1(nexternal),pdg2(nexternal)
      pdg_equal=.true.
      do i=1,nexternal
         if (pdg1(i).ne.pdg2(i)) then
            pdg_equal=.false.
            return
         endif
      enddo
      end
      
      logical function momenta_equal(p1,p2)
c Returns .true. only if the momenta p1 and p2 are equal. To save time,
c it only checks the 0th and 3rd components (energy and z-direction).
      implicit none
      include 'nexternal.inc'
      integer i,j
      double precision p1(0:3,nexternal),p2(0:3,nexternal),vtiny
      parameter (vtiny=1d-8)
      momenta_equal=.true.
      do i=1,nexternal
         do j=0,3,3
            if (p1(j,i).eq.0d0 .or. p2(j,i).eq.0d0) then
               if (abs(p1(j,i)-p2(j,i)).gt.vtiny) then
                  momenta_equal=.false.
                  return
               endif
            else
               if (abs((p1(j,i)-p2(j,i))/max(p1(j,i),p2(j,i))).gt.vtiny)
     &              then
                  momenta_equal=.false.
                  return
               endif
            endif
         enddo
      enddo
      end
      
      subroutine set_FxFx_scale(iterm,p)
c Sets the FxFx cluster scale and multiplies the f_* factors (computed
c by 'compute_prefactors_nbody' and 'compute_prefactors_n1body') by the
c Sudakov suppression. If called more than once with the same momenta
c and iterm, skip setting of the scales, and multiply the f_* factors by
c the cached Sudakovs.
c     iterm=  0 : reset the computation of the Sudakovs
c     iterm=  1 : Sudakov for n-body kinematics (f_b and f_nb)
c     iterm=  2 : Sudakov for n-body kinematics (all but f_b and f_nb)
c     iterm=  3 : Sudakov for n+1-body kinematics
c     iterm= -1 or -2 : only restore scales for n-body w/o recomputing
c     iterm= -3 : only restore scales for n+1-body w/o recomputing
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'timing_variables.inc'
      integer iterm,iterm_last_izero,iterm_last_mohdr,i,j
     &     ,nfxfx_ren_scales_izero,nfxfx_ren_scales_mohdr
      double precision p(0:3,nexternal),p_last_izero(0:3,nexternal)
     &     ,p_last_mohdr(0:3,nexternal),rewgt,rewgt_izero,rewgt_mohdr
     &     ,rewgt_exp_izero,rewgt_exp_mohdr
     &     ,fxfx_ren_scales_izero(0:nexternal),fxfx_fac_scale_izero(2)
     &     ,fxfx_ren_scales_mohdr(0:nexternal),fxfx_fac_scale_mohdr(2)
      logical setclscales,rewgt_izero_calculated,rewgt_mohdr_calculated
     &     ,momenta_equal,already_set
      external setclscales,rewgt,momenta_equal
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      double precision         fxfx_exp_rewgt
      common /c_fxfx_exp_regt/ fxfx_exp_rewgt
      integer                              nFxFx_ren_scales
      double precision     FxFx_ren_scales(0:nexternal),
     $                     FxFx_fac_scale(2)
      common/c_FxFx_scales/FxFx_ren_scales,nFxFx_ren_scales,
     $                     FxFx_fac_scale
      save rewgt_mohdr_calculated,rewgt_izero_calculated,p_last_izero
     &     ,p_last_mohdr,iterm_last_izero,iterm_last_mohdr
     &     ,fxfx_ren_scales_izero ,fxfx_ren_scales_mohdr
     &     ,fxfx_fac_scale_izero ,fxfx_fac_scale_mohdr
     &     ,nfxfx_ren_scales_izero ,nfxfx_ren_scales_mohdr
      call cpu_time(tBefore)
      ktscheme=1
      if (iterm.eq.0) then
         rewgt_mohdr_calculated=.false.
         rewgt_izero_calculated=.false.
         fxfx_exp_rewgt=0d0
         return
      endif
      already_set=.false.
      if (iterm.eq.1 .or. iterm.eq.2) then
c n-body momenta FxFx Sudakov factor (i.e. for S-events)
         if (rewgt_izero_calculated) then
            if (iterm_last_izero.ne.1 .and. iterm_last_izero.ne.2) then
               if (momenta_equal(p1_cnt(0,1,0),p_last_izero)) then
                  already_set=.true.
               endif
            endif
         endif
         if (.not.already_set) then
            if (.not. setclscales(p1_cnt(0,1,0))) then
               write (*,*) 'ERROR in setclscales izero'
               stop 1
            endif
            rewgt_izero=min(rewgt(p1_cnt(0,1,0),rewgt_exp_izero),1d0)
            fxfx_exp_rewgt=min(rewgt_exp_izero,0d0)
         endif
         rewgt_izero_calculated=.true.
         iterm_last_izero=iterm
         do i=1,nexternal
            do j=0,3
               p_last_izero(j,i)=p1_cnt(j,i,0)
            enddo
         enddo
         if (iterm.eq.1) then
            f_b =f_b *rewgt_izero
            f_nb=f_nb*rewgt_izero
         elseif(iterm.eq.2) then
            f_s =f_s *rewgt_izero
            f_c =f_c *rewgt_izero
            f_dc=f_dc*rewgt_izero
            f_sc=f_sc*rewgt_izero
            do i=1,4
               f_dsc(i)=f_dsc(i)*rewgt_izero
            enddo
            f_MC_S =f_MC_S *rewgt_izero
            f_s_MC_S =f_s_MC_S *rewgt_izero
            f_c_MC_S =f_c_MC_S *rewgt_izero
            f_sc_MC_S=f_sc_MC_S*rewgt_izero
         endif
         nFxFx_ren_scales_izero=nFxFx_ren_scales
         do i=0,nexternal
            FxFx_ren_scales_izero(i)=FxFx_ren_scales(i)
         enddo
         do i=1,2
            FxFx_fac_scale_izero(i)=FxFx_fac_scale(i)
         enddo
      elseif (iterm.eq.3) then
c n+1-body momenta FxFx Sudakov factor (i.e. for H-events)
         if (rewgt_mohdr_calculated) then
            if (iterm.eq.iterm_last_mohdr) then
               if (momenta_equal(p,p_last_mohdr)) then
                  already_set=.true.
               endif
            endif
         endif
         if (.not. already_set) then
            if (.not. setclscales(p)) then
               write (*,*) 'ERROR in setclscales mohdr'
               stop 1
            endif
            rewgt_mohdr=min(rewgt(p,rewgt_exp_mohdr),1d0)
         endif
         rewgt_mohdr_calculated=.true.
         iterm_last_mohdr=iterm
         do i=1,nexternal
            do j=0,3
               p_last_mohdr(j,i)=p(j,i)
            enddo
         enddo
         f_r=f_r*rewgt_mohdr
         f_MC_H =f_MC_H *rewgt_mohdr
         f_s_MC_H =f_s_MC_H *rewgt_izero
         f_c_MC_H =f_c_MC_H *rewgt_izero
         f_sc_MC_H=f_sc_MC_H*rewgt_izero
         nFxFx_ren_scales_mohdr=nFxFx_ren_scales
         do i=0,nexternal
            FxFx_ren_scales_mohdr(i)=FxFx_ren_scales(i)
         enddo
         do i=1,2
            FxFx_fac_scale_mohdr(i)=FxFx_fac_scale(i)
         enddo
         call cpu_time(tAfter)
         tFxFx=tFxFx+(tAfter-tBefore)
         return
      elseif (iterm.eq.-1 .or. iterm.eq.-2) then
c Restore scales for the n-body FxFx terms
         nFxFx_ren_scales=nFxFx_ren_scales_izero
         do i=0,nexternal
            FxFx_ren_scales(i)=FxFx_ren_scales_izero(i)
         enddo
         do i=1,2
            FxFx_fac_scale(i)=FxFx_fac_scale_izero(i)
         enddo
      elseif (iterm.eq.-3) then
c Restore scales for the n+1-body FxFx terms
         nFxFx_ren_scales=nFxFx_ren_scales_mohdr
         do i=0,nexternal
            FxFx_ren_scales(i)=FxFx_ren_scales_mohdr(i)
         enddo
         do i=1,2
            FxFx_fac_scale(i)=FxFx_fac_scale_mohdr(i)
         enddo
      else
         write (*,*) 'ERROR: unknown iterm in set_FxFx_scale',iterm
         stop 1
      endif
      call cpu_time(tAfter)
      tFxFx=tFxFx+(tAfter-tBefore)
      return
      end
      
      
      subroutine compute_prefactors_nbody(vegas_wgt)
c Compute all the relevant prefactors for the Born and the soft-virtual,
c i.e. all the nbody contributions. Also initialises the plots and
c bpower.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'reweight0.inc'
      include 'timing_variables.inc'
      double precision pi,unwgtfun,vegas_wgt,enhance,xnoborn_cnt,xtot
     $     ,bpower,cpower,tiny
      data xnoborn_cnt /0d0/
      integer inoborn_cnt,i
      data inoborn_cnt /0/
      double complex wgt_c(2)
      logical firsttime
      data firsttime /.true./
      parameter (pi=3.1415926535897932385d0)
      parameter (tiny=1d-6)
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision  xinorm_ev
      common /cxinormev/xinorm_ev
      double precision  xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     $                         fkssymmetryfactorDeg
      integer                  ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer            mapconfig(0:lmaxconfigs), iconfig
      common/to_mconfigs/mapconfig,                iconfig
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,          jamp2
      double precision   diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor
      double precision      f_b,f_nb
      common /factor_nbody/ f_b,f_nb
      integer iappl
      common /for_applgrid/ iappl
      logical needrndec
      parameter (needrndec=.true.)
      real*8 ran2
      external ran2
      real*8 rndec(10)
      common/crndec/rndec
      logical              fixed_order,nlo_ps
      common /c_fnlo_nlops/fixed_order,nlo_ps
      include "appl_common.inc" 
      call cpu_time(tBefore)
c Random numbers to be used in the plotting routine: these numbers will
c not change between events, counter events and n-body contributions.
      if(needrndec)then
         do i=1,10
            rndec(i)=ran2()
         enddo
      endif
      if (firsttime) then
c Put here call to compute bpower
         call compute_bpower(p_born,bpower)
         wgtbpower=bpower
c Store the power of alphas of the Born events in the appl common block.
         if(iappl.ne.0) appl_bpower = wgtbpower
c Initialize hiostograms for fixed order runs
         if (fixed_order) call initplot
c Compute cpower done for bottom Yukawa, routine needs to be adopted
c for other muR-dependendent factors
         call compute_cpower(p_born,cpower)
         if(dabs(cpower+1d0).lt.tiny) then
            wgtcpower=0d0
         else
            wgtcpower=cpower
         endif
c Check that things are done consistently
         if(wgtcpower.ne.cpowerinput.and.dabs(cpower+1d0).gt.tiny)then
           write(*,*)'Inconsistency in the computation of cpower',
     #               wgtcpower,cpowerinput
           write(*,*)'Check value in reweight0.inc'
           stop
         endif
         firsttime=.false.
      endif
c Compute the multi-channel enhancement factor 'enhance'.
      enhance=1.d0
      if (p_born(0,1).gt.0d0) then
         call sborn(p_born,wgt_c)
      elseif(p_born(0,1).lt.0d0)then
         enhance=0d0
      endif
      if (enhance.eq.0d0)then
         xnoborn_cnt=xnoborn_cnt+1.d0
         if(log10(xnoborn_cnt).gt.inoborn_cnt)then
            write (*,*) 'WARNING: no Born momenta more than 10**',
     $           inoborn_cnt,'times'
            inoborn_cnt=inoborn_cnt+1
         endif
      else
         xtot=0d0
         if (mapconfig(0).eq.0) then
            write (*,*) 'Fatal error in compute_prefactor_nbody:'/
     &           /' no Born diagrams ',mapconfig,
     &           '. Check bornfromreal.inc'
            write (*,*) 'Is fks_singular compiled correctly?'
            stop 1
         endif
         do i=1, mapconfig(0)
            xtot=xtot+amp2(mapconfig(i))
         enddo
         if (xtot.ne.0d0) then
            enhance=amp2(mapconfig(iconfig))/xtot
            enhance=enhance*diagramsymmetryfactor
         else
            enhance=0d0
         endif
      endif
      call unweight_function(p_born,unwgtfun)
      call set_cms_stuff(0)
c f_* multiplication factors for Born and nbody
      f_b=jac_cnt(0)*xinorm_ev/(min(xiimax_ev,xiBSVcut_used)*shat/(16
     $     *pi**2))*enhance*unwgtfun *fkssymmetryfactorBorn*vegas_wgt
      f_nb=f_b
      call cpu_time(tAfter)
      tf_nb=tf_nb+(tAfter-tBefore)
      return
      end

      subroutine compute_prefactors_n1body(vegas_wgt,jac_ev)
c Compute all relevant prefactors for the real emission and counter
c terms.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'fks_powers.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      double precision unwgtfun,vegas_wgt,enhance,xnoborn_cnt,xtot
     &     ,prefact,prefact_cnt_ssc,prefact_deg,prefact_c,prefact_coll
     &     ,jac_ev,pi,prefact_cnt_ssc_c,prefact_coll_c,prefact_deg_slxi
     &     ,prefact_deg_sxi,zero
      parameter (pi=3.1415926535897932385d0, zero=0d0)
      data xnoborn_cnt /0d0/
      integer inoborn_cnt,i
      data inoborn_cnt /0/
      double complex wgt_c(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision    xi_i_fks_ev,y_ij_fks_ev
      double precision    p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      integer            i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision   xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision  xinorm_ev
      common /cxinormev/xinorm_ev
      double precision  xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision   xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      double precision   xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt
      double precision    delta_used
      common /cdelta_used/delta_used
      double precision    xicut_used
      common /cxicut_used/xicut_used
      double precision     xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg
      integer                  ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer            mapconfig(0:lmaxconfigs), iconfig
      common/to_mconfigs/mapconfig,                iconfig
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,          jamp2
      double precision   diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision     f_r,f_s,f_c,f_dc,f_sc,f_dsc(4)
      common/factor_n1body/f_r,f_s,f_c,f_dc,f_sc,f_dsc
      double precision           f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      common/factor_n1body_NLOPS/f_s_MC_S,f_s_MC_H,f_c_MC_S,f_c_MC_H
     $     ,f_sc_MC_S,f_sc_MC_H,f_MC_S,f_MC_H
      double precision pmass(nexternal)
      include 'pmass.inc'
      call cpu_time(tBefore)
      enhance=1.d0
      if (p_born(0,1).gt.0d0) then
         call sborn(p_born,wgt_c)
      elseif(p_born(0,1).lt.0d0)then
         enhance=0d0
      endif
c Compute the multi-channel enhancement factor 'enhance'.
      if (enhance.eq.0d0)then
         xnoborn_cnt=xnoborn_cnt+1.d0
         if(log10(xnoborn_cnt).gt.inoborn_cnt)then
            write (*,*) 'WARNING: no Born momenta more than 10**',
     $           inoborn_cnt,'times'
            inoborn_cnt=inoborn_cnt+1
         endif
      else
         xtot=0d0
         if (mapconfig(0).eq.0) then
            write (*,*) 'Fatal error in compute_prefactor_n1body,'/
     &           /' no Born diagrams ',mapconfig
     &           ,'. Check bornfromreal.inc'
            write (*,*) 'Is fks_singular compiled correctly?'
            stop 1
         endif
         do i=1, mapconfig(0)
            xtot=xtot+amp2(mapconfig(i))
         enddo
         if (xtot.ne.0d0) then
            enhance=amp2(mapconfig(iconfig))/xtot
            enhance=enhance*diagramsymmetryfactor
         else
            enhance=0d0
         endif
      endif
      call unweight_function(p_born,unwgtfun)
      prefact=xinorm_ev/xi_i_fks_ev/(1-y_ij_fks_ev)

c f_* multiplication factors for real-emission, soft counter, ... etc.       
      f_r=prefact*jac_ev*enhance*unwgtfun*fkssymmetryfactor*vegas_wgt
      f_MC_S=f_r
      f_MC_H=f_r
      if (.not.nocntevents) then
         prefact_cnt_ssc=xinorm_ev/min(xiimax_ev,xiScut_used)*
     &        log(xicut_used/min(xiimax_ev,xiScut_used))/(1
     &        -y_ij_fks_ev)
         f_s=(prefact+prefact_cnt_ssc)*jac_cnt(0)*enhance
     $        *unwgtfun*fkssymmetryfactor*vegas_wgt
         f_s_MC_S=prefact*jac_cnt(0)*enhance
     $        *unwgtfun*fkssymmetryfactor*vegas_wgt
         f_s_MC_H=f_s_MC_S

         if (pmass(j_fks).eq.0d0) then
            prefact_c=xinorm_cnt(1)/xi_i_fks_cnt(1)/(1-y_ij_fks_ev)
            prefact_coll=xinorm_cnt(1)/xi_i_fks_cnt(1)*log(delta_used
     $           /deltaS)/deltaS
            f_c=(prefact_c+prefact_coll)*jac_cnt(1)
     $           *enhance*unwgtfun*fkssymmetryfactor*vegas_wgt
            f_c_MC_S=prefact_c*jac_cnt(1)
     $           *enhance*unwgtfun*fkssymmetryfactor*vegas_wgt
            f_c_MC_H=f_c_MC_S

            call set_cms_stuff(1)
            prefact_deg=xinorm_cnt(1)/xi_i_fks_cnt(1)/deltaS
            prefact_cnt_ssc_c=xinorm_cnt(1)/min(xiimax_cnt(1)
     &           ,xiScut_used)*log(xicut_used/min(xiimax_cnt(1)
     &           ,xiScut_used))*1/(1-y_ij_fks_ev)
            prefact_coll_c=xinorm_cnt(1)/min(xiimax_cnt(1),xiScut_used)
     $           *log(xicut_used/min(xiimax_cnt(1),xiScut_used))
     $           *log(delta_used/deltaS)/deltaS
            f_dc=jac_cnt(1)*prefact_deg/(shat/(32*pi**2))*enhance
     $           *unwgtfun*fkssymmetryfactorDeg*vegas_wgt
            f_sc=(prefact_c+prefact_coll+prefact_cnt_ssc_c
     &           +prefact_coll_c)*jac_cnt(2)*enhance*unwgtfun
     &           *fkssymmetryfactorDeg*vegas_wgt
            f_sc_MC_S=prefact_c*jac_cnt(2)
     $           *enhance*unwgtfun*fkssymmetryfactor*vegas_wgt
            f_sc_MC_H=f_sc_MC_S

            call set_cms_stuff(2)
            prefact_deg_sxi=xinorm_cnt(1)/min(xiimax_cnt(1),xiScut_used)
     &           *log(xicut_used/min(xiimax_cnt(1),xiScut_used))*1
     &           /deltaS
            prefact_deg_slxi=xinorm_cnt(1)/min(xiimax_cnt(1)
     &           ,xiScut_used)*( log(xicut_used)**2
     &           -log(min(xiimax_cnt(1),xiScut_used))**2 )*1/(2.d0
     &           *deltaS)
            f_dsc(1)=prefact_deg*jac_cnt(2)/(shat/(32*pi**2))*enhance
     &           *unwgtfun*fkssymmetryfactorDeg*vegas_wgt
            f_dsc(2)=prefact_deg_sxi*jac_cnt(2)/(shat/(32*pi**2))
     &           *enhance*unwgtfun*fkssymmetryfactorDeg*vegas_wgt
            f_dsc(3)=prefact_deg_slxi*jac_cnt(2)/(shat/(32*pi**2))
     &           *enhance*unwgtfun*fkssymmetryfactorDeg*vegas_wgt
            f_dsc(4)=( prefact_deg+prefact_deg_sxi )*jac_cnt(2)/(shat
     &           /(32*pi**2))*enhance*unwgtfun*fkssymmetryfactorDeg
     &           *vegas_wgt
         else
            f_c=0d0
            f_dc=0d0
            f_sc=0d0
            do i=1,4
               f_dsc(i)=0d0
            enddo
            f_c_MC_S=0d0
            f_c_MC_H=0d0
            f_sc_MC_S=0d0
            f_sc_MC_H=0d0
         endif
      else
         f_s=0d0
         f_c=0d0
         f_dc=0d0
         f_sc=0d0
         do i=1,4
            f_dsc(i)=0d0
         enddo
         f_s_MC_S=0d0
         f_s_MC_H=0d0
         f_c_MC_S=0d0
         f_c_MC_H=0d0
         f_sc_MC_S=0d0
         f_sc_MC_H=0d0
      endif
      call cpu_time(tAfter)
      tf_all=tf_all+(tAfter-tBefore)
      return
      end

      
      subroutine add_wgt(type,wgt1,wgt2,wgt3)
c Adds a contribution to the list in c_weight.inc. 'type' sets the type
c of the contribution and wgt1..wgt3 are the coefficients multiplying
c the logs. The arguments are:
c     type=1 : real-emission
c     type=2 : Born
c     type=3 : integrated counter terms
c     type=4 : soft counter-term
c     type=5 : collinear counter-term
c     type=6 : soft-collinear counter-term
c     type=7 : O(alphaS) expansion of Sudakov factor for NNLL+NLO
c     type=8 : soft counter-term (with n+1-body kin.)
c     type=9 : collinear counter-term (with n+1-body kin.)
c     type=10: soft-collinear counter-term (with n+1-body kin.)
c     type=11: real-emission (with n-body kin.)
c     type=12: MC subtraction with n-body kin.
c     type=13: MC subtraction with n+1-body kin.
c     type=14: virtual corrections
c     type=15: virt-trick: average born contribution
c     wgt1 : weight of the contribution not multiplying a scale log
c     wgt2 : coefficient of the weight multiplying the log[mu_R^2/Q^2]
c     wgt3 : coefficient of the weight multiplying the log[mu_F^2/Q^2]
c
c This subroutine increments the 'icontr' counter: each new call to this
c function makes sure that it's considered a new contribution. For each
c contribution, we save the
c     The type: itype(icontr)
c     The weights: wgt(1,icontr),wgt(2,icontr) and wgt(3,icontr) for
c         wgt1, wgt2 and wgt3, respectively.
c     The Bjorken x's: bjx(1,icontr), bjx(2,icontr)
c     The Ellis-Sexton scale squared used to compute the weight:
c        scales2(1,icontr)
c     The renormalisation scale squared used to compute the weight:
c        scales2(2,icontr)
c     The factorisation scale squared used to compute the weight:
c       scales2(3,icontr)
c     The value of the strong coupling: g_strong(icontr)
c     The FKS configuration: nFKS(icontr)
c     The boost to go from the momenta in the C.o.M. frame to the
c         laboratory frame: y_bst(icontr)      
c     The power of the strong coupling (g_strong) for the current
c       weight: QCDpower(icontr)
c     The momenta: momenta(j,i,icontr). For the Born contribution, the
c        counter-term momenta are used. This is okay for any IR-safe
c        observables.
c     The PDG codes: pdg(i,icontr). Always the ones with length
c        'nexternal' are used, because the momenta are also the 
c        'nexternal' ones. This is okay for IR-safe observables.
c     The PDG codes of the underlying Born process:
c        pdg_uborn(i,icontr). The PDGs of j_fks and i_fks are combined
c        to get the PDG code of the mother. The extra parton is given a
c        PDG=21 (gluon) code.
c     If the contribution belongs to an H-event or S-event:
c        H_event(icontr)
c     The weight of the born or real-emission matrix element
c        corresponding to this contribution: wgt_ME_tree. This weight does
c        include the 'ngluon' correction factor for the Born.
c
c Not set in this subroutine, but included in the c_weights common block
c are the
c     wgts(iwgt,icontr) : weights including scale/PDFs/logs. These are
c        normalised so that they can be used directly to compute cross
c        sections and fill plots. 'iwgt' goes from 1 to the maximum
c        number of weights obtained from scale and PDF reweighting, with
c        the iwgt=1 element being the central value.
c     plot_id(icontr) : =20 for Born, 11 for real-emission and 12 for
c        anything else.
c     plot_wgts(iwgt,icontr) : same as wgts(), but only non-zero for
c        unique contributions and non-unique are added to the unique
c        ones. 'Unique' here is defined that they would be identical in
c        an analysis routine (i.e. same momenta and PDG codes)
c     shower_scale(icontr) : The preferred shower starting scale for
c        this contribution
c     niproc(icontr) : number of combined subprocesses in parton_lum_*.f
c     parton_iproc(iproc,icontr) : value of the PDF for the iproc
c        contribution
c     parton_pdg(nexternal,iproc,icontr) : value of the PDG codes for
c        the iproc contribution
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'genps.inc'
      include 'coupl.inc'
      include 'fks_info.inc'
      include 'c_weight.inc'
      include 'q_es.inc'
      include 'reweight0.inc'
      integer type,i,j
      double precision wgt1,wgt2,wgt3
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision p_born(0:3,nexternal-1)
      common/pborn/    p_born
      double precision p_ev(0:3,nexternal)
      common/pev/      p_ev
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision    p1_cnt(0:3,nexternal,-2:2),wgt_cnt(-2:2)
     $                    ,pswgt_cnt(-2:2),jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision         fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg
      integer                                      ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
      double precision     iden_comp
      common /c_iden_comp/ iden_comp
      if (wgt1.eq.0d0 .and. wgt2.eq.0d0 .and. wgt3.eq.0d0) return
c Check for NaN's and INF's. Simply skip the contribution
      if (wgt1.ne.wgt1) return
      if (wgt2.ne.wgt2) return
      if (wgt3.ne.wgt3) return
      icontr=icontr+1
      if (icontr.gt.max_contr) then
         write (*,*) 'ERROR in add_wgt: too many contributions'
     &        ,max_contr
         stop 1
      endif
      itype(icontr)=type
      wgt(1,icontr)=wgt1
      wgt(2,icontr)=wgt2
      wgt(3,icontr)=wgt3
      bjx(1,icontr)=xbk(1)
      bjx(2,icontr)=xbk(2)
      scales2(1,icontr)=QES2
      scales2(2,icontr)=scale**2
      scales2(3,icontr)=q2fact(1)
      g_strong(icontr)=g
      nFKS(icontr)=nFKSprocess
      y_bst(icontr)=ybst_til_tolab
      call set_pdg(icontr,nFKSprocess)
      if (type.eq.2) then
c     Born contribution
         QCDpower(icontr)=nint(2*wgtbpower)
      else
c     Anything else
         QCDpower(icontr)=nint(2*wgtbpower+2)
      endif
c Compensate for the fact that in the Born matrix elements, we use the
c identical particle symmetry factor of the corresponding real emission
c matrix elements
c IDEN_COMP STUFF NEEDS TO BE UPDATED WHEN MERGING WITH 'FKS_EW' STUFF
      wgt_ME_tree(1,icontr)=wgt_me_born
      wgt_ME_tree(2,icontr)=wgt_me_real
      do i=1,nexternal
         do j=0,3
            if (p1_cnt(0,1,0).gt.0d0) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,0)
            elseif (p1_cnt(0,1,1).gt.0d0) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,1)
            elseif (p1_cnt(0,1,2).gt.0d0) then
               momenta_m(j,i,1,icontr)=p1_cnt(j,i,2)
            else
               if (i.lt.fks_i_d(nFKSprocess)) then
                  momenta_m(j,i,1,icontr)=p_born(j,i)
               elseif(i.eq.fks_i_d(nFKSprocess)) then
                  momenta_m(j,i,1,icontr)=0d0
               else
                  momenta_m(j,i,1,icontr)=p_born(j,i-1)
               endif
            endif
            momenta_m(j,i,2,icontr)=p_ev(j,i)
         enddo
      enddo
      if(type.eq.1 .or. type.eq. 8 .or. type.eq.9 .or. type.eq.10 .or.
     &     type.eq.13) then
c real emission and n+1-body kin. contributions to counter terms and MC
c subtr term
         do i=1,nexternal
            do j=0,3
               momenta(j,i,icontr)=momenta_m(j,i,2,icontr)
            enddo
         enddo
         H_event(icontr)=.true.
      elseif(type.ge.2 .and. type.le.7 .or. type.eq.11 .or. type.eq.12
     $        .or. type.eq.14 .or. type.eq.15)then
c Born, counter term, soft-virtual, or n-body kin. contributions to real
c and MC subtraction terms.
         do i=1,nexternal
            do j=0,3
               momenta(j,i,icontr)=momenta_m(j,i,1,icontr)
            enddo
         enddo
         H_event(icontr)=.false.
      else
         write (*,*) 'ERROR: unknown type in add_wgt',type
         stop 1
      endif
      return
      end

      subroutine include_veto_multiplier
      implicit none
c Multiply all the weights by the NNLL-NLO jet veto Sudakov factors,
c i.e., the term on the 2nd line of Eq.(20) of arXiv:1412.8408.
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight.inc'
      integer i,j
      if (H1_factor_virt.ne.0d0) then
         call compute_veto_multiplier(H1_factor_virt,1d0,1d0
     &        ,veto_multiplier)
         do i=1,icontr
            do j=1,3
               wgt(j,i)=wgt(j,i)*veto_multiplier
            enddo
         enddo
      else
         veto_multiplier=1d0
      endif
      end
      
      subroutine include_PDF_and_alphas
c Multiply the saved wgt() info by the PDFs, alpha_S and the scale
c dependence and saves the weights in the wgts() array. The weights in
c this array are now correctly normalised to compute the cross section
c or to fill histograms.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'c_weight.inc'
      include 'coupl.inc'
      include 'timing_variables.inc'
      include 'genps.inc'
      integer i,j,k
      double precision xlum,dlum,pi,mu2_r,mu2_f,mu2_q,rwgt_muR_dep_fac
     $     ,wgt_wo_pdf
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      do i=1,icontr
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
         mu2_r=scales2(2,i)
         mu2_f=scales2(3,i)
         q2fact(1)=mu2_f
         q2fact(2)=mu2_f
c call the PDFs
         xlum = dlum()
         if (iproc.gt.max_iproc) then
            write (*,*) 'ERROR iproc too large',iproc,max_iproc
            stop 1
         endif
c set_pdg_codes fills the niproc, parton_iproc, parton_pdg and parton_pdg_uborn
         call set_pdg_codes(iproc,pd,nFKSprocess,i)
c iwgt=1 is the central value (i.e. no scale/PDF reweighting).
         iwgt=1
         wgt_wo_pdf=(wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q) + wgt(3,i)
     &        *log(mu2_f/mu2_q))*g_strong(i)**QCDpower(i)
     &        *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
         wgts(iwgt,i)=xlum * wgt_wo_pdf
         do j=1,iproc
            parton_iproc(j,i)=parton_iproc(j,i) * wgt_wo_pdf
         enddo
         if (itype(i).eq.14) then
c Special for the soft-virtual needed for the virt-tricks. The
c *_wgt_mint variable should be directly passed to the mint-integrator
c and not be part of the plots nor computation of the cross section.
            virt_wgt_mint=virt_wgt_mint*xlum*g_strong(i)**QCDpower(i)
     &           *rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
            born_wgt_mint=born_wgt_mint*xlum*g_strong(i)**QCDpower(i)
     &           /(8d0*Pi**2)*rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
         endif
      enddo
      call cpu_time(tAfter)
      t_as=t_as+(tAfter-tBefore)
      return
      end


      subroutine set_pdg_codes(iproc,pd,iFKS,ict)
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'c_weight.inc'
      include 'fks_info.inc'
      integer j,k,iproc,ict,iFKS
      double precision  pd(0:maxproc),conv
      parameter (conv=389379660d0) ! conversion to picobarns
      include 'leshouche_decl.inc'
      common/c_leshouche_idup_d/ idup_d
c save also the separate contributions to the PDFs and the corresponding
c PDG codes
      niproc(ict)=iproc
      do j=1,iproc
         if (nincoming.eq.2) then
            parton_iproc(j,ict)=pd(j)*conv
         else
c           Keep GeV's for decay processes (no conv. factor needed)
            parton_iproc(j,ict)=pd(j)
         endif
         do k=1,nexternal
            parton_pdg(k,j,ict)=idup_d(iFKS,k,j)
            if (k.lt.fks_j_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k,j)
            elseif(k.eq.fks_j_d(iFKS)) then
               if ( abs(idup_d(iFKS,fks_i_d(iFKS),j)) .eq.
     &              abs(idup_d(iFKS,fks_j_d(iFKS),j)) ) then
                 parton_pdg_uborn(k,j,ict)=21
               elseif (abs(idup_d(iFKS,fks_i_d(iFKS),j)).eq.21) then
                 parton_pdg_uborn(k,j,ict)=idup_d(iFKS,fks_j_d(iFKS),j)
               elseif (idup_d(iFKS,fks_j_d(iFKS),j).eq.21) then
                 parton_pdg_uborn(k,j,ict)=-idup_d(iFKS,fks_i_d(iFKS),j)
               else
                 write (*,*)
     &                'ERROR in PDG assigment for underlying Born'
                 stop 1
               endif
            elseif(k.lt.fks_i_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k,j)
            elseif(k.eq.nexternal) then
               parton_pdg_uborn(k,j,ict)=0
            elseif(k.ge.fks_i_d(iFKS)) then
               parton_pdg_uborn(k,j,ict)=idup_d(iFKS,k+1,j)
            endif
         enddo
      enddo
      return
      end
      
      
      subroutine reweight_scale
c Use the saved c_weight info to perform scale reweighting. Extends the
c wgts() array to include the weights.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'c_weight.inc'
      include 'reweight.inc'
      include 'reweightNLO.inc'
      include 'timing_variables.inc'
      integer i,kr,kf,iwgt_save,dd
      double precision xlum(maxscales),dlum,pi,mu2_r(maxscales),c_mu2_r
     $     ,c_mu2_f,mu2_f(maxscales),mu2_q,alphas,g(maxscales)
     $     ,rwgt_muR_dep_fac
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c currently we have 'iwgt' weights in the wgts() array.
      iwgt_save=iwgt
c loop over all the contributions in the c_weights common block
      do i=1,icontr
         iwgt=iwgt_save
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
c Loop over the dynamical_scale_choices
         do dd=1,dyn_scale(0)
c renormalisation scale variation (requires recomputation of the strong
c coupling)
            call set_mu_central(i,dd,c_mu2_r,c_mu2_f)
            do kr=1,nint(scalevarR(0))
               if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
               mu2_r(kr)=c_mu2_r*scalevarR(kr)**2
               g(kr)=sqrt(4d0*pi*alphas(sqrt(mu2_r(kr))))
            enddo
c factorisation scale variation (require recomputation of the PDFs)
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               mu2_f(kf)=c_mu2_f*scalevarF(kf)**2
               q2fact(1)=mu2_f(kf)
               q2fact(2)=mu2_f(kf)
               xlum(kf) = dlum()
            enddo
            do kf=1,nint(scalevarF(0))
               if ((.not. lscalevar(dd)) .and. kf.ne.1) exit
               do kr=1,nint(scalevarR(0))
                  if ((.not. lscalevar(dd)) .and. kr.ne.1) exit
                  iwgt=iwgt+1   ! increment the iwgt for the wgts() array
                  if (iwgt.gt.max_wgt) then
                     write (*,*) 'ERROR too many weights in '/
     $                    /'reweight_scale',iwgt,max_wgt
                     stop 1
                  endif
c add the weights to the array
                  wgts(iwgt,i)=xlum(kf) * (wgt(1,i)+wgt(2,i)
     $                 *log(mu2_r(kr)/mu2_q)+wgt(3,i)*log(mu2_f(kf)
     $                 /mu2_q))*g(kr)**QCDpower(i)
                  wgts(iwgt,i)=wgts(iwgt,i)*rwgt_muR_dep_fac(
     &                 sqrt(mu2_r(kr)),sqrt(scales2(2,i)))
               enddo
            enddo
         enddo
      enddo
      call cpu_time(tAfter)
      tr_s=tr_s+(tAfter-tBefore)
      return
      end

      subroutine reweight_scale_NNLL
c Use the saved c_weight info to perform scale reweighting. Extends the
c wgts() array to include the weights. Special for the NNLL+NLO jet-veto
c computations (ickkw.eq.-1).
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'c_weight.inc'
      include 'reweight.inc'
      include 'reweightNLO.inc'
      include 'timing_variables.inc'
      integer i,ks,kh,iwgt_save
      double precision xlum(maxscales),dlum,pi,mu2_r(maxscales)
     &     ,mu2_f(maxscales),mu2_q,alphas,g(maxscales),rwgt_muR_dep_fac
     &     ,veto_multiplier_new(maxscales,maxscales)
     &     ,veto_compensating_factor_new
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      if (dyn_scale(0).gt.1) then
         write (*,*) "When doing NNLL+NLO veto, "/
     $        /"can only do one dynamical_scale_choice",dyn_scale(0)
         stop
      endif

c currently we have 'iwgt' weights in the wgts() array.
      iwgt_save=iwgt
c compute the new veto multiplier factor      
      do ks=1,nint(scalevarR(0))
         if ((.not. lscalevar(1)) .and. ks.ne.1) exit
         do kh=1,nint(scalevarF(0))
            if ((.not. lscalevar(1)) .and. kh.ne.1) exit
            if (H1_factor_virt.ne.0d0) then
               call compute_veto_multiplier(H1_factor_virt,scalevarR(ks)
     $              ,scalevarF(kh),veto_multiplier_new(ks,kh))
               veto_multiplier_new(ks,kh)=veto_multiplier_new(ks,kh)
     &              /veto_multiplier
            else
               veto_multiplier_new(ks,kh)=1d0
            endif
         enddo
      enddo
c loop over all the contributions in the c_weights common block
      do i=1,icontr
         iwgt=iwgt_save
         nFKSprocess=nFKS(i)
         xbk(1) = bjx(1,i)
         xbk(2) = bjx(2,i)
         mu2_q=scales2(1,i)
c Hard scale variation
         do kh=1,nint(scalevarF(0))
            if ((.not. lscalevar(1)) .and. kh.ne.1) exit
c soft scale variation
            do ks=1,nint(scalevarR(0))
               if ((.not. lscalevar(1)) .and. ks.ne.1) exit
               mu2_r(ks)=scales2(2,i)*scalevarR(ks)**2
               g(ks)=sqrt(4d0*pi*alphas(sqrt(mu2_r(ks))))
               mu2_f(ks)=scales2(2,i)*scalevarR(ks)**2
               q2fact(1)=mu2_f(ks)
               q2fact(2)=mu2_f(ks)
               xlum(ks) = dlum()
               iwgt=iwgt+1      ! increment the iwgt for the wgts() array
               if (iwgt.gt.max_wgt) then
                  write (*,*) 'ERROR too many weights in reweight_scale'
     &                 ,iwgt,max_wgt
                  stop 1
               endif
c add the weights to the array
               if (itype(i).ne.7) then
                  wgts(iwgt,i)=xlum(ks) * (wgt(1,i)+wgt(2,i)
     &                 *log(mu2_r(ks)/mu2_q)+wgt(3,i)*log(mu2_f(ks)
     &                 /mu2_q))*g(ks)**QCDpower(i)
               else
c special for the itype=7 (i.e, the veto-compensating factor)                  
                  call compute_veto_compensating_factor(H1_factor_virt
     &                 ,born_wgt_veto,scalevarR(ks),scalevarF(kh)
     &                 ,veto_compensating_factor_new)
                  wgts(iwgt,i)=xlum(ks) * wgt(1,i)*g(ks)**QCDpower(i)
     &                 /veto_compensating_factor
     &                 *veto_compensating_factor_new
               endif
               wgts(iwgt,i)=wgts(iwgt,i)*rwgt_muR_dep_fac(
     &              sqrt(mu2_r(ks)),sqrt(scales2(2,i)))
               wgts(iwgt,i)=wgts(iwgt,i)*veto_multiplier_new(ks,kh)
            enddo
         enddo
      enddo
      call cpu_time(tAfter)
      tr_s=tr_s+(tAfter-tBefore)
      return
      end

      subroutine reweight_pdf
c Use the saved c_weight info to perform PDF reweighting. Extends the
c wgts() array to include the weights.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'c_weight.inc'
      include 'reweight.inc'
      include 'reweightNLO.inc'
      include 'timing_variables.inc'
      integer n,i,nn
      double precision xlum,dlum,pi,mu2_r,mu2_f,mu2_q,rwgt_muR_dep_fac,g
     &     ,alphas
      external rwgt_muR_dep_fac
      parameter (pi=3.1415926535897932385d0)
      external dlum,alphas
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      do nn=1,lhaPDFid(0)
c Use as external loop the one over the PDF sets and as internal the one
c over the icontr. This reduces the number of calls to InitPDF and
c allows for better caching of the PDFs
         do n=0,nmemPDF(nn)
            iwgt=iwgt+1
            if (iwgt.gt.max_wgt) then
               write (*,*) 'ERROR too many weights in reweight_pdf',iwgt
     &              ,max_wgt
               stop 1
            endif
            call InitPDFm(nn,n)
            do i=1,icontr
               nFKSprocess=nFKS(i)
               xbk(1) = bjx(1,i)
               xbk(2) = bjx(2,i)
               mu2_q=scales2(1,i)
               mu2_r=scales2(2,i)
               mu2_f=scales2(3,i)
               q2fact(1)=mu2_f
               q2fact(2)=mu2_f
c Compute the luminosity
               xlum = dlum()
c Recompute the strong coupling: alpha_s in the PDF might change
               g=sqrt(4d0*pi*alphas(sqrt(mu2_r)))
c add the weights to the array
               wgts(iwgt,i)=xlum * (wgt(1,i) + wgt(2,i)*log(mu2_r/mu2_q)
     $              +wgt(3,i)*log(mu2_f/mu2_q))*g**QCDpower(i)
               wgts(iwgt,i)=wgts(iwgt,i)*
     &              rwgt_muR_dep_fac(sqrt(mu2_r),sqrt(mu2_r))
            enddo
         enddo
      enddo
      call InitPDFm(1,0)
      call cpu_time(tAfter)
      tr_pdf=tr_pdf+(tAfter-tBefore)
      return
      end

      subroutine fill_applgrid_weights(vegas_wgt)
c Fills the ApplGrid weights of appl_common.inc. This subroutine assumes
c that there is an unique PS configuration: at most one Born, one real
c and one set of counter terms. Among other things, this means that one
c must do MC over FKS directories.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'appl_common.inc'
      include 'nFKSconfigs.inc'
      include 'genps.inc'
      integer i
      double precision final_state_rescaling,vegas_wgt
      integer              flavour_map(fks_configs)
      common/c_flavour_map/flavour_map
      integer iproc_save(fks_configs),eto(maxproc,fks_configs),
     &     etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      if (icontr.gt.8) then
         write (*,*) 'ERROR: too many applgrid weights. '/
     &        /'Should have at most one of each itype.',icontr
         stop 1
      endif
      do i=1,4
         appl_w0(i)=0d0
         appl_wR(i)=0d0
         appl_wF(i)=0d0
         appl_wB(i)=0d0
         appl_x1(i)=0d0
         appl_x2(i)=0d0
         appl_QES2(i)=0d0
         appl_muR2(i)=0d0
         appl_muF2(i)=0d0
         appl_flavmap(i)=0
      enddo
      appl_event_weight = 0d0
      appl_vegaswgt = vegas_wgt
      if (icontr.eq.0) return
      do i=1,icontr
         appl_event_weight=appl_event_weight+wgts(1,i)/vegas_wgt
         final_state_rescaling = dble(iproc_save(nFKS(i))) /
     &        dble(appl_nproc(flavour_map(nFKS(i))))
         if (itype(i).eq.1) then
c     real
            appl_w0(1)=appl_w0(1)+wgt(1,i)*final_state_rescaling
            appl_x1(1)=bjx(1,i)
            appl_x2(1)=bjx(2,i)
            appl_flavmap(1) = flavour_map(nFKS(i))
            appl_QES2(1)=scales2(1,i)
            appl_muR2(1)=scales2(2,i)
            appl_muF2(1)=scales2(3,i)
         elseif (itype(i).eq.2) then
c     born
            appl_wB(2)=appl_wB(2)+wgt(1,i)*final_state_rescaling
            appl_x1(2)=bjx(1,i)
            appl_x2(2)=bjx(2,i)
            appl_flavmap(2) = flavour_map(nFKS(i))
            appl_QES2(2)=scales2(1,i)
            appl_muR2(2)=scales2(2,i)
            appl_muF2(2)=scales2(3,i)
         elseif (itype(i).eq.3 .or. itype(i).eq.4 .or. itype(i).eq.14
     &           .or. itype(i).eq.15)then
c     virtual, soft-virtual or soft-counter
            appl_w0(2)=appl_w0(2)+wgt(1,i)*final_state_rescaling
            appl_wR(2)=appl_wR(2)+wgt(2,i)*final_state_rescaling
            appl_wF(2)=appl_wF(2)+wgt(3,i)*final_state_rescaling
            appl_x1(2)=bjx(1,i)
            appl_x2(2)=bjx(2,i)
            appl_flavmap(2) = flavour_map(nFKS(i))
            appl_QES2(2)=scales2(1,i)
            appl_muR2(2)=scales2(2,i)
            appl_muF2(2)=scales2(3,i)
         elseif (itype(i).eq.5) then
c     collinear counter            
            appl_w0(3)=appl_w0(3)+wgt(1,i)*final_state_rescaling
            appl_wF(3)=appl_wF(3)+wgt(3,i)*final_state_rescaling
            appl_x1(3)=bjx(1,i)
            appl_x2(3)=bjx(2,i)
            appl_flavmap(3) = flavour_map(nFKS(i))
            appl_QES2(3)=scales2(1,i)
            appl_muR2(3)=scales2(2,i)
            appl_muF2(3)=scales2(3,i)
         elseif (itype(i).eq.6) then
c     soft-collinear counter            
            appl_w0(4)=appl_w0(4)+wgt(1,i)*final_state_rescaling
            appl_wF(4)=appl_wF(4)+wgt(3,i)*final_state_rescaling
            appl_x1(4)=bjx(1,i)
            appl_x2(4)=bjx(2,i)
            appl_flavmap(4) = flavour_map(nFKS(i))
            appl_QES2(4)=scales2(1,i)
            appl_muR2(4)=scales2(2,i)
            appl_muF2(4)=scales2(3,i)
         endif
      enddo
      return
      end
      

      subroutine set_pdg(ict,iFKS)
c fills the pdg and pdg_uborn variables. It uses only the 1st IPROC. For
c the pdg_uborn (the PDG codes for the underlying Born process) the PDG
c codes of i_fks and j_fks are combined to give the PDG code of the
c mother and the extra (n+1) parton is given the PDG code of the gluon.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'fks_info.inc'
      include 'genps.inc'
      integer k,ict,iFKS
      integer    maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     $     icolup(2,nexternal,maxflow),niprocs
      common /c_leshouche_inc/idup,mothup,icolup,niprocs
      do k=1,nexternal
         pdg(k,ict)=idup(k,1)
      enddo
      do k=1,nexternal
         if (k.lt.fks_j_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k,ict)
         elseif(k.eq.fks_j_d(iFKS)) then
            if ( abs(pdg(fks_i_d(iFKS),ict)) .eq.
     &           abs(pdg(fks_j_d(iFKS),ict)) ) then
c gluon splitting:  g -> XX
               pdg_uborn(k,ict)=21
            elseif (abs(pdg(fks_i_d(iFKS),ict)).eq.21) then
c final state gluon radiation:  X -> Xg
               pdg_uborn(k,ict)=pdg(fks_j_d(iFKS),ict)
            elseif (pdg(fks_j_d(iFKS),ict).eq.21) then
c initial state gluon splitting (gluon is j_fks):  g -> XX
               pdg_uborn(k,ict)=-pdg(fks_i_d(iFKS),ict)
            else
               write (*,*)
     &              'ERROR in PDG assigment for underlying Born'
               stop 1
            endif
         elseif(k.lt.fks_i_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k,ict)
         elseif(k.eq.nexternal) then
            pdg_uborn(k,ict)=21  ! give the extra particle a gluon PDG code
         elseif(k.ge.fks_i_d(iFKS)) then
            pdg_uborn(k,ict)=pdg(k+1,ict)
         endif
      enddo
      return
      end
      
      subroutine get_wgt_nbody(sig)
c Sums all the central weights that contribution to the nbody cross
c section
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      double precision sig
      integer i
      sig=0d0
      if (icontr.eq.0) return
      do i=1,icontr
         if (itype(i).eq.2 .or. itype(i).eq.3 .or. itype(i).eq.14 .or.
     &        itype(i).eq.7 .or. itype(i).eq.15) then
            sig=sig+wgts(1,i)
         endif
      enddo
      return
      end

      subroutine get_wgt_no_nbody(sig)
c Sums all the central weights that contribution to the cross section
c excluding the nbody contributions.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      double precision sig
      integer i
      sig=0d0
      if (icontr.eq.0) return
      do i=1,icontr
         if (itype(i).ne.2 .and. itype(i).ne.3 .and. itype(i).ne.14
     &        .and. itype(i).ne.7 .and. itype(i).ne.15) then
            sig=sig+wgts(1,i)
         endif
      enddo
      return
      end

      subroutine fill_plots
c Calls the analysis routine (which fill plots) for all the
c contributions in the c_weight common block. Instead of really calling
c it for all, it first checks if weights can be summed (i.e. they have
c the same PDG codes and the same momenta) before calling the analysis
c to greatly reduce the calls to the analysis routines.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      include 'timing_variables.inc'
      integer i,ii,j,max_weight
      logical momenta_equal,pdg_equal
      external momenta_equal,pdg_equal
      parameter (max_weight=maxscales*maxscales+maxpdfs+1)
      double precision www(max_weight)
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c fill the plots_wgts. Check if we can sum weights together before
c calling the analysis routines. This is the case if the PDG codes and
c the momenta are identical.
      do i=1,icontr
         do j=1,iwgt
            plot_wgts(j,i)=0d0
         enddo
         if (itype(i).eq.2) then
            plot_id(i)=20 ! Born
         elseif(itype(i).eq.1) then
            plot_id(i)=11 ! real-emission
         else
            plot_id(i)=12 ! soft-virtual and counter terms
         endif
c Loop over all previous icontr. If the plot_id, PDGs and momenta are
c equal to a previous icountr, add the current weight to the plot_wgts
c of that contribution and exit the do-loop. This loop extends to 'i',
c so if the current weight cannot be summed to a previous one, the ii=i
c contribution makes sure that it is added as a new element.
         do ii=1,i
            if (plot_id(ii).ne.plot_id(i)) cycle
            if (plot_id(i).eq.20 .or. plot_id(i).eq.12) then
               if (.not.pdg_equal(pdg_uborn(1,ii),pdg_uborn(1,i))) cycle
            else
               if (.not.pdg_equal(pdg(1,ii),pdg(1,i))) cycle
            endif
            if (.not.momenta_equal(momenta(0,1,ii),momenta(0,1,i)))cycle
            do j=1,iwgt
               plot_wgts(j,ii)=plot_wgts(j,ii)+wgts(j,i)
            enddo
            exit
         enddo
      enddo
      do i=1,icontr
         if (plot_wgts(1,i).ne.0d0) then
            if (iwgt.gt.max_weight) then
               write (*,*) 'ERROR too many weights in fill_plots',iwgt
     &              ,max_weight
               stop 1
            endif
            do j=1,iwgt
               www(j)=plot_wgts(j,i)
            enddo
c call the analysis/histogramming routines
            call outfun(momenta(0,1,i),y_bst(i),www,pdg(1,i),plot_id(i))
         endif
      enddo
      call cpu_time(tAfter)
      t_plot=t_plot+(tAfter-tBefore)
      return
      end

      subroutine fill_mint_function(f)
c Fills the function that is returned to the MINT integrator
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'mint.inc'
      integer i
      double precision f(nintegrals),sigint
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      sigint=0d0
      do i=1,icontr
         sigint=sigint+wgts(1,i)
      enddo
      f(1)=abs(sigint)
      f(2)=sigint
      f(3)=virt_wgt_mint
      f(4)=virtual_over_born
      f(5)=abs(virt_wgt_mint)
      f(6)=born_wgt_mint
      return
      end
      

      subroutine include_shape_in_shower_scale(p,iFKS)
c Includes the shape function from the MC counter terms in the shower
c starting scale. This function needs to be called (at least) once per
c FKS configuration that is included in the current PS point.
      implicit none
      include 'nexternal.inc'
      include 'run.inc'
      include 'c_weight.inc'
      include 'nFKSconfigs.inc'
      integer i,iFKS,Hevents,izero,mohdr
      double precision ddum(6),p(0:3,nexternal)
      logical ldum
      double precision    xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev(0:3)
     &                    ,p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision        ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      double precision    xm12
      integer                  ileg
      common/cscaleminmax/xm12,ileg
      character*4      abrv
      common /to_abrv/ abrv
      logical              MCcntcalled
      common/c_MCcntcalled/MCcntcalled
      double precision     SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      parameter (izero=0,mohdr=-100)
c Compute the shower starting scale including the shape function
      if ( (.not. MCcntcalled) .and.
     &     abrv.ne.'born' .and. ickkw.ne.4) then
         if(p(0,1).ne.-99d0)then
            call set_cms_stuff(mohdr)
            call assign_emsca(p,xi_i_fks_ev,y_ij_fks_ev)
            call kinematics_driver(xi_i_fks_ev,y_ij_fks_ev,shat,p,ileg,
     &           xm12,ddum(1),ddum(2),ddum(3),ddum(4),ddum(5),ddum(6)
     &           ,ldum)
         endif
      endif
      call set_cms_stuff(izero)
      call set_shower_scale(iFKS*2-1,.false.)
      call set_cms_stuff(mohdr)
      call set_shower_scale(iFKS*2,.true.)
c loop over all the weights and update the relevant ones
c (i.e. nFKS(i)=iFKS)
      do i=1,icontr
         if (nFKS(i).eq.iFKS) then
            if (H_event(i)) then
c H-event contribution
               shower_scale(i)=SCALUP(iFKS*2)
            else
c S-event contribution
               shower_scale(i)=SCALUP(iFKS*2-1)
            endif
         endif
      enddo
      return
      end


      subroutine sum_identical_contributions
c Sum contributions that would lead to an identical event before taking
c the ABS value. In particular this means adding the real emission with
c the MC counter terms for the H-events FKS configuration by FKS
c configuration, while for the S-events also contributions from the
c various FKS configurations can be summed together.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'timing_variables.inc'
      integer i,j,ii,jj,i_soft
      logical momenta_equal,pdg_equal,equal,found_S
      external momenta_equal,pdg_equal
      integer iproc_save(fks_configs),eto(maxproc,fks_configs),
     &     etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      logical               only_virt
      integer         imode
      common /c_imode/imode,only_virt
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      call cpu_time(tBefore)
      if (icontr.eq.0) return
c Find the contribution to sum all the S-event ones. This should be one
c that has a soft singularity. We set it to 'i_soft'.
      i_soft=0
      found_S=.false.
      do i=1,icontr
         if (H_event(i)) then
            cycle
         else
            found_S=.true.
         endif
         if (abs(pdg_type_d(nFKS(i),fks_i_d(nFKS(i)))).eq.21) then
            i_soft=i
            exit
         endif
      enddo
      if (found_S .and. i_soft.eq.0) then
         write (*,*) 'ERROR: S-event contribution found, '/
     $        /'but no FKS configuration with soft singularity'
         stop 1
      endif
c Main loop over contributions. For H-events we have to check explicitly
c to which contribution we can sum the current contribution (if any),
c while for the S-events we can sum it to the 'i_soft' one.
      do i=1,icontr
         do j=1,niproc(i)
            unwgt(j,i)=0d0
         enddo
         icontr_sum(0,i)=0
         if (H_event(i)) then
            do ii=1,i
               if (.not.H_event(ii)) cycle
c H-event. If PDG codes, shower starting scale and momenta are equal, we
c can sum them before taking ABS value.
               if (niproc(ii).ne.niproc(i)) cycle
               if (shower_scale(ii).ne.shower_scale(i)) cycle
               equal=.true.
               do j=1,niproc(ii)
                  if (.not.pdg_equal(parton_pdg(1,j,ii),
     &                               parton_pdg(1,j,i))) then
                     equal=.false.
                     exit
                  endif
               enddo
               if (.not. equal) cycle
               if (.not. momenta_equal(momenta(0,1,ii),
     &                                 momenta(0,1,i))) cycle
c     Identical contributions found: sum the contribution "i" to "ii"
               icontr_sum(0,ii)=icontr_sum(0,ii)+1
               icontr_sum(icontr_sum(0,ii),ii)=i
               do j=1,niproc(ii)
                  unwgt(j,ii)=unwgt(j,ii)+parton_iproc(j,i)
               enddo
               exit
            enddo
         else
c S-event: we can sum everything to 'i_soft': all the contributions to
c the S-events can be summed together. Ignore the shower_scale: this
c will be updated later
            icontr_sum(0,i_soft)=icontr_sum(0,i_soft)+1
            icontr_sum(icontr_sum(0,i_soft),i_soft)=i
            do j=1,niproc(i_soft)
               do jj=1,iproc_save(nFKS(i))
                  if (eto(jj,nFKS(i)).eq.j) then
c When computing upper bounding envelope (imode.eq.1) do not include the
c virtual corrections. Exception: when computing only the virtual, do
c include it here!
                     if (itype(i).eq.14 .and. imode.eq.1 .and. .not.
     $                    only_virt) exit
                     unwgt(j,i_soft)=unwgt(j,i_soft)+parton_iproc(jj,i)
                  endif
               enddo
            enddo
         endif
      enddo
      call cpu_time(tAfter)
      t_isum=t_isum+(tAfter-tBefore)
      return
      end

      subroutine update_shower_scale_Sevents
c When contributions from various FKS configrations are summed together
c for the S-events (see the sum_identical_contributions subroutine), we
c need to update the shower starting scale (because it is not
c necessarily the same for all of these summed FKS configurations). Take
c the weighted average over the FKS configurations as the shower scale
c for the summed contribution.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'nFKSconfigs.inc'
      integer i,j,ict
      double precision tmp_wgt(fks_configs),showerscale(fks_configs)
     $     ,temp_wgt,shsctemp
      do i=1,fks_configs
         tmp_wgt(i)=0d0
         showerscale(i)=-1d0
      enddo
c sum the weights that contribute to a single FKS configuration.
      do i=1,icontr
         if (H_event(i)) cycle
         if (icontr_sum(0,i).eq.0) cycle
         do j=1,icontr_sum(0,i)
            ict=icontr_sum(j,i)
            tmp_wgt(nFKS(ict))=tmp_wgt(nFKS(ict))+wgts(1,i)
            if (showerscale(nFKS(ict)).eq.-1d0) then
               showerscale(nFKS(ict))=shower_scale(ict)
c check that all the shower starting scales are identical for all the
c contribution to a given FKS configuration.
            elseif ( abs((showerscale(nFKS(ict))-shower_scale(ict))
     $                  /(showerscale(nFKS(ict))+shower_scale(ict)))
     $                                                  .gt. 1d-6 ) then
               write (*,*) 'ERROR in update_shower_scale_Sevents'
     $              ,showerscale(nFKS(ict)),shower_scale(ict)
               stop 1
            endif
         enddo
      enddo
c Compute the weighted average of the shower scale. Weight is given by
c the ABS cross section to given FKS configuration.
      temp_wgt=0d0
      shsctemp=0d0
      do i=1,fks_configs
         temp_wgt=temp_wgt+abs(tmp_wgt(i))
         shsctemp=shsctemp+abs(tmp_wgt(i))*showerscale(i)
      enddo
      if (temp_wgt.ne.0d0) then
         shsctemp=shsctemp/temp_wgt
      else
         shsctemp=0d0
      endif
c Overwrite the shower scale for the S-events
      do i=1,icontr
         if (H_event(i)) cycle
         if (icontr_sum(0,i).ne.0) shower_scale(i)=shsctemp
      enddo
      return
      end


      subroutine fill_mint_function_NLOPS(f,n1body_wgt)
c Fills the function that is returned to the MINT integrator. Depending
c on the imode we should or should not include the virtual corrections.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'mint.inc'
      integer i,j,ict
      double precision f(nintegrals),sigint,sigint1,sigint_ABS
     $     ,n1body_wgt,tmp_wgt,max_weight
      double precision           virt_wgt_mint,born_wgt_mint
      common /virt_born_wgt_mint/virt_wgt_mint,born_wgt_mint
      double precision virtual_over_born
      common /c_vob/   virtual_over_born
      logical               only_virt
      integer         imode
      common /c_imode/imode,only_virt
      sigint=0d0
      sigint1=0d0
      sigint_ABS=0d0
      n1body_wgt=0d0
      max_weight=0d0
      if (icontr.eq.0) then
         sigint_ABS=0d0
         sigint=0d0
         sigint1=0d0
      else
         do i=1,icontr
            sigint=sigint+wgts(1,i)
            max_weight=max(max_weight,abs(wgts(1,i)))
            if (icontr_sum(0,i).eq.0) cycle
            do j=1,niproc(i)
               sigint_ABS=sigint_ABS+abs(unwgt(j,i))
               sigint1=sigint1+unwgt(j,i) ! for consistency check
               max_weight=max(max_weight,abs(unwgt(j,i)))
            enddo
         enddo
c check the consistency of the results up to machine precision (10^-10 here)
         if (imode.ne.1 .or. only_virt) then
            if (abs((sigint-sigint1)/max_weight).gt.1d-10) then
               write (*,*) 'ERROR: inconsistent integrals #0',sigint
     $              ,sigint1,max_weight,abs((sigint-sigint1)/max_weight)
               do i=1, icontr
                  write (*,*) i,icontr_sum(0,i),niproc(i),wgts(1,i)
                  if (icontr_sum(0,i).eq.0) cycle
                  do j=1,niproc(i)
                     write (*,*) j,unwgt(j,i)
                  enddo
               enddo
               stop 1
            endif
         else
            sigint1=sigint1+virt_wgt_mint
            if (abs((sigint-sigint1)/max_weight).gt.1d-10) then
               write (*,*) 'ERROR: inconsistent integrals #1',sigint
     $              ,sigint1,max_weight,abs((sigint-sigint1)/max_weight)
     $              ,virt_wgt_mint
               do i=1, icontr
                  write (*,*) i,icontr_sum(0,i),niproc(i),wgts(1,i)
                  if (icontr_sum(0,i).eq.0) cycle
                  do j=1,niproc(i)
                     write (*,*) j,unwgt(j,i)
                  enddo
               enddo
               stop 1
            endif
         endif
c n1body_wgt is used for the importance sampling over FKS directories
         do i=1,icontr
            if (icontr_sum(0,i).eq.0) cycle
            tmp_wgt=0d0
            do j=1,icontr_sum(0,i)
               ict=icontr_sum(j,i)
               if ( itype(ict).ne.2  .and. itype(ict).ne.3 .and.
     $              itype(ict).ne.14 .and. itype(ict).ne.15)
     $                              tmp_wgt=tmp_wgt+wgts(1,ict)
            enddo
            n1body_wgt=n1body_wgt+abs(tmp_wgt)
         enddo
      endif
      f(1)=sigint_ABS
      f(2)=sigint
      f(3)=virt_wgt_mint
      f(4)=virtual_over_born
      f(5)=abs(virt_wgt_mint)
      f(6)=born_wgt_mint
      return
      end


      subroutine pick_unweight_contr(iFKS_picked)
c Randomly pick (weighted by the ABS values) the contribution to a given
c PS point that should be written in the event file.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      include 'timing_variables.inc'
      integer i,j,k,iFKS_picked,ict
      double precision tot_sum,rnd,ran2,current,target
      external ran2
      integer           i_process_addwrite
      common/c_addwrite/i_process_addwrite
      logical         Hevents
      common/SHevents/Hevents
      logical                 dummy
      double precision evtsgn
      common /c_unwgt/ evtsgn,dummy
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
      double precision     SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      call cpu_time(tBefore)
      if (icontr.eq.0) return
      tot_sum=0d0
      do i=1,icontr
         do j=1,niproc(i)
            tot_sum=tot_sum+abs(unwgt(j,i))
         enddo
      enddo
      rnd=ran2()
      current=0d0
      target=rnd*tot_sum
      i=1
      j=0
      do while (current.lt.target)
         j=j+1
         if (mod(j,niproc(i)+1).eq.0) then
            j=1
            i=i+1
         endif
         current=current+abs(unwgt(j,i))
      enddo
c found the contribution that should be written:
      icontr_picked=i
      iproc_picked=j
      if (H_event(icontr_picked)) then
         Hevents=.true.
         i_process_addwrite=iproc_picked
         iFKS_picked=nFKS(icontr_picked)
         SCALUP(iFKS_picked*2)=shower_scale(icontr_picked)
      else
         Hevents=.false.
         i_process_addwrite=etoi(iproc_picked,nFKS(icontr_picked))
         do k=1,icontr_sum(0,icontr_picked)
            ict=icontr_sum(k,icontr_picked)
            if (particle_type_d(nFKS(ict),fks_i_d(nFKS(ict))).eq.8) then
               iFKS_picked=nFKS(ict)
               exit
            endif
            if (k.eq.icontr_sum(0,icontr_picked)) then
               write (*,*) 'ERROR: no configuration with i_fks a gluon'
               stop 1
            endif
         enddo
         SCALUP(iFKS_picked*2-1)=shower_scale(icontr_picked)
      endif
      evtsgn=sign(1d0,unwgt(iproc_picked,icontr_picked))
      call cpu_time(tAfter)
      t_p_unw=t_p_unw+(tAfter-tBefore)
      return
      end


      subroutine fill_rwgt_lines
c Fills the lines, n_ctr_str, to be written in an event file with the
c (internal) information to perform scale and/or PDF reweighting. All
c information is available in each line to do the reweighting, apart
c from the momenta: these are put in the momenta_str_l() array, and a
c label in each of the n_ctr_str refers to a corresponding set of
c momenta in the momenta_str_l() array.
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      include 'fks_info.inc'
      integer k,i,ii,j,jj,ict,ipr,momenta_conf(2)
      logical momenta_equal,found
      double precision conv,momenta_str_l(0:3,nexternal,max_n_ctr)
      external momenta_equal
      character*512 procid,str_temp
      parameter (conv=389379660d0) ! conversion to picobarns
      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      logical         Hevents
      common/SHevents/Hevents
      wgtref=unwgt(iproc_picked,icontr_picked)
      n_ctr_found=0
      n_mom_conf=0
c Loop over all the contributions in the picked contribution (the latter
c is chosen in the pick_unweight_cont() subroutine)
      do i=1,icontr_sum(0,icontr_picked)
         ict=icontr_sum(i,icontr_picked)
c Check if the current set of momenta are already available in the
c momenta_str_l array. If not, add it.
         found=.false.
         do k=1,2
            do j=1,n_mom_conf
               if (momenta_m(0,1,k,ict).le.0d0) then
                  momenta_conf(k)=0
                  cycle
               endif
               if (momenta_equal(momenta_str_l(0,1,j),
     &                           momenta_m(0,1,k,ict))) then
                  momenta_conf(k)=j
                  found=.true.
                  exit
               endif
            enddo
            if (.not. found) then
               n_mom_conf=n_mom_conf+1
               do ii=1,nexternal
                  do jj=0,3
                     momenta_str(jj,ii,n_mom_conf)=
     &                                      momenta_m(jj,ii,k,ict)
                     momenta_str_l(jj,ii,n_mom_conf)=
     &                                      momenta_m(jj,ii,k,ict)
                  enddo
               enddo
               momenta_conf(k)=n_mom_conf
            endif
         enddo
         if (.not. Hevents) then
c For S-events, be careful to take all the IPROC that contribute to the
c iproc_picked:
            ipr=eto(etoi(iproc_picked,nFKS(ict)),nFKS(ict))
            do ii=1,iproc_save(nFKS(ict))
               if (eto(ii,nFKS(ict)).ne.ipr) cycle
               n_ctr_found=n_ctr_found+1

               if (nincoming.eq.2) then
                  write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &                 (wgt(j,ict)*conv,j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &                 nexternal
               else
                  write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &                 (wgt(j,ict),j=1,3),(wgt_me_tree(j,ict),j=1,2), 
     &                 nexternal
               endif

               procid=''
               do j=1,nexternal
                  write (str_temp,*) parton_pdg(j,ii,ict)
                  procid=trim(adjustl(procid))//' '
     &                 //trim(adjustl(str_temp))
               enddo
               n_ctr_str(n_ctr_found) =
     &              trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &              //trim(adjustl(procid))

               write (str_temp,'(i2,6(1x,d14.8),6(1x,i2),1x,i8,1x,d18.12)')
     &              QCDpower(ict),
     &              (bjx(j,ict),j=1,2),
     &              (scales2(j,ict),j=1,3),
     &              g_strong(ict),
     &              (momenta_conf(j),j=1,2),
     &              itype(ict),
     &              nFKS(ict),
     &              fks_i_d(nFKS(ict)),
     &              fks_j_d(nFKS(ict)),
     &              parton_pdg_uborn(fks_j_d(nFKS(ict)),ii,ict),
     &              parton_iproc(ii,ict)
               n_ctr_str(n_ctr_found) =
     &              trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &              //trim(adjustl(str_temp))
            enddo
         else
c H-event
            ipr=iproc_picked
            n_ctr_found=n_ctr_found+1

            if (nincoming.eq.2) then
               write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &              (wgt(j,ict)*conv,j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &              nexternal
            else
               write (n_ctr_str(n_ctr_found),'(5(1x,d18.12),1x,i2)')
     &              (wgt(j,ict),j=1,3),(wgt_me_tree(j,ict),j=1,2),
     &              nexternal
            endif

            procid=''
            do j=1,nexternal
               write (str_temp,*) parton_pdg(j,ipr,ict)
               procid=trim(adjustl(procid))//' '
     &              //trim(adjustl(str_temp))
            enddo
            n_ctr_str(n_ctr_found) =
     &           trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &           //trim(adjustl(procid))

            write (str_temp,'(i2,6(1x,d14.8),6(1x,i2),1x,i8,1x,d18.12)')
     &           QCDpower(ict),
     &           (bjx(j,ict),j=1,2),
     &           (scales2(j,ict),j=1,3),
     &           g_strong(ict),
     &           (momenta_conf(j),j=1,2),
     &           itype(ict),
     &           nFKS(ict),
     &           fks_i_d(nFKS(ict)),
     &           fks_j_d(nFKS(ict)),
     &           parton_pdg_uborn(fks_j_d(nFKS(ict)),ipr,ict),
     &           parton_iproc(ipr,ict)
            n_ctr_str(n_ctr_found) =
     &           trim(adjustl(n_ctr_str(n_ctr_found)))//' '
     &           //trim(adjustl(str_temp))


         endif
         if (n_ctr_found.ge.max_n_ctr) then
            write (*,*) 'ERROR: too many contributions in <rwgt>'
            stop1
         endif
      enddo
      end
      
      
      subroutine rotate_invar(pin,pout,cth,sth,cphi,sphi)
c Given the four momentum pin, returns the four momentum pout (in the same
c Lorentz frame) by performing a three-rotation of an angle theta 
c (cos(theta)=cth) around the y axis, followed by a three-rotation of an
c angle phi (cos(phi)=cphi) along the z axis. The components of pin
c and pout are given along these axes
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth-q2*sphi+q3*cphi*sth
      pout(2)=q1*sphi*cth+q2*cphi+q3*sphi*sth
      pout(3)=-q1*sth+q3*cth 
      pout(0)=pin(0)
      return
      end


      subroutine trp_rotate_invar(pin,pout,cth,sth,cphi,sphi)
c This subroutine performs a rotation in the three-space using a rotation
c matrix that is the transpose of that used in rotate_invar(). Thus, if
c called with the *same* angles, trp_rotate_invar() acting on the output
c of rotate_invar() will return the input of rotate_invar()
      implicit none
      real*8 cth,sth,cphi,sphi,pin(0:3),pout(0:3)
      real*8 q1,q2,q3
c
      q1=pin(1)
      q2=pin(2)
      q3=pin(3)
      pout(1)=q1*cphi*cth+q2*sphi*cth-q3*sth
      pout(2)=-q1*sphi+q2*cphi 
      pout(3)=q1*cphi*sth+q2*sphi*sth+q3*cth
      pout(0)=pin(0)
      return
      end


      subroutine getaziangles(p,cphi,sphi)
      implicit none
      real*8 p(0:3),cphi,sphi
      real*8 xlength,cth,sth
      double precision rho
      external rho
c
      xlength=rho(p)
      if(xlength.ne.0.d0)then
        cth=p(3)/xlength
        sth=sqrt(1-cth**2)
        if(sth.ne.0.d0)then
          cphi=p(1)/(xlength*sth)
          sphi=p(2)/(xlength*sth)
        else
          cphi=1.d0
          sphi=0.d0
        endif
      else
        cphi=1.d0
        sphi=0.d0
      endif
      return
      end

      subroutine phspncheck_born(ecm,xmass,xmom,pass)
c Checks four-momentum conservation.
c WARNING: works only in the partonic c.m. frame
      implicit none
      include 'nexternal.inc'
      real*8 ecm,xmass(nexternal-1),xmom(0:3,nexternal-1)
      real*8 tiny,xm,xlen4,xsum(0:3),xsuma(0:3),xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      integer jflag,npart,i,j,jj
      logical pass
c
      pass=.true.
      jflag=0
      npart=nexternal-1
      do i=0,3
        xsum(i)=0.d0
        xsuma(i)=0.d0
        do j=nincoming+1,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(i.eq.0)xsum(i)=xsum(i)-ecm
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        pass=.false.
        return
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(abs(xm-xmass(j))/ptmp(0).gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          pass=.false.
          return
        endif
      enddo
      return
      end


      subroutine phspncheck_nocms(npart,ecm,xmass,xmom,pass)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer npart,maxmom
      include "genps.inc"
      include "nexternal.inc"
      real*8 ecm,xmass(-max_branch:max_particles),
     # xmom(0:3,nexternal)
      real*8 tiny,vtiny,xm,xlen4,den,ecmtmp,xsum(0:3),xsuma(0:3),
     # xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-6)
      integer jflag,i,j,jj
      logical pass
      double precision dot
      external dot
c
      pass=.true.
      jflag=0
      do i=0,3
        if (nincoming.eq.2) then
          xsum(i)=-xmom(i,1)-xmom(i,2)
          xsuma(i)=abs(xmom(i,1))+abs(xmom(i,2))
        elseif(nincoming.eq.1) then
          xsum(i)=-xmom(i,1)
          xsuma(i)=abs(xmom(i,1))
        endif
        do j=nincoming+1,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved [nocms]'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(i2,1x,4(d14.8,1x))') j,(xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(a3,1x,4(d14.8,1x))') 'sum',(xsum(jj),jj=0,3)
        write(*,'(a3,1x,4(d14.8,1x))') 'rat',(xrat(jj),jj=0,3)
        pass=.false.
        return
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(ptmp(0).ge.1.d0)then
          den=ptmp(0)
        else
          den=1.d0
        endif
        if(abs(xm-xmass(j))/den.gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation [nocms]'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          pass=.false.
          return
        endif
      enddo
c
      if (nincoming.eq.2) then
         ecmtmp=sqrt(2d0*dot(xmom(0,1),xmom(0,2)))
      elseif (nincoming.eq.1) then
         ecmtmp=xmom(0,1)
      endif
      if(abs(ecm-ecmtmp).gt.vtiny)then
        write(*,*)'Inconsistent shat [nocms]'
        write(*,*)'ecm given=   ',ecm
        write(*,*)'ecm computed=',ecmtmp
        write(*,'(4(d14.8,1x))') (xmom(jj,1),jj=0,3)
        write(*,'(4(d14.8,1x))') (xmom(jj,2),jj=0,3)
        pass=.false.
        return
      endif

      return
      end


      function xlen4(v)
      implicit none
      real*8 xlen4,tmp,v(0:3)
c
      tmp=v(0)**2-v(1)**2-v(2)**2-v(3)**2
      xlen4=sign(1.d0,tmp)*sqrt(abs(tmp))
      return
      end


      subroutine set_shower_scale(iFKS,Hevents)
      implicit none
      include "nexternal.inc"
      include "madfks_mcatnlo.inc"
      integer iFKS
      logical Hevents
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision emsca,scalemin,scalemax,emsca_bare
      logical emscasharp
      common/cemsca/emsca,emsca_bare,emscasharp,scalemin,scalemax
      character*4 abrv
      common/to_abrv/abrv
      include 'nFKSconfigs.inc'
      double precision SCALUP(fks_configs*2)
      common /cshowerscale/SCALUP
      double precision shower_S_scale(fks_configs*2)
     &     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     &     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness

      double precision xm12
      integer ileg
      common/cscaleminmax/xm12,ileg

c Initialise
      SCALUP(iFKS)=0d0
c S events
      if(.not.Hevents)then
         if(abrv.ne.'born'.and.abrv.ne.'grid'.and.
     &      dampMCsubt.and.emsca.ne.0d0)then
            SCALUP(iFKS)=min(emsca,scalemax)
         else
            call assign_scaleminmax(shat_ev,xi_i_fks_ev,scalemin
     $           ,scalemax,ileg,xm12)
            SCALUP(iFKS)=scalemax
         endif
         SCALUP(iFKS)=min(SCALUP(iFKS),shower_S_scale(iFKS))
c H events
      else
         if(dampMCsubt.and.emsca.ne.0d0)then
            SCALUP(iFKS)=scalemax
         else
            call assign_scaleminmax(shat_ev,xi_i_fks_ev,scalemin
     $           ,scalemax,ileg,xm12)
            SCALUP(iFKS)=scalemax
         endif
         SCALUP(iFKS)=min(SCALUP(iFKS),max(shower_H_scale(iFKS),
     &                    ref_H_scale(iFKS)-min(emsca,scalemax)))
      endif
c Minimal starting scale
      SCALUP(iFKS)=max(SCALUP(iFKS),3d0)

      return
      end


      subroutine set_shower_scale_noshape(pp,iFKS)
      implicit none
      integer iFKS,j,i,iSH,nmax
      include "nexternal.inc"
      include "madfks_mcatnlo.inc"
      include 'run.inc'
      include 'nFKSconfigs.inc'
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev
      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision shower_S_scale(fks_configs*2)
     &     ,shower_H_scale(fks_configs*2),ref_H_scale(fks_configs*2)
     &     ,pt_hardness
      common /cshowerscale2/shower_S_scale,shower_H_scale,ref_H_scale
     &     ,pt_hardness
      double precision ptparton,pt,pp(0:3,nexternal),ppp(0:3,nexternal)
      external pt
c jet cluster algorithm
      integer NN,NJET,JET(nexternal)
      double precision pQCD(0:3,nexternal),PJET(0:3,nexternal),rfj,sycut
     $     ,palg,amcatnlo_fastjetdmergemax,di(nexternal)
      external amcatnlo_fastjetdmergemax

c Initialise
      NN=0
      ppp=0d0
      pQCD=0d0
      pt_hardness=0d0
      do j=1,nexternal
         if (j.gt.nincoming.and.is_a_j(j)) then
            NN=NN+1
            ptparton=pt(pp(0,j))
         endif
      enddo

c Unphysical situation
      if(NN.le.0)then
         write(*,*)'Error in set_shower_scale_noshape:'
         write(*,*)'not enough QCD partons in process ',NN
         stop
c Processes without jets at the Born
      elseif(NN.eq.1)then
         shower_S_scale(iFKS)=sqrtshat_cnt(0)
         shower_H_scale(iFKS)=sqrtshat_ev-ptparton
c$$$         shower_H_scale(iFKS)=sqrtshat_cnt(0)
         ref_H_scale(iFKS)=0d0
c Processes with jets at the Born (iSH = 1 (2) means S (H) events)
      else
         do iSH=1,2
            if(iSH.eq.1)then
               nmax=nexternal-1
               do j=1,nmax
                  do i=0,3
                     ppp(i,j)=p_born(i,j)
                  enddo
               enddo
            elseif(iSH.eq.2)then
               nmax=nexternal
               do j=1,nmax
                  do i=0,3
                     ppp(i,j)=pp(i,j)
                  enddo
               enddo
            else
               write(*,*)'Wrong iSH inset_shower_scale_noshape: ',iSH
               stop
            endif
            if(ppp(0,1).gt.0d0)then
c Put all (light) QCD partons in momentum array for jet clustering.
               NN=0
               do j=nincoming+1,nmax
                  if (is_a_j(j))then
                     NN=NN+1
                     do i=0,3
                        pQCD(i,NN)=ppp(i,j)
                     enddo
                  endif
               enddo
c One MUST use kt, and no lower pt cut. The radius parameter can be changed
               palg=1d0         ! jet algorithm: 1.0=kt, 0.0=C/A, -1.0 = anti-kt
               sycut=0d0        ! minimum jet pt
               rfj=1d0          ! the radius parameter
               call amcatnlo_fastjetppgenkt_timed(pQCD,NN,rfj,sycut,palg,
     &                                            pjet,njet,jet)
               do i=1,NN
                  di(i)=sqrt(amcatnlo_fastjetdmergemax(i-1))
                  if (i.gt.1.and.di(i).gt.di(i-1))then
                     write(*,*)'Error in set_shower_scale_noshape'
                     write(*,*)NN,i,di(i),di(i-1)
                     stop
                  endif
               enddo
               if(iSH.eq.1)shower_S_scale(iFKS)=di(NN)
               if(iSH.eq.2)then
                  ref_H_scale(iFKS)=di(NN-1)
                  pt_hardness=di(NN)
c$$$                  shower_H_scale(iFKS)=ref_H_scale(iFKS)-pt_hardness
                  shower_H_scale(iFKS)=ref_H_scale(iFKS)-pt_hardness/2d0
               endif
            else
               if(iSH.eq.1)shower_S_scale(iFKS)=sqrtshat_cnt(0)
               if(iSH.eq.2)then
                  ref_H_scale(iFKS)=shower_S_scale(iFKS)
                  shower_H_scale(iFKS)=ref_H_scale(iFKS)
               endif
            endif
         enddo
      endif

      return
      end



      subroutine sreal(pp,xi_i_fks,y_ij_fks,wgt)
c Wrapper for the n+1 contribution. Returns the n+1 matrix element
c squared reduced by the FKS damping factor xi**2*(1-y).
c Close to the soft or collinear limits it calls the corresponding
c Born and multiplies with the AP splitting function or eikonal factors.
      implicit none
      include "nexternal.inc"
      include "coupl.inc"

      double precision pp(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks

      double precision shattmp,dot
      integer i,j

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      logical softtest,colltest
      common/sctests/softtest,colltest

      double precision zero,tiny
      parameter (zero=0d0)
      
      integer icount
      data icount /0/

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision pmass(nexternal)
      include "pmass.inc"

      if (softtest.or.colltest) then
         tiny=1d-8
      else
         tiny=1d-6
      endif

      if(pp(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
        wgt=0.d0
        return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(pp(0,1),pp(0,2))
      else
         shattmp=pp(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      if (1d0-y_ij_fks.lt.tiny)then
         if (pmass(j_fks).eq.zero.and.j_fks.le.nincoming)then
            call sborncol_isr(pp,xi_i_fks,y_ij_fks,wgt)
         elseif (pmass(j_fks).eq.zero.and.j_fks.ge.nincoming+1)then
            call sborncol_fsr(pp,xi_i_fks,y_ij_fks,wgt)
         else
            wgt=0d0
         endif
      elseif (xi_i_fks.lt.tiny)then
         if (i_type.eq.8 .and. pmass(i_fks).eq.0d0)then
c i_fks is gluon
            call sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
         elseif (abs(i_type).eq.3)then
c i_fks is (anti-)quark
            wgt=0d0
         else
            write(*,*) 'FATAL ERROR #1 in sreal',i_type,i_fks
            stop
         endif
      else
         call smatrix_real(pp,wgt)
         wgt=wgt*xi_i_fks**2*(1d0-y_ij_fks)
      endif

      if(wgt.lt.0.d0)then
         icount=icount+1
         if (icount.le.10) then
            write(*,*) 'Warning, numerical problem found in sreal. '/
     $           /'Setting weight to zero',wgt,xi_i_fks,y_ij_fks
            do i=1,nexternal
               write(*,*) 'particle ',i,', ',(pp(j,i),j=0,3)
            enddo
            if (icount.eq.25) then
               write (*,*) 'ERROR 25 problems found... '/
     $              /'stopping the code'
               stop
            endif
         endif
         wgt=0d0
      endif

      return
      end



      subroutine sborncol_fsr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      logical rotategranny
      common/crotategranny/rotategranny

      double precision cthbe,sthbe,cphibe,sphibe
      common/cbeangles/cthbe,sthbe,cphibe,sphibe

      double precision p_born_rot(0:3,nexternal-1)

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      integer i,imother_fks
      double precision t,z,ap,E_j_fks,E_i_fks,Q,cphi_mother,
     # sphi_mother,pi(0:3),pj(0:3)
      double complex wgt1(2),W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))
      double precision iden_comp
      common /c_iden_comp/iden_comp
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_fsr"
         wgt=0.d0
         return
      endif

      E_j_fks = p(0,j_fks)
      E_i_fks = p(0,i_fks)
      z = 1d0 - E_i_fks/(E_i_fks+E_j_fks)
      t = z * shat/4d0
      if(rotategranny .and. nexternal-1.ne.3 .and. nincoming.eq.2)then
c Exclude 2->1 (at the Born level) processes: matrix elements are
c independent of the PS point, but non-zero helicity configurations
c might flip when rotating the momenta.
        do i=1,nexternal-1
          call trp_rotate_invar(p_born(0,i),p_born_rot(0,i),
     #                          cthbe,sthbe,cphibe,sphibe)
        enddo
        CalculatedBorn=.false.
        call sborn(p_born_rot,wgt1)
        CalculatedBorn=.false.
      else
        call sborn(p_born,wgt1)
      endif
      call AP_reduced(j_type,i_type,t,z,ap)
      if (abs(j_type).eq.3 .and. i_type.eq.8) then
         Q=0d0
         wgt1(2)=0d0
      elseif (m_type.eq.8) then
c Insert <ij>/[ij] which is not included by sborn()
         if (1d0-y_ij_fks.lt.vtiny)then
            azifact=xij_aor
         else
            do i=0,3
               pi(i)=p_i_fks_ev(i)
               pj(i)=p(i,j_fks)
            enddo
            if(rotategranny)then
              call trp_rotate_invar(pi,pi,cthbe,sthbe,cphibe,sphibe)
              call trp_rotate_invar(pj,pj,cthbe,sthbe,cphibe,sphibe)
            endif
            CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
            CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
            CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
            CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
            Wij_angle=(0d0,0d0)
            Wij_recta=(0d0,0d0)
            do i=1,4
               Wij_angle = Wij_angle + W1(i)*W2(i)
               Wij_recta = Wij_recta + W3(i)*W4(i)
            enddo
            azifact=Wij_angle/Wij_recta
         endif
c Insert the extra factor due to Madgraph convention for polarization vectors
         imother_fks=min(i_fks,j_fks)
         if(rotategranny)then
           call getaziangles(p_born_rot(0,imother_fks),
     #                       cphi_mother,sphi_mother)
         else
           call getaziangles(p_born(0,imother_fks),
     #                       cphi_mother,sphi_mother)
         endif
         wgt1(2) = -(cphi_mother-ximag*sphi_mother)**2 *
     #             wgt1(2) * azifact
         call Qterms_reduced_timelike(j_type, i_type, t, z, Q)
      else
         write(*,*) 'FATAL ERROR in sborncol_fsr',i_type,j_type,i_fks,j_fks
         stop
      endif
      wgt=dble(wgt1(1)*ap+wgt1(2)*Q)*iden_comp
      return
      end



      subroutine sborncol_isr(p,xi_i_fks,y_ij_fks,wgt)
      implicit none
      include "nexternal.inc"
      double precision p(0:3,nexternal),wgt
      double precision xi_i_fks,y_ij_fks
C  
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double complex xij_aor
      common/cxij_aor/xij_aor

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      double precision p_born_rot(0:3,nexternal-1)

      integer i
      double precision t,z,ap,Q,cphi_mother,sphi_mother,pi(0:3),pj(0:3)
      double complex wgt1(2),W1(6),W2(6),W3(6),W4(6),Wij_angle,Wij_recta
      double complex azifact

      double precision zero,vtiny
      parameter (zero=0d0)
      parameter (vtiny=1d-8)
      double complex ximag
      parameter (ximag=(0.d0,1.d0))
      double precision iden_comp
      common /c_iden_comp/iden_comp
C  
      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sborncol_isr"
         wgt=0.d0
         return
      endif

      z = 1d0 - xi_i_fks
c sreal return {\cal M} of FKS except for the partonic flux 1/(2*s).
c Thus, an extra factor z (implicit in the flux of the reduced Born
c in FKS) has to be inserted here
      t = z*shat/4d0
      if(j_fks.eq.2 .and. nexternal-1.ne.3 .and. nincoming.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed.
c Exclude 2->1 (at the Born level) processes: matrix elements are
c independent of the PS point, but non-zero helicity configurations
c might flip when rotating the momenta.
        do i=1,nexternal-1
          p_born_rot(0,i)=p_born(0,i)
          p_born_rot(1,i)=-p_born(1,i)
          p_born_rot(2,i)=p_born(2,i)
          p_born_rot(3,i)=-p_born(3,i)
        enddo
        CalculatedBorn=.false.
        call sborn(p_born_rot,wgt1)
        CalculatedBorn=.false.
      else
        call sborn(p_born,wgt1)
      endif
      call AP_reduced(m_type,i_type,t,z,ap)
      if (abs(m_type).eq.3) then
         Q=0d0
         wgt1(2)=0d0
      else
c Insert <ij>/[ij] which is not included by sborn()
         if (1d0-y_ij_fks.lt.vtiny)then
            azifact=xij_aor
         else
            do i=0,3
               pi(i)=p_i_fks_ev(i)
               pj(i)=p(i,j_fks)
            enddo
            if(j_fks.eq.2 .and. nincoming.eq.2)then
c Rotation according to innerpin.m. Use rotate_invar() if a more 
c general rotation is needed
               pi(1)=-pi(1)
               pi(3)=-pi(3)
               pj(1)=-pj(1)
               pj(3)=-pj(3)
            endif
            CALL IXXXSO(pi ,ZERO ,+1,+1,W1)        
            CALL OXXXSO(pj ,ZERO ,-1,+1,W2)        
            CALL IXXXSO(pi ,ZERO ,-1,+1,W3)        
            CALL OXXXSO(pj ,ZERO ,+1,+1,W4)        
            Wij_angle=(0d0,0d0)
            Wij_recta=(0d0,0d0)
            do i=1,4
               Wij_angle = Wij_angle + W1(i)*W2(i)
               Wij_recta = Wij_recta + W3(i)*W4(i)
            enddo
            azifact=Wij_angle/Wij_recta
         endif
c Insert the extra factor due to Madgraph convention for polarization vectors
         if(j_fks.eq.2 .and. nincoming.eq.2)then
           cphi_mother=-1.d0
           sphi_mother=0.d0
         else
           cphi_mother=1.d0
           sphi_mother=0.d0
         endif
         wgt1(2) = -(cphi_mother+ximag*sphi_mother)**2 *
     #             wgt1(2) * dconjg(azifact)
         call Qterms_reduced_spacelike(m_type, i_type, t, z, Q)
      endif
      wgt=dble(wgt1(1)*ap+wgt1(2)*Q)*iden_comp
      return
      end



      subroutine AP_reduced(part1, part2, t, z, ap)
c Returns Altarelli-Parisi splitting function summed/averaged over helicities
c times prefactors such that |M_n+1|^2 = ap * |M_n|^2. This means
c    AP_reduced = (1-z) P_{S(part1,part2)->part1+part2}(z) * gS^2/t
c Therefore, the labeling conventions for particle IDs are not as in FKS:
c part1 and part2 are the two particles emerging from the branching.
c part1 and part2 can be either gluon (8) or (anti-)quark (+-3). z is the
c fraction of the energy of part1 and t is the invariant mass of the mother.
      implicit none

      integer part1, part2
      double precision z,ap,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         ap = 2d0 * CA * ( (1d0-z)**2/z + z + z*(1d0-z)**2 )

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         ap = TR * ( z**2 + (1d0-z)**2 )*(1d0-z)
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         ap = CF * (1d0+z**2)

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         ap = CF * (1d0+(1d0-z)**2)*(1d0-z)/z
      else
         write (*,*) 'Fatal error in AP_reduced',part1,part2
         stop
      endif

      ap = ap*g**2/t

      return
      end



      subroutine AP_reduced_prime(part1, part2, t, z, apprime)
c Returns (1-z)*P^\prime * gS^2/t, with the same conventions as AP_reduced
      implicit none

      integer part1, part2
      double precision z,apprime,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         apprime = 0d0

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         apprime = -2 * TR * z * (1d0-z)**2
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         apprime = - CF * (1d0-z)**2

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         apprime = - CF * z * (1d0-z)
      else
         write (*,*) 'Fatal error in AP_reduced_prime',part1,part2
         stop
      endif

      apprime = apprime*g**2/t

      return
      end



      subroutine Qterms_reduced_timelike(part1, part2, t, z, Qterms)
c Eq's B.31 to B.34 of FKS paper, times (1-z)*gS^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced
      implicit none

      integer part1, part2
      double precision z,Qterms,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         Qterms = -4d0 * CA * z*(1d0-z)**2

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         Qterms = 4d0 * TR * z*(1d0-z)**2
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         Qterms = 0d0

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         Qterms = 0d0
      else
         write (*,*) 'Fatal error in Qterms_reduced_timelike',part1,part2
         stop
      endif

      Qterms = Qterms*g**2/t

      return
      end



      subroutine Qterms_reduced_spacelike(part1, part2, t, z, Qterms)
c Eq's B.42 to B.45 of FKS paper, times (1-z)*gS^2/t. The labeling
c conventions for particle IDs are the same as those in AP_reduced.
c Thus, part1 has momentum fraction z, and it is the one off-shell
c (see (FKS.B.41))
      implicit none

      integer part1, part2
      double precision z,Qterms,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         Qterms = -4d0 * CA * (1d0-z)**2/z

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         Qterms = 0d0
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         Qterms = 0d0

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         Qterms = -4d0 * CF * (1d0-z)**2/z
      else
         write (*,*) 'Fatal error in Qterms_reduced_spacelike',part1,part2
         stop
      endif

      Qterms = Qterms*g**2/t

      return
      end


      subroutine AP_reduced_SUSY(part1, part2, t, z, ap)
c Same as AP_reduced, except for the fact that it only deals with
c   go -> go g
c   sq -> sq g
c splittings in SUSY. We assume this function to be called with 
c part2==colour(i_fks)
      implicit none

      integer part1, part2
      double precision z,ap,t

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part2.ne.8)then
         write (*,*) 'Fatal error #0 in AP_reduced_SUSY',part1,part2
         stop
      endif

      if (part1.eq.8)then
c go->gog splitting
         ap = CA * (1d0+z**2)

      elseif(abs(part1).eq.3)then
c sq->sqg splitting
         ap = 2d0 * CF * z

      else
         write (*,*) 'Fatal error in AP_reduced_SUSY',part1,part2
         stop
      endif

      ap = ap*g**2/t

      return
      end



      subroutine AP_reduced_massive(part1, part2, t, z, q2, m2, ap)
c Returns massive Altarelli-Parisi splitting function summed/averaged over helicities
c times prefactors such that |M_n+1|^2 = ap * |M_n|^2. This means
c    AP_reduced = (1-z) P_{S(part1,part2)->part1+part2}(z) * gS^2/t
c Therefore, the labeling conventions for particle IDs are not as in FKS:
c part1 and part2 are the two particles emerging from the branching.
c part1 and part2 can be either gluon (8) or (anti-)quark (+-3). z is the
c fraction of the energy of part1 and t is the invariant mass of the mother.
      implicit none

      integer part1, part2
      double precision z,ap,t,q2,m2

      double precision CA,TR,CF
      parameter (CA=3d0,TR=1d0/2d0,CF=4d0/3d0)

      include "coupl.inc"

      if (part1.eq.8 .and. part2.eq.8)then
c g->gg splitting
         ap = 2d0 * CA * ( (1d0-z)**2/z + z + z*(1d0-z)**2 )

      elseif(abs(part1).eq.3 .and. abs(part2).eq.3)then
c g->qqbar splitting
         ap = TR * ( z**2 + (1d0-z)**2 )*(1d0-z) + TR * 2d0*m2/(z*q2)
         
      elseif(abs(part1).eq.3 .and. part2.eq.8)then
c q->qg splitting
         ap = CF * (1d0+z**2) - CF * 2d0*m2/(z*q2)

      elseif(part1.eq.8 .and. abs(part2).eq.3)then
c q->gq splitting
         ap = CF * (1d0+(1d0-z)**2)*(1d0-z)/z - CF * 2d0*m2/(z*q2)
      else
         write (*,*) 'Fatal error in AP_reduced',part1,part2
         stop
      endif

      ap = ap*g**2/t

      return
      end



      subroutine sbornsoft(pp,xi_i_fks,y_ij_fks,wgt)
      implicit none

      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "coupl.inc"

      integer m,n

      double precision softcontr,pp(0:3,nexternal),wgt,eik,xi_i_fks
     &     ,y_ij_fks
      double complex wgt1(2)
      integer i,j

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision zero,pmass(nexternal)
      parameter(zero=0d0)
      double precision iden_comp
      common /c_iden_comp/iden_comp
      include "pmass.inc"
c
c Call the Born to be sure that 'CalculatedBorn' is done correctly. This
c should always be done before calling the color-correlated Borns,
c because of the caching of the diagrams.
c
      call sborn(p_born(0,1),wgt1)
c
      softcontr=0d0
      do i=1,fks_j_from_i(i_fks,0)
         do j=1,i
            m=fks_j_from_i(i_fks,i)
            n=fks_j_from_i(i_fks,j)
            if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)) .and.
     &           n.ne.i_fks.and.m.ne.i_fks) then
               call sborn_sf(p_born,m,n,wgt)
               if (wgt.ne.0d0) then
                  call eikonal_reduced(pp,m,n,i_fks,j_fks,
     #                                 xi_i_fks,y_ij_fks,eik)
                  softcontr=softcontr+wgt*eik
               endif
            endif
         enddo
      enddo
      wgt=softcontr*iden_comp
c Add minus sign to compensate the minus in the color factor
c of the color-linked Borns (b_sf_0??.f)
c Factor two to fix the limits.
      wgt=-2d0*wgt
      return
      end






      subroutine eikonal_reduced(pp,m,n,i_fks,j_fks,xi_i_fks,y_ij_fks,eik)
c     Returns the eikonal factor
      implicit none

      include "nexternal.inc"
      double precision eik,pp(0:3,nexternal),xi_i_fks,y_ij_fks
      double precision dot,dotnm,dotni,dotmi,fact
      integer n,m,i_fks,j_fks,i
      integer softcol

      include "coupl.inc"

      external dot
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      real*8 phat_i_fks(0:3)

      double precision zero,pmass(nexternal),tiny
      parameter(zero=0d0)
      parameter(tiny=1d-6)
      include "pmass.inc"

c Define the reduced momentum for i_fks
      softcol=0
      if (1d0-y_ij_fks.lt.tiny)softcol=2
      if(p_i_fks_cnt(0,softcol).lt.0d0)then
        if(xi_i_fks.eq.0.d0)then
           write (*,*) 'Error #1 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
           stop
        endif
        if(pp(0,i_fks).ne.0.d0)then
          write(*,*)'WARNING in eikonal_reduced: no cnt momenta',
     #      softcol,xi_i_fks,y_ij_fks
          do i=0,3
            phat_i_fks(i)=pp(i,i_fks)/xi_i_fks
          enddo
        else
          write (*,*) 'Error #2 in eikonal_reduced',
     #                 softcol,xi_i_fks,y_ij_fks
          stop
        endif
      else
        do i=0,3
          phat_i_fks(i)=p_i_fks_cnt(i,softcol)
        enddo
      endif
c Calculate the eikonal factor
      dotnm=dot(pp(0,n),pp(0,m))
      if ((m.ne.j_fks .and. n.ne.j_fks) .or. pmass(j_fks).ne.ZERO) then
         dotmi=dot(pp(0,m),phat_i_fks)
         dotni=dot(pp(0,n),phat_i_fks)
         fact= 1d0-y_ij_fks
      elseif (m.eq.j_fks .and. n.ne.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=dot(pp(0,n),phat_i_fks)
         dotmi=sqrtshat/2d0 * pp(0,j_fks)
         fact= 1d0
      elseif (m.ne.j_fks .and. n.eq.j_fks .and.
     &        pmass(j_fks).eq.ZERO) then
         dotni=sqrtshat/2d0 * pp(0,j_fks)
         dotmi=dot(pp(0,m),phat_i_fks)
         fact= 1d0
      else
         write (*,*) 'Error #3 in eikonal_reduced'
         stop
      endif

      eik = dotnm/(dotni*dotmi)*fact

      eik = eik * g**2

      return
      end


      subroutine sreal_deg(p,xi_i_fks,y_ij_fks,
     #                     collrem_xi,collrem_lxi)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      include 'q_es.inc'
      include "run.inc"
      include 'reweight.inc'

      double precision p(0:3,nexternal),collrem_xi,collrem_lxi
      double precision xi_i_fks,y_ij_fks

      double complex wgt1(2)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision delta_used
      common /cdelta_used/delta_used

      double precision rwgt,shattmp,dot,born_wgt,oo2pi,z,t,ap,
     # apprime,xkkern,xnorm
      external dot

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type
      
      double precision one,pi
      parameter (one=1.d0)
      parameter (pi=3.1415926535897932385d0)
      double precision iden_comp
      common /c_iden_comp/iden_comp

      if(j_fks.gt.nincoming)then
c Do not include this contribution for final-state branchings
         collrem_xi=0.d0
         collrem_lxi=0.d0
         wgtdegrem_xi=0.d0
         wgtdegrem_lxi=0.d0
         wgtdegrem_muF=0.d0
         return
      endif

      if(p_born(0,1).le.0.d0)then
c Unphysical kinematics: set matrix elements equal to zero
         write (*,*) "No born momenta in sreal_deg"
         collrem_xi=0.d0
         collrem_lxi=0.d0
         wgtdegrem_xi=0.d0
         wgtdegrem_lxi=0.d0
         wgtdegrem_muF=0.d0
         return
      endif

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
      if (nincoming.eq.2) then
         shattmp=2d0*dot(p(0,1),p(0,2))
      else
         shattmp=p(0,1)**2
      endif
      if(abs(shattmp/shat-1.d0).gt.1.d-5)then
        write(*,*)'Error in sreal: inconsistent shat'
        write(*,*)shattmp,shat
        stop
      endif

      call sborn(p_born,wgt1)
      born_wgt=dble(wgt1(1))

c A factor gS^2 is included in the Altarelli-Parisi kernels
      oo2pi=one/(8d0*PI**2)

      z = 1d0 - xi_i_fks
      t = one
      call AP_reduced(m_type,i_type,t,z,ap)
      call AP_reduced_prime(m_type,i_type,t,z,apprime)

c Insert here proper functions for PDF change of scheme. With xkkern=0.d0
c one assumes MSbar
      xkkern=0.d0

      collrem_xi=ap*log(shat*delta_used/(2*q2fact(j_fks))) -
     #           apprime - xkkern 
      collrem_lxi=2*ap

c The partonic flux 1/(2*s) is inserted in genps. Thus, an extra 
c factor z (implicit in the flux of the reduced Born in FKS) 
c has to be inserted here
      xnorm=1.d0/z *iden_comp

      collrem_xi=oo2pi * born_wgt * collrem_xi * xnorm
      collrem_lxi=oo2pi * born_wgt * collrem_lxi * xnorm

      wgtdegrem_xi=ap*log(shat*delta_used/(2*QES2)) -
     #               apprime - xkkern 
      wgtdegrem_xi=oo2pi * born_wgt * wgtdegrem_xi * xnorm
      wgtdegrem_lxi=collrem_lxi
      wgtdegrem_muF= - oo2pi * born_wgt * ap * xnorm

      return
      end


      subroutine set_cms_stuff(icountevts)
      implicit none
      include "run.inc"

      integer icountevts

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision sqrtshat_ev,shat_ev
      common/parton_cms_ev/sqrtshat_ev,shat_ev

      double precision sqrtshat_cnt(-2:2),shat_cnt(-2:2)
      common/parton_cms_cnt/sqrtshat_cnt,shat_cnt

      double precision tau_ev,ycm_ev
      common/cbjrk12_ev/tau_ev,ycm_ev

      double precision tau_cnt(-2:2),ycm_cnt(-2:2)
      common/cbjrk12_cnt/tau_cnt,ycm_cnt

      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt

c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to lab frame --
c same for event and counterevents
c This is the rapidity that enters in the arguments of the sinh() and
c cosh() of the boost, in such a way that
c       y(k)_lab = y(k)_tilde - ybst_til_tolab
c where y(k)_lab and y(k)_tilde are the rapidities computed with a generic
c four-momentum k, in the lab frame and in the \tilde{k}_1+\tilde{k}_2 
c c.m. frame respectively
      ybst_til_tolab=-ycm_cnt(0)
      if(icountevts.eq.-100)then
c set Bjorken x's in run.inc for the computation of PDFs in auto_dsig
        xbk(1)=xbjrk_ev(1)
        xbk(2)=xbjrk_ev(2)
c shat=2*k1.k2 -- consistency of this assignment with momenta checked
c in phspncheck_nocms
        shat=shat_ev
        sqrtshat=sqrtshat_ev
c rapidity of boost from \tilde{k}_1+\tilde{k}_2 c.m. frame to 
c k_1+k_2 c.m. frame
        ybst_til_tocm=ycm_ev-ycm_cnt(0)
      else
c do the same as above for the counterevents
        xbk(1)=xbjrk_cnt(1,icountevts)
        xbk(2)=xbjrk_cnt(2,icountevts)
        shat=shat_cnt(icountevts)
        sqrtshat=sqrtshat_cnt(icountevts)
        ybst_til_tocm=ycm_cnt(icountevts)-ycm_cnt(0)
      endif
      return
      end

      subroutine get_mc_lum(j_fks,zhw_used,xi_i_fks,xlum_mc_fact)
      implicit none
      include "run.inc"
      include "nexternal.inc"
      integer j_fks
      double precision dlum
      external dlum
      double precision zhw_used,xi_i_fks,xlum_mc_fact
      double precision xbjrk_ev(2),xbjrk_cnt(2,-2:2)
      common/cbjorkenx/xbjrk_ev,xbjrk_cnt
      if(zhw_used.lt.0.d0.or.zhw_used.gt.1.d0)then
        write(*,*)'Error #1 in get_mc_lum',zhw_used
        stop
      endif
      if(j_fks.gt.nincoming)then
        xbk(1)=xbjrk_cnt(1,0)
        xbk(2)=xbjrk_cnt(2,0)
        xlum_mc_fact=1.d0
      elseif(j_fks.eq.1)then
        xbk(1)=xbjrk_cnt(1,0)/zhw_used
        xbk(2)=xbjrk_cnt(2,0)
c Note that this is true for Pythia since, due to event projection and to
c the definition of the shower variable x = zhw_used, the Bjorken x's for
c the event (to be used in H events) are the ones for the counterevent
c multiplied by 1/x (by 1) for the emitting (non emitting) leg 
        if(xbk(1).gt.1.d0)then
          xlum_mc_fact = 0.d0
        else
          xlum_mc_fact = (1-xi_i_fks)/zhw_used
        endif
      elseif(j_fks.eq.2)then
        xbk(1)=xbjrk_cnt(1,0)
        xbk(2)=xbjrk_cnt(2,0)/zhw_used
        if(xbk(2).gt.1.d0)then
          xlum_mc_fact = 0.d0
        else
          xlum_mc_fact = (1-xi_i_fks)/zhw_used
        endif
      else
        write(*,*)'Error in get_mc_lum: unknown j_fks',j_fks
        stop
      endif
      if( xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #    ( (xbk(1).gt.1.d0.or.xbk(2).gt.1.d0).and.
     #      j_fks.gt.nincoming ) .or.
     #    (xbk(2).gt.1.d0.and.j_fks.eq.1) .or.
     #    (xbk(1).gt.1.d0.and.j_fks.eq.2) )then
        write(*,*)'Error in get_mc_lum: x_i',xbk(1),xbk(2)
        stop
      endif
      return
      end


      subroutine xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,
     #                        p_i_fks_ev,p_i_fks_cnt,
     #                        xi_i_fks_ev,y_ij_fks_ev,pass)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer i_fks,j_fks
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision jac,jac_cnt(-2:2)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      double precision xi_i_fks_ev,y_ij_fks_ev
      integer izero,ione,itwo,iunit,isum
      logical verbose,pass,pass0
      parameter (izero=0)
      parameter (ione=1)
      parameter (itwo=2)
      parameter (iunit=6)
      parameter (verbose=.false.)
      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax
c
      isum=0
      if(jac_cnt(0).gt.0.d0)isum=isum+1
      if(jac_cnt(1).gt.0.d0)isum=isum+2
      if(jac_cnt(2).gt.0.d0)isum=isum+4
      pass=.true.
c
      if(isum.eq.0.or.isum.eq.1.or.isum.eq.2.or.isum.eq.4)then
c Nothing to be done: 0 or 1 configurations computed
        if(verbose)write(iunit,*)'none'
      elseif(isum.eq.3.or.isum.eq.5.or.isum.eq.7)then
c Soft is taken as reference
        if(isum.eq.7)then
          if(verbose)then
            write(iunit,*)'all'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,pass0,ione,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
          if(verbose)then
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,pass0,itwo,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        elseif(isum.eq.3)then
          if(verbose)then
            write(iunit,*)'C+S'
            write(iunit,*)'    '
            write(iunit,*)'C/S'
          endif
          call xmcompare(verbose,pass0,ione,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        elseif(isum.eq.5)then
          if(verbose)then
            write(iunit,*)'SC+S'
            write(iunit,*)'    '
            write(iunit,*)'SC/S'
          endif
          call xmcompare(verbose,pass0,itwo,izero,i_fks,j_fks,p,p1_cnt)
          pass=pass.and.pass0
        endif
      elseif(isum.eq.6)then
c Collinear is taken as reference
        if(verbose)then
          write(iunit,*)'SC+C'
          write(iunit,*)'    '
          write(iunit,*)'SC/C'
        endif
        call xmcompare(verbose,pass0,itwo,ione,i_fks,j_fks,p,p1_cnt)
        pass=pass.and.pass0
      else
        write(6,*)'Fatal error in xmom_compare',isum
        stop
      endif
      if(.not.pass)i_momcmp_count=i_momcmp_count +1
c
      if(jac_cnt(0).gt.0.d0.and.jac.gt.0.d0)
     #  call p_ev_vs_cnt(izero,i_fks,j_fks,p,p1_cnt,
     #                   p_i_fks_ev,p_i_fks_cnt,
     #                   xi_i_fks_ev,y_ij_fks_ev)
      if(jac_cnt(1).gt.0.d0.and.jac.gt.0.d0)
     #  call p_ev_vs_cnt(ione,i_fks,j_fks,p,p1_cnt,
     #                   p_i_fks_ev,p_i_fks_cnt,
     #                   xi_i_fks_ev,y_ij_fks_ev)
c
      return
      end


      subroutine xmcompare(verbose,pass0,inum,iden,i_fks,j_fks,p,p1_cnt)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'coupl.inc'
      logical verbose,pass0
      integer inum,iden,i_fks,j_fks,iunit,ipart,i,j,k
      double precision tiny,vtiny,xnum,xden,xrat
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      parameter (iunit=6)
      parameter (tiny=1.d-4)
      parameter (vtiny=1.d-10)
      double precision pmass(nexternal),zero
      parameter (zero=0d0)
      integer i_momcmp_count
      double precision xratmax
      common/ccheckcnt/i_momcmp_count,xratmax
      include "pmass.inc"
c
      pass0=.true.
      do ipart=1,nexternal
        do i=0,3
          xnum=p1_cnt(i,ipart,inum)
          xden=p1_cnt(i,ipart,iden)
          if(verbose)then
            if(i.eq.0)then
              write(iunit,*)' '
              write(iunit,*)'part=',ipart
            endif
            call xprintout(iunit,xnum,xden)
          else
            if(ipart.ne.i_fks.and.ipart.ne.j_fks)then
              if(xden.ne.0.d0)then
                xrat=abs(1-xnum/xden)
              else
                xrat=abs(xnum)
              endif
              if(abs(xnum).eq.0d0.and.abs(xden).le.vtiny)xrat=0d0
c The following line solves some problem as well, but before putting
c it as the standard, one should think a bit about it
              if(abs(xnum).le.vtiny.and.abs(xden).le.vtiny)xrat=0d0
              if(xrat.gt.tiny .and.
     &          (pmass(ipart).eq.0d0.or.xnum/pmass(ipart).gt.vtiny))then
                 write(*,*)'Kinematics of counterevents'
                 write(*,*)inum,iden
                 write(*,*)'is different. Particle:',ipart
                 write(*,*) xrat,xnum,xden
                 do j=1,nexternal
                    write(*,*) j,(p1_cnt(k,j,inum),k=0,3)
                 enddo
                 do j=1,nexternal
                    write(*,*) j,(p1_cnt(k,j,iden),k=0,3)
                 enddo
                 xratmax=max(xratmax,xrat)
                 pass0=.false.
              endif
            endif
          endif
        enddo
      enddo
      do i=0,3
        if(j_fks.gt.nincoming)then
          xnum=p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        else
          xnum=-p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
          xden=-p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        endif
        if(verbose)then
          if(i.eq.0)then
            write(iunit,*)' '
            write(iunit,*)'part=i+j'
          endif
          call xprintout(iunit,xnum,xden)
        else
          if(xden.ne.0.d0)then
            xrat=abs(1-xnum/xden)
          else
            xrat=abs(xnum)
          endif
          if(xrat.gt.tiny)then
            write(*,*)'Kinematics of counterevents'
            write(*,*)inum,iden
            write(*,*)'is different. Particle i+j'
            xratmax=max(xratmax,xrat)
            pass0=.false.
          endif
        endif
      enddo
      return
      end


      subroutine xmcompare_fsr(verbose,inum,iden,i_fks,j_fks,p,p1_cnt)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      logical verbose
      integer inum,iden,i_fks,j_fks,iunit,ipart,i
      double precision tiny,xnum,xden,xrat
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      parameter (iunit=6)
      parameter (tiny=1.d-4)
c
      do ipart=1,nexternal
        do i=0,3
          xnum=p1_cnt(i,ipart,inum)
          xden=p1_cnt(i,ipart,iden)
          if(verbose)then
            if(i.eq.0)then
              write(iunit,*)' '
              write(iunit,*)'part=',ipart
            endif
            call xprintout(iunit,xnum,xden)
          else
            if(ipart.ne.i_fks.and.ipart.ne.j_fks)then
              if(xden.ne.0.d0)then
                xrat=abs(1-xnum/xden)
              else
                xrat=abs(xnum)
              endif
              if(xrat.gt.tiny)then
                write(*,*)'Kinematics of counterevents'
                write(*,*)inum,iden
                write(*,*)'is different. Particle:',ipart
                stop
              endif
            endif
          endif
        enddo
      enddo
      do i=0,3
        xnum=p1_cnt(i,i_fks,inum)+p1_cnt(i,j_fks,inum)
        xden=p1_cnt(i,i_fks,iden)+p1_cnt(i,j_fks,iden)
        if(verbose)then
          if(i.eq.0)then
            write(iunit,*)' '
            write(iunit,*)'part=i+j'
          endif
          call xprintout(iunit,xnum,xden)
        else
          if(xden.ne.0.d0)then
            xrat=abs(1-xnum/xden)
          else
            xrat=abs(xnum)
          endif
          if(xrat.gt.tiny)then
            write(*,*)'Kinematics of counterevents'
            write(*,*)inum,iden
            write(*,*)'is different. Particle i+j'
            stop
          endif
        endif
      enddo
      return
      end


      subroutine xprintout(iunit,xv,xlim)
      implicit real*8(a-h,o-z)
c
      if(abs(xlim).gt.1.d-30)then
        write(iunit,*)xv/xlim,xv,xlim
      else
        write(iunit,*)xv,xlim
      endif
      return
      end


      subroutine p_ev_vs_cnt(icnt,i_fks,j_fks,p,p1_cnt,
     #                       p_i_fks_ev,p_i_fks_cnt,
     #                       xi_i_fks_ev,y_ij_fks_ev)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer icnt,i_fks,j_fks,ipart,i
      double precision p(0:3,-max_branch:max_particles)
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      double precision xi_i_fks_ev,y_ij_fks_ev,tiny
      double precision rat(0:3,nexternal+3),den(0:3,nexternal+3)
      integer maxrat
c
c This routine is obsolete; the convergence checks are done elsewhere
      return

      do ipart=1,nexternal
        do i=0,3
          den(i,ipart)=p1_cnt(i,ipart,icnt)
          if(den(i,ipart).ne.0.d0)then
            rat(i,ipart)=p(i,ipart)/den(i,ipart)
          else
            rat(i,ipart)=p(i,ipart)
          endif
        enddo
      enddo
c
      do i=0,3
        den(i,nexternal+1)=p1_cnt(i,i_fks,icnt)+p1_cnt(i,j_fks,icnt)
        if(den(i,nexternal+1).ne.0.d0)then
          rat(i,nexternal+1)=(p(i,i_fks)+p(i,j_fks))/den(i,nexternal+1)
        else
          rat(i,nexternal+1)=p(i,i_fks)+p(i,j_fks)
        endif
      enddo
c
      if(icnt.eq.0)then
        tiny=4*xi_i_fks_ev
        maxrat=nexternal+3
        do i=0,3
          den(i,nexternal+2)=p_i_fks_cnt(i,0)
          if(den(i,nexternal+2).ne.0.d0)then
            rat(i,nexternal+2)=p_i_fks_ev(i)/den(i,nexternal+2)
          else
            rat(i,nexternal+2)=p_i_fks_ev(i)
          endif
        enddo
        do i=0,3
          den(i,nexternal+3)=p_i_fks_cnt(i,0)
          if(den(i,nexternal+3).ne.0.d0)then
            rat(i,nexternal+3)=p(i,i_fks)/den(i,nexternal+3)
          else
            rat(i,nexternal+3)=p(i,i_fks)
          endif
        enddo
      else
        tiny=2*sqrt(1-y_ij_fks_ev)
        maxrat=nexternal+1
      endif
c
      return
      end


c The following has been derived with minor modifications from the
c analogous routine written for VBF
      subroutine checkres(xsecvc,xseclvc,wgt,wgtl,xp,lxp,
     #                    iflag,imax,iev,nexternal,i_fks,j_fks,iret)
c Checks that the sequence xsecvc(i), i=1,imax, converges to xseclvc.
c Due to numerical inaccuracies, the test is deemed OK if there are
c at least ithrs+1 consecutive elements in the sequence xsecvc(i)
c which are closer to xseclvc than the preceding element of the sequence.
c The counting is started when an xsecvc(i0) is encountered, which is
c such that |xsecvc(i0)/xseclvc-1|<0.1 if xseclvc#0, or such that
c |xsecvc(i0)|<0.1 if xseclvc=0. In order for xsecvc(i+1 )to be defined 
c closer to xseclvc than xsecvc(i), the condition
c   |xsecvc(i)/xseclvc-1|/|xsecvc(i+1)/xseclvc-1| > rat
c if xseclvc#0, or 
c   |xsecvc(i)|/|xsecvc(i+1)| > rat
c if xseclvc=0 must be fulfilled; the value of rat is set equal to 4 and to 2
c for soft and collinear limits respectively, since the cross section is 
c expected to scale as xii**2 and sqrt(1-yi**2), and the values of xii and yi
c are chosen as powers of 10 (thus, if scaling would be exact, rat should
c be set equal to 10 and sqrt(10)).
c If the test is passed, icount=ithrs, else icount<ithrs; in the former
c case iret=0, in the latter iret=1.
c When the test is not passed, one may choose to stop the program dead here;
c in such a case, set istop=1 below. Each time the test is not passed,
c the results are written onto fort.77; set iwrite=0 to prevent the writing
      implicit none
      real*8 xsecvc(15),xseclvc,wgt(15),wgtl,lxp(0:3,21),xp(15,0:3,21)
      real*8 ckc(15),rckc(15),rat
      integer iflag,imax,iev,nexternal,i_fks,j_fks,iret,ithrs,istop,
     # iwrite,i,k,l,imin,icount
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checkres: imax is too large',imax
        stop
      endif
      do i=1,imax
        if(xseclvc.eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=4.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checkres: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      do while(ckc(i).gt.0.1d0 .and. xseclvc.ne.0d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).ne.0.d0)then
          rckc(i)=ckc(i)/ckc(i+1)
        else
          rckc(i)=1.d8
        endif
      enddo
      icount=0
      i=imin
      do while(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      iret=0
      if(icount.ne.ithrs)then
        iret=1
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
        if(iwrite.eq.1)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'ME*wgt:'
          do i=1,imax
             call xprintout(77,xsecvc(i),xseclvc)
          enddo
          write(77,*)'wgt:'
          do i=1,imax
             call xprintout(77,wgt(i),wgtl)
          enddo
c
          write(78,*)'    '
          if(iflag.eq.0)then
            write(78,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(78,*)'Collinear #',iev
          endif
          do k=1,nexternal
            write(78,*)''
            write(78,*)'part:',k
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,k),lxp(l,k))
              enddo
            enddo
          enddo
          if(iflag.eq.0)then
            write(78,*)''
            write(78,*)'part: i_fks reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,nexternal+1),
     #                            lxp(l,nexternal+1))
              enddo
            enddo
            write(78,*)''
            write(78,*)'part: i_fks full/reduced'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks),
     #                            xp(i,l,nexternal+1))
              enddo
            enddo
          elseif(iflag.eq.1)then
            write(78,*)''
            write(78,*)'part: i_fks+j_fks'
            do l=0,3
              write(78,*)'comp:',l
              do i=1,imax
                call xprintout(78,xp(i,l,i_fks)+xp(i,l,j_fks),
     #                            lxp(l,i_fks)+lxp(l,j_fks))
              enddo
            enddo
          endif
        endif
      endif
      return
      end




      subroutine checksij(xsijvc,xsijlvc,xsijlim,
     #                    xsumvc,xsumlvc,xsumlim,
     #                    check,checkl,tolerance,
     #                    iflag,imax,iev,ki,kk,ll,
     #                    i_fks,j_fks,ilim,iret)
c Analogous to checkres. Relevant to S functions
      implicit none
      real*8 xsijvc(15),xsijlvc,xsumvc(15),xsumlvc,check(15),checkl
      real*8 xsijlim,xsumlim,tolerance
      real*8 xsecvc(15),xseclvc
      real*8 ckc(15),rckc(15),rat
      logical found
      integer iflag,imax,iev,ki,kk,ll,i_fks,j_fks,ilim,iret,ithrs,
     # istop,iwrite,i,imin,icount,itype
      parameter (ithrs=3)
      parameter (istop=0)
      parameter (iwrite=1)
c
      if(imax.gt.15)then
        write(6,*)'Error in checksij: imax is too large',imax
        stop
      endif
      itype=1
      iret=0
 100  continue
      if(itype.eq.1)then
        do i=1,imax
          xsecvc(i)=xsijvc(i)
        enddo
        xseclvc=xsijlvc
      elseif(itype.eq.2)then
        do i=1,imax
          xsecvc(i)=xsumvc(i)
        enddo
        xseclvc=xsumlvc
      else
        write(6,*)'Error in checksij: itype=',itype
        stop
      endif
      do i=1,imax
        if(xseclvc.eq.0.d0)then
          ckc(i)=abs(xsecvc(i))
        else
          ckc(i)=abs(xsecvc(i)/xseclvc-1.d0)
        endif
      enddo
      if(iflag.eq.0)then
        rat=8.d0
      elseif(iflag.eq.1)then
        rat=2.d0
      else
        write(6,*)'Error in checksij: iflag=',iflag
        write(6,*)' Must be 0 for soft, 1 for collinear'
        stop
      endif
c
      i=1
      dowhile(ckc(i).gt.0.1d0)
        i=i+1
      enddo
      imin=i
      do i=imin,imax-1
        if(ckc(i+1).gt.1.d-8)then
c If this condition is replaced by .eq.0, the test will fail if the series
c is made of elements all equal to the limit
          rckc(i)=ckc(i)/ckc(i+1)
        else
c Element #i+1 of series equal to the limit, so it must pass the test
          rckc(i)=rat*1.1d0
        endif
      enddo
      icount=0
      i=imin
      dowhile(icount.lt.ithrs.and.i.lt.imax)
        if(rckc(i).gt.rat)then
          icount=icount+1
        else
          icount=0
        endif
        i=i+1
      enddo
c
      if(icount.ne.ithrs)then
        iret=iret+itype
        if(istop.eq.1)then
          write(6,*)'Test failed',iflag
          write(6,*)'Event #',iev
          stop
        endif
      endif
      if(itype.eq.1.and.ki.eq.1.and.iflag.eq.0)then
        itype=2
        goto 100
      endif
c
      if(ki.eq.1.and.ilim.eq.1)then
        found=.false.
        i=0
        do while ((.not.found).and.i.lt.imax)
          i=i+1
          if(abs(check(i)-1.d0).gt.tolerance)then
            found=.true.
            itype=4
          endif
        enddo
        if(.not.found)then
          if(abs(checkl-1.d0).gt.tolerance)itype=4
        endif
        if(itype.eq.4)iret=iret+itype
      endif
c
      if( iwrite.eq.1 .and.
     #    iret.eq.1 .or.(iret.gt.1.and.ki.eq.1) )then
        if(iret.gt.7)then
          write(6,*)'Error in checksij: iret=',iret
          stop
        endif
        write(77,*)'    '
        if(iflag.eq.0)then
          write(77,*)'Soft #',iev
        elseif(iflag.eq.1)then
          write(77,*)'Collinear #',iev
        endif
        write(77,*)'iret:',iret
        write(77,*)'i_fks,j_fks:',i_fks,j_fks
        if(iret.eq.1.or.iret.eq.3.or.iret.eq.5.or.iret.eq.7)then
          write(77,*)'S_kl'
          write(77,*)'k,kk,ll',ki,kk,ll
          do i=1,imax
             call xprintout(77,xsijvc(i),xsijlvc)
          enddo
        endif
        if(iret.eq.2.or.iret.eq.3.or.iret.eq.6.or.iret.eq.7)then
          write(77,*)'sum of S'
          do i=1,imax
             call xprintout(77,xsumvc(i),xsumlvc)
          enddo
        endif
        if(iret.eq.4.or.iret.eq.5.or.iret.eq.6.or.iret.eq.7)then
          write(77,*)'check to one'
          do i=1,imax
             call xprintout(77,check(i),checkl)
          enddo
        endif
      endif
c
      if(ilim.eq.1)then
        if( abs(xsijlvc-xsijlim).gt.1.d-6 .and. 
     #    xsijlim.ne.-1.d0 )iret=iret+10
        if( abs(xsumlvc-xsumlim).gt.1.d-6 .and.
     #    xsumlim.ne.-1.d0 .and. iflag.eq.0)iret=iret+20
        if(iwrite.eq.1.and.iret.ge.10)then
          write(77,*)'    '
          if(iflag.eq.0)then
            write(77,*)'Soft #',iev
          elseif(iflag.eq.1)then
            write(77,*)'Collinear #',iev
          endif
          write(77,*)'iret:',iret
          write(77,*)'i_fks,j_fks:',i_fks,j_fks
          if((iret.ge.10.and.iret.lt.20).or.iret.ge.30)then
            write(77,*)'limit of S_kl'
            write(77,*)'k,kk,ll',ki,kk,ll
            write(77,*)xsijlvc,xsijlim
          endif
          if(iret.ge.20)then
            write(77,*)'limit of sum_j S_ij'
            write(77,*)xsumlvc,xsumlim
          endif
        endif
      endif
      return
      end


      subroutine bornsoftvirtual(p,bsv_wgt,virt_wgt,born_wgt)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
      include "coupl.inc"
      include 'q_es.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include "run.inc"
      include "fks_powers.inc"
      include 'reweight.inc'
      double precision p(0:3,nexternal),bsv_wgt,born_wgt,avv_wgt
      double precision pp(0:3,nexternal)
      
      double complex wgt1(2)
      double precision rwgt,ao2pi,Q,Ej,wgt,contr,eikIreg,m1l_W_finite_CDR
      double precision shattmp,dot
      integer i,j,aj,m,n,k

      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision xicut_used
      common /cxicut_used/xicut_used

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      double precision pi
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision double,single,xmu2
      logical ComputePoles,fksprefact
      parameter (ComputePoles=.false.)
      parameter (fksprefact=.true.)

      double precision beta0,ren_group_coeff
      common/cbeta0/beta0,ren_group_coeff

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      double precision virt_fraction_inc
      data virt_fraction_inc /1d0/

      integer include_virt
      double precision vol3

c For tests of virtuals
      double precision xnormsv
      common/cxnormsv/xnormsv
      double precision vrat

      double precision virt_wgt,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv

      logical ExceptPSpoint
      integer iminmax
      common/cExceptPSpoint/iminmax,ExceptPSpoint

      double precision average_virtual,virtual_fraction
      common/c_avg_virt/average_virtual,virtual_fraction
      double precision virtual_over_born
      common/c_vob/virtual_over_born

c timing statistics
      include "timing_variables.inc"

c For the MINT folding
      integer fold
      common /cfl/fold
      double precision virt_wgt_save
      save virt_wgt_save

      double precision pmass(nexternal),zero,tiny
      parameter (zero=0d0)
      parameter (tiny=1d-6)
      include "pmass.inc"

      ao2pi=g**2/(8d0*PI**2)

      if (particle_type(i_fks).eq.8 .or. abrv.eq.'grid') then

c Consistency check -- call to set_cms_stuff() must be done prior to
c entering this function
         if (nincoming.eq.2) then
            shattmp=2d0*dot(p(0,1),p(0,2))
         else
            shattmp=p(0,1)**2
         endif
         if(abs(shattmp/shat-1.d0).gt.1.d-5)then
           write(*,*)'Error in sreal: inconsistent shat'
           write(*,*)shattmp,shat
           stop
         endif

         call sborn(p_born,wgt1)

c Born contribution:
         bsv_wgt=dble(wgt1(1))
         born_wgt=dble(wgt1(1))
         virt_wgt=0d0
         avv_wgt=0d0 

         if (abrv.eq.'born' .or. abrv.eq.'grid') goto 549
         if (abrv.eq.'virt' .or. abrv.eq.'viSC' .or.
     #       abrv.eq.'viLC') goto 547

c Q contribution eq 5.5 and 5.6 of FKS
         Q=0d0
         do i=nincoming+1,nexternal
            if (i.ne.i_fks .and. particle_type(i).ne.1 .and. 
     #          pmass(i).eq.ZERO)then
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               endif
               Ej=p(0,i)
               if(abrv.eq.'novA')then
c 2+3+4
                  Q = Q
     &             -2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &             -( dlog(deltaO/2d0)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &               +2*dlog(xicut_used)**2*c(aj) )
     &             +gammap(aj)
     &             +2d0*c(aj)*dlog(2d0*Ej/sqrtshat)**2
     &             -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               elseif(abrv.eq.'novB')then
c 2+3+4_mu
                  Q = Q
     &             -2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &             -( dlog(deltaO/2d0)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &               +2*dlog(xicut_used)**2*c(aj) )
               elseif(abrv.eq.'viSA')then
c 1                
                  Q = Q
     &              -dlog(shat/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/sqrtshat) )
               elseif(abrv.eq.'viSB')then
c 1+4_L
                  Q = Q
     &              -dlog(shat/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/sqrtshat) )
     &             +gammap(aj)
     &             +2d0*c(aj)*dlog(2d0*Ej/sqrtshat)**2
     &             -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #                abrv.ne.'viLC')then
c 1+2+3+4
                  Q = Q+gammap(aj)
     &              -dlog(shat*deltaO/2d0/QES2)*( gamma(aj)-
     &                      2d0*c(aj)*dlog(2d0*Ej/xicut_used/sqrtshat) )
     &              +2d0*c(aj)*( dlog(2d0*Ej/sqrtshat)**2
     &              -dlog(xicut_used)**2 )
     &              -2d0*gamma(aj)*dlog(2d0*Ej/sqrtshat)
               else
                  write(*,*)'Error in bornsoftvirtual'
                  write(*,*)'abrv in Q:',abrv
                  stop
               endif
            endif
         enddo
c
         do i=1,nincoming
            if (particle_type(i).ne.1 .and. pmass(i).eq.ZERO) then
               if (particle_type(i).eq.8) then
                  aj=0
               elseif(abs(particle_type(i)).eq.3) then
                  aj=1
               endif
               if(abrv.eq.'novA'.or.abrv.eq.'novB')then
c 2+3+4 or 2+3+4_mu
                  Q=Q-2*dlog(shat/QES2)*dlog(xicut_used)*c(aj)
     &               -dlog(q2fact(i)/shat)*(
     &                  gamma(aj)+2d0*c(aj)*dlog(xicut_used) )
               elseif(abrv.eq.'viSA'.or.abrv.eq.'viSB')then
c 1 or 1+4_L
                  Q=Q-dlog(shat/QES2)*gamma(aj)
               elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #                abrv.ne.'viLC')then
c 1+2+3+4
                  Q=Q-dlog(q2fact(i)/QES2)*(
     &                 gamma(aj)+2d0*c(aj)*dlog(xicut_used))
               else
                  write(*,*)'Error in bornsoftvirtual'
                  write(*,*)'abrv in Q:',abrv
                  stop
               endif
            endif
         enddo

         bsv_wgt=bsv_wgt+ao2pi*Q*dble(wgt1(1))

c        If doing MC over helicities, must sum over the two
c        helicity contributions for the Q-terms of collinear limit.
 547     continue
         if (abrv.eq.'virt' .or. abrv.eq.'viSC' .or.
     #       abrv.eq.'viLC') goto 548
c
c I(reg) terms, eq 5.5 of FKS
         contr=0d0
         do i=1,fks_j_from_i(i_fks,0)
            do j=1,i
               m=fks_j_from_i(i_fks,i)
               n=fks_j_from_i(i_fks,j)
               if ((m.ne.n .or. (m.eq.n .and. pmass(m).ne.ZERO)).and.
     &              n.ne.i_fks.and.m.ne.i_fks) then
c To be sure that color-correlated Borns work well, we need to have
c *always* a call to sborn(p_born,wgt) just before. This is okay,
c because there is a call above in this subroutine
                  call sborn_sf(p_born,m,n,wgt)
                  if (wgt.ne.0d0) then
                     call eikonal_Ireg(p,m,n,xicut_used,eikIreg)
                     contr=contr+wgt*eikIreg
                  endif
               endif
            enddo
         enddo

C WARNING: THE FACTOR -2 BELOW COMPENSATES FOR THE MISSING -2 IN THE
C COLOUR LINKED BORN -- SEE ALSO SBORNSOFT().
C If the colour-linked Borns were normalized as reported in the paper
c we should set
c   bsv_wgt=bsv_wgt+ao2pi*contr  <-- DO NOT USE THIS LINE
c
         bsv_wgt=bsv_wgt-2*ao2pi*contr

 548     continue
c Finite part of one-loop corrections
c convert to Binoth Les Houches Accord standards
         virt_wgt=0d0
         if (fold.eq.0) then
            if ((ran2().le.virtual_fraction .and.
     $           abrv(1:3).ne.'nov').or.abrv(1:4).eq.'virt') then
               call cpu_time(tBefore)
               Call BinothLHA(p_born,born_wgt,virt_wgt)
c$$$               virt_wgt=m1l_W_finite_CDR(p_born,born_wgt)
               call cpu_time(tAfter)
               tOLP=tOLP+(tAfter-tBefore)
               virtual_over_born=virt_wgt/(born_wgt*ao2pi)
               if (ickkw.ne.-1)
     &              virt_wgt=virt_wgt-average_virtual*born_wgt*ao2pi
               if (abrv.ne.'virt') then
                  virt_wgt=virt_wgt/virtual_fraction
               endif
               virt_wgt_save=virt_wgt
c$$$               bsv_wgt=bsv_wgt+virt_wgt_save
            endif
         elseif(fold.eq.1) then
            virt_wgt=virt_wgt_save
c$$$            bsv_wgt=bsv_wgt+virt_wgt_save
         endif
         if (abrv(1:4).ne.'virt' .and. ickkw.ne.-1)
     &        avv_wgt=average_virtual*born_wgt*ao2pi

c eq.(MadFKS.C.13)
         if(abrv.eq.'viSA'.or.abrv.eq.'viSB')then
           bsv_wgt=bsv_wgt + 2*pi*(beta0*wgtbpower
     #      +ren_group_coeff*wgtcpower)*log(shat/QES2)*ao2pi*dble(wgt1(1))
         elseif(abrv.eq.'novA'.or.abrv.eq.'novB')then
           bsv_wgt=bsv_wgt + 2*pi*(beta0*wgtbpower
     #      +ren_group_coeff*wgtcpower)*log(q2fact(1)/shat)*ao2pi*dble(wgt1(1))
         elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #          abrv.ne.'viLC')then
           bsv_wgt=bsv_wgt + 2*pi*(beta0*wgtbpower
     #      +ren_group_coeff*wgtcpower)*log(q2fact(1)/QES2)*ao2pi*dble(wgt1(1))
         endif
c eq.(MadFKS.C.14)
         if(abrv(1:2).ne.'vi')then
           bsv_wgt=bsv_wgt - 2*pi*(beta0*wgtbpower
     #      +ren_group_coeff*wgtcpower)*log(q2fact(1)/scale**2)*ao2pi*dble(wgt1(1))
         endif


 549     continue

         wgtwnstmpmuf=0.d0
         if(abrv.ne.'born' .and. abrv.ne.'grid')then
            if(abrv(1:2).eq.'vi')then
               wgtwnstmpmur=0.d0
            else
               do i=1,nincoming
                  if (particle_type(i).ne.1)then
                     if (particle_type(i).eq.8) then
                        aj=0
                     elseif(abs(particle_type(i)).eq.3) then
                        aj=1
                     endif
                     wgtwnstmpmuf=wgtwnstmpmuf-
     #                   ( gamma(aj)+2d0*c(aj)*dlog(xicut_used) )
                  endif
               enddo
               wgtwnstmpmuf=ao2pi*wgtwnstmpmuf*dble(wgt1(1))
               wgtwnstmpmur=2*pi*(beta0*wgtbpower
     #         +ren_group_coeff*wgtcpower)*ao2pi*dble(wgt1(1))
            endif
c bsv_wgt here always contains the Born; must subtract it, since 
c we need the pure NLO terms only
            wgtnstmp=bsv_wgt-born_wgt-
     #                wgtwnstmpmuf*log(q2fact(1)/QES2)-
     #                wgtwnstmpmur*log(scale**2/QES2)
            wgtnstmp_avgvirt = avv_wgt
         else
            wgtnstmp=0d0
            wgtwnstmpmur=0.d0
            wgtnstmp_avgvirt = 0d0
         endif

         if (abrv(1:2).eq.'vi') then
            bsv_wgt=bsv_wgt-born_wgt
            born_wgt=0d0
         endif

         if (ComputePoles) then
            call sborn(p_born,wgt1)
            born_wgt=dble(wgt1(1))

            print*,"           "
            write(*,123)((p(i,j),i=0,3),j=1,nexternal)
            xmu2=q2fact(1)
            call getpoles(p,xmu2,double,single,fksprefact)
            print*,"BORN",born_wgt!/conv
            print*,"DOUBLE",double/born_wgt/ao2pi
            print*,"SINGLE",single/born_wgt/ao2pi
c            print*,"LOOP",virt_wgt!/born_wgt/ao2pi*2d0
c            print*,"LOOP2",(virtcor+born_wgt*4d0/3d0-double*pi**2/6d0)
c            stop
 123        format(4(1x,d22.16))
         endif


      else
         bsv_wgt=0d0
         virt_wgt=0d0
         born_wgt=0d0
         wgtnstmp=0d0
         wgtwnstmpmuf=0d0
         wgtwnstmpmur=0d0
      endif

      return
      end


      subroutine compute_bpower(p_born,bpower)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"

      double precision p_born(0:3,nexternal-1)
      double precision bpower,born_wgt
      double complex wgt1(2)

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      integer isum_hel_orig
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      double precision tiny
      parameter (tiny=1d-6)

c Make sure that we sum over helicities (such that we do get a
c non-zero Born)
      isum_hel_orig = isum_hel
      isum_hel=0
      call get_helicity(i_fks,j_fks)

      calculatedBorn=.false.
      call sborn(p_born,wgt1)
c Born contribution:
      born_wgt=dble(wgt1(1))
      
c Multiply the strong coupling by 10
      if (g.ne.0d0) then
         g=10d0*g
      else
         write(*,*)'Error in bornsoftvirtual'
         write(*,*)'Strong coupling is zero'
         stop
      endif

c Update alphaS-dependent couplings
      call update_as_param()

c recompute the Born with the new couplings
      calculatedBorn=.false.
      call sborn(p_born,wgt1)

c Compute bpower
      bpower=Log10(dble(wgt1(1))/born_wgt)/2d0
      if(abs(bpower-dble(nint(bpower))) .gt. tiny) then
         write(*,*)'Error in computation of bpower:'
         write(*,*)' not an integer',bpower
         stop
      elseif (bpower.lt.-tiny) then
         write(*,*)'Error in computation of bpower:'
         write(*,*)' negative value',bpower
         stop
      else
c set it to the integer exactly
         bpower=dble(nint(bpower))
         write(*,*)'bpower is', bpower
      endif

c Change couplings back and recompute the Born to make sure that 
c nothing funny happens later on
      g=g/10d0
      call update_as_param()
      isum_hel=isum_hel_orig
      calculatedBorn=.false.
      call sborn(p_born,wgt1)

      return
      end

c       This function computes the power of a muR-dependent factor which
c       is stored in cpower. You need to modify it when you try to 
c       reweight your cross section with a muR-dependent factor
c       (runfac=1 in reweight0.inc)
c Note: The implementation below only works for the Bottom Yukawa in
c       the SM where "GC_33" contains the Yukawa, for other models
c       or general muR-dependent factors you need to change GC_33
c       to the corresponding coupling.
      subroutine compute_cpower(p_born,cpower)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include 'reweight.inc'

      double precision p_born(0:3,nexternal-1)
      double precision cpower,born_wgt
      double complex wgt1(2)

      integer isum_hel
      logical multi_channel
      common/to_matrix/isum_hel, multi_channel
      integer isum_hel_orig
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn

      double precision tiny
      parameter (tiny=1d-6)
c comment these lines to calculate cpower
      cpower = -1d0
      return
c comment these lines to calculate cpower

c   The following is relevant for a muR-dependent bottom-mass in Yukawa.
c$$$
c$$$c Make sure that we sum over helicities (such that we do get a
c$$$c non-zero Born)
c$$$      isum_hel_orig = isum_hel
c$$$      isum_hel=0
c$$$      call get_helicity(i_fks,j_fks)
c$$$
c$$$      calculatedBorn=.false.
c$$$      call sborn(p_born,wgt1)
c$$$c Born contribution:
c$$$      born_wgt=dble(wgt1(1))
c$$$      
c$$$c Multiply the Yukawa by 10 (If you use this,
c$$$c double check that GC_33 is the yukawa! (also below))
c$$$      if (GC_33.ne.0d0) then
c$$$         GC_33 = GC_33 * 10d0
c$$$      else
c$$$         write(*,*)'Warning In Bornsoftvirtual'
c$$$         Write(*,*)'Yukawa Is Zero - Cpower Set To Zero'
c$$$         Cpower = 0d0
c$$$         Return
c$$$      Endif
c$$$
c$$$c recompute the Born with the new Yukawa
c$$$      calculatedBorn=.false.
c$$$      call sborn(p_born,wgt1)
c$$$
c$$$c Compute cpower
c$$$      cpower=Log10(dble(wgt1(1))/born_wgt)
c$$$      if(abs(cpower-dble(nint(cpower))) .gt. tiny) then
c$$$         write(*,*)'Error in computation of cpower:'
c$$$         write(*,*)' not an integer',cpower
c$$$         stop
c$$$      elseif (cpower.lt.-tiny) then
c$$$         write(*,*)'Error in computation of cpower:'
c$$$         write(*,*)' negative value',cpower
c$$$         stop
c$$$      else
c$$$c set it to the integer exactly
c$$$         cpower=dble(nint(cpower))
c$$$         write(*,*)'cpower is', cpower
c$$$c Check consistency with value used in reweighting
c$$$c$$$         if( (doreweight.or.doNLOreweight) .and.
c$$$c$$$     &        abs(cpower-wgtcpower).gt.tiny )then
c$$$c$$$            write(*,*)'Error in compute_cpower'
c$$$c$$$            write(*,*)'cpower(s) are:',cpower,wgtcpower
c$$$c$$$            stop
c$$$c$$$         endif
c$$$      endif
c$$$
c$$$c Change couplings back and recompute the Born to make sure that 
c$$$c nothing funny happens later on
c$$$      GC_33 = GC_33 / 10d0
c$$$      isum_hel=isum_hel_orig
c$$$      calculatedBorn=.false.
c$$$      call sborn(p_born,wgt1)
c$$$
c$$$      return
      end


      subroutine eikonal_Ireg(p,m,n,xicut_used,eikIreg)
      implicit none
      double precision zero,pi,pi2
      parameter (zero=0.d0)
      parameter (pi=3.1415926535897932385d0)
      parameter (pi2=pi**2)
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal),xicut_used,eikIreg
      integer m,n

      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat

      character*4 abrv
      common /to_abrv/ abrv

      double precision Ei,Ej,kikj,rij,tmp,xmj,betaj,betai,xmi2,xmj2,
     # vij,xi0,alij,tHVvl,tHVv,arg1,arg2,arg3,arg4,xi1a,xj1a,
     # dot,ddilog
      external dot,ddilog

      double precision pmass(nexternal)
      include "pmass.inc"

      tmp=0.d0
      if(pmass(m).eq.0.d0.and.pmass(n).eq.0.d0)then
        if(m.eq.n)then
          write(*,*)'Error #2 in eikonal_Ireg',m,n
          stop
        endif
        Ei=p(0,n)
        Ej=p(0,m)
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)
        if(abs(rij-1.d0).gt.1.d-6)then
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2+
     #          2*dlog(xicut_used)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2+
     #          2*dlog(xicut_used)*dlog(rij)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1d0/2d0*dlog(shat/QES2)**2+
     #          dlog(shat/QES2)*dlog(rij)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1d0/2d0*dlog(shat/QES2)**2+
     #          dlog(shat/QES2)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2+
     #          dlog(xicut_used**2*shat/QES2)*dlog(rij)-
     #          ddilog(rij)+1d0/2d0*dlog(rij)**2-
     #          dlog(1-rij)*dlog(rij)
          else
             write(*,*)'Error #11 in eikonal_Ireg',abrv
             stop
          endif
        else
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2-pi2/6.d0
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(shat/QES2)*dlog(xicut_used)+
     #          2*dlog(xicut_used)**2
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1d0/2d0*dlog(shat/QES2)**2
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1d0/2d0*dlog(shat/QES2)**2-pi2/6.d0
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1d0/2d0*dlog(xicut_used**2*shat/QES2)**2-pi2/6.d0
          else
             write(*,*)'Error #12 in eikonal_Ireg',abrv
             stop
          endif
        endif
      elseif( (pmass(m).ne.0.d0.and.pmass(n).eq.0.d0) .or.
     #        (pmass(m).eq.0.d0.and.pmass(n).ne.0.d0) )then
        if(m.eq.n)then
          write(*,*)'Error #3 in eikonal_Ireg',m,n
          stop
        endif
        if(pmass(m).ne.0.d0.and.pmass(n).eq.0.d0)then
          Ei=p(0,n)
          Ej=p(0,m)
          xmj=pmass(m)
          betaj=sqrt(1-xmj**2/Ej**2)
        else
          Ei=p(0,m)
          Ej=p(0,n)
          xmj=pmass(n)
          betaj=sqrt(1-xmj**2/Ej**2)
        endif
        kikj=dot(p(0,n),p(0,m))
        rij=kikj/(2*Ei*Ej)

        if(abrv.eq.'novA')then
c 2+3+4
          tmp=dlog(xicut_used)*dlog(shat/QES2)+
     #        dlog(xicut_used)**2+
     #        2*dlog(xicut_used)*dlog(kikj/(xmj*Ei))-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2-pi2/12.d0-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        elseif(abrv.eq.'novB')then
c 2+3+4_mu
          tmp=dlog(xicut_used)*dlog(shat/QES2)+
     #        dlog(xicut_used)**2+
     #        2*dlog(xicut_used)*dlog(kikj/(xmj*Ei))
        elseif(abrv.eq.'viSA')then
c 1                
          tmp=1/4.d0*dlog(shat/QES2)**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))
        elseif(abrv.eq.'viSB')then
c 1+4_L
          tmp=1/4.d0*dlog(shat/QES2)**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2-pi2/12.d0-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #         abrv.ne.'viLC')then
c 1+2+3+4
          tmp=dlog(xicut_used)*( dlog(xicut_used*shat/QES2)+
     #                           2*dlog(kikj/(xmj*Ei)) )-
     #        ddilog(1-(1+betaj)/(2*rij))+ddilog(1-2*rij/(1-betaj))+
     #        1/2.d0*log(2*rij/(1-betaj))**2+
     #        dlog(shat/QES2)*dlog(kikj/(xmj*Ei))-pi2/12.d0+
     #        1/4.d0*dlog(shat/QES2)**2-
     #        1/4.d0*dlog((1+betaj)/(1-betaj))**2
        else
           write(*,*)'Error #13 in eikonal_Ireg',abrv
           stop
        endif
      elseif(pmass(m).ne.0.d0.and.pmass(n).ne.0.d0)then
        if(n.eq.m)then
          Ei=p(0,n)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          if(abrv.eq.'novA')then
c 2+3+4
            tmp=2*dlog(xicut_used)-
     #          1/betai*dlog((1+betai)/(1-betai))
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=2*dlog(xicut_used)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=dlog(shat/QES2)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=dlog(shat/QES2)-
     #          1/betai*dlog((1+betai)/(1-betai))
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            if (betai.gt.1d-6) then
               tmp=dlog(xicut_used**2*shat/QES2)-
     &              1/betai*dlog((1+betai)/(1-betai))
            else
               tmp=dlog(xicut_used**2*shat/QES2)-
     &              2d0*(1d0+betai**2/3d0+betai**4/5d0)
            endif
          else
             write(*,*)'Error #14 in eikonal_Ireg',abrv
             stop
          endif
        else
          Ei=p(0,n)
          Ej=p(0,m)
          betai=sqrt(1-pmass(n)**2/Ei**2)
          betaj=sqrt(1-pmass(m)**2/Ej**2)
          xmi2=pmass(n)**2
          xmj2=pmass(m)**2
          kikj=dot(p(0,n),p(0,m))
          vij=sqrt(1-xmi2*xmj2/kikj**2)
          alij=kikj*(1+vij)/xmi2
          tHVvl=(alij**2*xmi2-xmj2)/2.d0
          tHVv=tHVvl/(alij*Ei-Ej)
          arg1=alij*Ei
          arg2=arg1*betai
          arg3=Ej
          arg4=arg3*betaj
          xi0=1/vij*log((1+vij)/(1-vij))
          xi1a=kikj**2*(1+vij)/xmi2*( xj1a(arg1,arg2,tHVv,tHVvl)-
     #                                xj1a(arg3,arg4,tHVv,tHVvl) )

          if(abrv.eq.'novA')then
c 2+3+4
            tmp=xi0*dlog(xicut_used)+1/2.d0*xi1a
          elseif(abrv.eq.'novB')then
c 2+3+4_mu
            tmp=xi0*dlog(xicut_used)
          elseif(abrv.eq.'viSA')then
c 1                
            tmp=1/2.d0*xi0*dlog(shat/QES2)
          elseif(abrv.eq.'viSB')then
c 1+4_L
            tmp=1/2.d0*xi0*dlog(shat/QES2)+1/2.d0*xi1a
          elseif(abrv.ne.'virt' .and. abrv.ne.'viSC' .and.
     #           abrv.ne.'viLC')then
c 1+2+3+4
            tmp=1/2.d0*xi0*dlog(xicut_used**2*shat/QES2)+1/2.d0*xi1a
          else
             write(*,*)'Error #15 in eikonal_Ireg',abrv
             stop
          endif
        endif
      else
        write(*,*)'Error #4 in eikonal_Ireg',m,n,pmass(m),pmass(n)
        stop
      endif
      eikIreg=tmp
      return
      end


      function xj1a(x,y,tHVv,tHVvl)
      implicit none
      real*8 xj1a,x,y,tHVv,tHVvl,ddilog
      external ddilog
c
      xj1a=1/(2*tHVvl)*( dlog((x-y)/(x+y))**2+4*ddilog(1-(x+y)/tHVv)+
     #                   4*ddilog(1-(x-y)/tHVv) )
      return
      end


      FUNCTION DDILOG(X)
*
* $Id: imp64.inc,v 1.1.1.1 1996/04/01 15:02:59 mclareni Exp $
*
* $Log: imp64.inc,v $
* Revision 1.1.1.1  1996/04/01 15:02:59  mclareni
* Mathlib gen
*
*
* imp64.inc
*
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION C(0:19)
      PARAMETER (Z1 = 1, HF = Z1/2)
      PARAMETER (PI = 3.14159 26535 89793 24D0)
      PARAMETER (PI3 = PI**2/3, PI6 = PI**2/6, PI12 = PI**2/12)
      DATA C( 0) / 0.42996 69356 08136 97D0/
      DATA C( 1) / 0.40975 98753 30771 05D0/
      DATA C( 2) /-0.01858 84366 50145 92D0/
      DATA C( 3) / 0.00145 75108 40622 68D0/
      DATA C( 4) /-0.00014 30418 44423 40D0/
      DATA C( 5) / 0.00001 58841 55418 80D0/
      DATA C( 6) /-0.00000 19078 49593 87D0/
      DATA C( 7) / 0.00000 02419 51808 54D0/
      DATA C( 8) /-0.00000 00319 33412 74D0/
      DATA C( 9) / 0.00000 00043 45450 63D0/
      DATA C(10) /-0.00000 00006 05784 80D0/
      DATA C(11) / 0.00000 00000 86120 98D0/
      DATA C(12) /-0.00000 00000 12443 32D0/
      DATA C(13) / 0.00000 00000 01822 56D0/
      DATA C(14) /-0.00000 00000 00270 07D0/
      DATA C(15) / 0.00000 00000 00040 42D0/
      DATA C(16) /-0.00000 00000 00006 10D0/
      DATA C(17) / 0.00000 00000 00000 93D0/
      DATA C(18) /-0.00000 00000 00000 14D0/
      DATA C(19) /+0.00000 00000 00000 02D0/
      IF(X .EQ. 1) THEN
       H=PI6
      ELSEIF(X .EQ. -1) THEN
       H=-PI12
      ELSE
       T=-X
       IF(T .LE. -2) THEN
        Y=-1/(1+T)
        S=1
        A=-PI3+HF*(LOG(-T)**2-LOG(1+1/T)**2)
       ELSEIF(T .LT. -1) THEN
        Y=-1-T
        S=-1
        A=LOG(-T)
        A=-PI6+A*(A+LOG(1+1/T))
       ELSE IF(T .LE. -HF) THEN
        Y=-(1+T)/T
        S=1
        A=LOG(-T)
        A=-PI6+A*(-HF*A+LOG(1+T))
       ELSE IF(T .LT. 0) THEN
        Y=-T/(1+T)
        S=-1
        A=HF*LOG(1+T)**2
       ELSE IF(T .LE. 1) THEN
        Y=T
        S=1
        A=0
       ELSE
        Y=1/T
        S=-1
        A=PI6+HF*LOG(T)**2
       ENDIF
       H=Y+Y-1
       ALFA=H+H
       B1=0
       B2=0
       DO 1 I = 19,0,-1
       B0=C(I)+ALFA*B1-B2
       B2=B1
    1  B1=B0
       H=-(S*(B0-H*B2)+A)
      ENDIF
      DDILOG=H
      RETURN
      END


      subroutine getpoles(p,xmu2,double,single,fksprefact)
c Returns the residues of double and single poles according to 
c eq.(B.1) and eq.(B.2) if fksprefact=.true.. When fksprefact=.false.,
c the prefactor (mu2/Q2)^ep in eq.(B.1) is expanded, and giving an
c extra contribution to the single pole
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal),xmu2,double,single
      logical fksprefact
      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double complex wgt1(2)
      double precision born,wgt,kikj,dot,vij,aso2pi
      integer aj,i,j,m,n
      double precision pmass(nexternal),zero,pi
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0d0)
      include "pmass.inc"
c
      double=0.d0
      single=0.d0
c Born terms
      call sborn(p_born,wgt1)
      born=dble(wgt1(1))
      do i=1,nexternal
        if(i.ne.i_fks .and. particle_type(i).ne.1)then
          if (particle_type(i).eq.8) then
             aj=0
          elseif(abs(particle_type(i)).eq.3) then
             aj=1
          endif
          if(pmass(i).eq.ZERO)then
            double=double-c(aj)
            single=single-gamma(aj)
          else
            single=single-c(aj)
          endif
        endif
      enddo

      double=double*born
      single=single*born
c Colour-linked Born terms
      do i=1,fks_j_from_i(i_fks,0)
        do j=1,i
          m=fks_j_from_i(i_fks,i)
          n=fks_j_from_i(i_fks,j)
          if( m.ne.n .and. n.ne.i_fks .and. m.ne.i_fks )then
            call sborn_sf(p_born,m,n,wgt)
c The factor -2 compensate for that missing in sborn_sf
            wgt=-2*wgt
            if(wgt.ne.0.d0)then
              if(pmass(m).eq.zero.and.pmass(n).eq.zero)then
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).ne.zero.and.pmass(n).eq.zero)then
                single=single-0.5d0*log(pmass(m)**2/QES2)*wgt
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).eq.zero.and.pmass(n).ne.zero)then
                single=single-0.5d0*log(pmass(n)**2/QES2)*wgt
                kikj=dot(p(0,n),p(0,m))
                single=single+log(2*kikj/QES2)*wgt
              elseif(pmass(m).ne.zero.and.pmass(n).ne.zero)then
                kikj=dot(p(0,n),p(0,m))
                vij=sqrt(1-(pmass(n)*pmass(m)/kikj)**2)
                if (vij .gt. 1d-6) then
                   single=single+0.5d0*1/vij*log((1+vij)/(1-vij))*wgt
                else
                   single=single+(1d0+vij**2/3d0+vij**4/5d0)*wgt
                endif
              else
                write(*,*)'Error in getpoles',i,j,n,m,pmass(n),pmass(m)
                stop
              endif
            endif
          endif
        enddo
      enddo
      aso2pi=g**2/(8*pi**2)
      double=double*aso2pi
      single=single*aso2pi
      if(.not.fksprefact)single=single+double*log(xmu2/QES2)
c
      return
      end


      function m1l_finite_CDR(p,born)
c Returns the finite part of virtual contribution, according to the
c definitions given in (B.1) and (B.2). This function must include
c the factor as/(2*pi)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal-1),m1l_finite_CDR,born
      double precision CF,pi,aso2pi,shat,dot,xlgq2os
      parameter (CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)
c
      aso2pi=g**2/(8*pi**2)
c This is relevant to e+e- --> qqbar
      shat=2d0*dot(p(0,1),p(0,2))
      xlgq2os=log(QES2/shat)
      m1l_finite_CDR=-aso2pi*CF*(xlgq2os**2+3*xlgq2os-pi**2+8.d0)*born
      return
      end


      function m1l_W_finite_CDR(p,born)
c Returns the finite part of virtual contribution, according to the
c definitions given in (B.1) and (B.2). This function must include
c the factor as/(2*pi)
      implicit none
      include "genps.inc"
      include 'nexternal.inc'
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include 'coupl.inc'
      include 'q_es.inc'
      double precision p(0:3,nexternal-1),m1l_W_finite_CDR,born
      double precision CF,pi,aso2pi,shat,dot,xlgq2os
      parameter (CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)
c
      aso2pi=g**2/(8*pi**2)
      shat=2d0*dot(p(0,1),p(0,2))
      xlgq2os=log(QES2/shat)

c This is relevant to qqbar -> W 
      m1l_W_finite_CDR=aso2pi*CF*(-xlgq2os**2-3d0*xlgq2os+pi**2-8d0)
      m1l_W_finite_CDR=m1l_W_finite_CDR*born

c This is relevant to gg -> H
c$$$      m1l_W_finite_CDR=aso2pi*(-3d0*xlgq2os**2+11d0+3d0*pi**2)
c$$$      m1l_W_finite_CDR=m1l_W_finite_CDR*born

c This is relevant to bbbar -> H
c$$$      m1l_W_finite_CDR=aso2pi
c$$$     f     * (-4d0/3d0*xlgq2os**2
c$$$     f        -8d0/3d0+(16d0/3d0+8d0/3d0)*pi**2/6d0)
c$$$      m1l_W_finite_CDR=m1l_W_finite_CDR*born
      return
      end


      subroutine setfksfactor(iconfig,match_to_shower)
      implicit none

      double precision CA,CF, PI
      parameter (CA=3d0,CF=4d0/3d0)
      parameter (pi=3.1415926535897932385d0)

      double precision c(0:1),gamma(0:1),gammap(0:1)
      common/fks_colors/c,gamma,gammap

      double precision beta0,ren_group_coeff
      common/cbeta0/beta0,ren_group_coeff

      logical softtest,colltest
      common/sctests/softtest,colltest

      integer config_fks,i,j,iconfig,fac1,fac2
      logical match_to_shower

      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks

      double precision iden_comp
      common /c_iden_comp/iden_comp

      include 'coupl.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'fks_powers.inc'
      include 'nFKSconfigs.inc'
      include 'c_weight.inc'
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      include 'reweight0.inc'
      include 'run.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS

      integer mapconfig(0:lmaxconfigs), this_config
      integer iforest(2,-max_branch:-1,lmaxconfigs)
      integer sprop(-max_branch:-1,lmaxconfigs)
      integer tprid(-max_branch:-1,lmaxconfigs)
      include "born_conf.inc"

      logical firsttime,firsttime_nFKSprocess(fks_configs)
      data firsttime,firsttime_nFKSprocess/.true.,fks_configs*.true./

      double precision xicut_used
      common /cxicut_used/xicut_used
      double precision delta_used
      common /cdelta_used/delta_used
      double precision xiScut_used,xiBSVcut_used
      common /cxiScut_used/xiScut_used,xiBSVcut_used
      logical rotategranny
      common/crotategranny/rotategranny
      double precision diagramsymmetryfactor
      common /dsymfactor/diagramsymmetryfactor

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      character*1 integrate
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer fac_i,fac_j,i_fks_pdg,j_fks_pdg,iden(nexternal)

      integer fac_i_FKS(fks_configs),fac_j_FKS(fks_configs)
     &     ,i_type_FKS(fks_configs),j_type_FKS(fks_configs)
     &     ,m_type_FKS(fks_configs),ngluons_FKS(fks_configs)
     &     ,iden_real_FKS(fks_configs),iden_born_FKS(fks_configs)
      save fac_i_FKS,fac_j_FKS,i_type_FKS,j_type_FKS,m_type_FKS
     $     ,ngluons_FKS,iden_real_FKS,iden_born_FKS

      character*13 filename

      character*4 abrv
      common /to_abrv/ abrv

      logical nbody
      common/cnbody/nbody

      integer fold
      common /cfl/fold

c Particle types (=color) of i_fks, j_fks and fks_mother
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

c The value of rotategranny may be superseded later if phase space
c parametrization allows it
      rotategranny=.false.

      softtest=.false.
      colltest=.false.
      fold=0

      if (j_fks.gt.nincoming)then
         delta_used=deltaO
      else
         delta_used=deltaI
      endif
      
      xicut_used=xicut
      xiScut_used=xiScut
      if( nbody .or. (abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &     abrv(1:2).eq.'vi') )then
        xiBSVcut_used=1.d0
      else
        xiBSVcut_used=xiBSVcut
      endif

      c(0)=CA
      c(1)=CF
      gamma(0)=( 11d0*CA-2d0*Nf )/6d0
      gamma(1)=CF*3d0/2d0
      gammap(0)=( 67d0/9d0 - 2d0*PI**2/3d0 )*CA - 23d0/18d0*Nf
      gammap(1)=( 13/2d0 - 2d0*PI**2/3d0 )*CF
            
c Beta_0 defined according to (MadFKS.C.5)
      beta0=gamma(0)/(2*pi)
c ren_group_coeff defined accordingly
      ren_group_coeff=ren_group_coeff_in/(2*pi)

      if (firsttime_nFKSprocess(nFKSprocess)) then
         firsttime_nFKSprocess(nFKSprocess)=.false.
c---------------------------------------------------------------------
c              Symmetry Factors
c---------------------------------------------------------------------
c fkssymmetryfactor:
c Calculate the FKS symmetry factors to be able to reduce the number
c of directories to (maximum) 4 (neglecting quark flavors):
c     1. i_fks=gluon, j_fks=gluon 
c     2. i_fks=gluon, j_fks=quark
c     3. i_fks=gluon, j_fks=anti-quark
c     4. i_fks=quark, j_fks=anti-quark (or vice versa).
c This sets the fkssymmetryfactor (in which the quark flavors are taken
c into account) for the subtracted reals.
c
c fkssymmetryfactorBorn:
c Note that in the Born's included here, the final state identical
c particle factor is set equal to the identical particle factor
c for the real contribution to be able to get the correct limits for the
c subtraction terms and the approximated real contributions.
c However when we want to calculate the Born contributions only, we
c have to correct for this difference. Since we only include the Born
c related to a soft limit (this uniquely defines the Born for a given real)
c the difference is always n!/(n-1)!=n, where n is the number of final state
c gluons in the real contribution.
c
c Furthermore, because we are not integrating all the directories, we also
c have to include a fkssymmetryfactor for the Born contributions. However,
c this factor is not the same as the factor defined above, because in this
c case i_fks is fixed to the extra gluon (which goes soft and defines the
c Born contribution) and should therefore not be taken into account when
c calculating the symmetry factor. Together with the factor n above this
c sets the fkssymmetryfactorBorn equal to the fkssymmetryfactor for the
c subtracted reals.
c
c We set fkssymmetryfactorBorn to zero when i_fks not a gluon
c
         i_fks_pdg=pdg_type(i_fks)
         j_fks_pdg=pdg_type(j_fks)
      
         fac_i_FKS(nFKSprocess)=0
         fac_j_FKS(nFKSprocess)=0
         do i=nincoming+1,nexternal
            if (i_fks_pdg.eq.pdg_type(i)) fac_i_FKS(nFKSprocess) =
     $           fac_i_FKS(nFKSprocess) + 1
            if (j_fks_pdg.eq.pdg_type(i)) fac_j_FKS(nFKSprocess) =
     $           fac_j_FKS(nFKSprocess) + 1
         enddo
c Overwrite if initial state singularity
         if(j_fks.le.nincoming) fac_j_FKS(nFKSprocess)=1

c i_fks and j_fks of the same type? -> subtract 1 to avoid double counting
         if (j_fks.gt.nincoming .and. i_fks_pdg.eq.j_fks_pdg)
     $        fac_j_FKS(nFKSprocess)=fac_j_FKS(nFKSprocess)-1

c THESE TESTS WORK ONLY FOR FINAL STATE SINGULARITIES
         if (j_fks.gt.nincoming) then
            if ( i_fks_pdg.eq.j_fks_pdg .and. i_fks_pdg.ne.21) then
               write (*,*) 'ERROR, if PDG type of i_fks and j_fks '//
     &              'are equal, they MUST be gluons',
     &              i_fks,j_fks,i_fks_pdg,j_fks_pdg
               stop
            elseif(abs(particle_type(i_fks)).eq.3) then
               if ( particle_type(i_fks).ne.-particle_type(j_fks) .or.
     &              pdg_type(i_fks).ne.-pdg_type(j_fks)) then
                  write (*,*) 'ERROR, if i_fks is a color triplet,'//
     &                 ' j_fks must be its anti-particle,'//
     &                 ' or an initial state gluon.',
     &                 i_fks,j_fks,particle_type(i_fks),
     &                 particle_type(j_fks),pdg_type(i_fks),pdg_type(j_fks)
                  stop
               endif
            elseif(abs(i_fks_pdg).ne.21) then ! if not already above, it MUST be a gluon
               write (*,*) 'ERROR, i_fks is not a gluon and falls not'//
     $              ' in other categories',i_fks,j_fks,i_fks_pdg
     $              ,j_fks_pdg
            endif
         endif

         ngluons_FKS(nFKSprocess)=0
         do i=nincoming+1,nexternal
            if (pdg_type(i).eq.21) ngluons_FKS(nFKSprocess)
     $           =ngluons_FKS(nFKSprocess)+1
         enddo



c Set color types of i_fks, j_fks and fks_mother.
         i_type=particle_type(i_fks)
         j_type=particle_type(j_fks)
         if (abs(i_type).eq.abs(j_type)) then
            m_type=8
            if ( (j_fks.le.nincoming .and.
     &           abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &           (j_fks.gt.nincoming .and.
     &           abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
               write(*,*)'Flavour mismatch #1 in setfksfactor',
     &              i_fks,j_fks,i_type,j_type
               stop
            endif
         elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
            if(j_fks.le.nincoming)then
               m_type=-i_type
            else
               write (*,*) 'Error in setfksfactor: (i,j)=(q,g)'
               stop
            endif
         elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
            if (j_fks.le.nincoming) then
               m_type=j_type
            else
               m_type=j_type
            endif
         elseif(i_type.eq.8.and.j_type.eq.1.and.pdg_type(i_fks).eq.-21)then
         ! dirty trick for LOonly processes without colored legs
            m_type=j_type
         else
            write(*,*)'Flavour mismatch #2 in setfksfactor',
     &           i_type,j_type,m_type
            stop
         endif
         i_type_FKS(nFKSprocess)=i_type
         j_type_FKS(nFKSprocess)=j_type
         m_type_FKS(nFKSprocess)=m_type


c Compute the identical particle symmetry factor that is in the
c real-emission matrix elements.
         iden_real_FKS(nFKSprocess)=1
         do i=1,nexternal
            iden(i)=1
         enddo
         do i=nincoming+2,nexternal
            do j=nincoming+1,i-1
               if (pdg_type(j).eq.pdg_type(i)) then
                  iden(j)=iden(j)+1
                  iden_real_FKS(nFKSprocess)=
     &                 iden_real_FKS(nFKSprocess)*iden(j)
                  exit
               endif
            enddo
         enddo
c Compute the identical particle symmetry factor that is in the
c Born matrix elements.
         iden_born_FKS(nFKSprocess)=1
         call set_pdg(0,nFKSprocess)
         do i=1,nexternal
            iden(i)=1
         enddo
         do i=nincoming+2,nexternal-1
            do j=nincoming+1,i-1
               if (pdg_uborn(j,0).eq.pdg_uborn(i,0)) then
                  iden(j)=iden(j)+1
                  iden_born_FKS(nFKSprocess)=
     &                 iden_born_FKS(nFKSprocess)*iden(j)
                  exit
               endif
            enddo
         enddo
      endif

      i_type=i_type_FKS(nFKSprocess)
      j_type=j_type_FKS(nFKSprocess)
      m_type=m_type_FKS(nFKSprocess)

c Difference in identical particle factor in the Born and real emission
c matrix elements. To define wgt_ME_tree for the Born, we need to
c include this factor, because in the current Born the symmetry factor
c for the real is used. THIS NEEDS TO BE CHANGED WHEN MERGING WITH THE
c 'FKS_EW' STUFF
      iden_comp=dble(iden_born_FKS(nFKSprocess))/
     &          dble(iden_real_FKS(nFKSprocess))
      
      
c Set matrices used by MC counterterms
      if (match_to_shower) call set_mc_matrices

      fac_i=fac_i_FKS(nFKSprocess)
      fac_j=fac_j_FKS(nFKSprocess)
      ngluons=ngluons_FKS(nFKSprocess)
c Setup the FKS symmetry factors. 
      if (nbody.and.pdg_type(i_fks).eq.21) then
         fkssymmetryfactor=dble(ngluons)
         fkssymmetryfactorDeg=dble(ngluons)
         fkssymmetryfactorBorn=1d0
      elseif(pdg_type(i_fks).eq.-21) then
         fkssymmetryfactor=1d0
         fkssymmetryfactorDeg=1d0
         fkssymmetryfactorBorn=1d0
      else
         fkssymmetryfactor=dble(fac_i*fac_j)
         fkssymmetryfactorDeg=dble(fac_i*fac_j)
         if (pdg_type(i_fks).eq.21) then
            fkssymmetryfactorBorn=dble(fac_i*fac_j)
         else
            fkssymmetryfactorBorn=0d0
         endif
         if (abrv.eq.'grid') then
            fkssymmetryfactorBorn=1d0
            fkssymmetryfactor=0d0
            fkssymmetryfactorDeg=0d0
         endif
      endif

      if (firsttime) then
c Check to see if this channel needs to be included in the multi-channeling
         diagramsymmetryfactor=0d0
         if (multi_channel) then
            open (unit=19,file="symfact.dat",status="old",err=14)
            do i=1,mapconfig(0)
               read (19,*,err=23) fac1,fac2
               if (i.eq.iconfig) then
                  if (mapconfig(iconfig).ne.fac1) then
                     write (*,*) 'inconsistency in symfact.dat',i
     $                    ,iconfig,mapconfig(iconfig),fac1
                     stop
                  endif
                  diagramsymmetryfactor=dble(fac2)
               endif
            enddo
            close(19)
         else                   ! no multi_channel
            diagramsymmetryfactor=1d0
         endif
 12      continue
         firsttime=.false.
      endif

      return

 99   continue
      write (*,*) '"integrate.fks" or "nbodyonly.fks" not found.'
      write (*,*) 'make and run "genint_fks" first.'
      stop
 23   continue
      write (*,*) '"symfact.dat" is not of the correct format'
      stop
 14   continue
      diagramsymmetryfactor=1d0
      goto 12
      end


      subroutine get_helicity(i_fks,j_fks)
      implicit none
      include "nexternal.inc"
      include "born_nhel.inc"
      include "madfks_mcatnlo.inc"
      integer NHEL(nexternal,max_bhel*2),IHEL
chel  include "helicities.inc"
      include 'nFKSconfigs.inc'
      double precision hel_fac
      integer get_hel,skip(fks_configs)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer hel_wgt,hel_wgt_born,hel_wgt_real
      integer nhelreal(nexternal,4),goodhelreal(4)
      integer nhelrealall(nexternal,max_bhel*2)
      common /c_nhelreal/ nhelreal,nhelrealall,goodhelreal,hel_wgt_real
      integer nhelborn(nexternal-1,2),goodhelborn(2)
      integer nhelbornall(nexternal-1,max_bhel)
      common /c_nhelborn/ nhelborn,nhelbornall,goodhelborn,hel_wgt_born

      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel

      integer i,nexthel,j,i_fks,j_fks,ngood,k
      data nexthel /0/
      data ngood /0/
      logical done,firsttime,all_set,chckr
      data firsttime/.true./
      integer goodhelr(0:4,max_bhel/2),goodhelb(0:2,max_bhel/2)
      save goodhelr,goodhelb,all_set,chckr
      double precision rnd,ran2
      external ran2

      character*4 abrv
      common /to_abrv/ abrv
      logical Hevents
      common/SHevents/Hevents
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint

c Do not change these two lines, because ./bin/compile_madfks.sh might
c need to change them automatically
      logical HelSum
      parameter (HelSum=.true.)

c************
c goodhelr=2, real emission matrix element not yet calculated
c             for this helicity
c goodhelr=1, real emission matrix element calculated and non-zero
c goodhelr=0, real emission matrix element calculated and zero,
c             so can be skipped next time.
c************
      if (HelSum) return

      if (isum_hel.ne.0) then ! MC over helicities
c First, set the goodhelr and goodhelb to their starting values
      if (firsttime) then
         if ((mint .and. (.not.Hevents) .and. (abrv(1:2).eq.'vi' .or.
     &        abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &        (.not.UseSudakov))) .or. (.not.mint .and. (abrv.eq.'born'
     &        .or. abrv.eq.'grid' .or. abrv(1:2).eq.'vi'))) then
c           if computing only the Born diagrams, should not
c           consider real emission helicities            
            chckr=.false.
         else
            chckr=.true.
         endif
         do i=1,fks_configs
            skip(i)=1
         enddo
c read from file if possible
         open(unit=65,file='goodhel.dat',status='old',err=532)
         all_set=.true.
         do j=0,4
            read (65,*,err=532) (goodhelr(j,i),i=1,max_bhel/2)
         enddo
         do j=0,2
            read (65,*,err=532) (goodhelb(j,i),i=1,max_bhel/2)
         enddo
         read(65,*,err=532) hel_wgt
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt
         do i=1,max_bhel/2
            if ((chckr .and.
     &           (goodhelb(0,i).eq.2 .or. goodhelr(0,i).eq.2)) .or.
     &           (.not.chckr.and.goodhelb(0,i).eq.2)) all_set=.false.
         enddo
         close(65)
         goto 533
c if file does not exist or has wrong format, set all to 2
 532     close(65)
         write (*,*) 'Good helicities not found in file'
         all_set=.false.
         do j=0,4
            do i=1,max_bhel/2
               goodhelr(j,i)=2
            enddo
         enddo
         do j=0,2
            do i=1,max_bhel/2
               goodhelb(j,i)=2
            enddo
         enddo
         hel_wgt=max_bhel/2
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt
 533     continue
         firsttime=.false.
         goto 534 ! no previous event, so skip to the next helicity
      endif

c From previous event, check if there is an update
      if (.not.all_set) then
c real emission
         if(goodhelr(0,ngood).eq.2) then
            if ( goodhelreal(1).eq.0 .and.
     &           goodhelreal(2).eq.0 .and.
     &           goodhelreal(3).eq.0 .and.
     &           goodhelreal(4).eq.0 ) then
               do j=0,4
                  goodhelr(j,ngood)=0
               enddo
            elseif( goodhelreal(1).le.1 .and.
     &              goodhelreal(2).le.1 .and.
     &              goodhelreal(3).le.1 .and.
     &              goodhelreal(4).le.1 ) then
               goodhelr(0,ngood)=1
               do j=1,4
                  goodhelr(j,ngood)=goodhelreal(j)
               enddo
            elseif (.not.(goodhelreal(1).eq.2 .and.
     &                    goodhelreal(2).eq.2 .and.
     &                    goodhelreal(2).eq.2 .and.
     &                    goodhelreal(2).eq.2) ) then
               write (*,*) 'Error #2 in get_helicities',
     &              ngood,(goodhelr(j,ngood),j=0,4)
               stop
            endif
         endif
c Born and counter events
         if(goodhelb(0,ngood).eq.2) then
            if ( goodhelborn(1).eq.0 .and.
     &           goodhelborn(2).eq.0 ) then
               do j=0,2
                  goodhelb(j,ngood)=0
               enddo
            elseif( goodhelborn(1).le.1 .and.
     &              goodhelborn(2).le.1 ) then
               goodhelb(0,ngood)=1
               do j=1,2
                  goodhelb(j,ngood)=goodhelborn(j)
               enddo
            elseif (.not.(goodhelborn(1).eq.2 .and.
     &                    goodhelborn(2).eq.2) ) then
               write (*,*) 'Error #3 in get_helicities',
     &              nexthel,(goodhelb(j,ngood),j=0,2)
               stop
            endif
         endif

c Calculate new hel_wgt
         hel_wgt=0
         do i=1,max_bhel/2
            if((chckr .and.
     &           (goodhelb(0,i).ge.1.or.goodhelr(0,i).ge.1)) .or.
     &           (.not.chckr .and. goodhelb(0,i).ge.1)) then
               hel_wgt=hel_wgt+1
            endif
         enddo
         hel_wgt_born=hel_wgt
         hel_wgt_real=hel_wgt

c check if all have been set, if so -> write to file
         all_set=.true.
         do i=1,max_bhel/2
            if ((chckr .and.
     &           (goodhelb(0,i).eq.2 .or. goodhelr(0,i).eq.2)) .or.
     &           (.not.chckr.and.goodhelb(0,i).eq.2)) all_set=.false.
         enddo
         if (all_set) then
            write (*,*) 'All good helicities have been found.',hel_wgt
            open(unit=65,file='goodhel.dat',status='unknown')
            do j=0,4
               write (65,*) (goodhelr(j,i),i=1,max_bhel/2)
            enddo
            do j=0,2
               write (65,*) (goodhelb(j,i),i=1,max_bhel/2)
            enddo
            write(65,*) hel_wgt
            close(65)
         endif
      else
         do i=1,4
            if (goodhelr(i,ngood).ne.goodhelreal(i)) then
               write (*,*)'Error #4 in get_helicities',i,ngood
               stop
            endif
         enddo
         do i=1,2
            if (goodhelb(i,ngood).ne.goodhelborn(i)) then
               write (*,*)'Error #5 in get_helicities',i,ngood
               stop
            endif
         enddo
      endif

c Get the next helicity
 534  continue
      done=.false.
      do while (.not.done)
         if (nexthel.eq.max_bhel*2) nexthel=0
         nexthel=nexthel+1
         if(nhel(i_fks,nexthel).eq.1.and.nhel(j_fks,nexthel).eq.1) then
            if (ngood.eq.max_bhel/2) ngood=0
            ngood=ngood+1
            if((chckr .and.
     &           (goodhelr(0,ngood).ge.1.or.goodhelb(0,ngood).ge.1)).or.
     &           (.not.chckr .and. goodhelb(0,ngood).ge.1)) then
c Using random number to see if we have to go to the next.
c Probably this is an overkill, but have to make sure that there is
c no bias considering the *semi*-random numbers from VEGAS.
               rnd=ran2()
               if (rnd.le.1d0/dble(hel_wgt)) then
                  done=.true.
               endif
            endif
         endif
      enddo

      do i=1,nexternal
         if (i.eq.i_fks) then
            nhelreal(i,1)=1
            nhelreal(i,2)=1
            nhelreal(i,3)=-1
            nhelreal(i,4)=-1
         elseif (i.eq.j_fks) then
            nhelreal(i,1)=1
            nhelreal(i,2)=-1
            nhelreal(i,3)=1
            nhelreal(i,4)=-1
         else
            nhelreal(i,1)=nhel(i,nexthel)
            nhelreal(i,2)=nhel(i,nexthel)
            nhelreal(i,3)=nhel(i,nexthel)
            nhelreal(i,4)=nhel(i,nexthel)
         endif
      enddo
      do j=1,4
         goodhelreal(j)=goodhelr(j,ngood)
      enddo

      do i=1,nexternal-1
         if (i.eq.min(i_fks,j_fks)) then
            nhelborn(i,1)=1
            nhelborn(i,2)=-1
         elseif(i.lt.max(i_fks,j_fks)) then
            nhelborn(i,1)=nhel(i,nexthel)
            nhelborn(i,2)=nhel(i,nexthel)
         else
            nhelborn(i,1)=nhel(i+1,nexthel)
            nhelborn(i,2)=nhel(i+1,nexthel)
         endif
      enddo
      do j=1,2
         goodhelborn(j)=goodhelb(j,ngood)
      enddo

      else !isum_hel is zero, sum explicitly over helicities

      do i=1,nexternal
         do j=1,max_bhel*2
            nhelrealall(i,j)=nhel(i,j)
         enddo
      enddo
      do i=1,nexternal-1
         k=0
         do j=1,max_bhel*2
            if (nhel(i_fks,j).eq.-1) then
               k=k+1
               if (i.lt.i_fks) then
                  nhelbornall(i,k)=nhel(i,j)                  
               elseif(i.ge.i_fks) then
                  nhelbornall(i,k)=nhel(i+1,j)
               endif
            endif
         enddo
      enddo

      endif
      return
      end

      function get_ptrel(pp,i_fks,j_fks)
      implicit none
      include 'nexternal.inc'
      double precision get_ptrel,pp(0:3,nexternal)
      integer i_fks,j_fks
      double precision tmp,psum(3)
      integer i
c
      if(j_fks.le.2)then
        tmp=sqrt(pp(1,i_fks)**2+pp(2,i_fks)**2)
      else
        do i=1,3
          psum(i)=pp(i,i_fks)+pp(i,j_fks)
        enddo
        tmp=( pp(2,i_fks)*psum(1)-pp(1,i_fks)*psum(2) )**2+
     #      ( pp(3,i_fks)*psum(1)-pp(1,i_fks)*psum(3) )**2+
     #      ( pp(3,i_fks)*psum(2)-pp(2,i_fks)*psum(3) )**2
        if(tmp.ne.0.d0)tmp=sqrt( tmp/
     #       (psum(1)**2+psum(2)**2+psum(3)**2) )
      endif
      get_ptrel=tmp
      return
      end



      FUNCTION FK88RANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      REAL*8 MINV,FK88RANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      FK88RANDOM = SEED*MINV
      END


      subroutine set_mu_central(ic,dd,c_mu2_r,c_mu2_f)
      implicit none
      include 'nexternal.inc'
      include 'c_weight.inc'
      include 'reweight0.inc'
      include 'run.inc'
      integer ic,dd,i,j
      double precision c_mu2_r,c_mu2_f,muR,muF,pp(0:3,nexternal)
      if (dd.eq.1) then
         c_mu2_r=scales2(2,ic)
         c_mu2_f=scales2(3,ic)
      else
c need to recompute the scales using the momenta
         dynamical_scale_choice=dyn_scale(dd)
         do i=1,nexternal
            do j=0,3
               pp(j,i)=momenta(j,i,ic)
            enddo
         enddo
         call set_ren_scale(pp,muR)
         c_mu2_r=muR**2
         call set_fac_scale(pp,muF)
         c_mu2_f=muF**2
c     reset the default dynamical_scale_choice
         dynamical_scale_choice=dyn_scale(1)
      endif
      return
      end
