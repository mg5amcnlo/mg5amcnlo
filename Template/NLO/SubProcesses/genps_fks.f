      subroutine generate_momenta(ndim,iconfig,wgt,x,p)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'nFKSconfigs.inc'
      include 'timing_variables.inc'
      integer ndim,iconfig
      double precision wgt,x(99),p(0:3,nexternal)
      double precision pmass(-nexternal:0,lmaxconfigs,0:fks_configs)
      double precision pwidth(-nexternal:0,lmaxconfigs,0:fks_configs)
      integer iforest(2,-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer sprop(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer tprid(-max_branch:-1,lmaxconfigs,0:fks_configs)
      integer mapconfig(0:lmaxconfigs,0:fks_configs)
      common /c_configurations/pmass,pwidth,iforest,sprop,tprid
     $     ,mapconfig
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),jac
      integer i,j
      double precision zero
      parameter (zero=0d0)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      integer iconfig0
      common/ciconfig0/iconfig0
      double precision qmass_common(-nexternal:0),qwidth_common(
     &     -nexternal:0)
      common /c_qmass_qwidth/qmass_common,qwidth_common
      double precision xvar(99)
      common /c_vegas_x/xvar
      integer            this_config
      common/to_mconfigs/this_config
c     
      call cpu_time(tBefore)
      this_config=iconfig
      iconf=iconfig
      iconfig0=iconfig
      do i=-max_branch,-1
         do j=1,2
            itree(j,i)=iforest(j,i,iconfig,0)
         enddo
      enddo

      do i=-nexternal,0
         qmass(i)=pmass(i,iconfig,0)
         qwidth(i)=pwidth(i,iconfig,0)
         qmass_common(i)=qmass(i)
         qwidth_common(i)=qwidth(i)
      enddo
      do i=1,ndim
         xvar(i)=x(i)
      enddo
c
      call generate_momenta_conf_wrapper(ndim,jac,x,itree,qmass,qwidth,p)
c If the input weight 'wgt' to this subroutine was not equal to one,
c make sure we update all the (counter-event) jacobians and return also
c the updated wgt (i.e. the jacobian for the event)
      do i=-2,2
         jac_cnt(i)=jac_cnt(i)*wgt
      enddo
      wgt=wgt*jac
c
      call cpu_time(tAfter)
      tGenPS=tGenPS+(tAfter-tBefore)
      return
      end

      double precision function virtgranny(virtgrannybar)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision virtgrannybar
      integer i,j,igranny_fail
      data igranny_fail /0/
      double precision dot,dummy,pgranny(0:3),jac
      external dot
      double precision rat_xi_orig
      common /c_rat_xi/ rat_xi_orig
      double precision granny_m2_red(-1:1)
      common /to_virtgranny/granny_m2_red
c common block that is filled by this subroutine
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
c arguments for the generate_momenta_conf subroutine from common blocks
      double precision p(0:3,nexternal)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      integer nndim
      common/tosigint/nndim
      double precision qmass_common(-nexternal:0),qwidth_common(
     &     -nexternal:0)
      common /c_qmass_qwidth/qmass_common,qwidth_common
      double precision xvar(99)
      common /c_vegas_x/xvar
c      
      granny_m2_red(0)=virtgrannybar
      if (virtgrannybar.le.granny_m2_red(-1) .or.
     &     virtgrannybar.ge.granny_m2_red(1) ) then
         igranny_fail=igranny_fail+1
         virtgranny=0d0
         return
      endif
      call generate_momenta_conf(.true.,nndim,jac,xvar,granny_m2_red
     &     ,rat_xi_orig,itree,qmass_common,qwidth_common,p)
      if (jac.gt.0d0) then
         do j=0,3
            pgranny(j)=0d0
            do i=1,nexternal
               if (granny_chain_real_final(i)) pgranny(j)=pgranny(j)+p(j
     &              ,i)
            enddo
         enddo
         virtgranny=dot(pgranny,pgranny)
      else
         virtgranny=0d0
      endif
      return
      end

      double precision function virtgranny_red(virtgrannybar)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision virtgrannybar
      integer i,j,igranny_fail
      data igranny_fail /0/
      double precision dot,rho,dummy,pgranny_bar(0:3),p_mother_bar3(3)
     &     ,pcm(0:3),df1(0:3),jac
      external dot,rho
      double precision granny_m2_red(-1:1)
      common /to_virtgranny/granny_m2_red
      double precision rat_xi_orig
      common /c_rat_xi/ rat_xi_orig
c common block that is filled by this subroutine
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
c arguments for the generate_momenta_conf subroutine from common blocks
      double precision p(0:3,nexternal)
      integer itree(2,-max_branch:-1),iconf
      common /to_itree/itree,iconf
      integer nndim
      common/tosigint/nndim
      double precision qmass_common(-nexternal:0),qwidth_common(
     &     -nexternal:0)
      common /c_qmass_qwidth/qmass_common,qwidth_common
c
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     &                        sqrtshat,shat
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision shybst,chybst,chybstmo
      common /virtgranny_boost/shybst,chybst,chybstmo
      double precision xvar(99)
      common /c_vegas_x/xvar
c      
      granny_m2_red(0)=virtgrannybar
      if (virtgrannybar.le.granny_m2_red(-1) .or.
     &     virtgrannybar.ge.granny_m2_red(1) ) then
         igranny_fail=igranny_fail+1
         virtgranny_red=0d0
         return
      endif
      call generate_momenta_conf(.true.,nndim,jac,xvar,granny_m2_red
     &     ,rat_xi_orig,itree,qmass_common,qwidth_common,p)
      if (jac.gt.0d0) then
         do j=1,3
            pcm(j)=0d0
            p_mother_bar3(j)=p_born_l(j,j_fks)/rho(p_born_l(0,j_fks))
         enddo
         pcm(0)=sqrtshat
         call boostwdir2(chybst,-shybst,chybstmo,p_mother_bar3,pcm,df1)
         do j=0,3
            df1(j)=df1(j)-pcm(j)
         enddo
         do j=0,3
            pgranny_bar(j)=0d0
            do i=1,nexternal-1
               if (granny_chain(i))
     &              pgranny_bar(j)=pgranny_bar(j)+p_born_l(j,i)
            enddo
         enddo
         virtgranny_red=dot(df1,df1)+2*dot(pgranny_bar,df1)
      else
         virtgranny_red=0d0
      endif
      return
      end

      function xinv_virtgranny(valxmbe2)
c Any call to this function must be preceded by a call to fillcblk
      implicit none
      real*8 xinv_virtgranny,valxmbe2
      real*8 tmp,tolerance
      parameter (tolerance=1.d-15)
      integer ierr,mxf,mode
      parameter (mxf=500)
      parameter (mode=2)
      real*8 xmbemin2,xmbemax2
      common/cgrannyrange/xmbemin2,xmbemax2
      real*8 offset
      common/coffset/offset
      real*8 dzerox,off_virtgranny
      external off_virtgranny
c
      offset=valxmbe2
      tmp=dzerox(xmbemin2,xmbemax2,tolerance,mxf,
     #           off_virtgranny,mode,ierr)
      if(ierr.ne.0)tmp=0.d0
      xinv_virtgranny=tmp
      return
      end


      function xinv_redvirtgranny(valxmbe2)
c Any call to this function must be preceded by a call to fillcblk
      implicit none
      real*8 xinv_redvirtgranny,valxmbe2
      real*8 tmp,tolerance
      parameter (tolerance=1.d-15)
      integer ierr,mxf,mode
      parameter (mxf=500)
      parameter (mode=2)
      real*8 xmbemin2,xmbemax2
      common/cgrannyrange/xmbemin2,xmbemax2
      real*8 offset
      common/coffset/offset
      real*8 dzerox,off_redvirtgranny
      external off_redvirtgranny
c
      offset=valxmbe2
      tmp=dzerox(xmbemin2,xmbemax2,tolerance,mxf,
     #           off_redvirtgranny,mode,ierr)
      if(ierr.ne.0)tmp=0.d0
      xinv_redvirtgranny=tmp
      return
      end


      function off_virtgranny(virtgrannybar)
c Any call to this function must be preceded by a call to fillcblk
      implicit none
      real*8 off_virtgranny,virtgrannybar
      real*8 tmp,virtgranny,offset
      common/coffset/offset
      external virtgranny
c
      tmp=virtgranny(virtgrannybar)-offset
      off_virtgranny=tmp
      return
      end


      function off_redvirtgranny(virtgrannybar)
c Any call to this function must be preceded by a call to fillcblk
      implicit none
      real*8 off_redvirtgranny,virtgrannybar
      real*8 tmp,virtgranny_red,offset
      common/coffset/offset
      external virtgranny_red
c
      tmp=virtgranny_red(virtgrannybar)+virtgrannybar-offset
      off_redvirtgranny=tmp
      return
      end


      
      subroutine generate_momenta_conf_wrapper(nndim,jac,x,itree,qmass
     $     ,qwidth,p)
      use mint_module
      implicit none
      include 'nexternal.inc'
      include 'genps.inc'
      include 'nFKSconfigs.inc'
      integer nndim
      double precision jac,x(99),p(0:3,nexternal)
      integer itree(2,-max_branch:-1),i,j
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),del1
     &     ,del2,del3,del30,der,derivative,errder,random,ran2,virtgranny
     &     ,virtgranny_red,MC_sum_factor,xmbe2hatlow,xmbe2hatupp
     &     ,xmbe2inv,xmbe2inv_temp,xinv_redvirtgranny,xinv_virtgranny
      external derivative,ran2,virtgranny,xinv_redvirtgranny
     &     ,xinv_virtgranny
c     granny stuff
      double precision tiny,granny_m2(-1:1),step,granny_m2_red_local(
     &     -1:1)
      double precision granny_m2_red(-1:1)
      common /to_virtgranny/granny_m2_red
      real*8 xmbemin2,xmbemax2,xmbemin2_0,xmbemax2_0
      common/cgrannyrange/xmbemin2,xmbemax2
      logical input_granny_m2,compute_mapped,compute_non_shifted
      parameter (tiny=1d-3)
      integer irange,idir
      data irange/0/
      parameter (idir=0,step=1d-2)
c common block that is filled by this subroutine
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
      logical only_event_phsp,skip_event_phsp
      common /c_skip_only_event_phsp/only_event_phsp,skip_event_phsp
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision rat_xi,rat_xi_orig
      common /c_rat_xi/ rat_xi_orig
c     debug stuff
      double precision temp
      logical debug_granny
      parameter (debug_granny=.false.)
      double precision deravg,derstd,dermax,xi_i_fks_ev_der_max
     &     ,y_ij_fks_ev_der_max
      integer ntot_granny,derntot,ncase(0:6)
      common /c_granny_counters/ ntot_granny,ncase,derntot,deravg,derstd
     &     ,dermax,xi_i_fks_ev_der_max,y_ij_fks_ev_der_max
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      logical do_mapping_granny
      logical softtest,colltest
      common/sctests/softtest,colltest
      integer              nFKSprocess
      common/c_nFKSprocess/nFKSprocess
c Common block with information to determine if we should not write a
c possible resonance.
      logical write_granny(fks_configs)
      integer which_is_granny(fks_configs)
      common/write_granny_resonance/which_is_granny,write_granny
      integer isolsign
      common /c_isolsign/isolsign
      double precision border,border_massive,border_massless,fborder
      parameter (border_massive=2d0,border_massless=0.1d0,fborder=0.02d0)
      logical firsttime
      data firsttime/.true./
      integer icase
      logical case_0or1or6
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision pmass(nexternal)
      common /to_mass/pmass
      character*4 abrv
      common /to_abrv/ abrv
      character*10 shower_mc
      common /cMonteCarloType/shower_mc
c
      write_granny(nFKSprocess)=.true.
      which_is_granny(nFKSprocess)=0
c
      do i=-1,1
         granny_m2_red(i)=-99d99
      enddo
      rat_xi=-99d99
c By default always try to do the mapping if need be. Change the logical
c 'do_mapping_granny' to false to never do the phase-space mapping to
c keep the invariant mass of the granny fixed.
      do_mapping_granny=.true.
         
c When doing only the Born, or when matching to Pythia8, never do the
c granny phase-space mapping.
      if ( abrv(1:4).eq.'born' .or.
     &     (nlo_ps .and. shower_mc(1:7).eq.'PYTHIA8') )
     &        do_mapping_granny=.false.

c Set the minimal tau = x1*x2. This also checks if granny is a resonance
      call set_tau_min()

      if (granny_is_res) then
         which_is_granny(nFKSprocess)=igranny
         if (.not. do_mapping_granny) then
            compute_non_shifted=.true.
            compute_mapped=.false.
            MC_sum_factor=1d0
         else
            ntot_granny=ntot_granny+1
c This computes the event kinematics and sets the range for the granny
c inv. mass, in terms of the integration variable (which is not the
c physical range of the invariant mass in the event!)
            only_event_phsp=.true.
            skip_event_phsp=.false.
            input_granny_m2=.false.
            call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $           ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
            if (nint(jac).eq.-222) return ! numerical inaccuracy: not
                                          ! even Born momenta generated.
            rat_xi_orig=rat_xi
            granny_m2_red_local( 0)=granny_m2_red( 0)
            granny_m2_red_local(-1)=granny_m2_red(-1)
            granny_m2_red_local( 1)=granny_m2_red( 1)
            xmbemin2_0=granny_m2_red_local(-1)
            xmbemax2_0=granny_m2_red_local(1)
            if (pmass(j_fks).gt.0d0) then
               border=max(border_massive ,fborder*(sqrt(xmbemax2_0)
     &                                            -sqrt(xmbemin2_0)))
            else
               border=max(border_massless,fborder*(sqrt(xmbemax2_0)
     &                                            -sqrt(xmbemin2_0)))
            endif
            xmbemin2=(sqrt(xmbemin2_0)+border)**2
            xmbemax2=(max(0.d0,sqrt(xmbemax2_0)-border))**2
            if (firsttime) then
               write (*,*) 'In phase-space generator, the border'/
     &              /' for the mapping range is set to ',border,
     &              'for the first PS point.'
               firsttime=.false.
            endif
            if(xmbemin2.ge.xmbemax2)then
               icase=0
               goto 111
            endif
            granny_m2(0) =virtgranny_red(granny_m2_red_local( 0))
     &           +granny_m2_red_local(0) ! central value
            granny_m2(1) =virtgranny_red(xmbemax2)
     &           +xmbemax2      ! upper limit
            granny_m2(-1)=virtgranny_red(xmbemin2)
     &           +xmbemin2      ! lower limit
            
            if (debug_granny) then
               temp =virtgranny(granny_m2_red_local( 0))
               if (abs((temp-granny_m2(0))/temp).gt.1d-3)then
                  write (*,*) 'DEBUG error: virtgranny,virtgranny_red'
     &                 ,temp,granny_m2(0),granny_m2_red_local(0)
c$$$                  stop
               endif
            endif
            if(granny_m2(-1).gt.granny_m2(1))then
               icase=0
            else
               if(granny_m2(1).le.xmbemin2)then
                  icase=1
               elseif( granny_m2(-1).le.xmbemin2.and.
     &                 granny_m2( 1).gt.xmbemin2.and.
     &                 granny_m2( 1).le.xmbemax2 )then
                  icase=2
               elseif( granny_m2(-1).le.xmbemin2.and.
     &                 granny_m2( 1).gt.xmbemax2 )then
                  icase=3
               elseif( granny_m2(-1).gt.xmbemin2.and.
     &                 granny_m2(-1).le.xmbemax2.and.
     &                 granny_m2( 1).le.xmbemax2 )then
                  icase=4
               elseif( granny_m2(-1).gt.xmbemin2.and.
     &                 granny_m2(-1).le.xmbemax2.and.
     &                 granny_m2( 1).gt.xmbemax2 )then
                  icase=5
               elseif( granny_m2(-1).gt.xmbemax2 )then
                  icase=6
               else
                  write(*,*)'Error in determining cases in '/
     &                 /'phase-space generation', granny_m2(-1)
     &                 ,granny_m2(1),xmbemin2,xmbemax2
                  stop
               endif
            endif
 111        continue
            ncase(icase)=ncase(icase)+1
            case_0or1or6=.false.
            if(icase.eq.0 .or. icase.eq.1 .or. icase.eq.6)then
               case_0or1or6=.true.
               xmbe2hatlow=xmbemax2_0 ! makes sure that we always go to 'compute_non_shifted'
               xmbe2hatupp=xmbemax2_0
            elseif(icase.eq.2)then
               xmbe2hatlow=xinv_redvirtgranny(xmbemin2)
               xmbe2hatupp=xmbemax2
            elseif(icase.eq.3)then
               xmbe2hatlow=xinv_redvirtgranny(xmbemin2)
               xmbe2hatupp=xinv_redvirtgranny(xmbemax2)
            elseif(icase.eq.4)then
               xmbe2hatlow=xmbemin2
               xmbe2hatupp=xmbemax2
            elseif(icase.eq.5)then
               xmbe2hatlow=xmbemin2
               xmbe2hatupp=xinv_redvirtgranny(xmbemax2)
            endif

            
c Check that granny_m2_red_local is covering the whole integration range
c of granny_m2. If that's the case, we do the special granny trick, if
c not, integrate as normal.
            if ( (granny_m2_red_local(0).lt.xmbe2hatlow .or.
     &            granny_m2_red_local(0).gt.xmbe2hatupp) .or.
     &           isolsign.eq.-1) then
c     2nd term in eq.70 of the note
               compute_non_shifted=.true.
            else
               compute_non_shifted=.false.
            endif
            if ( granny_m2_red_local(0).gt.max(granny_m2(-1),xmbemin2) .and.
     &           granny_m2_red_local(0).lt.min(granny_m2(1),xmbemax2) .and.
     &           isolsign.ne.-1 .and. (.not.case_0or1or6) ) then
c     1st term in eq.70 of the note
               compute_mapped=.true.
            else
               compute_mapped=.false.
            endif
            MC_sum_factor=1d0
            if (compute_mapped.and.compute_non_shifted) then
c     Could add importance sampling here
               random=ran2()
               if (random.lt.0.5d0) then
                  compute_mapped=.false.
               else
                  compute_non_shifted=.false.
               endif
               MC_sum_factor=2d0
            endif
         endif
         if (.not. compute_mapped .and. .not. compute_non_shifted) then
c only counter-event exists.
            input_granny_m2=.false.
            only_event_phsp=.false.
c In principle skip_event_phsp can be set to .true. to save time, but
c need to set it to .false. to fill shat_ev (et al) to be able to assign
c a shower scale. The kinematics won't be used, because below jac=-222
c and p(0,1)=-99d0.
            skip_event_phsp=.false. 
            call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $           ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
            jac=-222
            p(0,1)=-99d0
            return
         endif
         if (compute_non_shifted) then
c integrate as normal
            input_granny_m2=.false.
            only_event_phsp=.false.
            skip_event_phsp=.false.
            do i=-1,1
               granny_m2_red(i)=-99d99
            enddo
            call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $           ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
c In this case, we should not write the grandmother in the event file,
c because the shower should not keep its inv. mass fixed.
            write_granny(nFKSprocess)=.false.
         endif
         if (compute_mapped) then
c Special Phase-space generation for granny stuff: keep its invariant
c mass fixed.
c Apply the theta functions on the range of granny_m2_red also to the
c corresponding granny_m2.
c compute the derivative numerically (to compute the Jacobian)
            only_event_phsp=.true.
            skip_event_phsp=.false.
            xmbe2inv=xinv_redvirtgranny(granny_m2_red_local(0))
            der=derivative(virtgranny_red,xmbe2inv,step,idir,xmbemin2
     &           ,xmbemax2,errder)
            if(abs(der).lt.1.d-8)der=0.d0
            der=1.d0/(1.d0+der)
            derntot=derntot+1
            deravg=(deravg*(derntot-1)+abs(der))/dfloat(derntot)
            derstd=(derstd*(derntot-1)+der**2)/dfloat(derntot)
            if (abs(der).gt.dermax) then
               dermax=abs(der)
               xi_i_fks_ev_der_max=xi_i_fks_ev
               y_ij_fks_ev_der_max=y_ij_fks_ev
            endif
            if (errder.gt.0.1d0) then
               write (*,*) 'WARNING: uncertainty is large in the'/
     $              /' computation of the derivative',errder,der
            endif
c compute the event kinematics using xmbe2inv as mass for the
c grandmother of the Born (this will give granny_m2_red_local(0) mass to
c the event).
            input_granny_m2=.true.
            skip_event_phsp=.false.
            only_event_phsp=.true.
            granny_m2_red(0)=xmbe2inv
            call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $           ,granny_m2_red,rat_xi_orig,itree,qmass,qwidth,p)
c multiply event jacobian by the numerically computed jacobian for the
c derivative
            jac=jac*abs(der)
c counter-event kinematics: even though it shouldn't change from above,
c better compute it again to set all the common blocks correctly.
            input_granny_m2=.false.
            only_event_phsp=.false.
            skip_event_phsp=.true.
            call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $           ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
         endif
         jac=jac*MC_sum_factor
      else
         skip_event_phsp=.false.
         only_event_phsp =.false.
         input_granny_m2=.false.
         call generate_momenta_conf(input_granny_m2,nndim,jac,x
     $        ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
      endif
      end


      subroutine generate_momenta_conf(input_granny_m2,ndim,jac,x
     &     ,granny_m2_red,rat_xi,itree,qmass,qwidth,p)
c
c x(1)...x(ndim-5) --> invariant mass & angles for the Born
c x(ndim-4) --> tau_born
c x(ndim-3) --> y_born
c x(ndim-2) --> xi_i_fks
c x(ndim-1) --> y_ij_fks
c x(ndim) --> phi_i
c
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      include 'run.inc'
c arguments
      integer ndim
      double precision jac,x(99),p(0:3,nexternal)
      integer itree(2,-max_branch:-1)
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
     &     ,granny_m2_red(-1:1)
      logical input_granny_m2
c common
c     Arguments have the following meanings:
c     -2 soft-collinear, incoming leg, - direction as in FKS paper
c     -1 collinear, incoming leg, - direction as in FKS paper
c     0 soft
c     1 collinear
c     2 soft-collinear
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      double precision p_born_l(0:3,nexternal-1)
      common/pborn_l/p_born_l
      double precision p_born_ev(0:3,nexternal-1)
      common/pborn_ev/p_born_ev
      double precision p_ev(0:3,nexternal)
      common/pev/p_ev
      double precision xi_i_fks_ev,y_ij_fks_ev
      double precision p_i_fks_ev(0:3),p_i_fks_cnt(0:3,-2:2)
      common/fksvariables/xi_i_fks_ev,y_ij_fks_ev,p_i_fks_ev,p_i_fks_cnt
      double precision xi_i_fks_cnt(-2:2)
      common /cxiifkscnt/xi_i_fks_cnt
      double precision xi_i_hat_ev,xi_i_hat_cnt(-2:2)
      common /cxi_i_hat/xi_i_hat_ev,xi_i_hat_cnt
      double complex xij_aor
      common/cxij_aor/xij_aor
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     &                        sqrtshat,shat
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
      logical softtest,colltest
      common/sctests/softtest,colltest
      logical nocntevents
      common/cnocntevents/nocntevents
      double precision xiimax_ev
      common /cxiimaxev/xiimax_ev
      double precision xiimax_cnt(-2:2)
      common /cxiimaxcnt/xiimax_cnt
      logical nbody
      common/cnbody/nbody
      double precision xinorm_ev
      common /cxinormev/xinorm_ev
      double precision xinorm_cnt(-2:2)
      common /cxinormcnt/xinorm_cnt
      integer iconfig0,iconfigsave
      common/ciconfig0/iconfig0
      save iconfigsave
c Masses of particles. Should be filled in setcuts.f
      double precision pmass(nexternal)
      common /to_mass/pmass
c local
      integer i,j,nbranch,ns_channel,nt_channel,ionebody
     &     ,fksconfiguration,icountevts,imother,ixEi,ixyij,ixpi,isolsign
      double precision M(-max_branch:max_particles),totmassin,totmass
     &     ,stot,xp(0:3,nexternal),pb(0:3,-max_branch:nexternal-1),xjac0
     &     ,tau_born,ycm_born,ycmhat,fksmass,xbjrk_born(2),shat_born
     &     ,sqrtshat_born,S(-max_branch:max_particles),xpswgt0
     &     ,m_born(nexternal-1),m_j_fks,xmrec2,xjac,xpswgt,phi_i_fks
     &     ,tau,ycm,xbjrk(2),xiimax,xinorm,xi_i_fks ,y_ij_fks,flux
     &     ,p_i_fks(0:3),pwgt,p_born_CHECK(0:3,nexternal-1),xi_i_hat
     &     ,rat_xi
      logical one_body,pass,check_cnt
c external
      double precision lambda
      external lambda
c parameters
      logical fks_as_is
      parameter (fks_as_is=.false.)
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      logical firsttime
      data firsttime/.true./
      double precision zero
      parameter (zero=0d0)
c saves
      save m,stot,totmassin,totmass,ns_channel,nt_channel,one_body
     &     ,ionebody,fksmass,nbranch
      common /c_isolsign/isolsign
c Conflicting BW stuff
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      common/c_conflictingBW/cBW_mass,cBW_width,cBW_level_max,cBW
     $     ,cBW_level
c Common block with granny information
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
      logical only_event_phsp,skip_event_phsp
      common /c_skip_only_event_phsp/only_event_phsp,skip_event_phsp
      
      pass=.true.
      do i=1,nexternal-1
         if (i.lt.i_fks) then
            m(i)=pmass(i)
         else
            m(i)=pmass(i+1)
         endif
      enddo
      if( firsttime .or. iconfig0.ne.iconfigsave ) then
         if (nincoming.eq.2) then
            stot = 4d0*ebeam(1)*ebeam(2)
         else
            stot=pmass(1)**2
         endif
c Make sure have enough mass for external particles
         totmassin=0d0
         do i=1,nincoming
            totmassin=totmassin+m(i)
         enddo
         totmass=0d0
         nbranch = nexternal-3 ! nexternal is for n+1-body, while itree uses n-body
         do i=nincoming+1,nexternal-1
            totmass=totmass+m(i)
         enddo
         fksmass=totmass
         if (stot .lt. max(totmass,totmassin)**2) then
            write (*,*) 'Fatal error #0 in one_tree:'/
     &           /'insufficient collider energy'
            stop
         endif
c Determine number of s- and t-channel branches, at this point it
c includes the s-channel p1+p2
         ns_channel=1
         do while(itree(1,-ns_channel).ne.1 .and.
     &        itree(1,-ns_channel).ne.2 .and. ns_channel.lt.nbranch)
            m(-ns_channel)=0d0                 
            ns_channel=ns_channel+1         
         enddo
         ns_channel=ns_channel - 1
         nt_channel=nbranch-ns_channel-1
c If no t-channles, ns_channels is one less, because we want to exclude
c the s-channel p1+p2
         if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
            ns_channel=ns_channel-1
         endif
c Set one_body to true if it's a 2->1 process at the Born (i.e. 2->2 for the n+1-body)
         if((nexternal-nincoming).eq.2)then
            one_body=.true.
            ionebody=nexternal-1
            ns_channel=0
            nt_channel=0
         elseif((nexternal-nincoming).gt.2)then
            one_body=.false.
         else
            write(*,*)'Error #1 in genps_fks.f',nexternal,nincoming
            stop
         endif
         firsttime=.false.
         iconfigsave=iconfig0
      endif                     ! firsttime
c
      xjac0=1d0
      xpswgt0=1d0
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c Generate Bjorken x's if need be cccccccccccccccccccccccccccccccccccccccccccc
ccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      if (abs(lpp(1)).ne.abs(lpp(2))) then
          write(*,*) 'Different beams not implemented', lpp
          stop 1
      endif

      if (abs(lpp(1)).ge.1 .and. abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         if (abs(lpp(1)).ne.4.and.abs(lpp(1)).ne.3) then ! this is for pp collisions
c x(ndim-1) -> tau_cnt(0); x(ndim) -> ycm_cnt(0)
            if (one_body) then
c tau is fixed by the mass of the final state particle
               call compute_tau_one_body(totmass,stot,tau_born,xjac0)
            else
               if(nt_channel.eq.0 .and. qwidth(-ns_channel-1).ne.0.d0 .and.
     $           cBW(-ns_channel-1).ne.2)then
c Generate tau according to a Breit-Wiger function
                  call generate_tau_BW(stot,ndim-4,x(ndim-4),qmass(
     $              -ns_channel-1),qwidth(-ns_channel-1),cBW(-ns_channel
     $              -1),cBW_mass(-1, -ns_channel-1),cBW_width(-1,
     $              -ns_channel-1),tau_born,xjac0)
               else 
c     not a Breit Wigner
                  call generate_tau(stot,ndim-4,x(ndim-4),tau_born,xjac0)
               endif
            endif
         
c Generate the rapditity of the Born system
           call generate_y(tau_born,x(ndim-3),ycm_born,ycmhat,xjac0)

        else                    ! this is for dressed ee collisions
           call generate_ee_tau_y(x(ndim-4), x(ndim-3), one_body,
     $        stot, nt_channel, qmass(-ns_channel-1),qwidth(-ns_channel-1),
     $        cBW(-ns_channel-1),cBW_mass(-1, -ns_channel-1),
     $        cBW_width(-1,-ns_channel-1),
     $        tau_born, ycm_born, ycmhat, xjac0)
           ! for non-physical configurations, xjac=-1000
           if (xjac0.eq.-1000d0) goto 222
         endif
      elseif (abs(lpp(1)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x1 not implemented in one_tree'
         stop
      elseif (abs(lpp(2)).ge.1 .and.
     &     .not.(softtest.or.colltest)) then
         write(*,*)'Option x2 not implemented in one_tree'
         stop
      else
c No PDFs (also use fixed energy when performing tests)
         call compute_tau_y_epem(j_fks,one_body,fksmass,stot,
     &        tau_born,ycm_born,ycmhat)
         if (j_fks.le.nincoming .and. .not.(softtest.or.colltest)) then
            write (*,*) 'Process has incoming j_fks, but fixed shat: '/
     &           /'not allowed for processes generated at NLO.'
            stop 1
         endif
      endif
c Compute Bjorken x's from tau and y
      xbjrk_born(1)=sqrt(tau_born)*exp(ycm_born)
      xbjrk_born(2)=sqrt(tau_born)*exp(-ycm_born)
c Compute shat and sqrt(shat)
      if(.not.one_body)then
        shat_born=tau_born*stot
        sqrtshat_born=sqrt(shat_born)
      else
c Trivial, but prevents loss of accuracy
        shat_born=fksmass**2
        sqrtshat_born=fksmass
      endif
c Generate the momenta for the initial state of the Born system
      if(nincoming.eq.2) then
        call mom2cx(sqrtshat_born,m(1),m(2),1d0,0d0,pb(0,1),pb(0,2))
      else
         pb(0,1)=sqrtshat_born
         do i=1,2
            pb(i,1)=0d0
         enddo
         p(3,1)=1e-14           ! For HELAS routine ixxxxx for neg. mass
      endif
      s(-nbranch)  = shat_born
      m(-nbranch)  = sqrtshat_born
      pb(0,-nbranch)= m(-nbranch)
      pb(1,-nbranch)= 0d0
      pb(2,-nbranch)= 0d0
      pb(3,-nbranch)= 0d0
c     
c Generate Born-level momenta
c
c Start by generating all the invariant masses of the s-channels
      if (granny_is_res) then
         call generate_inv_mass_sch_granny(input_granny_m2,ns_channel
     &        ,itree,m,granny_m2_red,sqrtshat_born,totmass,qwidth,qmass
     &        ,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      else
         call generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     $        ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0
     $        ,pass)
      endif
      if (.not.pass) goto 222
c If only s-channels, also set the p1+p2 s-channel
      if (nt_channel .eq. 0 .and. nincoming .eq. 2) then
         s(-nbranch+1)=s(-nbranch) 
         m(-nbranch+1)=m(-nbranch)       !Basic s-channel has s_hat 
         pb(0,-nbranch+1) = m(-nbranch+1)!and 0 momentum
         pb(1,-nbranch+1) = 0d0
         pb(2,-nbranch+1) = 0d0
         pb(3,-nbranch+1) = 0d0
      endif
c
c     Next do the T-channel branchings
c
      if (nt_channel.ne.0) then
         call generate_t_channel_branchings(ns_channel,nbranch,itree
     $        ,m,s,x,pb,xjac0,xpswgt0,pass)
         if (.not.pass) goto 222
      endif
c
c     Now generate momentum for all intermediate and final states
c     being careful to calculate from more massive to less massive states
c     so the last states done are the final particle states.
c
      call fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      if (.not.pass) goto 222
c
c  Now I have the Born momenta
c
      do i=1,nexternal-1
         do j=0,3
            p_born_l(j,i)=pb(j,i)
            p_born_CHECK(j,i)=pb(j,i)
         enddo
         m_born(i)=m(i)
      enddo
      call phspncheck_born(sqrtshat_born,m_born,p_born_CHECK,pass)
      if (.not.pass) then
         xjac0=-142
         goto 222
      endif

      if (.not.only_event_phsp) then
         do i=1,nexternal-1
            do j=0,3
               p_born(j,i)=p_born_l(j,i)
            enddo
         enddo
      endif

      if (.not. skip_event_phsp) then
         do i=1,nexternal-1
            do j=0,3
               p_born_ev(j,i)=p_born_l(j,i)
            enddo
         enddo
      endif
c
c
c Here we start with the FKS Stuff
c
c
c
c icountevts=-100 is the event, -2 to 2 the counterevents
      icountevts = -100
c if event/counterevents will not be generated, the following
c energy components will stay negative. Also set the upper limits of
c the xi ranges to negative values to force crash if something
c goes wrong. The jacobian of the counterevents are set negative
c to prevent using those skipped because e.g. m(j_fks)#0
      if (skip_event_phsp) then
         xi_i_hat=xi_i_hat_ev
         if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
            icountevts=-2
         else
            icountevts=0
         endif
c     skips counterevents when integrating over second fold for massive
c     j_fks
c     FIXTHIS FIXTHIS FIXTHIS FIXTHIS:         
         if( isolsign.eq.-1 )then
            write (*,*) 'ERROR, when doing 2nd fold of massive j_fks,'
     &           //' cannot skip event_phsp'
            stop
         endif
      else
         p_i_fks_ev(0)=-1.d0
         xiimax_ev=-1.d0
      endif
      if (.not.only_event_phsp) then
         do i=-2,2
            p_i_fks_cnt(0,i)=-1.d0
            xiimax_cnt(i)=-1.d0
            jac_cnt(i)=-1.d0
         enddo
      endif
c set cm stuff to values to make the program crash if not set elsewhere
      ybst_til_tolab=1.d14
      ybst_til_tocm=1.d14
      sqrtshat=0.d0
      shat=0.d0
c if collinear counterevent will not be generated, the following
c quantity will stay zero
      if (.not.only_event_phsp) xij_aor=(0.d0,0.d0)
c
c These will correspond to the vegas x's for the FKS variables xi_i,
c y_ij and phi_i (changing this also requires changing folding parameters)
      ixEi=ndim-2
      ixyij=ndim-1
      ixpi=ndim
c
      imother=min(j_fks,i_fks)
      m_j_fks=pmass(j_fks)
c
c For final state j_fks, compute the recoil invariant mass
      if (j_fks.gt.nincoming) then
         call get_recoil(p_born_l,imother,shat_born,xmrec2,pass)
         if (.not.pass) then
            xjac0=-44
            goto 222
         endif
      endif

c Here is the beginning of the loop over the momenta for the event and
c counter-events. This will fill the xp momenta with the event and
c counter-event momenta.
 111  continue
      xjac   = xjac0
      xpswgt = xpswgt0
c
c Put the Born momenta in the xp momenta, making sure that the mapping
c is correct; put i_fks momenta equal to zero.
      do i=1,nexternal
         if(i.lt.i_fks) then
            do j=0,3
               xp(j,i)=p_born_l(j,i)
            enddo
            m(i)=m_born(i)
         elseif(i.eq.i_fks) then
            do j=0,3
               xp(j,i)=0d0
            enddo
            m(i)=0d0
         elseif(i.ge.i_fks) then
            do j=0,3
               xp(j,i)=p_born_l(j,i-1)
            enddo
            m(i)=m_born(i-1)
         endif
      enddo
c
c set-up phi_i_fks
c
      phi_i_fks=2d0*pi*x(ixpi)
      xjac=xjac*2d0*pi
c To keep track of the special phase-space region with massive j_fks
      isolsign=0
c
c consider the three cases:
c case 1: j_fks is massless final state
c case 2: j_fks is massive final state
c case 3: j_fks is initial state
      if (j_fks.gt.nincoming) then
         shat=shat_born
         sqrtshat=sqrtshat_born
         tau=tau_born
         ycm=ycm_born
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
         if (m_j_fks.eq.0d0) then
            isolsign=1
            call generate_momenta_massless_final(icountevts,i_fks,j_fks
     &           ,p_born_l(0,imother),shat,sqrtshat,x(ixEi),xmrec2,xp
     &           ,phi_i_fks,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat
     &           ,p_i_fks,xjac,xpswgt,pass)
            if (.not.pass) goto 112
         elseif(m_j_fks.gt.0d0) then
            call generate_momenta_massive_final(icountevts,isolsign
     &           ,input_granny_m2,rat_xi,i_fks,j_fks,p_born_l(0,imother)
     &           ,shat,sqrtshat,m_j_fks,x(ixEi),xmrec2,xp,phi_i_fks
     &           ,xiimax,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac
     &           ,xpswgt,pass)
            if (.not.pass) goto 112
         endif
      elseif(j_fks.le.nincoming) then
         isolsign=1
         call generate_momenta_initial(icountevts,i_fks,j_fks,xbjrk_born
     &        ,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks,xp,x(ixEi)
     &        ,shat,stot,sqrtshat,tau,ycm,xbjrk,p_i_fks,xiimax,xinorm
     &        ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt,xjac ,pass)
         if (.not.pass) goto 112
      else
         write (*,*) 'Error #2 in genps_fks.f',j_fks
         stop
      endif
c At this point, the phase space lacks a factor xi_i_fks, which need be 
c excluded in an NLO computation according to FKS, being taken into 
c account elsewhere
c$$$      xpswgt=xpswgt*xi_i_fks
c
c All done, so check four-momentum conservation
      if(xjac.gt.0.d0)then
         call phspncheck_nocms(nexternal,sqrtshat,m,xp,pass)
         if (.not.pass) then
            xjac=-199
            goto 112
         endif
      endif
c      
      if(nincoming.eq.2)then
         flux  = 1d0 /(2.D0*SQRT(LAMBDA(shat,m(1)**2,m(2)**2)))
      else                      ! Decays
         flux = 1d0/(2d0*sqrtshat)
      endif
c The pi-dependent factor inserted below is due to the fact that the
c weight computed above is relevant to R_n, as defined in Kajantie's
c book, eq.(III.3.1), while we need the full n-body phase space
      flux  = flux / (2d0*pi)**(3 * (nexternal-nincoming) - 4)
c This extra pi-dependent factor is due to the fact that the phase-space
c part relevant to i_fks and j_fks does contain all the pi's needed for 
c the correct normalization of the phase space
      flux  = flux * (2d0*pi)**3
      pwgt=max(xjac*xpswgt,1d-99)
      xjac = pwgt*flux
c
 112  continue
c Catch the points for which there is no viable phase-space generation
c (still fill the common blocks with some information that is needed
c (e.g. ycm_cnt)).
      if (xjac .le. 0d0 ) then
         xp(0,1)=-99d0
      endif
c
c Fill common blocks
      if (icountevts.eq.-100) then
         tau_ev=tau
         ycm_ev=ycm
         shat_ev=shat
         sqrtshat_ev=sqrtshat
         xbjrk_ev(1)=xbjrk(1)
         xbjrk_ev(2)=xbjrk(2)
         xiimax_ev=xiimax
         xinorm_ev=xinorm
         xi_i_fks_ev=xi_i_fks
         xi_i_hat_ev=xi_i_hat
         do i=0,3
            p_i_fks_ev(i)=p_i_fks(i)
         enddo
         y_ij_fks_ev=y_ij_fks
         do i=1,nexternal
            do j=0,3
               p(j,i)=xp(j,i)
               p_ev(j,i)=xp(j,i)
            enddo
         enddo
         jac=xjac
      else
         tau_cnt(icountevts)=tau
c Special fix in the case the soft counter-events are not generated but
c the Born and real are. (This can happen if ptj>0 in the
c run_card). This fix is needed for set_cms_stuff to work properly.
         if (icountevts.eq.0) then
            ycm=ycm_born
         endif
         ycm_cnt(icountevts)=ycm
         shat_cnt(icountevts)=shat
         sqrtshat_cnt(icountevts)=sqrtshat
         xbjrk_cnt(1,icountevts)=xbjrk(1)
         xbjrk_cnt(2,icountevts)=xbjrk(2)
         xiimax_cnt(icountevts)=xiimax
         xinorm_cnt(icountevts)=xinorm
         xi_i_fks_cnt(icountevts)=xi_i_fks
         xi_i_hat_cnt(icountevts)=xi_i_hat
         do i=0,3
            p_i_fks_cnt(i,icountevts)=p_i_fks(i)
         enddo
         do i=1,nexternal
            do j=0,3
               p1_cnt(j,i,icountevts)=xp(j,i)
            enddo
         enddo
         jac_cnt(icountevts)=xjac
c the following two are obsolete, but still part of some common block:
c so give some non-physical values
         wgt_cnt(icountevts)=-1d99
         pswgt_cnt(icountevts)=-1d99
      endif
c
      if(icountevts.eq.-100)then
         if( (j_fks.eq.1.or.j_fks.eq.2).and.fks_as_is )then
            icountevts=-2
         else
            icountevts=0
         endif
c skips counterevents when integrating over second fold for massive
c j_fks
         if( isolsign.eq.-1 )icountevts=5
         if (only_event_phsp) return
      else
         icountevts=icountevts+1
      endif
      if( (icountevts.le.2.and.m_j_fks.eq.0.d0.and.(.not.nbody)).or.
     &    (icountevts.eq.0.and.m_j_fks.eq.0.d0.and.nbody) .or.
     &    (icountevts.eq.0.and.m_j_fks.ne.0.d0) )then
         goto 111
      elseif(icountevts.eq.5) then
c icountevts=5 only when integrating over the second fold with j_fks
c massive. The counterevents have been skipped, so make sure their
c momenta are unphysical. Born are physical if event was generated, and
c must stay so for the computation of enhancement factors.
         do i=0,2
            jac_cnt(i)=-299
            p1_cnt(0,1,i)=-99
         enddo
      endif
      nocntevents=(jac_cnt(0).le.0.d0) .and.
     &            (jac_cnt(1).le.0.d0) .and.
     &            (jac_cnt(2).le.0.d0)
      call xmom_compare(i_fks,j_fks,jac,jac_cnt,p,p1_cnt,
     &                  p_i_fks_ev,p_i_fks_cnt,
     &                  xi_i_fks_ev,y_ij_fks_ev,check_cnt)
c check_cnt=.false. is an exceedingly rare situation -- just dump the event
      if(.not.check_cnt)goto 222
c
c If all went well, we are done and can exit
c
      return
c
 222  continue
c
c Born momenta have not been generated. Neither events nor counterevents exist.
c Set all to negative values and exit
      jac=-222
      jac_cnt(0)=-222
      jac_cnt(1)=-222
      jac_cnt(2)=-222
      p(0,1)=-99
      do i=-2,2
        p1_cnt(0,1,i)=-99
      enddo
      if (.not.only_event_phsp) p_born(0,1)=-99
      nocntevents=.true.
      return
      end
      

      subroutine generate_momenta_massless_final(icountevts,i_fks,j_fks
     &     ,p_born_imother,shat,sqrtshat,x,xmrec2,xp,phi_i_fks,xiimax
     &     ,xinorm,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt
     &     ,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision E_i_fks,x3len_i_fks,x3len_j_fks,x3len_fks_mother
     &     ,costh_i_fks,sinth_i_fks,xpifksred(0:3),th_mother_fks
     &     ,costh_mother_fks,sinth_mother_fks, phi_mother_fks
     &     ,cosphi_mother_fks,sinphi_mother_fks,recoil(0:3),sumrec
     &     ,sumrec2,betabst,gammabst,shybst,chybst,chybstmo,xdir(3)
     &     ,veckn,veckbarn,xp_mother(0:3),cosphi_i_fks
     &     ,sinphi_i_fks,xiimax_save
      save xiimax_save
      double complex resAoR0
      common /virtgranny_boost/shybst,chybst,chybstmo
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      double precision stiny,sstiny,qtiny,ctiny,cctiny
      double complex ximag
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (ctiny=5d-7)
      parameter (ximag=(0d0,1d0))
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
c
c set-up y_ij_fks
c
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &             (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks=y_ij_fks_fix
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
      else
         write(*,*)'Error #3 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(2)*2d0

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute maximum allowed xi_i_fks
      xiimax=1-xmrec2/shat
      xinorm=xiimax
c
c Define xi_i_fks
c
      if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &     (.not.softtest)  )then
         if(icountevts.eq.-100)then
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->0
            xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         endif
c in the case of counter events, xi_i_hat is an input to this function
         xi_i_fks=xi_i_hat*xiimax
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
c This is to keep xi_i_hat, rather than xi_i, fixed in the tests.
c Changed in the context of granny stuff       
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix*xiimax
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        softtest )then
         if(xi_i_fks_fix.lt.1d0)then
            xi_i_fks=xi_i_fks_fix*xiimax
         else
            xjac=-102
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=xi_i_fks_matrix(icountevts)
      else
         write(*,*)'Error #4 in genps_fks.f',icountevts
         stop
      endif
c remove the following if no importance sampling towards soft
c singularity is performed when integrating over xi_i_hat
      xjac=xjac*2d0*x(1)

c Check that xii is in the allowed range
      if( icountevts.eq.-100 .or. abs(icountevts).eq.1 )then
         if(xi_i_fks.gt.(1-xmrec2/shat))then
            xjac=-101
            pass=.false.
            return
         endif
      elseif(icountevts.eq.0 .or. abs(icountevts).eq.2)then
c May insert here a check on whether xii<xicut, rather than doing it 
c in the cross sections
         continue
      endif
c
c Compute costh_i_fks from xi_i_fks et al.
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      x3len_j_fks=(shat-xmrec2-2*sqrtshat*x3len_i_fks)/
     &             (2*(sqrtshat-x3len_i_fks*(1-y_ij_fks)))
      x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     &                       2*x3len_i_fks*x3len_j_fks*y_ij_fks )
      if(xi_i_fks.lt.qtiny)then
         costh_i_fks=y_ij_fks+shat*(1-y_ij_fks**2)*xi_i_fks/
     &                                          (shat-xmrec2)
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
      elseif(1-y_ij_fks.lt.qtiny)then
         costh_i_fks=1-(shat*(1-xi_i_fks)-xmrec2)**2*(1-y_ij_fks)/
     &                                          (shat-xmrec2)**2
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=1.d0
      else
         costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)
     &               /(2*x3len_fks_mother*x3len_i_fks)
         if(abs(costh_i_fks).gt.1.d0)then
            if(abs(costh_i_fks).le.(1.d0+1.d-5))then
               costh_i_fks=sign(1.d0,costh_i_fks)
            else
               write(*,*)'Fatal error #5 in one_tree',
     &              costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
               stop
            endif
         endif
      endif
      sinth_i_fks=sqrt(1-costh_i_fks**2)
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c The momentum if i_fks and j_fks
c
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c  
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be 
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
      sumrec=recoil(0)+rho(recoil)
      sumrec2=sumrec**2
      betabst=-(shat-sumrec2)/(shat+sumrec2)
      gammabst=1/sqrt(1-betabst**2)
      shybst=-(shat-sumrec2)/(2*sumrec*sqrtshat)
      chybst=(shat+sumrec2)/(2*sumrec*sqrtshat)
c cosh(y) is very often close to one, so define cosh(y)-1 as well
      chybstmo=(sqrtshat-sumrec)**2/(2*sumrec*sqrtshat)
      do j=1,3
         xdir(j)=xp_mother(j)/x3len_fks_mother
      enddo
c Perform the boost here
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Collinear limit of <ij>/[ij]. See innerp3.m. 
      if( ( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) ) )then
         resAoR0=-exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c The term O(srt(1-y)) is formally correct but may be numerically large
c Set it to zero
c$$$          resAoR5=-ximag*sqrt(2.d0)*
c$$$       &          sinphi_i_fks*tan(th_mother_fks/2.d0)*
c$$$       &          exp( 2*ximag*(phi_mother_fks+phi_i_fks) )
c$$$          xij_aor=resAoR0+resAoR5*sqrt(1-y_ij_fks)
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif 
c
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)
      return
      end

      subroutine generate_momenta_massive_final(icountevts,isolsign
     &     ,input_granny_m2,rat_xi,i_fks,j_fks,p_born_imother,shat
     &     ,sqrtshat,m_j_fks,x,xmrec2,xp,phi_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,p_i_fks,xjac,xpswgt,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks,isolsign
      double precision shat,sqrtshat,x(2),xmrec2,xp(0:3,nexternal)
     &     ,y_ij_fks,p_born_imother(0:3),m_j_fks,phi_i_fks,xi_i_hat
      double precision xiimax,xinorm,xi_i_fks,p_i_fks(0:3),xjac,xpswgt
      logical pass,input_granny_m2
c common blocks
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j
      double precision xmj,xmj2,xmjhat,xmhat,xim,cffA2,cffB2,cffC2
     $     ,cffDEL2,xiBm,ximax,xirplus,xirminus,rat_xi,xjactmp,xitmp1
     $     ,E_i_fks,x3len_i_fks,b2m4ac,x3len_j_fks_num,x3len_j_fks_den
     $     ,x3len_j_fks,x3len_fks_mother,costh_i_fks,sinth_i_fks
     $     ,xpifksred(0:3),recoil(0:3),xp_mother(0:3),sumrec,expybst
     $     ,shybst,chybst,chybstmo,xdir(3),veckn,veckbarn ,cosphi_i_fks
     $     ,sinphi_i_fks,cosphi_mother_fks,costh_mother_fks
     $     ,phi_mother_fks,sinphi_mother_fks,th_mother_fks,xitmp2
     $     ,sinth_mother_fks
      save xjactmp
      common /virtgranny_boost/shybst,chybst,chybstmo
c external
      double precision rho
      external rho
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      double precision stiny,sstiny,qtiny,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (ctiny=5d-7)
c
      if(colltest .or.
     &     abs(icountevts).eq.1.or.abs(icountevts).eq.2)then
         write(*,*)'Error #5 in genps_fks.f:'
         write(*,*)
     &        'This parametrization cannot be used in FS coll limits'
         stop
      endif
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
c
c set-up y_ij_fks
c
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &             (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = -2d0*(cctiny+(1-cctiny)*x(2)**2)+1d0
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks=y_ij_fks_fix
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
      else
         write(*,*)'Error #6 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*2d0*x(2)*2d0

      call getangles(p_born_imother,
     &     th_mother_fks,costh_mother_fks,sinth_mother_fks,
     &     phi_mother_fks,cosphi_mother_fks,sinphi_mother_fks)
c
c Compute the maximum allowed xi_i_fks
c
      xmj=m_j_fks
      xmj2=xmj**2
      xmjhat=xmj/sqrtshat
      xmhat=sqrt(xmrec2)/sqrtshat
      xim=(1-xmhat**2-2*xmjhat+xmjhat**2)/(1-xmjhat)
      cffA2=1-xmjhat**2*(1-y_ij_fks**2)
      cffB2=-2*(1-xmhat**2-xmjhat**2)
      cffC2=(1-(xmhat-xmjhat)**2)*(1-(xmhat+xmjhat)**2)
      cffDEL2=cffB2**2-4*cffA2*cffC2
      xiBm=(-cffB2-sqrt(cffDEL2))/(2*cffA2)
      ximax=1-(xmhat+xmjhat)**2
      if(xiBm.lt.(xim-1.d-8).or.xim.lt.0.d0.or.xiBm.lt.0.d0.or.
     &     xiBm.gt.(ximax+1.d-8).or.ximax.gt.1.or.ximax.lt.0.d0)then
         write(*,*)'WARNING #4 in one_tree',xim,xiBm,ximax
         xjac=-104d0
         pass=.false.
         return
      endif
      if(y_ij_fks.ge.0.d0)then
         xirplus=xim
         xirminus=0.d0
      else
         xirplus=xiBm
         xirminus=xiBm-xim
      endif
      xiimax=xirplus
      xinorm=xirplus+xirminus
c If inputing the granny mass, do not update rat_xi, but use the
c inputted rat_xi instead.
      if (.not.input_granny_m2) then
         rat_xi=xiimax/xinorm
      endif
c
c Generate xi_i_fks
c
      if( icountevts.eq.-100 .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &      (.not.softtest)  )then
         xjactmp=1.d0
         xitmp1=x(1)
c Map regions (0,A) and (A,1) in xitmp1 onto regions (0,rat_xi) and (rat_xi,1)
c in xi_i_hat respectively. The parameter A is free, but it appears to be 
c convenient to choose A=rat_xi
         if(xitmp1.le.rat_xi)then
            xitmp1=xitmp1/rat_xi
            xjactmp=xjactmp/rat_xi
c importance sampling towards soft singularity
c insert here further importance samplings
            xitmp2=sstiny+(1-sstiny)*xitmp1**2
            xjactmp=xjactmp*2*xitmp1
            xi_i_hat=xitmp2*rat_xi
            xjactmp=xjactmp*rat_xi
            xi_i_fks=xinorm*xi_i_hat
            isolsign=1
         else
c insert here further importance samplings
            xi_i_hat=xitmp1
            xi_i_fks=-xinorm*xi_i_hat+2*xiimax
            isolsign=-1
         endif
      elseif( icountevts.eq.-100 .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
         xjactmp=1.d0
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
         isolsign=1
      elseif( (icountevts.eq.-100) .and.
     &        softtest )then
         xjactmp=1.d0
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xjac=-102
            pass=.false.
            return
         endif
         isolsign=1
      elseif(icountevts.eq.0)then
c Don't set xjactmp here, because we should use the same as what was
c used for the (real-emission) event
         xi_i_fks=xi_i_fks_matrix(icountevts)
         isolsign=1
      else
         write(*,*)'Error #7 in genps_fks.f',icountevts
         stop
      endif
      xjac=xjac*xjactmp
c
      if(isolsign.eq.0)then
         write(*,*)'Fatal error #11 in one_tree',isolsign
         stop
      endif
c
c Compute costh_i_fks
c
      E_i_fks=xi_i_fks*sqrtshat/2d0
      x3len_i_fks=E_i_fks
      b2m4ac=xi_i_fks**2*cffA2 + xi_i_fks*cffB2 + cffC2
      if(b2m4ac.le.0.d0)then
         if(abs(b2m4ac).lt.1.d-3)then
            b2m4ac=0.d0
         else
            write(*,*)'Fatal error #6 in one_tree'
            write(*,*)b2m4ac,xi_i_fks,cffA2,cffB2,cffC2
            write(*,*)y_ij_fks,xim,xiBm
            stop
         endif
      endif
      x3len_j_fks_num=-xi_i_fks*y_ij_fks*
     &                (1-xmhat**2+xmjhat**2-xi_i_fks) +
     &                (2-xi_i_fks)*sqrt(b2m4ac)*isolsign
      x3len_j_fks_den=(2-xi_i_fks*(1-y_ij_fks))*
     &                (2-xi_i_fks*(1+y_ij_fks))
      x3len_j_fks=sqrtshat*x3len_j_fks_num/x3len_j_fks_den
      if(x3len_j_fks.lt.0.d0)then
         write(*,*)'WARNING #7 in one_tree',
     &        x3len_j_fks_num,x3len_j_fks_den,xi_i_fks,y_ij_fks
         xjac=-107d0
         pass=.false.
         return
      endif
      x3len_fks_mother=sqrt( x3len_i_fks**2+x3len_j_fks**2+
     &                       2*x3len_i_fks*x3len_j_fks*y_ij_fks )
      if(xi_i_fks.lt.qtiny)then
         costh_i_fks=y_ij_fks+(1-y_ij_fks**2)*xi_i_fks/sqrt(cffC2)
         if(abs(costh_i_fks).gt.1.d0)costh_i_fks=y_ij_fks
      else
         costh_i_fks=(x3len_fks_mother**2-x3len_j_fks**2+x3len_i_fks**2)
     $        /(2*x3len_fks_mother*x3len_i_fks)
         if(abs(costh_i_fks).gt.1.d0+qtiny)then
            write(*,*)'Fatal error #8 in one_tree',
     &           costh_i_fks,xi_i_fks,y_ij_fks,xmrec2
            stop
         elseif(abs(costh_i_fks).gt.1.d0)then
            costh_i_fks = sign(1d0,costh_i_fks)
         endif
      endif
      sinth_i_fks=sqrt(1-costh_i_fks**2)
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks
      xpifksred(2)=sinth_i_fks*sinphi_i_fks
      xpifksred(3)=costh_i_fks
c
c Generate momenta for j_fks and i_fks
c     
      xp(0,i_fks)=E_i_fks
      xp(0,j_fks)=sqrt(x3len_j_fks**2+m_j_fks**2)
      p_i_fks(0)=sqrtshat/2d0
      do j=1,3
         p_i_fks(j)=sqrtshat/2d0*xpifksred(j)
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         if(j.ne.3)then
            xp(j,j_fks)=-xp(j,i_fks)
         else
            xp(j,j_fks)=x3len_fks_mother-xp(j,i_fks)
         endif
      enddo
c  
      call rotate_invar(xp(0,i_fks),xp(0,i_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(xp(0,j_fks),xp(0,j_fks),
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
      call rotate_invar(p_i_fks,p_i_fks,
     &                  costh_mother_fks,sinth_mother_fks,
     &                  cosphi_mother_fks,sinphi_mother_fks)
c
c Now the xp four vectors of all partons except i_fks and j_fks will be 
c boosted along the direction of the mother; start by redefining the
c mother four momenta
      do i=0,3
         xp_mother(i)=xp(i,i_fks)+xp(i,j_fks)
         if (nincoming.eq.2) then
            recoil(i)=xp(i,1)+xp(i,2)-xp_mother(i)
         else
            recoil(i)=xp(i,1)-xp_mother(i)
         endif
      enddo
c
      sumrec=recoil(0)+rho(recoil)
      if(xmrec2.lt.1.d-16*shat)then
         expybst=sqrtshat*sumrec/(shat-xmj2)*
     &           (1+xmj2*xmrec2/(shat-xmj2)**2)
      else
         expybst=sumrec/(2*sqrtshat*xmrec2)*
     &           (shat+xmrec2-xmj2-shat*sqrt(cffC2))
      endif
      if(expybst.le.0.d0)then
         write(*,*)'Fatal error #10 in one_tree',expybst
         stop
      endif
      shybst=(expybst-1/expybst)/2.d0
      chybst=(expybst+1/expybst)/2.d0
      chybstmo=chybst-1.d0
c
      do j=1,3
         xdir(j)=xp_mother(j)/x3len_fks_mother
      enddo
c Boost the momenta
      do i=nincoming+1,nexternal
         if(i.ne.i_fks.and.i.ne.j_fks.and.shybst.ne.0.d0)
     &      call boostwdir2(chybst,shybst,chybstmo,xdir,xp(0,i),xp(0,i))
      enddo
c
c Phase-space factor for (xii,yij,phii)
      veckn=rho(xp(0,j_fks))
      veckbarn=rho(p_born_imother)
c
c Qunatities to be passed to montecarlocounter (event kinematics)
      if(icountevts.eq.-100)then
         veckn_ev=veckn
         veckbarn_ev=veckbarn
         xp0jfks=xp(0,j_fks)
      endif 
c
      xpswgt=xpswgt*2*shat/(4*pi)**3*veckn/veckbarn/
     &     ( 2-xi_i_fks*(1-xp(0,j_fks)/veckn*y_ij_fks) )
      xpswgt=abs(xpswgt)
      return
      end


      subroutine generate_momenta_initial(icountevts,i_fks,j_fks,
     &     xbjrk_born,tau_born,ycm_born,ycmhat,shat_born,phi_i_fks ,xp,x
     &     , shat,stot,sqrtshat,tau,ycm,xbjrk ,p_i_fks,xiimax,xinorm
     &     ,xi_i_fks,y_ij_fks,xi_i_hat,xpswgt ,xjac ,pass)
      implicit none
      include 'nexternal.inc'
c arguments
      integer icountevts,i_fks,j_fks
      double precision xbjrk_born(2),tau_born,ycm_born,ycmhat,shat_born
     &     ,phi_i_fks,xpswgt,xjac,xiimax,xinorm,xp(0:3,nexternal),stot
     &     ,x(2),y_ij_fks,xi_i_hat
      double precision shat,sqrtshat,tau,ycm,xbjrk(2),p_i_fks(0:3)
      logical pass
c common blocks
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      double precision  veckn_ev,veckbarn_ev,xp0jfks
      common/cgenps_fks/veckn_ev,veckbarn_ev,xp0jfks
      double complex xij_aor
      common/cxij_aor/xij_aor
      logical softtest,colltest
      common/sctests/softtest,colltest
      double precision xi_i_fks_fix,y_ij_fks_fix
      common/cxiyfix/xi_i_fks_fix,y_ij_fks_fix
c local
      integer i,j,idir
      double precision yijdir,costh_i_fks,x1bar2,x2bar2,yij_sol,xi1,xi2
     $     ,ximaxtmp,omega,bstfact,shy_tbst,chy_tbst,chy_tbstmo
     $     ,xdir_t(3),cosphi_i_fks,sinphi_i_fks,shy_lbst,chy_lbst
     $     ,encmso2,E_i_fks,sinth_i_fks,xpifksred(0:3),xi_i_fks
     $     ,xiimin,yij_upp,yij_low,y_ij_fks_upp,y_ij_fks_low
      double complex resAoR0
c external
c
c parameters
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      double precision xi_i_fks_matrix(-2:2)
      data xi_i_fks_matrix/0.d0,-1.d8,0.d0,-1.d8,0.d0/
      double precision y_ij_fks_matrix(-2:2)
      data y_ij_fks_matrix/-1.d0,-1.d0,-1.d8,1.d0,1.d0/
      logical fks_as_is
      parameter (fks_as_is=.false.)
      double complex ximag
      parameter (ximag=(0d0,1d0))
      double precision stiny,sstiny,qtiny,zero,ctiny,cctiny
      parameter (stiny=1d-6)
      parameter (qtiny=1d-7)
      parameter (zero=0d0)
      parameter (ctiny=5d-7)
c
      pass=.true.
      if(softtest)then
        sstiny=0.d0
      else
        sstiny=stiny
      endif
c
c FKS for left or right incoming parton
c
      idir=0
      if(.not.fks_as_is)then
         if(j_fks.eq.1)then
            idir=1
         elseif(j_fks.eq.2)then
            idir=-1
         endif
      else
         idir=1
         write(*,*)'One_tree: option not checked'
         stop
      endif
c
c set-up lower and upper bounds on y_ij_fks
c
      if( tau_born.le.tau_lower_bound .and.ycm_born.gt.
     &         (0.5d0*log(tau_born)-log(tau_lower_bound)) )then
         yij_upp= (tau_lower_bound+tau_born)*
     &        ( 1-exp(2*ycm_born)*tau_lower_bound ) /
     &                  ( (tau_lower_bound-tau_born)*
     &                    (1+exp(2*ycm_born)*tau_lower_bound) )
      else
         yij_upp=1.d0
      endif
      if( tau_born.le.tau_lower_bound .and. ycm_born.lt.
     &        (-0.5d0*log(tau_born)+log(tau_lower_bound)) )then
         yij_low=-(tau_lower_bound+tau_born)*
     &        ( 1-exp(-2*ycm_born)*tau_lower_bound ) / 
     &                   ( (tau_lower_bound-tau_born)*
     &                     (1+exp(-2*ycm_born)*tau_lower_bound) )
      else
         yij_low=-1.d0
      endif
c
      if(idir.eq.1)then
         y_ij_fks_upp=yij_upp
         y_ij_fks_low=yij_low
      elseif(idir.eq.-1)then
         y_ij_fks_upp=-yij_low
         y_ij_fks_low=-yij_upp
      endif
      
c
c set-up y_ij_fks
c
      if(colltest)then
        cctiny=0.d0
      else
        cctiny=ctiny
      endif
      if( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &     ((.not.softtest) .or. 
     &            (softtest.and.y_ij_fks_fix.eq.-2.d0)) .and.
     &     (.not.colltest)  )then
c importance sampling towards collinear singularity
c insert here further importance sampling towards y_ij_fks->1
         y_ij_fks = y_ij_fks_upp -
     &        (y_ij_fks_upp-y_ij_fks_low)*(cctiny+(1-cctiny)*x(2)**2)
      elseif( (icountevts.eq.-100.or.icountevts.eq.0) .and.
     &        ((softtest.and.y_ij_fks_fix.ne.-2.d0) .or.
     &          colltest)  )then
         y_ij_fks = y_ij_fks_fix
         if ( y_ij_fks.gt.y_ij_fks_upp+1d-12 .or.
     &        y_ij_fks.lt.y_ij_fks_low-1d-12) then
            xjac=-33d0
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.abs(icountevts).eq.1)then
         y_ij_fks=y_ij_fks_matrix(icountevts)
c Check that y_ij_fks is in the allowed range. If not, counter events
c cannot be generated
         if ( y_ij_fks.gt.y_ij_fks_upp+1d-12 .or.
     &        y_ij_fks.lt.y_ij_fks_low-1d-12) then
            xjac=-33d0
            pass=.false.
            return
         endif
      else
         write(*,*)'Error #8 in genps_fks.f',icountevts
         stop
      endif
c importance sampling towards collinear singularity
      xjac=xjac*(y_ij_fks_upp-y_ij_fks_low)*x(2)*2d0
c
c Compute costh_i_fks
c
      yijdir=idir*y_ij_fks
      costh_i_fks=yijdir
c
c Compute maximal xi_i_fks
c
      x1bar2=xbjrk_born(1)**2
      x2bar2=xbjrk_born(2)**2
      if(1-tau_born.gt.1.d-5)then
         yij_sol=-sinh(ycm_born)*(1+tau_born)/
     &            ( cosh(ycm_born)*(1-tau_born) )
      else
         yij_sol=-ycmhat
      endif
      if(abs(yij_sol).gt.1.d0)then
         write(*,*)'Error #9 in genps_fks.f',yij_sol,icountevts
         write(*,*)xbjrk_born(1),xbjrk_born(2),yijdir
      endif
      if(yijdir.ge.yij_sol)then
         xi1=2*(1+yijdir)*x1bar2/(
     &        sqrt( ((1+x1bar2)*(1-yijdir))**2+16*yijdir*x1bar2 ) +
     &        (1-yijdir)*(1-x1bar2) )
         ximaxtmp=1-xi1
      elseif(yijdir.lt.yij_sol)then
         xi2=2*(1-yijdir)*x2bar2/(
     &        sqrt( ((1+x2bar2)*(1+yijdir))**2-16*yijdir*x2bar2 ) +
     &        (1+yijdir)*(1-x2bar2) )
         ximaxtmp=1-xi2
      else
         write(*,*)'Fatal error #14 in one_tree: unknown option'
         write(*,*)y_ij_fks,yij_sol,idir
         stop
      endif
      xiimax=ximaxtmp
c
c Lower bound on xi_i_fks
c
      if (tau_born.lt.tau_lower_bound) then
         xiimin=1d0-tau_born/tau_lower_bound
      else
         xiimin=0d0
      endif
      if (xiimax.lt.xiimin) then
         write (*,*) 'WARNING #10 in genps_fks.f',icountevts,xiimax
     $        ,xiimin
         xjac=-342d0
         pass=.false.
         return
      endif

      xinorm=xiimax-xiimin
      if( icountevts.ge.1 .and.
     &     ( (idir.eq.1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(1))).gt.1.d-5) .or.
     &     (idir.eq.-1.and.
     &     abs(ximaxtmp-(1-xbjrk_born(2))).gt.1.d-5) ) )then 
         write(*,*)'Fatal error #15 in one_tree'
         write(*,*)ximaxtmp,xbjrk_born(1),xbjrk_born(2),idir
         stop
      endif
c
c Define xi_i_fks
c
      if( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &     ((.not.colltest) .or. 
     &     (colltest.and.xi_i_fks_fix.eq.-2.d0)) .and.
     &     (.not.softtest)  )then
         if(icountevts.eq.-100)then
c importance sampling towards soft singularity
c insert here further importance sampling towards xi_i_hat->0
            xi_i_hat=sstiny+(1-sstiny)*x(1)**2
         endif
         xi_i_fks=xiimin+(xiimax-xiimin)*xi_i_hat
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        (colltest.and.xi_i_fks_fix.ne.-2.d0) .and.
     &        (.not.softtest)  )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xi_i_fks=xi_i_fks_fix*xiimax
         endif
      elseif( (icountevts.eq.-100.or.abs(icountevts).eq.1) .and.
     &        softtest )then
         if(xi_i_fks_fix.lt.xiimax)then
            xi_i_fks=xi_i_fks_fix
         else
            xjac=-102
            pass=.false.
            return
         endif
      elseif(abs(icountevts).eq.2.or.icountevts.eq.0)then
         xi_i_fks=xi_i_fks_matrix(icountevts)
c Check that xi_i_fks is in the allowed range. If not, counter events
c cannot be generated
         if ( xi_i_fks.gt.xiimax+1d-12 .or.
     &        xi_i_fks.lt.xiimin-1d-12 ) then
            xjac=-34d0
            pass=.false.
            return
         endif
      else
         write(*,*)'Error #11 in genps_fks.f',icountevts
         stop
      endif
c remove the following if no importance sampling towards soft
c singularity is performed when integrating over xi_i_hat
      xjac=xjac*2d0*x(1)
c
c Initial state variables are different for events and counterevents. Update them here.
c
      omega=sqrt( (2-xi_i_fks*(1+yijdir))/
     &     (2-xi_i_fks*(1-yijdir)) )
      if (icountevts.ne.0) then
         tau=tau_born/(1-xi_i_fks)
         ycm=ycm_born-log(omega)
         shat=tau*stot
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)/(sqrt(1-xi_i_fks)*omega)
         xbjrk(2)=xbjrk_born(2)*omega/sqrt(1-xi_i_fks)
      else
         tau=tau_born
         ycm=ycm_born
         shat=shat_born
         sqrtshat=sqrt(shat)
         xbjrk(1)=xbjrk_born(1)
         xbjrk(2)=xbjrk_born(2)
      endif
c
c Define the boost factor here
c
      bstfact=sqrt( (2-xi_i_fks*(1-yijdir))*(2-xi_i_fks*(1+yijdir)) )
      shy_tbst=-xi_i_fks*sqrt(1-yijdir**2)/(2*sqrt(1-xi_i_fks))
      chy_tbst=bstfact/(2*sqrt(1-xi_i_fks))
      chy_tbstmo=chy_tbst-1.d0
      cosphi_i_fks=cos(phi_i_fks)
      sinphi_i_fks=sin(phi_i_fks)
      xdir_t(1)=-cosphi_i_fks
      xdir_t(2)=-sinphi_i_fks
      xdir_t(3)=zero
c
      shy_lbst=-xi_i_fks*yijdir/bstfact
      chy_lbst=(2-xi_i_fks)/bstfact
c Boost the momenta
      do i=3,nexternal
         if(i.ne.i_fks.and.shy_tbst.ne.0.d0)
     &        call boostwdir2(chy_tbst,shy_tbst,chy_tbstmo,xdir_t,
     &                        xp(0,i),xp(0,i))
      enddo
c
      encmso2=sqrtshat/2.d0
      p_i_fks(0)=encmso2
      E_i_fks=xi_i_fks*encmso2
      sinth_i_fks=sqrt(1-costh_i_fks**2)
c
      xp(0,1)=encmso2*(chy_lbst-shy_lbst)
      xp(1,1)=0.d0
      xp(2,1)=0.d0
      xp(3,1)=xp(0,1)
c
      xp(0,2)=encmso2*(chy_lbst+shy_lbst)
      xp(1,2)=0.d0
      xp(2,2)=0.d0
      xp(3,2)=-xp(0,2)
c
      xp(0,i_fks)=E_i_fks*(chy_lbst-shy_lbst*yijdir)
      p_i_fks(0)=p_i_fks(0)*(chy_lbst-shy_lbst*yijdir)
      xpifksred(1)=sinth_i_fks*cosphi_i_fks    
      xpifksred(2)=sinth_i_fks*sinphi_i_fks    
      xpifksred(3)=chy_lbst*yijdir-shy_lbst
c
      do j=1,3
         xp(j,i_fks)=E_i_fks*xpifksred(j)
         p_i_fks(j)=encmso2*xpifksred(j)
      enddo
c
c Collinear limit of <ij>/[ij]. See innerpin.m. 
      if( icountevts.eq.-100 .or.
     &     (icountevts.eq.1.and.xij_aor.eq.0) )then
         resAoR0=-exp( 2*idir*ximag*phi_i_fks )
         xij_aor=resAoR0
      endif
c
c Phase-space factor for (xii,yij,phii) * (tau,ycm)
      xpswgt=xpswgt*shat
      xpswgt=xpswgt/(4*pi)**3/(1-xi_i_fks)
      xpswgt=abs(xpswgt)
c
      return
      end

         
      subroutine getangles(pin,th,cth,sth,phi,cphi,sphi)
      implicit none
      real*8 pin(0:3),th,cth,sth,phi,cphi,sphi,xlength
c
      xlength=pin(1)**2+pin(2)**2+pin(3)**2
      if(xlength.eq.0)then
        th=0.d0
        cth=1.d0
        sth=0.d0
        phi=0.d0
        cphi=1.d0
        sphi=0.d0
      else
        xlength=sqrt(xlength)
        cth=pin(3)/xlength
        th=acos(cth)
        if(cth.ne.1.d0)then
          sth=sqrt(1-cth**2)
          phi=atan2(pin(2),pin(1))
          cphi=cos(phi)
          sphi=sin(phi)
        else
          sth=0.d0
          phi=0.d0
          cphi=1.d0
          sphi=0.d0
        endif
      endif
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
            write(*,*) 'Warning #12 in genps_fks.f',pp
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
          write(6,*)'Error #1 in function Lambda:',s,ma2,mb2
          stop
        endif
        rat=1-(sqrt(ma2)+sqrt(mb2))/s
        if(rat.gt.-tiny)then
          tmp=0.d0
        else
          write(6,*)'Error #2 in function Lambda:',s,ma2,mb2,rat
        endif
      endif
      LAMBDA=tmp
      RETURN
      END


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


      subroutine compute_tau_one_body(totmass,stot,tau,jac)
      implicit none
      double precision totmass,stot,tau,jac,roH
      roH=totmass**2/stot
      tau=roH
c Jacobian due to delta() of tau_born
      jac=jac*2*totmass/stot
      return
      end


      subroutine generate_tau_BW(stot,idim,x,mass,width,cBW,BWmass
     $     ,BWwidth,tau,jac)
      implicit none
      integer cBW,idim
      double precision stot,x,tau,jac,mass,width,BWmass(-1:1),BWwidth(
     $     -1:1),s_mass,s
      double precision smax,smin
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     &     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     &     ,tau_lower_bound_resonance,tau_lower_bound
      if (cBW.eq.1 .and. width.gt.0d0 .and. BWwidth(1).gt.0d0) then
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(5,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      else
         smin=tau_Born_lower_bound*stot
         smax=stot
         s_mass=smin
         call trans_x(3,idim,x,smin,smax,s_mass,mass,width,BWmass(
     $        -1),BWwidth(-1),jac,s)
         tau=s/stot
         jac=jac/stot
      endif
      return
      end


      subroutine generate_tau(stot,idim,x,tau,jac)
      implicit none
      integer idim
      double precision x,tau,jac,smin,smax,s_mass,s,tiny,dum,dum3(-1:1)
     $     ,stot
      parameter (tiny=1d-8)
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound
      smin=tau_born_lower_bound*stot
      smax=stot
      s_mass=tau_lower_bound_resonance*stot
      if (s_mass.gt.smin*(1d0+tiny)) then
         call trans_x(2,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      elseif(abs(s_mass-smin).lt.tiny*smin) then
         call trans_x(7,idim,x,smin,smax,s_mass,dum,dum
     $        ,dum3,dum3,jac,s)
      else
         write (*,*) 'ERROR #39 in genps_fks.f',s_mass,smin,smax
         jac=-1d0
      endif
      tau=s/stot
      jac=jac/stot
      return
      end


      subroutine generate_y(tau,x,ycm,ycmhat,jac)
      implicit none
      double precision tau,x,ycm,jac
      double precision ylim,ycmhat
      ylim=-0.5d0*log(tau)
      ycmhat=2*x-1
      ycm=ylim*ycmhat
      jac=jac*ylim*2
      return
      end

      
      subroutine compute_tau_y_epem(j_fks,one_body,fksmass,
     &                              stot,tau,ycm,ycmhat)
      implicit none
      include 'nexternal.inc'
      integer j_fks
      logical one_body
      double precision fksmass,stot,tau,ycm,ycmhat
      if(j_fks.le.nincoming)then
c This should never happen in normal integration: when no PDFs, j_fks
c cannot be initial state (but needed for testing). If tau set to one,
c integration range in xi_i_fks will be zero, so lower it artificially
c when too large
         if(one_body)then
            tau=fksmass**2/stot
         else
            tau=max((0.85d0)**2,fksmass**2/stot)
         endif
         ycm=0.d0
      else
c For e+e- collisions, set tau to one and y to zero
         tau=1.d0
         ycm=0.d0
      endif
      ycmhat=0.d0
      return
      end

      
      subroutine generate_inv_mass_sch(ns_channel,itree,m,sqrtshat_born
     $     ,totmass,qwidth,qmass,cBW,cBW_mass,cBW_width,s,x,xjac0,pass)
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      integer ns_channel
      double precision qmass(-nexternal:0),qwidth(-nexternal:0)
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision sqrtshat_born,totmass,xjac0
      integer itree(2,-max_branch:-1)
      integer i,j,ii,order(-nexternal:0)
      double precision smin,smax,totalmass
      logical pass
      integer cBW(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),cBW_width(-1:1,
     $     -nexternal:-1)
      double precision s_mass(-nexternal:nexternal)
      common/to_phase_space_s_channel/s_mass
      pass=.true.
      totalmass=totmass
      do ii = -1,-ns_channel,-1
c Randomize the order with which to generate the s-channel masses:
         call sChan_order(ns_channel,order)
         i=order(ii)
c     Generate invariant masses for all s-channel branchings of the Born
         smin = (m(itree(1,i))+m(itree(2,i)))**2
         smax = (sqrtshat_born-totalmass+sqrt(smin))**2
         if(smax.lt.smin.or.smax.lt.0.d0.or.smin.lt.0.d0)then
            write(*,*)'Error #13 in genps_fks.f'
            write(*,*)smin,smax,i
            stop
         endif
         call generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     &        ,qwidth,x,xjac0,s_mass)
c If numerical inaccuracy, quit loop
         if (xjac0 .lt. 0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31 in genps_fks.f',i,s(i),smin,smax
     $              ,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         if (s(i) .lt. smin) then
            write (*,*) 'WARNING #32 in genps_fks.f',i,s(i),smin,smax,x(
     $           -i)
            xjac0=-5
            pass=.false.
            return
         endif
c
c     fill masses, update totalmass
c
         m(i) = sqrt(s(i))
         totalmass=totalmass+m(i)-
     &        m(itree(1,i))-m(itree(2,i))
         if ( totalmass.gt.sqrtshat_born )then
            write (*,*) 'WARNING #33 in genps_fks.f',i,totalmass
     $           ,sqrtshat_born,s(i)
            xjac0 = -4
            pass=.false.
            return
         endif
      enddo
      return
      end


      subroutine generate_inv_mass_sch_granny(input_granny_m2,ns_channel
     &     ,itree,m,granny_m2_red,sqrtshat_born,totmass,qwidth,qmass,cBW
     &     ,cBW_mass,cBW_width,s,x,xjac0,pass)
c Identical to generate_inv_mass_sch, but that it generates the masses
c from the inside to the outside (i.e., first the largest one, and then
c the smaller ones). All the daughters of the granny are generated
c next-to-last, with granny itself as the final one.
      implicit none
      include 'genps.inc'
      include 'nexternal.inc'
      double precision qmass(-nexternal:0),qwidth(-nexternal:0),M(
     &     -max_branch:max_particles),x(99),s(-max_branch:max_particles)
     &     ,sqrtshat_born,totmass,xjac0,smin,smax,totalmass,min_m(
     &     -nexternal:nexternal),max_m(-nexternal:nexternal)
     &     ,granny_m2_red(-1:1)
      integer ns_channel,i,j,itree(2,-max_branch:-1),do_granny_daughters
      logical pass,start_s_chan(-nexternal:nexternal),input_granny_m2
      integer cBW_level_max,cBW(-nexternal:-1),cBW_level(-nexternal:-1)
      double precision cBW_mass(-1:1,-nexternal:-1),
     &     cBW_width(-1:1,-nexternal:-1)
      double precision s_mass(-nexternal:nexternal)
      common/to_phase_space_s_channel/s_mass
c Common block with granny information
      logical granny_is_res
      integer igranny,iaunt
      logical granny_chain(-nexternal:nexternal)
     &     ,granny_chain_real_final(-nexternal:nexternal)
      common /c_granny_res/igranny,iaunt,granny_is_res,granny_chain
     &     ,granny_chain_real_final
c
      totalmass=0d0
      do i=nexternal-1,-ns_channel,-1
         if (i.gt.0) then
            min_m(i)=m(i)
            if (i.gt.nincoming) totalmass=totalmass+m(i)
         elseif (i.lt.0) then
c     "Bare" integration ranges. 'max_m' will be updated below as soon
c     as invariant mass of mother has been generated.
            min_m(i)=min_m(itree(1,i))+min_m(itree(2,i))
            max_m(i)=sqrtshat_born-totalmass+min_m(i)
c     At the of the loop 'start_s_chan' is .true. only for s-channel
c     propagators attached directly to the t-channel.
            start_s_chan(i)=.true.
            start_s_chan(itree(1,i))=.false.
            start_s_chan(itree(2,i))=.false.
         else
            min_m(i)=0d0
            max_m(i)=sqrtshat_born
         endif
      enddo
c
c Generate the s-channel masses for everything.
c
      pass=.true.
      do do_granny_daughters=0,1
         do i = -ns_channel,-1
c Skip granny and its daughters if do_granny_daughters is 0,
c skip everything else if do_granny_daughters is 1
            if ( (do_granny_daughters.eq.0 .and. granny_chain(i)) .or.
     $           (do_granny_daughters.eq.1 .and. .not. granny_chain(i)))
     $           cycle
c Skip the granny
            if (i.eq.igranny) then
               if (do_granny_daughters.eq.1) then
c     once we have done all the masses except granny and daughters, we
c     have to update the integration ranges on imother and iaunt, using
c     the maximal allowed range, i.e. granny is as heavy as possible
                  if (itree(1,i).lt.0) then
                     max_m(itree(1,i))=max_m(i)-min_m(itree(2,i))
                  endif
                  if (itree(2,i).lt.0) then
                     max_m(itree(2,i))=max_m(i)-min_m(itree(1,i))
                  endif
               endif
               cycle
            endif
c Generate invariant masses for all s-channel branchings of the Born
            if ( max_m(i).lt.min_m(i) .or. max_m(i).lt.0.d0
     &           .or. min_m(i).lt.0.d0)then
               write(*,*) 'Error #13 in genps_fks.f (granny)'
               write(*,*) min_m(i),max_m(i),i
               stop
            endif
            smin = min_m(i)**2
            smax = max_m(i)**2
            call generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     &           ,qwidth,x,xjac0,s_mass)
c     If numerical inaccuracy, quit loop
            if ( xjac0.lt.0d0 .or.
     &           s(i) .lt. smin .or. s(i).gt.smax) then
               xjac0=-5
               pass=.false.
               return
            endif
c fill masses, update (upper) integration boundary for the next s-channel
            m(i) = sqrt(s(i))
c     update the range for the two daughters of the current s-channel
            if (itree(1,i).lt.0) then
               max_m(itree(1,i))=m(i)-min_m(itree(2,i))
            endif
            if (itree(2,i).lt.0) then
               max_m(itree(2,i))=m(i)-min_m(itree(1,i))
            endif
c     update the range for the sister
            do j=-ns_channel,i
               if (itree(1,j).eq.i .and.
     &              itree(2,j).lt.0 .and. itree(2,j).gt.i) then
c                 1st daughter of "j" is "i" --> 2nd is sister
                  max_m(itree(2,j))=m(j)-m(i)
               elseif( itree(2,j).eq.i .and. 
     &              itree(1,j).lt.0 .and. itree(1,j).gt.i) then
c                 2nd daughter of "j" is "i" --> 1st is sister
                  max_m(itree(1,j))=m(j)-m(i)
               endif
            enddo
c     update the range for all the other starts of s-channels chains if
c     the current one is the start of an s-channel chain.
            if (start_s_chan(i)) then
               do j=i,-1
                  if (start_s_chan(j))
     &                 max_m(j)=max_m(j)-(m(i)-min_m(i))
               enddo
c     be sure to also update the range for granny: it always computed
c     later
               if (start_s_chan(igranny).and.igranny.lt.i) then
                  max_m(igranny)=max_m(igranny)-(m(i)-min_m(i))
               endif
            endif
         enddo
      enddo
c At the end, compute the grandmother invariant mass
      smin = (m(itree(1,igranny))+m(itree(2,igranny)))**2
      smax = max_m(igranny)**2
      if (.not. input_granny_m2) then
         call generate_si(igranny,smin,smax,s,cBW,cBW_width,cBW_mass
     &        ,qmass,qwidth,x,xjac0,s_mass)
c     if numerical inaccuracy, quit.
         if ( xjac0.lt.0d0 ) then
            xjac0=-5
            pass=.false.
            return
         endif
         granny_m2_red( 0)=s(igranny)
         granny_m2_red(-1)=smin
         granny_m2_red( 1)=smax
      else
c     call this function just to get the right Jacobian.
         call generate_si(igranny,smin,smax,s,cBW,cBW_width,cBW_mass
     &        ,qmass,qwidth,x,xjac0,s_mass)
c     if numerical inaccuracy, quit.
         if ( xjac0.lt.0d0 ) then
            xjac0=-5
            pass=.false.
            return
         endif
c     overwrite the mass with the granny_m2_red(0).
         s(igranny)=granny_m2_red(0)
      endif
c     Check that this is a valid invariant, i.e. sum of daughter masses
c     is smaller than granny mass.
      if (s(igranny) .lt. smin .or. s(igranny).gt.smax) then
         xjac0=-5
         pass=.false.
         return
      endif
      m(igranny) = sqrt(s(igranny))
      return
      end


      subroutine generate_si(i,smin,smax,s,cBW,cBW_width,cBW_mass,qmass
     $     ,qwidth,x,xjac0,s_mass)
      implicit none 
      include 'genps.inc'
      include 'nexternal.inc'
      integer i
      double precision smin,smax,s(-max_branch:max_particles),qwidth(
     &     -nexternal:0),qmass(-nexternal:0),cBW_width(-1:1,-nexternal:
     &     -1),cBW_mass(-1:1,-nexternal:-1),xjac0,x(99),s_mass(
     &     -nexternal:nexternal)
      integer cBW(-nexternal:-1)
c Choose the appropriate s given our constraints smin,smax
      if(qwidth(i).ne.0.d0 .and. cBW(i).ne.2)then
c Breit Wigner
         if (cBW(i).eq.1 .and.
     &        cBW_width(1,i).gt.0d0 .and. cBW_width(-1,i).gt.0d0) then
c     conflicting BW on both sides
            call trans_x(6,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (cBW(i).eq.1.and.cBW_width(1,i).gt.0d0) then
c     conflicting BW with alternative mass larger
            call trans_x(5,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (cBW(i).eq.1.and.cBW_width(-1,i).gt.0d0) then
c     conflicting BW with alternative mass smaller
            call trans_x(4,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         else
c     normal BW
            call trans_x(3,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         endif
      else
c not a Breit Wigner
         if (smin.eq.0d0 .and. s_mass(i).eq.0d0) then
c     no lower limit on invariant mass from cuts or final state masses:
c     use flat distribution
            call trans_x(1,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (smin.ge.s_mass(i) .and. smin.gt.0d0) then
c     A lower limit on smin, which is larger than lower limit from cuts
c     or masses. Use 1/x importance sampling
            call trans_x(7,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         elseif (smin.lt.s_mass(i) .and. s_mass(i).gt.0d0) then
c     Use flat grid between smin and s_mass(i), and 1/x^nsamp above
c     s_mass(i)
            call trans_x(2,-i,x(-i),smin,smax,s_mass(i),qmass(i)
     &           ,qwidth(i),cBW_mass(-1,i),cBW_width(-1,i),xjac0,s(i))
         else
            write (*,*) "ERROR in genps_fks.f:"/
     $           /" cannot set s-channel without BW",i,smin,s_mass(i)
            stop 1
         endif
      endif
      return
      end

      subroutine generate_t_channel_branchings(ns_channel,nbranch,itree
     $     ,m,s,x,pb,xjac0,xpswgt0,pass)
c First we need to determine the energy of the remaining particles this
c is essentially in place of the cos(theta) degree of freedom we have
c with the s channel decay sequence
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      double precision xjac0,xpswgt0
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      integer ns_channel,nbranch
      logical pass
c
      double precision totalmass,smin,smax,s1,ma2,mbq,m12,mnq,tmin,tmax
     &     ,t,tmax_temp,phi,dum,dum3(-1:1),s_m,tm,tiny
      parameter (tiny=1d-8)
      integer i,ibranch,idim
      double precision lambda,dot
      external lambda,dot
      double precision s_mass(-nexternal:nexternal)
      common/to_phase_space_s_channel/s_mass
c 
      pass=.true.
      totalmass=0d0
      s_m=0d0
      do ibranch = -ns_channel-1,-nbranch,-1
         totalmass=totalmass+m(itree(2,ibranch))
         s_m=s_m+sqrt(s_mass(itree(2,ibranch)))
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
         idim=(nbranch-1+(-ibranch)*2)
         s_m=s_m-sqrt(s_mass(itree(2,ibranch)))
         if (abs(smin-s_m**2).lt.tiny) then
            call trans_x(1,idim,x(idim),smin,smax,s_m**2,dum
     $           ,dum,dum3(-1),dum3(-1),xjac0,s1)
         else
            call trans_x(1,idim,x(idim),smin,smax,s_m**2,dum
     $           ,dum,dum3(-1),dum3(-1),xjac0,s1)
         endif
         if (xjac0.le.0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31a in genps_fks.f',ibranch,s1
     $              ,smin,smax,s_m**2,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         m(ibranch-1)=sqrt(s1)
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
c
      do ibranch=-ns_channel-1,-nbranch+1,-1
         s1  = m(ibranch)**2    !Total mass available
         ma2 = m(2)**2
         mbq = dot(pb(0,itree(1,ibranch)),pb(0,itree(1,ibranch)))
         m12 = m(itree(2,ibranch))**2
         mnq = m(ibranch-1)**2
         call yminmax(s1,t,m12,ma2,mbq,mnq,tmin,tmax)
         call trans_x(1,-ibranch,x(-ibranch),-tmax,-tmin,s_mass(ibranch)
     $        ,dum,dum,dum3(-1),dum3(-1),xjac0,tm)
         if (xjac0.le.0d0) then
            if ((xjac0.gt.-400d0 .or. xjac0.le.-500d0) .and.
     $           xjac0.ne.0d0)then
               write (*,*) 'WARNING #31b in genps_fks.f',ibranch,tm
     $              ,-tmax,-tmin,xjac0
            endif
            xjac0 = -6
            pass=.false.
            return
         endif
         t=-tm
         if (t .lt. tmin .or. t .gt. tmax) then
            write (*,*) "WARNING #35 in genps_fks.f",t,tmin,tmax
            xjac0=-3d0
            pass=.false.
            return
         endif
         phi = 2d0*pi*x(nbranch+(-ibranch-1)*2)
         xjac0 = xjac0*2d0*pi
c Finally generate the momentum. The call is of the form
c pa+pb -> p1+ p2; t=(pa-p1)**2;   pr = pa-p1
c gentcms(pa,pb,t,phi,m1,m2,p1,pr) 
         call gentcms(pb(0,itree(1,ibranch)),pb(0,2),t,phi,
     &        m(itree(2,ibranch)),m(ibranch-1),pb(0,itree(2,ibranch)),
     &        pb(0,ibranch),xjac0)
c
         if (xjac0 .lt. 0d0) then
            write(*,*) 'Failed gentcms',ibranch,xjac0
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0/(4d0*dsqrt(lambda(s1,ma2,mbq)))
      enddo
c We need to get the momentum of the last external particle.  This
c should just be the sum of p(0,2) and the remaining momentum from our
c last t channel 2->2
      do i=0,3
         pb(i,itree(2,-nbranch)) = pb(i,-nbranch+1)+pb(i,2)
      enddo
      return
      end


      subroutine fill_born_momenta(nbranch,nt_channel,one_body,ionebody
     &     ,x,itree,m,s,pb,xjac0,xpswgt0,pass)
      implicit none
      real*8 pi
      parameter (pi=3.1415926535897932d0)
      include 'genps.inc'
      include 'nexternal.inc'
      integer nbranch,nt_channel,ionebody
      double precision M(-max_branch:max_particles),x(99)
      double precision s(-max_branch:max_particles)
      double precision pb(0:3,-max_branch:nexternal-1)
      integer itree(2,-max_branch:-1)
      double precision xjac0,xpswgt0
      logical pass,one_body
c
      double precision one
      parameter (one=1d0)
      double precision costh,phi,xa2,xb2
      integer i,ix
      double precision lambda,dot
      external lambda,dot
      double precision vtiny
      parameter (vtiny=1d-12)
c
      pass=.true.
      do i = -nbranch+nt_channel+(nincoming-1),-1
         ix = nbranch+(-i-1)*2+(2-nincoming)
         if (nt_channel .eq. 0) ix=ix-1
         costh= 2d0*x(ix)-1d0
         phi  = 2d0*pi*x(ix+1)
         xjac0 = xjac0 * 4d0*pi
         xa2 = m(itree(1,i))*m(itree(1,i))/s(i)
         xb2 = m(itree(2,i))*m(itree(2,i))/s(i)
         if (m(itree(1,i))+m(itree(2,i)) .ge. m(i)) then
            xjac0=-8
            pass=.false.
            return
         endif
         xpswgt0 = xpswgt0*.5D0*PI*SQRT(LAMBDA(ONE,XA2,XB2))/(4.D0*PI)
         call mom2cx(m(i),m(itree(1,i)),m(itree(2,i)),costh,phi,
     &        pb(0,itree(1,i)),pb(0,itree(2,i)))
c If there is an extremely large boost needed here, skip the phase-space point
c because of numerical stabilities.
         if (dsqrt(abs(dot(pb(0,i),pb(0,i))))/pb(0,i) 
     &        .lt.vtiny) then
            xjac0=-81
            pass=.false.
            return
         else
            call boostm(pb(0,itree(1,i)),pb(0,i),m(i),pb(0,itree(1,i)))
            call boostm(pb(0,itree(2,i)),pb(0,i),m(i),pb(0,itree(2,i)))
         endif
      enddo
c
c
c Special phase-space fix for the one_body
      if (one_body) then
c Factor due to the delta function in dphi_1
         xpswgt0=pi/m(ionebody)
c Kajantie's normalization of phase space (compensated below in flux)
         xpswgt0=xpswgt0/(2*pi)
         do i=0,3
            pb(i,3) = pb(i,1)+pb(i,2)
         enddo
      endif
      return
      end


      subroutine get_recoil(p_born,imother,shat_born,xmrec2,pass)
      implicit none
      include 'nexternal.inc'
      double precision p_born(0:3,nexternal-1),xmrec2,shat_born
      logical pass
      integer imother,i
      double precision recoilbar(0:3),dot
      external dot
      pass=.true.
      do i=0,3
         if (nincoming.eq.2) then
            recoilbar(i)=p_born(i,1)+p_born(i,2)-p_born(i,imother)
         else
            recoilbar(i)=p_born(i,1)-p_born(i,imother)
         endif
      enddo
      xmrec2=dot(recoilbar,recoilbar)
      if(xmrec2.lt.0.d0)then
         if(abs(xmrec2).gt.(1.d-4*shat_born))then
            write(*,*)'Fatal error #14 in genps_fks.f',xmrec2,imother
            stop
         else
            write(*,*)'Error #15 in genps_fks.f',xmrec2,imother
            pass=.false.
            return
         endif
      endif
      if (xmrec2.ne.xmrec2) then
         write (*,*) 'Error #16 in setting up event in genps_fks.f,'//
     &        ' skipping event'
         pass=.false.
         return
      endif
      return
      end


      
      subroutine trans_x(itype,idim,x,smin,smax,s_mass,qmass,qwidth
     $     ,cBW_mass,cBW_width,jac,s)
c Given the input random number 'x', returns the corresponding value of
c the invariant mass squared 's'. 
c
c     itype=1: flat transformation
c     itype=2: flat between 0 and s_mass/stot, 1/x above
c     itype=3: Breit-Wigner
c     itype=4: Conflicting BW, with alternative mass smaller
c     itype=5: Conflicting BW, with alternative mass larger
c     itype=6: Conflicting BW on both sides
c
      implicit none
      integer itype,idim
      double precision x,smin,smax,s_mass,qmass,qwidth,cBW_mass(-1:1)
     $     ,cBW_width(-1:1),jac,s
      double precision fract,A,B,C,D,E,F,G,bs(-1:1),maxi,mini
      integer j
c
      if (itype.eq.1) then
c     flat transformation:
         A=smax-smin
         B=smin
         s=A*x+B
         jac=jac*A
      elseif (itype.eq.2) then
         fract=0.25d0
         if (s_mass.eq.0d0) then
            write (*,*) 's_mass is zero',itype,idim
         endif
         if (x.lt.fract) then
c     flat transformation:
            if (s_mass.lt.smin) then
               jac=-421d0
               return
            endif
            maxi=min(s_mass,smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         else
c     S=A/(B-x) transformation:
            if (s_mass.ge.smax) then
               jac=-422d0
               return
            endif
            mini=max(s_mass,smin)
            A=mini*smax*(1d0-fract)/(smax-mini)
            B=(smax-fract*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif(itype.eq.3) then
c     Normal Breit-Wigner, i.e.
c        \int_smin^smax ds g(s)/((s-qmass^2)^2-qmass^2*qwidth^2) =
c        \int_0^1 dx g(s(x))
         A=atan((qmass-smin/qmass)/qwidth)
         B=atan((qmass-smax/qmass)/qwidth)
         s=qmass*(qmass-qwidth*tan(A-(A-B)*x))
         jac=jac*qmass*qwidth*(A-B)/(cos(A-(A-B)*x))**2
      elseif(itype.eq.4) then
c     Conflicting BW, with alternative mass smaller than current
c     mass. That is, we need to throw also many events at smaller masses
c     than the peak of the current BW. Split 'x' at 'bs(-1)', using a
c     flat distribution below the split, and a BW above the split.
         fract=0.3d0
         bs(-1)=(cBW_mass(-1)-qmass)/
     &        (qwidth+cBW_width(-1)) ! bs(-1) is negative here
         bs(-1)=qmass+bs(-1)*qwidth
         bs(-1)=bs(-1)**2
         if (x.lt.fract) then
            if(smin.gt.bs(-1)) then
               jac=-441d0
               return
            endif
            maxi=min(bs(-1),smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         else
            if(smax.lt.bs(-1)) then
               jac=-442d0
               return
            endif
            mini=max(bs(-1),smin)
            A=atan((qmass-mini/qmass)/qwidth)
            B=atan((qmass-smax/qmass)/qwidth)
            C=((1d0-x)*A+(x-fract)*B)/(1d0-fract)
            s=qmass*(qmass-qwidth*tan(C))
            jac=jac*qmass*qwidth*(A-B)/((cos(C))**2*(1d0-fract))
         endif
      elseif(itype.eq.5) then
c     Conflicting BW, with alternative mass larger than current
c     mass. That is, we need to throw also many events at larger masses
c     than the peak of the current BW. Split 'x' at 'bs(1)' and the
c     alternative mass. Use a BW below bs(1), a flat distribution
c     between bs(1) and the alternative mass, and a 1/x above the
c     alternative mass.
         fract=0.35d0
         bs(1)=(cBW_mass(1)-qmass)/
     &        (qwidth+cBW_width(1))
         bs(1)=qmass+bs(1)*qwidth
         bs(1)=bs(1)**2
         if (x.lt.fract) then
            if(smin.gt.bs(1)) then
               jac=-451d0
               return
            endif
            maxi=min(bs(1),smax)
            A=atan((qmass-smin/qmass)/qwidth)
            B=atan((qmass-maxi/qmass)/qwidth)
            C=((B-A)*x+fract*A)/fract
            s=qmass*(qmass-qwidth*tan(C))
            jac=jac*qmass*qwidth*(A-B)/((cos(C))**2*fract)
         elseif (x.lt.1d0-fract) then
            if(smin.gt.cBW_mass(1)**2 .or. smax.lt.bs(1)) then
               jac=-452d0
               return
            endif
            maxi=min(cBW_mass(1)**2,smax)
            mini=max(bs(1),smin)
            A=(maxi-mini)/(1d0-2d0*fract)
            B=((1d0-fract)*mini-fract*maxi)/(1d0-2d0*fract)
            s=A*x+B
            jac=jac*A
         else
            if(smax.le.cBW_mass(1)**2) then
               jac=-453d0
               return
            endif
            mini=max(cBW_mass(1)**2,smin)
            A=mini*smax*fract/(smax-mini)
            B=(smax-(1d0-fract)*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif(itype.eq.6) then
         fract=0.3d0
c     Conflicting BW on both sides. Use flat below bs(-1); BW between
c     bs(-1) and bs(1); flat between bs(1) and alternative mass; and 1/x
c     above alternative mass.
         do j=-1,1,2
            bs(j)=(cBW_mass(j)-qmass)/
     &           (qwidth+cBW_width(j))
            bs(j)=qmass+bs(j)*qwidth
            bs(j)=bs(j)**2
         enddo
         if (x.lt.fract) then
            if(smin.gt.bs(-1)) then
               jac=-461d0
               return
            endif
            maxi=min(bs(-1),smax)
            A=(maxi-smin)/fract
            B=smin
            s=A*x+B
            jac=jac*A
         elseif(x.lt.1d0-fract) then
            if(smin.gt.bs(1) .or. smax.lt.bs(-1)) then
               jac=-462d0
               return
            endif
            maxi=min(bs(1),smax)
            mini=max(bs(-1),smin)
            A=atan((qmass-mini/qmass)/qwidth)
            B=atan((qmass-maxi/qmass)/qwidth)
            C=((1d0-fract-x)*A+(x-fract)*B)/(1d0-2d0*fract)
            s=qmass*(qmass-qwidth*tan(C))
            jac=-jac*qmass*qwidth*(B-A)/((cos(C))**2*(1d0-2d0*fract))
         elseif(x.lt.1d0-fract/2d0) then
            if(smin.gt.cBW_mass(1)**2 .or. smax.lt.bs(1)) then
               jac=-463d0
               return
            endif
            maxi=min(cBW_mass(1)**2,smax)
            mini=max(bs(1),smin)
            A=2d0*(maxi-mini)/fract
            B=2d0*maxi-mini-2d0*(maxi-mini)/fract
            s=A*x+B
            jac=jac*A
         else
            if(smax.le.cBW_mass(1)**2) then
               jac=-464d0
               return
            endif
            mini=max(cBW_mass(1)**2,smin)
            A=mini*smax*fract/(2d0*(smax-mini))
            B=(smax-(1d0-fract/2d0)*mini)/(smax-mini)
            s=A/(B-x)
            jac=jac*s**2/A
         endif
      elseif (itype.eq.7) then
c     S=A/(B-x) transformation:
         if (smin.le.0d0) then
            jac=-471d0
            return
         endif
         A=smin*smax/(smax-smin)
         B=smax/(smax-smin)
         s=A/(B-x)
         jac=jac*s**2/A
      endif
      return
      end

      subroutine get_tau_y_from_x12(x1, x2, omx1, omx2, tau, ycm, ycmhat, jac) 
      implicit none
      double precision x1, x2, omx1, omx2, tau, ycm, ycmhat, jac
      double precision ylim
      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound
      double precision tolerance
      parameter (tolerance=1e-3)
      double precision y_settozero
      parameter (y_settozero=1e-12)
      double precision lx1, lx2
      double precision ylim0, ycm0

      tau = x1*x2

      ! ycm=-log(tau)/2 ;  ylim = log(x1/x2)/2
      if (1d0-x1.gt.tolerance) then
        lx1 = dlog(x1)
      else
        lx1 = -omx1-omx1**2/2d0-omx1**3/3d0-omx1**4/4d0-omx1**5/5d0
      endif
      ylim = -0.5d0*lx1
      ycm = 0.5d0*lx1

      if (1d0-x2.gt.tolerance) then
        lx2 = dlog(x2)
      else
        lx2 = -omx2-omx2**2/2d0-omx2**3/3d0-omx2**4/4d0-omx2**5/5d0
      endif
      ylim = ylim-0.5d0*lx2
      ycm = ycm-0.5d0*lx2

      ycmhat = ycm / ylim

      ! this is to prevent numerical inaccuracies
      ! when botn x->1
      if (ylim.lt.y_settozero) then
        ylim = 0d0
        ycm = 0d0
        ycmhat = 1d0
      endif

      if (abs(ycmhat).gt.1d0) then
        if (abs(ycmhat).gt.1d0 + tolerance) then
          write(*,*) 'ERROR YCMHAT', ycmhat, x1, x2
          stop 1 
        else
          ycmhat = sign(1d0, ycmhat)
        endif
      endif

      if (tau.lt.tau_born_lower_bound) then
        write(*,*) 'get_tau_y_from_x12: Warning, unphysical tau',
     $  tau, tau_born_lower_bound
        jac = -1000d0
      endif

      return 
      end

      subroutine generate_x_ee(rnd, xmin, x, omx, jac)
      implicit none
      ! generates the momentum fraction with importance
      !  sampling suitable for ee collisions
      ! rnd is generated uniformly in [0,1], 
      ! x is generated according to (1 -rnd)^-expo, starting
      ! from xmin
      ! jac is the corresponding jacobian
      ! omx is 1-x, stored to improve numerical accuracy
      double precision rnd, x, omx, jac, xmin
      double precision expo
      double precision get_ee_expo
      double precision tolerance
      parameter (tolerance=1.d-5)

      expo = get_ee_expo()

      x = 1d0 - rnd ** (1d0/(1d0-expo))
      omx = rnd ** (1d0/(1d0-expo))
      if (x.ge.1d0) then
        if (x.lt.1d0+tolerance) then
          x=1d0
        else
          write(*,*) 'ERROR in generate_x_ee', rnd, x
          stop 1
        endif
      endif
      jac = 1d0/(1d0-expo) 
      ! then rescale it between xmin and 1
      x = x * (1d0 - xmin) + xmin
      omx = omx * (1d0 - xmin)
      jac = jac * (1d0 - xmin)**(1d0-expo)

      return 
      end


      subroutine generate_ee_tau_y(rnd1_in, rnd2_in, one_body, stot, nt_channel,
     $     qmass, qwidth, cBW, cBW_mass, cBW_width,
     $     tau_born, ycm_born, ycmhat, xjac0)
      implicit none
      double precision rnd1_in, rnd2_in
      double precision rnd1, rnd2, stot, qmass, qwidth
      double precision cBW_mass(-1:1), cBW_width(-1:1)
      integer nt_channel, cBW
      logical one_body
      double precision tau_born, ycm_born, ycmhat, xjac0

      logical bw_exists, generate_with_bw
      common /to_ee_generatebw/ generate_with_bw
      double precision frac_bw
      parameter (frac_bw=0.5d0)
      integer idim_dum
C dressed lepton stuff
      double precision x1_ee, x2_ee, jac_ee
      
      double precision omx_ee(2)
      common /to_ee_omx1/ omx_ee

      double precision tau_Born_lower_bound,tau_lower_bound_resonance
     $     ,tau_lower_bound
      common/ctau_lower_bound/tau_Born_lower_bound
     $     ,tau_lower_bound_resonance,tau_lower_bound

      double precision get_ee_expo
      double precision tau_m, tau_w

      ! these common blocks are never used
      ! we leave them here for the moment 
      ! as e.g. one may want to plot random numbers, etc.
      double precision r1, r2, x1bk, x2bk
      common /to_random_numbers/r1,r2, x1bk, x2bk


      ! copy the random numbers, as they may be rescaled
      ! (avoids side effects)
      rnd1=rnd1_in
      rnd2=rnd2_in

      ! these lines store the random numbers in the common
      ! block (may be removed)
      r1=rnd1
      r2=rnd2

      ! define the analogous of tau for mass and width
      tau_m = qmass**2/stot
      tau_w = qwidth**2/stot

      bw_exists = nt_channel.eq.0.and.qwidth.ne.0.d0.and.cBW.ne.2
     $ .and.tau_m.lt.1d0
      generate_with_bw=.false.
      ! if there are BWs, decide whether to generate flat or
      ! to use the BW-specific generation (half and half, or
      ! as determined by frac_bw)
      if (bw_exists) then
        generate_with_bw = rnd1.lt.frac_bw
        if (generate_with_bw) then
            rnd1 = rnd1 / frac_bw
            xjac0 = xjac0 / frac_bw
        else
            rnd1 = (rnd1 - frac_bw) / (1d0 - frac_bw)
            xjac0 = xjac0 / (1d0 - frac_bw)
        endif
      endif

      if (one_body) then
        write(*,*) 'one body with ee collisions not implemented'
        stop 1
      endif

      if(generate_with_bw) then
        ! here we treat the case of resonances

        ! first generate tau with the dedicated function
        idim_dum = 1000 ! this is never used in practice
        call generate_tau_BW(stot,idim_dum,rnd1,qmass,qwidth,cBW,cBW_mass,
     $       cBW_width,tau_born,xjac0)
        ! multiply the jacobian by a multichannel factor
        xjac0 = xjac0 * (1d0/((tau_born-tau_m)**2 + tau_m*tau_w)) / 
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (1d0-tau_born)**(1d0-2*get_ee_expo()))

        ! then pick either x1 or x2 and generate it the usual way;
        ! Note that:
        ! - setting xmin=sqrt(tau_born) ensures that the largest 
        !    bjorken x is being generated.
        ! -  there is a jacobian for x1 x2 -> tau x1(2)
        ! -  we must include the factor 1/(1-x)^get_ee_expo,
        !    (x is the bjorken x which is not generated)
        !    because the compute_eepdf function assumes that
        !    this is the case in general
        if (rnd2.lt.0.5d0) then
          call generate_x_ee(rnd2*2d0, dsqrt(tau_born), x1_ee, omx_ee(1), jac_ee)
          x2_ee = tau_born / x1_ee
          omx_ee(2) = 1d0 - x2_ee
          xjac0 = xjac0 / x1_ee * 2d0 * jac_ee / (1d0-x2_ee)**get_ee_expo()
        else
          call generate_x_ee(1d0-2d0*(rnd2-0.5d0), dsqrt(tau_born), x2_ee, omx_ee(2), jac_ee)
          x1_ee = tau_born / x2_ee
          omx_ee(1) = 1d0 - x1_ee
          xjac0 = xjac0 / x2_ee * 2d0  * jac_ee / (1d0-x1_ee)**get_ee_expo()
        endif
      else
        ! standard (without resonances) generation:
        ! for dressed ee collisions the generation is different
        ! wrt the pp case. In the pp case, tau and y_cm are generated, 
        ! while in the ee case x1 and x2 are generated first.

        call generate_x_ee(rnd1, tau_born_lower_bound,
     $      x1_ee, omx_ee(1), jac_ee)
        xjac0 = xjac0 * jac_ee
        call generate_x_ee(rnd2, tau_born_lower_bound/x1_ee,
     $      x2_ee, omx_ee(2), jac_ee)
        xjac0 = xjac0 * jac_ee

        tau_born = x1_ee * x2_ee
        ! multiply the jacobian by a multichannel factor if the 
        ! generation with resonances is also possible
        if (bw_exists) xjac0 = xjac0 * (1d0-tau_born)**(1d0-2*get_ee_expo()) / 
     $       ( 1d0/((tau_born-tau_m)**2 + tau_m*tau_w) + (1d0-tau_born)**(1d0-2*get_ee_expo()))
      endif

      ! Check here if the bjorken x's are physical (may not be so
      ! because of instabilities
      if (x1_ee.gt.1d0.or.x2_ee.gt.1d0) then
        write(*,*) 'generate_ee_tau_y: Warning, unphysical x:', 
     $   x1_ee, x2_ee, generate_with_bw
        xjac0 = -1000d0
        return
      endif

      ! now we are done. We must call the following function 
      ! in order to (re-)generate tau and ycm
      ! from x1 and x2. It also (re-)checks that tau_born 
      ! is pysical, and otherwise sets xjac0=-1000
      call get_tau_y_from_x12(x1_ee, x2_ee, omx_ee(1), omx_ee(2), tau_born, ycm_born, ycmhat, xjac0) 

      x1bk=x1_ee
      x2bk=x2_ee

      return
      end


