c This file contains the routines relevant to reweighting 
c NLO and aMC@NLO results, for the computation of scale 
c and PDF uncertainties

      
      subroutine reweight_fill_extra_inverse()
c Fills arrays with dimensions nexternal, equating them with their
c counterparts with dimensions maxparticles; this is thus the inverse
c of reweight_fill_extra(). Needed before calling routines that
c include reweight1.inc, whose content may have not be filled
c (for example, by read_lhef_event(), which only fills reweight0.inc)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "reweight.inc"
      integer i,j,k
c
      do k=1,4
        do i=1,nexternal
          do j=0,3
            wgtkin(j,i,k)=wgtkinE(j,i,k)
          enddo
        enddo
      enddo
c
      do k=1,nexternal
        wgtwmcxsec(k)=wgtwmcxsecE(k)
        wgtmcxbj(1,k)=wgtmcxbjE(1,k)
        wgtmcxbj(2,k)=wgtmcxbjE(2,k)
      enddo
c
      return
      end


      subroutine reweight_settozero()
c Set all reweight variables equal to zero
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "reweight.inc"
      logical all
      integer i,j,k
c
      wgtref=0.d0
      wgtref_nbody=0.d0
      do k=1,4
        wgtqes2(k)=0.d0
        wgtxbj(1,k)=0.d0
        wgtxbj(2,k)=0.d0
        do i=1,nexternal
          do j=0,3
            wgtkin(j,i,k)=0.d0
          enddo
        enddo
c Special value, consistent with genps_fks. The kinematic configuration
c should not be used if wgtkin(0,1,*)=-99
        wgtkin(0,1,k)=-99.d0
        wgtmuR2(k)=0.d0
        wgtmuF12(k)=0.d0
        wgtmuF22(k)=0.d0
        wgtwreal(k)=0.d0
        wgtwdeg(k)=0.d0
        wgtwdegmuf(k)=0.d0
        if(k.eq.2)then
           wgtwborn(k)=0.d0
           wgtwns(k)=0.d0
           wgtwnsmuf(k)=0.d0
           wgtwnsmur(k)=0.d0
        endif
      enddo
      wgtdegrem_xi=0.d0
      wgtdegrem_lxi=0.d0
      wgtdegrem_muF=0.d0
      wgtnstmp=0.d0
      wgtwnstmpmuf=0.d0
      wgtwnstmpmur=0.d0
      do k=1,nexternal
        wgtwmcxsec(k)=0.d0
        wgtmcxbj(1,k)=0.d0
        wgtmcxbj(2,k)=0.d0
      enddo
      iwgtnumpartn=0
      jwgtinfo=0
      mexternal=0
      wgtNLO11=0d0
      wgtNLO12=0d0
      wgtNLO20=0d0
      return
      end


      subroutine reweight_settozero_all(iFKS,all)
c Set all reweight variables equal to zero
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "reweight_all.inc"
      logical all
      integer i,j,k,iFKS,l
c
      do l=1,maxproc_save
         wgtref_all(iFKS,l)=0.d0
         if (all) wgtref_nbody_all(l)=0.d0
      enddo
      do k=1,4
         wgtqes2_all(k,iFKS)=0.d0
         wgtxbj_all(1,k,iFKS)=0.d0
         wgtxbj_all(2,k,iFKS)=0.d0
         do i=1,nexternal
            do j=0,3
               wgtkin_all(j,i,k,iFKS)=0.d0
            enddo
         enddo
c Special value, consistent with genps_fks. The kinematic configuration
c should not be used if wgtkin(0,1,*)=-99
         wgtkin_all(0,1,k,iFKS)=-99.d0
         wgtmuR2_all(k,iFKS)=0.d0
         wgtmuF12_all(k,iFKS)=0.d0
         wgtmuF22_all(k,iFKS)=0.d0
         wgtwreal_all(k,iFKS)=0.d0
         wgtwdeg_all(k,iFKS)=0.d0
         wgtwdegmuf_all(k,iFKS)=0.d0
      enddo
      wgtdegrem_xi=0.d0
      wgtdegrem_lxi=0.d0
      wgtdegrem_muF=0.d0
      if (all) then
         wgtwborn_all=0.d0
         wgtwns_all=0.d0
         wgtwnsmuf_all=0.d0
         wgtwnsmur_all=0.d0
         wgtnstmp=0.d0
         wgtwnstmpmuf=0.d0
         wgtwnstmpmur=0.d0
         wgtref_nbody=0.d0
         do k=1,4
            do i=1,nexternal
               do j=0,3
                  wgtkin_all(j,i,k,0)=0.d0
               enddo
            enddo
            wgtxbj_all(1,k,0)=0.d0
            wgtxbj_all(2,k,0)=0.d0
            wgtkin_all(0,1,k,0)=-99.d0
            wgtmuR2_all(k,0)=0.d0
            wgtmuF12_all(k,0)=0.d0
            wgtmuF22_all(k,0)=0.d0
            wgtqes2_all(k,0)=0.d0
         enddo
      endif
      do k=1,nexternal
         wgtwmcxsec_all(k,iFKS)=0.d0
         wgtmcxbj_all(1,k,iFKS)=0.d0
         wgtmcxbj_all(2,k,iFKS)=0.d0
      enddo
      iwgtnumpartn_all(iFKS)=0
      jwgtinfo=0
      mexternal=0
      wgtNLO11=0d0
      wgtNLO12=0d0
      wgtNLO20=0d0
      return
      end
      

      subroutine fill_reweight0inc_nbody(iproc)
c Set all reweight variables equal to zero
      implicit none
      include "nexternal.inc"
      include "reweight_all.inc"
      integer iproc
      logical debug
      parameter (debug=.false.)
      if (debug) write (*,*) 'wgtref_nbody',wgtref_nbody
     $     ,wgtref_nbody_all(iproc)
      call reweight_overwrite(wgtref_nbody,wgtref_nbody_all(iproc),2)
      return
      end


      subroutine fill_reweight0inc(iFKS,iproc)
c Set all reweight variables equal to zero
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "reweight_all.inc"
      integer i,j,k,iFKS,iproc
      logical debug
      parameter (debug=.false.)
c
      if (debug) write (*,*) 'wgtref',iFKS,wgtref,wgtref_all(iFKS,iproc)
      call reweight_overwrite(wgtref,wgtref_all(iFKS,iproc),0)
c$$$      if (debug) write (*,*) 'wgtref_nbody',wgtref_nbody
c$$$     $     ,wgtref_nbody_all(iproc)
c$$$      call reweight_overwrite(wgtref_nbody,wgtref_nbody_all(iproc),2)
      do k=1,4
         if (debug) write (*,*) 'wgtqes2',k,iFKS,wgtqes2(k)
     &        ,wgtqes2_all(k,iFKS)
         call reweight_overwrite(wgtqes2(k),wgtqes2_all(k,iFKS),0)
         if (debug) write (*,*) 'wgtxbj1',k,iFKS,wgtxbj(1,k)
     &        ,wgtxbj_all(1,k,iFKS)
         call reweight_overwrite(wgtxbj(1,k),wgtxbj_all(1,k,iFKS),0)
         if (debug) write (*,*) 'wgtxbj2',k,iFKS,wgtxbj(2,k)
     &        ,wgtxbj_all(2,k,iFKS)
         call reweight_overwrite(wgtxbj(2,k),wgtxbj_all(2,k,iFKS),0)
        do i=1,nexternal
          do j=0,3
             if (i.eq.1.and.j.eq.0) then
                if (debug) write (*,*) 'wgtkin',j,i,k,iFKS,wgtkin(j,i,k)
     &               ,wgtkin_all(j,i,k,iFKS)
                call reweight_overwrite
     &               (wgtkin(j,i,k),wgtkin_all(j,i,k,iFKS),1)
             else
                if (debug) write (*,*) 'wgtkin',j,i,k,iFKS,wgtkin(j,i,k)
     &               ,wgtkin_all(j,i,k,iFKS)
                call reweight_overwrite
     &               (wgtkin(j,i,k),wgtkin_all(j,i,k,iFKS),0)
             endif
          enddo
        enddo
        if (debug) write (*,*) 'wgtmuR2',k,iFKS,wgtmuR2(k),wgtmuR2_all(k
     &       ,iFKS)
        call reweight_overwrite(wgtmuR2(k),wgtmuR2_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtmuF12',k,iFKS,wgtmuF12(k)
     &       ,wgtmuF12_all(k,iFKS)
        call reweight_overwrite(wgtmuF12(k),wgtmuF12_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtmuF22',k,iFKS,wgtmuF22(k)
     &       ,wgtmuF22_all(k,iFKS)
        call reweight_overwrite(wgtmuF22(k),wgtmuF22_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtwreal',k,iFKS,wgtwreal(k)
     &       ,wgtwreal_all(k,iFKS)
        call reweight_overwrite(wgtwreal(k),wgtwreal_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtwdeg',k,iFKS,wgtwdeg(k),wgtwdeg_all(k
     &       ,iFKS)
        call reweight_overwrite(wgtwdeg(k),wgtwdeg_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtwdegmuF',k,iFKS,wgtwdegmuf(k)
     &       ,wgtwdegmuf_all(k,iFKS)
        call reweight_overwrite(wgtwdegmuf(k),wgtwdegmuf_all(k,iFKS),0)
c$$$        if(k.eq.2)then
c$$$           if (debug) write (*,*) 'wgtwborn',k,wgtwborn(k),wgtwborn_all
c$$$           call reweight_overwrite(wgtwborn(k),wgtwborn_all,0)
c$$$           if (debug) write (*,*) 'wgtwns',k,wgtwns(k),wgtwns_all
c$$$           call reweight_overwrite(wgtwns(k),wgtwns_all,0)
c$$$           if (debug) write (*,*) 'wgtwnsmuf',k,wgtwnsmuf(k)
c$$$     &          ,wgtwnsmuf_all
c$$$           call reweight_overwrite(wgtwnsmuf(k),wgtwnsmuf_all,0)
c$$$           if (debug) write (*,*) 'wgtwnsmur',k,wgtwnsmur(k)
c$$$     &          ,wgtwnsmur_all
c$$$          call reweight_overwrite(wgtwnsmur(k),wgtwnsmur_all,0)
c$$$        endif
      enddo
      do k=1,nexternal
         if (debug) write (*,*) 'wgtwmcxsec',k,iFKS,wgtwmcxsec(k)
     &        ,wgtwmcxsec_all(k,iFKS)
        call reweight_overwrite(wgtwmcxsec(k),wgtwmcxsec_all(k,iFKS),0)
        if (debug) write (*,*) 'wgtwmcxbj1',k,iFKS,wgtmcxbj(1,k)
     &       ,wgtmcxbj_all(1,k,iFKS)
        call reweight_overwrite(wgtmcxbj(1,k),wgtmcxbj_all(1,k,iFKS),0)
        if (debug) write (*,*) 'wgtwmcxbj2',k,iFKS,wgtmcxbj(2,k)
     &       ,wgtmcxbj_all(2,k,iFKS)
        call reweight_overwrite(wgtmcxbj(2,k),wgtmcxbj_all(2,k,iFKS),0)
      enddo
      if (debug) write (*,*) 'iwgtnumpartn',iFKS,iwgtnumpartn
     &     ,iwgtnumpartn_all(iFKS)
      call reweight_overwrite_int(iwgtnumpartn,iwgtnumpartn_all(iFKS))
      return
      end

      subroutine reweight_overwrite_int(a,b)
      implicit none
      integer a,b
      if (a.eq.0) then
         a=b
      elseif (a.ne.b) then
         write (*,*) 'Error #1 in reweight_overwrite_int',a,b
         stop
      endif
      return
      end


      subroutine reweight_overwrite(a,b,i)
c     i=0: overwrite 'a' by 'b' if 'a' is zero
c     i=1: overwrite 'a' by 'b' if 'a' is equal to -99
c     i=2: overwrite 'a' by 'b'
      implicit none
      integer i
      double precision a,b
      double precision vtiny
      parameter (vtiny=1d-10)
      if (i.eq.0) then
         if (a.eq.0d0) then
            a=b
         elseif (abs((a-b)/a).gt.vtiny) then
            write (*,*) 'Error #1 in reweight_overwrite',a,b
            stop
         endif
      elseif (i.eq.1) then
         if (a.eq.-99d0) then
            a=b
         elseif (a.eq.0d0) then
            write (*,*) 'Error #2 in reweight_overwrite',a,b
            stop
         elseif (abs((a-b)/a).gt.vtiny) then
            write (*,*) 'Error #3 in reweight_overwrite',a,b
            stop
         endif
      elseif (i.eq.2) then
         a=b
      else
         write (*,*) 'Error #4 in reweight_overwrite',i,a,b
         stop
      endif
      return
      end

      function compute_rwgt_wgt_Hev(xmuR_over_ref,xmuF1_over_ref,
     #                              xmuF2_over_ref,xQES_over_ref,
     #                              kwgtinfo)
c Recomputes the H-event cross section using the weights saved, and compares
c with the reference weight
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      include 'run.inc'
      include "reweight.inc"
      include 'nFKSconfigs.inc'

      double precision compute_rwgt_wgt_Hev
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,alphas,temp
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      integer i,j,k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)

      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379660D0)  !CONV TO PICOBARNS             
      integer i_process
      common/c_i_process/i_process

      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer save_nFKSprocess
c FxFx merging
      logical rewgt_mohdr_calculated,rewgt_izero_calculated
      double precision rewgt_mohdr,rewgt_izero,rewgt_exp_mohdr
     $     ,rewgt_exp_izero
      logical setclscales
      double precision rewgt
      external setclscales,rewgt
      double precision rwgt_muR_dep_fac
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
      save_nFKSprocess=nFKSprocess
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if(kwgtinfo.ne.4.and.kwgtinfo.ne.5) wgtbpower=rwgtbpower
c
      xsec=0.d0

      temp=0d0
      do i=1,iwgtnumpartn
         temp=temp+abs(wgtwmcxsec(i))
      enddo
      if (temp.eq.0d0) goto 541

      call set_cms_stuff(izero)
      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(1).ne.0.d0) .or.
     $     ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and.wgtkin(0,1,1).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.3.or.kwgtinfo.le.5)then
          call set_cms_stuff(mohdr)
          if (ickkw.eq.3) then
              mu_r=sqrt(wgtmuR2(1))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(1)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(1)
           else
              call set_alphaS(wgtkin(0,1,1))
           endif
        else
          write(*,*)'Error #0a in compute_rwgt_wgt_Hev',kwgtinfo
          stop
        endif
        do i=1,iwgtnumpartn
          xbk(1) = wgtmcxbj(1,i)
          xbk(2) = wgtmcxbj(2,i)
          if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #       xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
            if(wgtwmcxsec(i).ne.0.d0)then
              write(*,*)'Error #1 in compute_rwgt_wgt_Hev'
              write(*,*)i,xbk(1),xbk(2),wgtwmcxsec(i)
              stop
            endif
          else
            nFKSprocess=nFKSprocess_used
            xlum = dlum()
            xsec=xsec+CONV*PD(i_process)*wgtwmcxsec(i)*g**(2*wgtbpower
     $           +2.d0) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
          endif
        enddo
      endif
c
 541  continue
      if (wgtwreal(2).eq.0d0 .and. wgtwreal(3).eq.0d0 .and.
     $     wgtwreal(4).eq.0d0) goto 542

      call set_cms_stuff(izero)
      if( (kwgtinfo.eq.1.and.wgtmuR2(2).ne.0.d0) .or.
     $     ((kwgtinfo.ge.2.or.kwgtinfo.le.5).and.wgtkin(0,1,2).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(2))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(2) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(2) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.2.or.kwgtinfo.le.5)then
           if (ickkw.eq.3) then
              mu_r=sqrt(wgtmuR2(2))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(2)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(2)
           else
              call set_alphaS(wgtkin(0,1,2))
           endif
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_Hev',kwgtinfo
          stop
        endif
        do k=2,4
          xbk(1) = wgtxbj(1,k)
          xbk(2) = wgtxbj(2,k)
          if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #       xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
            if(wgtwreal(k).ne.0.d0)then
              write(*,*)'Error #2 in compute_rwgt_wgt_Hev'
              write(*,*)k,xbk(1),xbk(2),wgtwreal(k)
              stop
            endif
          else
            nFKSprocess=nFKSprocess_used
            xlum = dlum()
            xsec=xsec+CONV*PD(i_process)*wgtwreal(k)*g**(2*wgtbpower
     $           +2.d0) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
          endif
        enddo
      endif
c
 542  continue
      if (wgtwreal(1).eq.0d0) goto 543
      
      call set_cms_stuff(mohdr)
      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(1).ne.0.d0) .or.
     $     ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and.wgtkin(0,1,1).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.3.or.kwgtinfo.le.5)then
           if (ickkw.eq.3) then 
              mu_r=sqrt(wgtmuR2(1))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(1)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(1)
           else
              call set_alphaS(wgtkin(0,1,1))
           endif
        else
          write(*,*)'Error #0c in compute_rwgt_wgt_Hev',kwgtinfo
          stop
        endif
        xbk(1) = wgtxbj(1,1)
        xbk(2) = wgtxbj(2,1)
        if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #     xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
          if(wgtwreal(1).ne.0.d0)then
            write(*,*)'Error #3 in compute_rwgt_wgt_Hev'
            write(*,*)xbk(1),xbk(2),wgtwreal(1)
            stop
          endif
        else
          nFKSprocess=nFKSprocess_used
          xlum = dlum()
          xsec=xsec+CONV*PD(i_process)*wgtwreal(1)*g**(2*wgtbpower+2.d0)
     f         * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
        endif
      endif
c
 543  continue
c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
      nFKSprocess=save_nFKSprocess
c
      compute_rwgt_wgt_Hev=xsec
c
      return
      end


      function compute_rwgt_wgt_Sev(xmuR_over_ref,xmuF1_over_ref,
     #                              xmuF2_over_ref,xQES_over_ref,
     #                              kwgtinfo)
c Recomputes the S-event cross section using the weights saved, and compares
c with the reference weight
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      include 'run.inc'
      include "reweight.inc"
      include 'nFKSconfigs.inc'

      double precision compute_rwgt_wgt_Sev
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,xlgmuf,xlgmur,alphas,temp
      double precision QES2_local
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision tiny,pi
      parameter (tiny=1.d-2)
      parameter (pi=3.14159265358979323846d0)

      integer i,j,k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)

      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379660D0)  !CONV TO PICOBARNS             
      integer i_process
      common/c_i_process/i_process

      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer save_nFKSprocess
c FxFx merging
      logical rewgt_mohdr_calculated,rewgt_izero_calculated
      double precision rewgt_mohdr,rewgt_izero,rewgt_exp_mohdr
     $     ,rewgt_exp_izero
      logical setclscales
      double precision rewgt
      external setclscales,rewgt
      double precision rwgt_muR_dep_fac
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
      save_nFKSprocess=nFKSprocess
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if(kwgtinfo.ne.4.and.kwgtinfo.ne.5) wgtbpower=rwgtbpower
c
      xsec=0.d0
     
      temp=0d0
      do i=1,iwgtnumpartn
         temp=temp+abs(wgtwmcxsec(i))
      enddo
      if (temp.eq.0d0) goto 541
      call set_cms_stuff(izero)

      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     $     (kwgtinfo.eq.2.and.wgtkin(0,1,1).gt.0.d0) .or.
     $     ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and.wgtkin(0,1,1).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.2.or.kwgtinfo.le.5)then
           call set_cms_stuff(mohdr)
           if (ickkw.eq.3) then
              mu_r=sqrt(wgtmuR2(1))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(1)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(1)
           else
              call set_alphaS(wgtkin(0,1,1))
           endif
        else
          write(*,*)'Error #0a in compute_rwgt_wgt_Sev',kwgtinfo
          stop
        endif
        do i=1,iwgtnumpartn
          xbk(1) = wgtmcxbj(1,i)
          xbk(2) = wgtmcxbj(2,i)
          if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #       xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
            if(wgtwmcxsec(i).ne.0.d0)then
              write(*,*)'Error #1 in compute_rwgt_wgt_Sev'
              write(*,*)i,xbk(1),xbk(2),wgtwmcxsec(i)
              stop
            endif
          else
            nFKSprocess=nFKSprocess_used
            xlum = dlum()
            do j=1,iproc_save(nFKSprocess)
               if (eto(j,nFKSprocess).eq.i_process) then
                  xsec=xsec+CONV*PD(j)*wgtwmcxsec(i)*g**(2*wgtbpower
     $                 +2.d0) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
               endif
            enddo
          endif
        enddo
      endif
c
 541  continue
      if ( wgtwreal(2).eq.0d0 .and.
     $     wgtwreal(3).eq.0d0 .and. wgtwreal(4).eq.0d0 .and.
     $     wgtwdeg(2).eq.0d0 .and.
     $     wgtwdeg(3).eq.0d0 .and. wgtwdeg(4).eq.0d0 .and.
     $     wgtwdegmuf(2).eq.0d0 .and.
     $     wgtwdegmuf(3).eq.0d0 .and. wgtwdegmuf(4).eq.0d0 .and.
     $     wgtwborn(2).eq.0d0 .and. wgtwns(2).eq.0d0 .and.
     $     wgtwnsmuf(2).eq.0d0 .and. wgtwnsmur(2).eq.0d0) goto 542

      call set_cms_stuff(izero)

      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(2).ne.0.d0) .or.
     $     ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and. wgtkin(0,1,2).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(2))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(2) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(2) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.3.or.kwgtinfo.le.5)then
           if (ickkw.eq.3) then
              mu_r=sqrt(wgtmuR2(2))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(2)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(2)
              qes2=wgtqes2(2)
           else
              call set_alphaS(wgtkin(0,1,2))
           endif
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_Sev',kwgtinfo
          stop
        endif
        QES2_local=wgtqes2(2)
        if(QES2_local.eq.0.d0)then
          if(wgtwdegmuf(3).ne.0.d0.or.
     #       wgtwdegmuf(4).ne.0.d0.or.
     #       wgtwnsmuf(2).ne.0.d0.or.
     #       wgtwnsmur(2).ne.0.d0)then
            write(*,*)'Error in compute_rwgt_wgt_Sev'
            write(*,*)' Ellis-Sexton scale was not set'
            write(*,*)wgtwdegmuf(3),wgtwdegmuf(4),
     #                wgtwnsmuf(2),wgtwnsmur(2)
            stop
          endif
          xlgmuf=0.d0
          xlgmur=0.d0
        else
           if(abs(QES2/QES2_local-1.d0).gt.tiny.and.
     &          (kwgtinfo.ge.3.or.kwgtinfo.le.5))then
              write(*,*)'Error in compute_rwgt_wgt_Sev'
              write(*,*)' Mismatch in ES scale',QES2,QES2_local
              stop
           endif
          xlgmuf=log(q2fact(1)/QES2_local)
          xlgmur=log(scale**2/QES2_local)
        endif
        do k=2,4
          xbk(1) = wgtxbj(1,k)
          xbk(2) = wgtxbj(2,k)
          if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #       xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
            if(wgtwreal(k).ne.0.d0.or.
     #         wgtwdeg(k).ne.0.d0.or.
     #         wgtwdegmuf(k).ne.0.d0.or.
     #         (k.eq.2.and.(wgtwborn(2).ne.0.d0.or.
     #                      wgtwns(2).ne.0.d0.or.
     #                      wgtwnsmuf(2).ne.0.d0.or.
     #                      wgtwnsmur(2).ne.0.d0)))then
              write(*,*)'Error #2 in compute_rwgt_wgt_Sev'
              write(*,*)k,xbk(1),xbk(2)
              write(*,*)wgtwreal(k),wgtwdeg(k),wgtwdegmuf(k)
              if(k.eq.2)write(*,*)wgtwborn(k),wgtwns(k),
     #                            wgtwnsmuf(k),wgtwnsmur(k)
              stop
            endif
          else
            nFKSprocess=nFKSprocess_used
            xlum = dlum()
            do j=1,iproc_save(nFKSprocess)
               if (eto(j,nFKSprocess).eq.i_process) then
                  xsec=xsec+CONV*PD(j)*( wgtwreal(k)+wgtwdeg(k)
     $                 +wgtwdegmuf(k)*xlgmuf )*g**(2*wgtbpower+2.d0)
     f                 * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
               endif
            enddo
            if(k.eq.2)then
              nFKSprocess=nFKSprocess_used_Born
              xlum = dlum()
              do j=1,iproc_save(nFKSprocess)
                 if (eto(j,nFKSprocess).eq.i_process) then
                    if(wgtbpower.gt.0)then
                       xsec=xsec+CONV*PD(j)*wgtwborn(k)*g**(2*wgtbpower)
     f                      * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                    else
                       xsec=xsec+CONV*PD(j)*wgtwborn(k)
     f                      * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                    endif
                    xsec=xsec+CONV*PD(j)*( wgtwns(k)+ wgtwnsmuf(k)
     $                   *xlgmuf+wgtwnsmur(k)*xlgmur )*g**(2*wgtbpower
     $                   +2.d0) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                 endif
              enddo
            endif
          endif
        enddo
      endif
c
 542  continue
      if (wgtwreal(1).eq.0d0) goto 543
      call set_cms_stuff(mohdr)

      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     $     ((kwgtinfo.ge.2.or.kwgtinfo.ge.5).and.wgtkin(0,1,1).gt.0.d0)
     $     )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.ge.2.or.kwgtinfo.le.5)then
           if (ickkw.eq.3) then
              mu_r=sqrt(wgtmuR2(1))*muR_over_ref
              scale=mu_r
              g=sqrt(4d0*pi*alphas(scale))
              call update_as_param()
              q2fact(1)=muF1_over_ref**2*wgtmuF12(1)
              q2fact(2)=muF2_over_ref**2*wgtmuF22(1)
           else
              call set_alphaS(wgtkin(0,1,1))
           endif
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_Sev',kwgtinfo
          stop
        endif
        xbk(1) = wgtxbj(1,1)
        xbk(2) = wgtxbj(2,1)
        if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #     xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
          if(wgtwreal(1).ne.0.d0)then
            write(*,*)'Error #3 in compute_rwgt_wgt_Sev'
            write(*,*)xbk(1),xbk(2),wgtwreal(1)
            stop
          endif
        else
          nFKSprocess=nFKSprocess_used
          xlum = dlum()
          do j=1,iproc_save(nFKSprocess)
             if (eto(j,nFKSprocess).eq.i_process) then
                xsec=xsec+CONV*PD(j)*wgtwreal(1)*g**(2*wgtbpower+2.d0)
     f               * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
             endif
          enddo
        endif
      endif
c
 543  continue
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
      nFKSprocess=save_nFKSprocess
c
      compute_rwgt_wgt_Sev=xsec
c
      return
      end


      function compute_rwgt_wgt_Sev_nbody(xmuR_over_ref,xmuF1_over_ref,
     &     xmuF2_over_ref,xQES_over_ref, kwgtinfo)
c Recomputes the S-event cross section using the weights saved, and compares
c with the reference weight
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      include 'run.inc'
      include 'nFKSconfigs.inc'
      include "reweight_all.inc"

      double precision compute_rwgt_wgt_Sev_nbody
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,xlgmuf,xlgmur,alphas
      double precision QES2_local
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision tiny,pi
      parameter (tiny=1.d-2)
      parameter (pi=3.14159265358979323846d0)

      integer i,j,k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)

      integer iproc_save(fks_configs),eto(maxproc,fks_configs)
     $     ,etoi(maxproc,fks_configs),maxproc_found
      common/cproc_combination/iproc_save,eto,etoi,maxproc_found
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379660D0)  !CONV TO PICOBARNS             
      integer i_process
      common/c_i_process/i_process

      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer save_nFKSprocess
c FxFx merging
      logical rewgt_mohdr_calculated,rewgt_izero_calculated
      double precision rewgt_mohdr,rewgt_izero,rewgt_exp_mohdr
     $     ,rewgt_exp_izero
      logical setclscales
      double precision rewgt
      external setclscales,rewgt
      double precision rwgt_muR_dep_fac
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
      save_nFKSprocess=nFKSprocess
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if (kwgtinfo.ne.5) then
         write (*,*) 'nbody-reweighting only with kwgtinfo.eq.5'
     &        ,kwgtinfo
      endif
c
      xsec=0.d0

      if ( wgtwborn_all.eq.0d0 .and. wgtwns_all.eq.0d0 .and.
     $     wgtwnsmuf_all.eq.0d0 .and. wgtwnsmur_all.eq.0d0) goto 541

      call set_cms_stuff(izero)

      if( wgtkin_all(0,1,2,0).gt.0d0 )then
         if (ickkw.eq.3) then
            mu_r=sqrt(wgtmuR2_all(2,0))*muR_over_ref
            scale=mu_r
            g=sqrt(4d0*pi*alphas(scale))
            call update_as_param()
            q2fact(1)=muF1_over_ref**2*wgtmuF12_all(2,0)
            q2fact(2)=muF2_over_ref**2*wgtmuF22_all(2,0)
            qes2=wgtqes2_all(2,0)
         else
            call set_alphaS(wgtkin_all(0,1,2,0))
         endif
         QES2_local=wgtqes2_all(2,0)
         if (QES2_local.ne.0d0) then
            if(abs(QES2/QES2_local-1.d0).gt.tiny)then
               write(*,*)'Error in compute_rwgt_wgt_Sev_nbody'
               write(*,*)' Mismatch in ES scale',QES2,QES2_local
               stop
            endif
            xlgmuf=log(q2fact(1)/QES2_local)
            xlgmur=log(scale**2/QES2_local)
            xbk(1) = wgtxbj_all(1,2,0)
            xbk(2) = wgtxbj_all(2,2,0)
            if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #         xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
               if(wgtwborn_all.ne.0d0 .or. wgtwns_all.ne.0d0 .or.
     $           wgtwnsmuf_all.ne.0d0 .or. wgtwnsmur_all.ne.0d0)then
                  write(*,*)'Error #3 in compute_rwgt_wgt_Sev_nbody'
                  write(*,*) QES2_local,QES2,wgtwborn_all,wgtwns_all
     $                 ,wgtwnsmuf_all,wgtwnsmur_all
                  stop
               endif
            else
               nFKSprocess=nFKSprocess_used_Born
               xlum = dlum()
               do j=1,iproc_save(nFKSprocess)
                  if (eto(j,nFKSprocess).eq.i_process) then
                     if(wgtbpower.gt.0)then
                        xsec=xsec+CONV*PD(j)*wgtwborn_all*g**(2
     $                       *wgtbpower) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                     else
                        xsec=xsec+CONV*PD(j)*wgtwborn_all
     f                       * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                     endif
                     xsec=xsec+CONV*PD(j)*( wgtwns_all+ wgtwnsmuf_all
     $                    *xlgmuf+wgtwnsmur_all*xlgmur )*g**(2*wgtbpower
     $                    +2.d0) * rwgt_muR_dep_fac(scale,scale)
com-- muR-dependent fac is reweighted here
                  endif
               enddo
            endif
         else
            if (wgtwborn_all.ne.0d0 .or. wgtwns_all.ne.0d0 .or.
     $           wgtwnsmuf_all.ne.0d0 .or. wgtwnsmur_all.ne.0d0) then
               write(*,*)'ES scale is zero, but weights are not'
               write(*,*) QES2_local,QES2,wgtwborn_all,wgtwns_all
     $              ,wgtwnsmuf_all,wgtwnsmur_all
               stop
            endif
         endif
      endif
 541  continue
c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
      nFKSprocess=save_nFKSprocess
c
      compute_rwgt_wgt_Sev_nbody=xsec
c
      return
      end


      subroutine setup_fill_rwgt_NLOplot()
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'run.inc'
      include "reweight.inc"
      include "reweightNLO.inc"
      include "../../Source/pdf.inc"
      integer i,itmp,nsets
      double precision delta
c
      yQES_over_ref=QES_over_ref
      ymuR_over_ref=muR_over_ref
      ymuF1_over_ref=muF1_over_ref
      ymuF2_over_ref=muF2_over_ref
c
      if(.not.do_rwgt_scale)goto 111
      numscales=3
      if(numscales.gt.maxscales)then
        write(*,*)'Error #1 in setup_fill_rwgt_NLOplot'
        write(*,*)' Increase maxscales in reweight0.inc'
        stop
      endif
      yfactF(1)=1d0
      yfactF(2)=rw_Fscale_up
      yfactF(3)=rw_Fscale_down
      yfactR(1)=1d0
      yfactR(2)=rw_Rscale_up
      yfactR(3)=rw_Rscale_down
c
 111  continue
      if(.not.do_rwgt_pdf)goto 222
      idpdf(0)=lhaid
      idpdf(1)=pdf_set_min
      itmp=pdf_set_max
      nsets=itmp-idpdf(1)+1
      if(mod(nsets,2).ne.0)then
        write(*,*)'The number of error sets must be even',nsets
        stop
      else
        numPDFpairs=nsets/2
      endif
      do i=2,nsets
        idpdf(i)=idpdf(1)+i-1
      enddo
      if(nsets.gt.maxPDFs)then
        write(*,*)'Error #2 in setup_fill_rwgt_NLOplot'
        write(*,*)' Increase maxPDFs in reweight0.inc'
        stop
      endif
      numPDFs=nsets+1
c
 222  continue
      return
      end

c     This function defines the reweighting of the cross section to
c     include a muR-dependent pre-factor. Multiply by the muR-dependent
c     factor and devide by the muR-independent one.
c Note: This is implemented below for the Bottom Yukawa in the SM.
c       Change it to the factor you need to reweight.
      Double precision function rwgt_muR_dep_fac(scale,central)
      implicit none
      double precision scale,vev,mbmb,apimuR,apimZ,apimb,mbmuR,alphas,pi
      parameter (pi=3.14159265358979323846d0)
      include "nexternal.inc"
      include "genps.inc"
      include "reweight.inc"
      include "coupl.inc"
      include "../../Source/MODEL/input.inc"
      integer i
      double precision central,tootiny,apicentral,mbcentral
      parameter (tootiny=1d-9)
      rwgt_muR_dep_fac = 1d0
c     This is relevant for a muR-dependent bottom-mass in Yukawa.
      IF(wgtcpower .ne. 0d0 .and. runfac .eq. 1) THEN
c$$$      vev    = 246.21845813429518469305d0 !vev in aMC@NLO from y_b->m_b
c$$$      mbmb = MDL_YB*vev/dsqrt(2d0)
c$$$com-- mbmb input for fixed Yukawa bmass in param_card.dat is used here
c$$$com-- as start value of running and to remove it from the cross section
c$$$c new settings NLO
c$$$      apimZ  = alphas(MDL_MZ)/pi
c$$$
c$$$      if(dabs(scale/central-1d0).lt.tootiny) then
c$$$c if scale muR is the same as the central scale of muR, get 
c$$$c "input value" mb(muR) with highest possible accuracy
c$$$         CALL runalpha(apimZ,MDL_MZ,central,4d0,4,0,apimuR)
c$$$         CALL runalpha(apimZ,MDL_MZ,mbmb,4d0,4,0,apimb)
c$$$         CALL runmass(mbmb,apimb,apimuR,4d0,4,mbmuR)
c$$$      else
c$$$c if scale and central are different (muR variations) do two steps:
c$$$c step 1: get "input value" mb(central scale) from most accurate running
c$$$         CALL runalpha(apimZ,MDL_MZ,central,4d0,4,0,apicentral)
c$$$         CALL runalpha(apimZ,MDL_MZ,mbmb,4d0,4,0,apimb)
c$$$         CALL runmass(mbmb,apimb,apicentral,4d0,4,mbcentral)
c$$$c step 2: get variation around central value, ie mb(muR), with loop 
c$$$c         order consistent with computation LO: 1-loop, NLO: 2-loop
c$$$         CALL runalpha(apicentral,central,scale,4d0,2,0,apimuR)
c$$$         CALL runmass(mbcentral,apicentral,apimuR,4d0,2,mbmuR)
c$$$      endif
c$$$      rwgt_muR_dep_fac = (mbmuR/mbmb)**wgtcpower
      ELSE
         return
      ENDIF
      END

C-{{{ routines for running of alphas:

C-{{{ subroutine odeint:

      SUBROUTINE odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,
     *rkqs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER nbad,nok,nvar,KMAXX,MAXSTP,NMAX
      REAL*8 eps,h1,hmin,x1,x2,ystart(nvar),TINY
      EXTERNAL derivs,rkqs
      PARAMETER (MAXSTP=10000,NMAX=50,KMAXX=200,TINY=1.e-30)
      INTEGER i,kmax,kount,nstp
      REAL*8 dxsav,h,hdid,hnext,x,xsav,dydx(NMAX),xp(KMAXX),y(NMAX),
     *yp(NMAX,KMAXX),yscal(NMAX)
      COMMON /path/ kmax,kount,dxsav,xp,yp
      x=x1
      h=sign(h1,x2-x1)
      nok=0
      nbad=0
      kount=0
      do 11 i=1,nvar
        y(i)=ystart(i)
11    continue
      if (kmax.gt.0) xsav=x-2.*dxsav
      do 16 nstp=1,MAXSTP
        call derivs(x,y,dydx)
        do 12 i=1,nvar
          yscal(i)=dabs(y(i))+dabs(h*dydx(i))+TINY
12      continue
        if(kmax.gt.0)then
          if(dabs(x-xsav).gt.dabs(dxsav)) then
            if(kount.lt.kmax-1)then
              kount=kount+1
              xp(kount)=x
              do 13 i=1,nvar
                yp(i,kount)=y(i)
13            continue
              xsav=x
            endif
          endif
        endif
        if((x+h-x2)*(x+h-x1).gt.0.) h=x2-x
        call rkqs(y,dydx,nvar,x,h,eps,yscal,hdid,hnext,derivs)
        if(hdid.eq.h)then
          nok=nok+1
        else
          nbad=nbad+1
        endif
        if((x-x2)*(x2-x1).ge.0.)then
          do 14 i=1,nvar
            ystart(i)=y(i)
14        continue
          if(kmax.ne.0)then
            kount=kount+1
            xp(kount)=x
            do 15 i=1,nvar
              yp(i,kount)=y(i)
15          continue
          endif
          return
        endif
        if(dabs(hnext).lt.hmin) write(6,*)
     *       'stepsize smaller than minimum in odeint'
        h=hnext
16    continue
      write(6,*) 'too many steps in odeint'
      stop
      return
      END

C-}}}
C-{{{ subroutine rkck:

      SUBROUTINE rkck(y,dydx,n,x,h,yout,yerr,derivs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER n,NMAX
      REAL*8 h,x,dydx(n),y(n),yerr(n),yout(n)
      EXTERNAL derivs
      PARAMETER (NMAX=50)
CU    USES derivs
      INTEGER i
      REAL*8 ak2(NMAX),ak3(NMAX),ak4(NMAX),ak5(NMAX),ak6(NMAX),
     *ytemp(NMAX),A2,A3,A4,A5,A6,B21,B31,B32,B41,B42,B43,B51,B52,B53,
     *B54,B61,B62,B63,B64,B65,C1,C3,C4,C6,DC1,DC3,DC4,DC5,DC6
      PARAMETER (A2=.2,A3=.3,A4=.6,A5=1.,A6=.875,B21=.2,B31=3./40.,
     *B32=9./40.,B41=.3,B42=-.9,B43=1.2,B51=-11./54.,B52=2.5,
     *B53=-70./27.,B54=35./27.,B61=1631./55296.,B62=175./512.,
     *B63=575./13824.,B64=44275./110592.,B65=253./4096.,C1=37./378.,
     *C3=250./621.,C4=125./594.,C6=512./1771.,DC1=C1-2825./27648.,
     *DC3=C3-18575./48384.,DC4=C4-13525./55296.,DC5=-277./14336.,
     *DC6=C6-.25)
      do 11 i=1,n
        ytemp(i)=y(i)+B21*h*dydx(i)
11    continue
      call derivs(x+A2*h,ytemp,ak2)
      do 12 i=1,n
        ytemp(i)=y(i)+h*(B31*dydx(i)+B32*ak2(i))
12    continue
      call derivs(x+A3*h,ytemp,ak3)
      do 13 i=1,n
        ytemp(i)=y(i)+h*(B41*dydx(i)+B42*ak2(i)+B43*ak3(i))
13    continue
      call derivs(x+A4*h,ytemp,ak4)
      do 14 i=1,n
        ytemp(i)=y(i)+h*(B51*dydx(i)+B52*ak2(i)+B53*ak3(i)+B54*ak4(i))
14    continue
      call derivs(x+A5*h,ytemp,ak5)
      do 15 i=1,n
        ytemp(i)=y(i)+h*(B61*dydx(i)+B62*ak2(i)+B63*ak3(i)+B64*ak4(i)+
     *B65*ak5(i))
15    continue
      call derivs(x+A6*h,ytemp,ak6)
      do 16 i=1,n
        yout(i)=y(i)+h*(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i))
16    continue
      do 17 i=1,n
        yerr(i)=h*(DC1*dydx(i)+DC3*ak3(i)+DC4*ak4(i)+DC5*ak5(i)+DC6*
     *ak6(i))
17    continue
      return
      END

C-}}}
C-{{{ subroutine rkqs:

      SUBROUTINE rkqs(y,dydx,n,x,htry,eps,yscal,hdid,hnext,derivs)
c..(C) Copr. 1986-92 Numerical Recipes Software 5,".
c..   transscribed to real*8 by R. Harlander, Feb.2002
      implicit real*8 (a-z)
      INTEGER n,NMAX
      REAL*8 eps,hdid,hnext,htry,x,dydx(n),y(n),yscal(n)
      EXTERNAL derivs
      PARAMETER (NMAX=50)
CU    USES derivs,rkck
      INTEGER i
      REAL*8 errmax,h,htemp,xnew,yerr(NMAX),ytemp(NMAX),SAFETY,PGROW,
     *PSHRNK,ERRCON
      PARAMETER (SAFETY=0.9,PGROW=-.2,PSHRNK=-.25,ERRCON=1.89d-4)
      h=htry
1     call rkck(y,dydx,n,x,h,ytemp,yerr,derivs)
      errmax=0.d0
      do 11 i=1,n
        errmax=max(errmax,dabs(yerr(i)/yscal(i)))
11    continue
      errmax=errmax/eps
      if(errmax.gt.1.)then
        htemp=SAFETY*h*(errmax**PSHRNK)
        h=sign(max(dabs(htemp),0.1*dabs(h)),h)
        xnew=x+h
        if(xnew.eq.x) write(6,*) 'stepsize underflow in rkqs'
        goto 1
      else
        if(errmax.gt.ERRCON)then
          hnext=SAFETY*h*(errmax**PGROW)
        else
          hnext=5.*h
        endif
        hdid=h
        x=x+h
        do 12 i=1,n
          y(i)=ytemp(i)
12      continue
        return
      endif
      END

C-}}}
C-{{{ subroutine runalpha:

      subroutine runalpha(api0,mu0,mu,nf,nloop,verb,apiout)
C..
c..   NEEDS:  rkck.f rkqs.f odeint.f  (from Numerical Recipes)
c..   
c..   Note:  api = {\alpha_s \over \pi}
C..
c..   purpose : computes the value of api(mu) from api(mu0)
c..   method  : solving RG-equation by adaptive Runge-Kutta method
c..   uses    : odeint.for  from Numerical Recipes
C..
c..   api0  :  api(mu0)
c..   nf    :  number of flavors
c..   nloop :  number of loops
c..   verb  :  0=quiet,  1=verbose
c..   apiout:  api(mu)    
C..
      implicit real*8 (a-h,o-z)
      INTEGER KMAXX,NMAX,NVAR
      PARAMETER (KMAXX=200,NMAX=50,NVAR=1)
      INTEGER kmax,kount,nbad,nok,nrhs,verb
      REAL*8 dxsav,eps,h1,hmin,x,y,apif(NVAR),api0,apiout,pi
      real*8 mu,mu0,l0,lf,nf
c..   /path/  is for odeint.for:
      COMMON /path/ kmax,kount,dxsav,x(KMAXX),y(NMAX,KMAXX)
      common /bfunc/ beta0,beta1,beta2,beta3
      COMMON /cbnrhs/nrhs 
      data pi/3.14159265358979323846264338328d0/
      EXTERNAL rhs,rkqs

      if (nloop.eq.0) then
         apiout = api0
         return
      endif

      nrhs=0

c..   integration bounds (note that log(mu^2) is the integration variable)
      l0 = 0.d0
      lf = 2.*dlog(mu/mu0)
      apif(1)=api0

c..   see documentation for odeint.for:
      eps=1.0d-8
      h1=dabs(lf-l0)/10.d0
      hmin=0.0d0
      kmax=100
      dxsav=dabs(lf-l0)/20.d0

c..   initialize beta-function (common block /bfunc/):
      call inibeta(nf,nloop)

c..   check if input values are reasonable
      dlam = mu0*dexp(-1.d0/(2.d0*beta0*api0))
      if (mu.le.dlam) then
         write(6,2001) dlam,mu,mu0,api0*pi
      endif

c..   integrate RG-equation:

      call odeint(apif,NVAR,l0,lf,eps,h1,hmin,nok,nbad,rhs,rkqs)

      if (verb.eq.1) then
         write(6,'(/1x,a,t30,i3)') 'Successful steps:',nok
         write(6,'(1x,a,t30,i3)') 'Bad steps:',nbad
         write(6,'(1x,a,t30,i3)') 'Function evaluations:',nrhs
         write(6,'(1x,a,t30,i3)') 'Stored intermediate values:',kount
      endif

c..   api(mu):
      apiout = apif(1)

 2001 format(' -> <subroutine runalpha>',/,
     &     ' - WARNING: mu-value too low.',/,
     &     ' -     Should be significantly larger than  ',1f8.3,'.',/,
     &     ' -             mu = ',1f8.3,' GeV',/,
     &     ' -            mu0 = ',1f8.3,' GeV',/,
     &     ' -        api0*pi = ',1f8.3,/,
     &     ' -     Integration might break down.',/,
     &     '<- <subroutine runalpha>'
     &     )

      END

C-}}}
C-{{{ subroutine rhs:

      subroutine rhs(logmumu0,api,ainteg)
C..
c..   RG-equation:   (d api)/(d log(mu^2)) = api*beta(api)
C..
      implicit real*8 (a-h,o-z)
      integer nrhs
      real*8 api(*),ainteg(*),logmumu0
      common /bfunc/ beta0,beta1,beta2,beta3
      COMMON nrhs
      nrhs=nrhs+1
      ainteg(1) = api(1)*(- beta0*api(1) - beta1*api(1)**2 - beta2
     &     *api(1)**3 - beta3*api(1)**4)
      end

C-}}}
C-{{{ subroutine inibeta:

      subroutine inibeta(nf,nloopin)
C..
c..   initialize beta function
C..
      implicit real*8 (a-h,o-z)
      real*8 nf
      data z3/1.2020569031595942853997/
      common /bfunc/ beta0,beta1,beta2,beta3

      beta0 = (33 - 2*nf)/12.d0
      beta1 = (102 - (38*nf)/3.d0)/16.d0
      beta2 = (2857/2.d0 - (5033*nf)/18.d0 + (325*nf**2)/54.d0)/64.d0
      beta3 = (149753/6.d0 + (1093*nf**3)/729.d0 + 3564*z3 + nf**2
     &     *(50065/162.d0 + (6472*z3)/81.d0) - nf*(1078361/162.d0 +
     &     (6508*z3)/27.d0))/256.d0

      
      nloop=nloopin

      if (nloop.gt.4) then
         write(6,*) '-> <subroutine inibeta>:'
         write(6,*)
     &        ' - 5-loop beta function unknown. Using 4-loop instead.'
         write(6,*) '<- <subroutine inibeta>'
         nloop=4
      endif
      if (nloop.lt.4) then
         beta3 = 0d0
         if (nloop.lt.3) then
            beta2 = 0d0
            if (nloop.lt.2) then
               beta1 = 0d0
               if (nloop.lt.1) then
                  beta0=0d0
               endif
            endif
         endif
      endif
      end

C-}}}

C-}}}
C-{{{ subroutine runmass:

      subroutine runmass(mass0,api0,apif,nf,nloop,massout)
c..
c..   evaluates the running of the MS-bar quark mass
c..   by expanding the equation
c..   
c..   m(mu) = m(mu0) * exp( \int_a0^af dx gammam(x)/x/beta(x) )
c..   
c..   in terms of alpha_s. The results agree with RunDec.m.
c..   
c..   
c..   Input:
c..   ------
c..   mass0  :  m(mu0)
c..   api0   :  alpha_s(mu0)/pi
c..   apif   :  alpha_s(muf)/pi
c..   nf     :  number of flavors
c..   nloop  :  order of calculation (nloop=1..4)
c..
c..   Output:
c..   -------
c..   massout:  m(muf)
c..   
      implicit real*8 (a-h,o-z)
      real*8 mass0,massout,massfun
      real*8 nf
      external massfun
      parameter(accmass=1.d-6)
      common /bfunc/ beta0,beta1,beta2,beta3
      common /gfunc/ gamma0,gamma1,gamma2,gamma3

      if (nloop.eq.0) then
         massout = mass0
         return
      endif

      call inigamma(nf,nloop)
      call inibeta(nf,nloop)

      bb1 = beta1/beta0
      bb2 = beta2/beta0
      bb3 = beta3/beta0

      cc0 = gamma0/beta0
      cc1 = gamma1/beta0
      cc2 = gamma2/beta0
      cc3 = gamma3/beta0

      cfunc1 = 1.d0
      cfunc2 = cc1 - bb1*cc0
      cfunc3 = 1/2.d0*((cc1-bb1*cc0)**2 + cc2 - bb1*cc1 + bb1**2*cc0 -
     &     bb2*cc0)
      cfunc4 = (1/6*(cc1 - bb1*cc0)**3 + 1/2*(cc1 - bb1*cc0)*(cc2 - bb1
     &     *cc1 + bb1**2*cc0 - bb2*cc0) + 1/3*(cc3 - bb1*cc2 + bb1**2
     &     *cc1 - bb2*cc1 - bb1**3*cc0 + 2*bb1*bb2*cc0 - bb3*cc0))

      if (nloop.lt.4) then
         cfunc4 = 0.d0
         if (nloop.lt.3) then
            cfunc3 = 0.d0
            if (nloop.lt.2) then
               cfunc2 = 0.d0
               if (nloop.lt.1) then
                  cfunc1 = 0.d0
               endif
            endif
         endif
      endif

      cfuncmu0 = cfunc1 + cfunc2*api0 + cfunc3*api0**2 + cfunc4*api0**3
      cfuncmuf = cfunc1 + cfunc2*apif + cfunc3*apif**2 + cfunc4*apif**3


      massout = mass0*(apif/api0)**cc0*cfuncmuf/cfuncmu0
      
      return
      end

C-}}}
C-{{{ subroutine inigamma:

      subroutine inigamma(nfin,nloopin)
C
C     initialize beta function
C
      implicit real*8 (a-h,o-z)
      real*8 nf,nfin
      data z3/1.2020569031595942853997/,
     &     z5/1.0369277551433699263/,
     &     pi/3.1415926535897932381/
      common /gfunc/ gamma0,gamma1,gamma2,gamma3

      nf = nfin

      gamma0 = 1.d0
      gamma1 = (67.33333333333333d0 - (20*nf)/9.d0)/16.d0
      gamma2 = (1249.d0 - (140*nf**2)/81.d0 + 2*nf*(-20.59259259259259d0
     &     - 48*z3) +(8*nf*(-46 + 48*z3))/9.d0)/64.d0
      gamma3 = (28413.91975308642d0 + (135680*z3)/27.d0 + nf**3*(-1
     &     .3662551440329218d0 + (64*z3)/27.d0) + nf**2*(21
     &     .57201646090535d0 - (16*Pi**4)/27.d0 + (800*z3)/9.d0) - 8800
     &     *z5 + nf*(-3397.1481481481483d0 + (88*Pi**4)/9.d0 - (34192
     &     *z3)/9.d0 + (18400*z5)/9.d0))/256.d0

      nloop=nloopin

      if (nloop.gt.4) then
         write(6,*) '-> <subroutine inigamma>:'
         write(6,*)
     &        ' - 5-loop gamma function unknown. Using 4-loop instead.'
         write(6,*) '<- <subroutine inigamma>'
         nloop=4
      endif
      if (nloop.lt.4) then
         gamma3 = 0d0
         if (nloop.lt.3) then
            gamma2 = 0d0
            if (nloop.lt.2) then
               gamma1 = 0d0
               if (nloop.lt.1) then
                  gamma0 = 0d0
               endif
            endif
         endif
      endif
      end


