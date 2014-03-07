c This file contains the routines relevant to reweighting 
c NLO and aMC@NLO results, for the computation of scale 
c and PDF uncertainties

      subroutine reweight_fillkin(pp,itype)
c Fills four-momenta common blocks, according to itype
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "reweight.inc"

      double precision pp(0:3,nexternal)
      integer itype

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      integer i,j,icnt,ione
      parameter (ione=1)
c
      if(itype.eq.1)then
        do i=1,nexternal
          do j=0,3
            wgtkin(j,i,ione)=pp(j,i)
          enddo
        enddo
      elseif(itype.ge.2.and.itype.le.4)then
        icnt=itype-2
        do i=1,nexternal
          do j=0,3
            wgtkin(j,i,itype)=p1_cnt(j,i,icnt)
          enddo
        enddo
      else
        write(*,*)'Error in reweight_fillkin: itype=',itype
        stop
      endif
      return
      end


      subroutine reweight_fillkin_all(pp,itype,iFKS)
c Fills four-momenta common blocks, according to itype
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "reweight_all.inc"

      double precision pp(0:3,nexternal)
      integer itype,iFKS

      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      integer i,j,icnt,ione
      parameter (ione=1)
c
      if(itype.eq.1)then
        do i=1,nexternal
          do j=0,3
            wgtkin_all(j,i,ione,iFKS)=pp(j,i)
          enddo
        enddo
      elseif(itype.ge.2.and.itype.le.4)then
        icnt=2-itype
        do i=1,nexternal
          do j=0,3
            wgtkin_all(j,i,itype,iFKS)=p1_cnt(j,i,icnt)
          enddo
        enddo
      else
        write(*,*)'Error in reweight_fillkin: itype=',itype
        stop
      endif
      return
      end


      subroutine reweight_replkin_cnt()
c Fills wgtkin(*,*,2) with the content of wgtkin(*,*,3) or wgtkin(*,*,4).
c This is needed since only wgtkin(*,*,0) is used in the computations of
c cross sections through weights, and counterterms may be non zero
c only for collinear or soft-collinear configurations (eg because of
c choices of xicut and delta's). This is fine so long as the counterevent
c kinematic configurations are identical for the computations performed
c here 
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "reweight.inc"

      integer i,j,icnt,itwo
      parameter (itwo=2)
c
      if (wgtkin(0,1,2).gt.0.d0 .or.
     #   (wgtkin(0,1,3).lt.0.d0.and.wgtkin(0,1,4).lt.0.d0) )then
        write(*,*)'Error in reweight_replkin_cnt'
        write(*,*)wgtkin(0,1,2),wgtkin(0,1,3),wgtkin(0,1,4)
        stop
      endif
c
      if(wgtkin(0,1,4).gt.0.d0)then
        icnt=4
      else
        icnt=3
      endif
c
      do i=1,nexternal
        do j=0,3
          wgtkin(j,i,itwo)=wgtkin(j,i,icnt)
        enddo
      enddo
c
      return
      end


      subroutine reweight_fill_extra()
c Fills arrays with dimensions maxparticles, equating them with their
c counterparts with dimensions nexternal. Needed before calling routines
c that only include reweight0.inc, such as read_lhef_event() and 
c write_lhef_event()
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "reweight.inc"
      integer i,j,k
c
      do k=1,4
        do i=1,nexternal
          do j=0,3
            wgtkinE(j,i,k)=wgtkin(j,i,k)
          enddo
        enddo
      enddo
c
      do k=1,nexternal
        wgtwmcxsecE(k)=wgtwmcxsec(k)
        wgtmcxbjE(1,k)=wgtmcxbj(1,k)
        wgtmcxbjE(2,k)=wgtmcxbj(2,k)
      enddo
c
      return
      end


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


      subroutine sum_reweight(iFKS_s,iFKS,iproc)
c Set all reweight variables equal to zero
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include "nFKSconfigs.inc"
      include "reweight_all.inc"
      integer i,j,k,iFKS_s,iFKS,iproc
      logical debug
      parameter (debug=.false.)
c
      if (debug) write (*,*) 'wgtref',iFKS_s,iFKS,wgtref_all(iFKS_s
     $     ,iproc),wgtref_all(iFKS,iproc)
      call reweight_sum(wgtref_all(iFKS_s,iproc),wgtref_all(iFKS,iproc))
      if (debug) write (*,*) 'wgtref_nbody',iFKS_s,iFKS
     $     ,wgtref_nbody_all(iproc),wgtref_nbody_all(iproc)
      call reweight_sum(wgtref_nbody_all(iproc),wgtref_nbody_all(iproc))
      do k=1,4
         if (debug) write (*,*) 'wgtqes2',k,iFKS_s,iFKS,wgtqes2_all(k
     &        ,iFKS_s),wgtqes2_all(k,iFKS)
         call reweight_check_equal(wgtqes2_all(k,iFKS_s),wgtqes2_all(k
     &        ,iFKS))
         if (debug) write (*,*) 'wgtxbj1',k,iFKS_s,iFKS,wgtxbj_all(1,k
     &        ,iFKS_s),wgtxbj_all(1,k,iFKS)
         if (k.eq.1.or.k.eq.2) then
            call reweight_check_equal(wgtxbj_all(1,k,iFKS_s)
     &           ,wgtxbj_all(1,k,iFKS))
         else
            call reweight_overwrite(wgtxbj_all(1,k,iFKS_s)
     &           ,wgtxbj_all(1,k,iFKS),2)
         endif
         if (debug) write (*,*) 'wgtxbj2',k,iFKS_s,iFKS,wgtxbj_all(2,k
     &        ,iFKS_s),wgtxbj_all(2,k,iFKS)
         if (k.eq.1.or.k.eq.2) then
            call reweight_check_equal(wgtxbj_all(2,k,iFKS_s)
     &           ,wgtxbj_all(2,k,iFKS))
         else
            call reweight_overwrite(wgtxbj_all(2,k,iFKS_s)
     &           ,wgtxbj_all(2,k,iFKS),2)
         endif
        do i=1,nexternal
          do j=0,3
             if (debug) write (*,*) 'wgtkin',j,i,k,iFKS_s,iFKS
     &            ,wgtkin_all(j,i,k,iFKS_s),wgtkin_all(j,i,k,iFKS)
             if (k.eq.1.or.k.eq.2) then
                call reweight_check_equal(wgtkin_all(j,i,k,iFKS_s)
     &               ,wgtkin_all(j,i,k,iFKS))
             else
                call reweight_overwrite(wgtkin_all(j,i,k,iFKS_s)
     &               ,wgtkin_all(j,i,k,iFKS),2)
             endif
          enddo
        enddo
        if (debug) write (*,*) 'wgtmuR2',k,iFKS_s,iFKS,wgtmuR2_all(k
     &       ,iFKS_s),wgtmuR2_all(k,iFKS)
        call reweight_check_equal(wgtmuR2_all(k,iFKS_s),wgtmuR2_all(k
     &       ,iFKS))
        if (debug) write (*,*) 'wgtmuF12',k,iFKS_s,iFKS,wgtmuF12_all(k
     &       ,iFKS_s),wgtmuF12_all(k,iFKS)
        call reweight_check_equal(wgtmuF12_all(k,iFKS_s),wgtmuF12_all(k
     &       ,iFKS))
        if (debug) write (*,*) 'wgtmuF22',k,iFKS_s,iFKS,wgtmuF22_all(k
     &       ,iFKS_s),wgtmuF22_all(k,iFKS)
        call reweight_check_equal(wgtmuF22_all(k,iFKS_s),wgtmuF22_all(k
     &       ,iFKS))
        if (debug) write (*,*) 'wgtwreal',k,iFKS_s,iFKS,wgtwreal_all(k
     &       ,iFKS_s),wgtwreal_all(k,iFKS)
        call reweight_sum(wgtwreal_all(k,iFKS_s),wgtwreal_all(k,iFKS))
        if (debug) write (*,*) 'wgtwdeg',k,iFKS_s,iFKS,wgtwdeg_all(k
     &       ,iFKS_s),wgtwdeg_all(k,iFKS)
        call reweight_sum(wgtwdeg_all(k,iFKS),wgtwdeg_all(k,iFKS))
        if (debug) write (*,*) 'wgtwdegmuF',k,iFKS_s,iFKS
     &       ,wgtwdegmuf_all(k,iFKS_s),wgtwdegmuf_all(k,iFKS)
        call reweight_sum(wgtwdegmuf_all(k,iFKS),wgtwdegmuf_all(k,iFKS))
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
         if (debug) write (*,*) 'wgtwmcxsec',k,iFKS_s,iFKS
     &        ,wgtwmcxsec_all(k,iFKS_s),wgtwmcxsec_all(k,iFKS)
         call reweight_sum(wgtwmcxsec_all(k,iFKS_s),wgtwmcxsec_all(k
     &        ,iFKS))
         if (debug) write (*,*) 'wgtwmcxbj1',k,iFKS_s,iFKS
     &        ,wgtmcxbj_all(1,k,iFKS_s),wgtmcxbj_all(1,k,iFKS)
         call reweight_check_equal(wgtmcxbj_all(1,k,iFKS_s)
     &        ,wgtmcxbj_all(1,k,iFKS))
         if (debug) write (*,*) 'wgtwmcxbj2',k,iFKS_s,iFKS
     &        ,wgtmcxbj_all(2,k,iFKS_s),wgtmcxbj_all(2,k,iFKS)
         call reweight_check_equal(wgtmcxbj_all(2,k,iFKS_s)
     &        ,wgtmcxbj_all(2,k,iFKS))
      enddo
      if (debug) write (*,*) 'iwgtnumpartn',iFKS
     &     ,iwgtnumpartn_all(iFKS_s),iwgtnumpartn_all(iFKS)
      call reweight_check_equal_int(iwgtnumpartn_all(iFKS_s)
     &     ,iwgtnumpartn_all(iFKS))
      return
      end

      subroutine reweight_check_equal_int(a,b)
      implicit none
      integer a,b
      if (a.ne.b) then
         write (*,*) 'Error #1 in reweight_check_equal_int',a,b
         stop
      endif
      return
      end

      subroutine reweight_check_equal(a,b)
      implicit none
      double precision a,b
      double precision vtiny
      parameter (vtiny=1d-10)
      if (a.ne.0) then
         if (abs(a-b)/abs(a).gt.vtiny) then
            write (*,*) 'Error #1 in reweight_check_equal',a,b
            stop
         endif
      elseif (b.ne.0) then
         if (abs(a-b)/abs(b).gt.vtiny) then
            write (*,*) 'Error #2 in reweight_check_equal',a,b
            stop
         endif
      endif
      return
      end

      subroutine reweight_sum(a,b)
      implicit none
      double precision a,b
      a=a+b
      return
      end


      function compute_rwgt_wgt_NLO(xmuR_over_ref,xmuF1_over_ref,
     #                              xmuF2_over_ref,xQES_over_ref,
     #                              kwgtinfo)
c Recomputes the NLO cross section using the weights saved, and compares
c with the reference weight
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'q_es.inc'
      include 'run.inc'
      include "reweight.inc"

      double precision compute_rwgt_wgt_NLO
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,xlgmuf,xlgmur,alphas
      double precision xsec11,xsec12,xsec20
      double precision QES2_local
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision tiny,pi
      parameter (tiny=1.d-2)
      parameter (pi=3.14159265358979323846d0)

      integer k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)
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
      xsec11=0.d0
      xsec12=0.d0
      xsec20=0.d0

      if (wgtwreal(1).eq.0d0) goto 541

      call set_cms_stuff(mohdr)
      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5)) )then
        if(kwgtinfo.eq.1)then
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
          write(*,*)'Error #0a in compute_rwgt_wgt_NLO',kwgtinfo
          stop
        endif
        xbk(1) = wgtxbj(1,1)
        xbk(2) = wgtxbj(2,1)
        if(xbk(1).le.0.d0.or.xbk(2).le.0.d0.or.
     #     xbk(1).gt.1.d0.or.xbk(2).gt.1.d0)then
          if(wgtwreal(1).ne.0.d0)then
            write(*,*)'Error #1 in compute_rwgt_wgt_NLO'
            write(*,*)xbk(1),xbk(2),wgtwreal(1)
            stop
          endif
        else
          nFKSprocess=nFKSprocess_used
          xlum = dlum()
          xsec11=xsec11+xlum*wgtwreal(1)*g**(2*wgtbpower+2.d0)
        endif
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
      if( (kwgtinfo.eq.1.and.wgtmuR2(2).ne.0.d0) .or.
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5)) )then
        if(kwgtinfo.eq.1)then
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
           else
              call set_alphaS(wgtkin(0,1,2))
           endif
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_NLO',kwgtinfo
          stop
        endif
        QES2_local=wgtqes2(2)
        if(QES2_local.eq.0.d0)then
          if(wgtwdegmuf(3).ne.0.d0.or.
     #       wgtwdegmuf(4).ne.0.d0.or.
     #       wgtwnsmuf(2).ne.0.d0.or.
     #       wgtwnsmur(2).ne.0.d0)then
            write(*,*)'Error in compute_rwgt_wgt_NLO'
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
              write(*,*)'Error in compute_rwgt_wgt_NLO'
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
              write(*,*)'Error #2 in compute_rwgt_wgt_NLO'
              write(*,*)k,xbk(1),xbk(2)
              write(*,*)wgtwreal(k),wgtwdeg(k),wgtwdegmuf(k)
              if(k.eq.2)write(*,*)wgtwborn(k),wgtwns(k),
     #                            wgtwnsmuf(k),wgtwnsmur(k)
              stop
            endif
          else
            nFKSprocess=nFKSprocess_used
            xlum = dlum()
            xsec12=xsec12+xlum*( wgtwreal(k)+wgtwdeg(k)+
     #                           wgtwdegmuf(k)*xlgmuf )*
     #                g**(2*wgtbpower+2.d0)
            if(k.eq.2)then
               nFKSprocess=nFKSprocess_used_born
               xlum = dlum()
              if(wgtbpower.gt.0)then
                xsec20=xsec20+xlum*wgtwborn(k)*g**(2*wgtbpower)
              else
                xsec20=xsec20+xlum*wgtwborn(k)
              endif
              xsec12=xsec12+xsec20
              xsec12=xsec12+xlum*( wgtwns(k)+
     #                         wgtwnsmuf(k)*xlgmuf+
     #                         wgtwnsmur(k)*xlgmur )*
     #                  g**(2*wgtbpower+2.d0)
            endif
          endif
        enddo
      endif

 542  continue

c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
      nFKSprocess=save_nFKSprocess
c
      wgtNLO11=xsec11
      wgtNLO12=xsec12-xsec20
      wgtNLO20=xsec20
      xsec=xsec11+xsec12
      compute_rwgt_wgt_NLO=xsec
c
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
      common/c_addwrite/i_process

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
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and.
     #     wgtkin(0,1,1).gt.0.d0) )then
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
     $           +2.d0)
          endif
        enddo
      endif
c
 541  continue
      if (wgtwreal(2).eq.0d0 .and. wgtwreal(3).eq.0d0 .and.
     $     wgtwreal(4).eq.0d0) goto 542

      call set_cms_stuff(izero)
      if( (kwgtinfo.eq.1.and.wgtmuR2(2).ne.0.d0) .or.
     #    ((kwgtinfo.ge.2.or.kwgtinfo.le.5)) )then
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
     $           +2.d0)
          endif
        enddo
      endif
c
 542  continue
      if (wgtwreal(1).eq.0d0) goto 543
      
      call set_cms_stuff(mohdr)
      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5)) )then
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
      common/c_addwrite/i_process

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
     #    (kwgtinfo.eq.2.and.wgtkin(0,1,1).gt.0.d0) .or.
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5).and.
     #     wgtkin(0,1,1).gt.0.d0) )then
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
     $                 +2.d0)
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
     #    ((kwgtinfo.ge.3.or.kwgtinfo.le.5)) )then
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
               endif
            enddo
            if(k.eq.2)then
              nFKSprocess=nFKSprocess_used_Born
              xlum = dlum()
              do j=1,iproc_save(nFKSprocess)
                 if (eto(j,nFKSprocess).eq.i_process) then
                    if(wgtbpower.gt.0)then
                       xsec=xsec+CONV*PD(j)*wgtwborn(k)*g**(2*wgtbpower)
                    else
                       xsec=xsec+CONV*PD(j)*wgtwborn(k)
                    endif
                    xsec=xsec+CONV*PD(j)*( wgtwns(k)+ wgtwnsmuf(k)
     $                   *xlgmuf+wgtwnsmur(k)*xlgmur )*g**(2*wgtbpower
     $                   +2.d0)
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
     #    ((kwgtinfo.ge.2.or.kwgtinfo.ge.5)) )then
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
      common/c_addwrite/i_process

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
     $                       *wgtbpower)
                     else
                        xsec=xsec+CONV*PD(j)*wgtwborn_all
                     endif
                     xsec=xsec+CONV*PD(j)*( wgtwns_all+ wgtwnsmuf_all
     $                    *xlgmuf+wgtwnsmur_all*xlgmur )*g**(2*wgtbpower
     $                    +2.d0)
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


      subroutine check_rwgt_wgt(idstring)
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'run.inc'
      include "reweight.inc"
      character*3 idstring
      double precision compute_rwgt_wgt_NLO,compute_rwgt_wgt_Hev,
     #                 compute_rwgt_wgt_Sev,compute_rwgt_wgt_Sev_nbody
      double precision wgtnew,tiny
      parameter (tiny=1.d-2)
c
      INTEGER              IPROC
      DOUBLE PRECISION PD(0:MAXPROC)
      COMMON /SUBPROC/ PD, IPROC
      integer iproc_save
      save iproc_save
c
      iproc_save=iproc
      if(idstring.eq."NLO")then
        wgtnew=compute_rwgt_wgt_NLO(muR_over_ref,muF1_over_ref,
     #                              muF2_over_ref,QES_over_ref,
     #                              iwgtinfo)
      elseif(idstring.eq."Hev")then
        wgtnew=compute_rwgt_wgt_Hev(muR_over_ref,muF1_over_ref,
     #                              muF2_over_ref,QES_over_ref,
     #                              iwgtinfo)
      elseif(idstring.eq."Sev")then
        wgtnew=compute_rwgt_wgt_Sev(muR_over_ref,muF1_over_ref,
     #                              muF2_over_ref,QES_over_ref,
     #                              iwgtinfo)
      elseif(idstring.eq."nbd")then
        wgtnew=compute_rwgt_wgt_Sev_nbody(muR_over_ref,muF1_over_ref,
     #                              muF2_over_ref,QES_over_ref,
     #                              iwgtinfo)
      else
        write(*,*)'Error in check_rwgt_wgt'
        write(*,*)' Unknown function: ',idstring
        stop
      endif
c
      if (idstring.eq."nbd") then
      if( (abs(wgtref_nbody).ge.1.d0 .and.
     #     abs(1.d0-wgtnew/wgtref_nbody).gt.tiny) .or.
     #    (abs(wgtref_nbody).lt.1.d0 .and.
     #     abs(wgtnew-wgtref_nbody).gt.tiny) )then
         write(*,*)'Error in check_rwgt_wgt: ',idstring,wgtref_nbody
     &        ,wgtnew
        stop
      endif
      else
      if( (abs(wgtref).ge.1.d0 .and.
     #     abs(1.d0-wgtnew/wgtref).gt.tiny) .or.
     #    (abs(wgtref).lt.1.d0 .and.
     #     abs(wgtnew-wgtref).gt.tiny) )then
        write(*,*)'Error in check_rwgt_wgt: ',idstring,wgtref,wgtnew
        stop
      endif
      endif
      iproc=iproc_save
c
      return
      end


      subroutine fill_rwgt_NLOplot()
      implicit none
      include "genps.inc"
      include "nexternal.inc"
      include 'coupl.inc'
      include 'run.inc'
      include "reweight.inc"
      include "reweightNLO.inc"
      integer izero
      parameter (izero=0)
      integer kr,kf,n
      double precision pr_muR_over_ref,pr_muF1_over_ref,
     # pr_muF2_over_ref,dummy,compute_rwgt_wgt_NLO
c
      dummy=compute_rwgt_wgt_NLO(ymuR_over_ref,ymuF1_over_ref,
     #                           ymuF2_over_ref,yQES_over_ref,
     #                           iwgtinfo)
      wgtrefNLO11=wgtNLO11
      wgtrefNLO12=wgtNLO12
      wgtrefNLO20=wgtNLO20
c
      if(do_rwgt_scale)then
        do kr=1,numscales
          do kf=1,numscales
            pr_muR_over_ref=ymuR_over_ref*yfactR(kr)
            pr_muF1_over_ref=ymuF1_over_ref*yfactF(kf)
            pr_muF2_over_ref=pr_muF1_over_ref
            dummy=compute_rwgt_wgt_NLO(pr_muR_over_ref,pr_muF1_over_ref,
     #                                 pr_muF2_over_ref,yQES_over_ref,
     #                                 iwgtinfo)
            wgtNLOxsecmu(1,kr,kf)=wgtNLO11
            wgtNLOxsecmu(2,kr,kf)=wgtNLO12
            wgtNLOxsecmu(3,kr,kf)=wgtNLO20
          enddo
        enddo
      endif
c
      if(do_rwgt_pdf)then
        do n=0,numPDFs-1
          call InitPDF(n)
          dummy=compute_rwgt_wgt_NLO(ymuR_over_ref,ymuF1_over_ref,
     #                               ymuF2_over_ref,yQES_over_ref,
     #                               iwgtinfo)
          wgtNLOxsecPDF(1,n)=wgtNLO11
          wgtNLOxsecPDF(2,n)=wgtNLO12
          wgtNLOxsecPDF(3,n)=wgtNLO20
        enddo
c Restore default PDFs
        call InitPDF(izero)
      endif
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
