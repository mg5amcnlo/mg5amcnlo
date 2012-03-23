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
        icnt=2-itype
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

      integer i,j,k
c
      wgtref=0.d0
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

      logical passcuts
      double precision compute_rwgt_wgt_NLO
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,xlgmuf,xlgmur,alphas
      double precision QES2_local
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision tiny,pi
      parameter (tiny=1.d-4)
      parameter (pi=3.14159265358979323846d0)

      integer k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if(kwgtinfo.ne.4) wgtbpower=rwgtbpower
c
      xsec=0.d0
      call set_cms_stuff(mohdr)
      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,1),rwgt)) )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,1))
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
          xlum = dlum()
          xsec=xsec+xlum*wgtwreal(1)*g**(2*wgtbpower+2.d0)
        endif
      endif
c
      call set_cms_stuff(izero)
      if( (kwgtinfo.eq.1.and.wgtmuR2(2).ne.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,2),rwgt)) )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(2))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(2) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(2) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,2))
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_NLO',kwgtinfo
          stop
        endif
        QES2_local=wgtqes2(2)
        if(abs(QES2/QES2_local-1.d0).gt.tiny.and.
     &       (kwgtinfo.eq.3.or.kwgtinfo.eq.4))then
          write(*,*)'Error in compute_rwgt_wgt_NLO'
          write(*,*)' Mismatch in ES scale',QES2,QES2_local
          stop
        endif
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
     #         (k.eq.2.and.(wgtwborn(k).ne.0.d0.or.
     #                      wgtwns(k).ne.0.d0.or.
     #                      wgtwnsmuf(k).ne.0.d0.or.
     #                      wgtwnsmur(k).ne.0.d0)))then
              write(*,*)'Error #2 in compute_rwgt_wgt_NLO'
              write(*,*)k,xbk(1),xbk(2)
              write(*,*)wgtwreal(k),wgtwdeg(k),wgtwdegmuf(k)
              if(k.eq.2)write(*,*)wgtwborn(k),wgtwns(k),
     #                            wgtwnsmuf(k),wgtwnsmur(k)
              stop
            endif
          else
            xlum = dlum()
            xsec=xsec+xlum*( wgtwreal(k)+wgtwdeg(k)+
     #                       wgtwdegmuf(k)*xlgmuf )*
     #                g**(2*wgtbpower+2.d0)
            if(k.eq.2)then
              if(wgtbpower.gt.0)then
                xsec=xsec+xlum*wgtwborn(k)*g**(2*wgtbpower)
              else
                xsec=xsec+xlum*wgtwborn(k)
              endif
              xsec=xsec+xlum*( wgtwns(k)+
     #                         wgtwnsmuf(k)*xlgmuf+
     #                         wgtwnsmur(k)*xlgmur )*
     #                  g**(2*wgtbpower+2.d0)
            endif
          endif
        enddo
      endif
c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
c
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

      logical passcuts
      double precision compute_rwgt_wgt_Hev
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,alphas
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision pi
      parameter (pi=3.14159265358979323846d0)

      integer i,k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if(kwgtinfo.ne.4) wgtbpower=rwgtbpower
c
      xsec=0.d0
      call set_cms_stuff(izero)
      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,2),rwgt).and.
     #     wgtkin(0,1,1).gt.0.d0) )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_cms_stuff(mohdr)
          call set_alphaS(wgtkin(0,1,1))
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
            xlum = dlum()
            xsec=xsec+xlum*wgtwmcxsec(i)*g**(2*wgtbpower+2.d0)
          endif
        enddo
      endif
c
      call set_cms_stuff(izero)
      if( (kwgtinfo.eq.1.and.wgtmuR2(2).ne.0.d0) .or.
     #    ((kwgtinfo.eq.2.or.kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.
     #     passcuts(wgtkin(0,1,2),rwgt)) )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(2))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(2) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(2) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.2.or.kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,2))
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
            xlum = dlum()
            xsec=xsec+xlum*wgtwreal(k)*g**(2*wgtbpower+2.d0)
          endif
        enddo
      endif
c
      call set_cms_stuff(mohdr)
      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,1),rwgt)) )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,1))
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
          xlum = dlum()
          xsec=xsec+xlum*wgtwreal(1)*g**(2*wgtbpower+2.d0)
        endif
      endif
c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
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

      logical passcuts
      double precision compute_rwgt_wgt_Sev
      double precision xmuR_over_ref,xmuF1_over_ref,
     #                 xmuF2_over_ref,xQES_over_ref
      integer kwgtinfo
      double precision rwgt,xsec,xlum,dlum,xlgmuf,xlgmur,alphas
      double precision QES2_local
      double precision save_murrat,save_muf1rat,save_muf2rat,save_qesrat
      double precision tiny,pi
      parameter (tiny=1.d-4)
      parameter (pi=3.14159265358979323846d0)

      integer i,k,izero,mohdr
      parameter (izero=0)
      parameter (mohdr=-100)
c
      save_murrat=muR_over_ref
      save_muf1rat=muF1_over_ref
      save_muf2rat=muF2_over_ref
      save_qesrat=QES_over_ref
c
      muR_over_ref=xmuR_over_ref 
      muF1_over_ref=xmuF1_over_ref
      muF2_over_ref=xmuF2_over_ref
      QES_over_ref=xQES_over_ref 
c
      if(kwgtinfo.ne.4) wgtbpower=rwgtbpower
c
      xsec=0.d0
      call set_cms_stuff(izero)

      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     #    (kwgtinfo.eq.2.and.wgtkin(0,1,1).gt.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,2),rwgt).and.
     #     wgtkin(0,1,1).gt.0.d0) )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.2.or.kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_cms_stuff(mohdr)
          call set_alphaS(wgtkin(0,1,1))
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
            xlum = dlum()
            xsec=xsec+xlum*wgtwmcxsec(i)*g**(2*wgtbpower+2.d0)
          endif
        enddo
      endif
c
      call set_cms_stuff(izero)

      if( ((kwgtinfo.eq.1.or.kwgtinfo.eq.2).and.wgtmuR2(2).ne.0.d0) .or.
     #    ((kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.passcuts(wgtkin(0,1,2),rwgt)) )then
        if(kwgtinfo.eq.1.or.kwgtinfo.eq.2)then
          scale=muR_over_ref*sqrt(wgtmuR2(2))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(2) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(2) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,2))
        else
          write(*,*)'Error #0b in compute_rwgt_wgt_Sev',kwgtinfo
          stop
        endif
        QES2_local=wgtqes2(2)
        if(abs(QES2/QES2_local-1.d0).gt.tiny.and.
     &       (kwgtinfo.eq.3.or.kwgtinfo.eq.4))then
          write(*,*)'Error in compute_rwgt_wgt_Sev'
          write(*,*)' Mismatch in ES scale',QES2,QES2_local
          stop
        endif
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
     #         (k.eq.2.and.(wgtwborn(k).ne.0.d0.or.
     #                      wgtwns(k).ne.0.d0.or.
     #                      wgtwnsmuf(k).ne.0.d0.or.
     #                      wgtwnsmur(k).ne.0.d0)))then
              write(*,*)'Error #2 in compute_rwgt_wgt_Sev'
              write(*,*)k,xbk(1),xbk(2)
              write(*,*)wgtwreal(k),wgtwdeg(k),wgtwdegmuf(k)
              if(k.eq.2)write(*,*)wgtwborn(k),wgtwns(k),
     #                            wgtwnsmuf(k),wgtwnsmur(k)
              stop
            endif
          else
            xlum = dlum()
            xsec=xsec+xlum*( wgtwreal(k)+wgtwdeg(k)+
     #                       wgtwdegmuf(k)*xlgmuf )*
     #                g**(2*wgtbpower+2.d0)
            if(k.eq.2)then
              if(wgtbpower.gt.0)then
                xsec=xsec+xlum*wgtwborn(k)*g**(2*wgtbpower)
              else
                xsec=xsec+xlum*wgtwborn(k)
              endif
              xsec=xsec+xlum*( wgtwns(k)+
     #                         wgtwnsmuf(k)*xlgmuf+
     #                         wgtwnsmur(k)*xlgmur )*
     #                  g**(2*wgtbpower+2.d0)
            endif
          endif
        enddo
      endif
c
      call set_cms_stuff(mohdr)

      if( (kwgtinfo.eq.1.and.wgtmuR2(1).ne.0.d0) .or.
     #    ((kwgtinfo.eq.2.or.kwgtinfo.eq.3.or.kwgtinfo.eq.4).and.
     #     passcuts(wgtkin(0,1,1),rwgt)) )then
        if(kwgtinfo.eq.1)then
          scale=muR_over_ref*sqrt(wgtmuR2(1))
          g=sqrt(4d0*pi*alphas(scale))
          q2fact(1)=wgtmuF12(1) * muF1_over_ref**2
          q2fact(2)=wgtmuF22(1) * muF2_over_ref**2
c Should cause the code to crash if used
          QES2=0.d0
        elseif(kwgtinfo.eq.2.or.kwgtinfo.eq.3.or.kwgtinfo.eq.4)then
          call set_alphaS(wgtkin(0,1,1))
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
          xlum = dlum()
          xsec=xsec+xlum*wgtwreal(1)*g**(2*wgtbpower+2.d0)
        endif
      endif
c
      muR_over_ref=save_murrat
      muF1_over_ref=save_muf1rat
      muF2_over_ref=save_muf2rat
      QES_over_ref=save_qesrat
c
      compute_rwgt_wgt_Sev=xsec
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
     #                 compute_rwgt_wgt_Sev
      double precision wgtnew,tiny
      parameter (tiny=1.d-4)
c
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
      else
        write(*,*)'Error in check_rwgt_wgt'
        write(*,*)' Unknown function: ',idstring
        stop
      endif
c
      if( (abs(wgtref).ge.1.d0 .and.
     #     abs(1.d0-wgtnew/wgtref).gt.tiny) .or.
     #    (abs(wgtref).lt.1.d0 .and.
     #     abs(wgtnew-wgtref).gt.tiny) )then
        write(*,*)'Error in check_rwgt_wgt: ',idstring,wgtref,wgtnew
        write(*,*)wgtkin(0,3,1),wgtkin(1,3,1),
     #            wgtkin(2,3,1),wgtkin(3,3,1)
        stop
      endif
c
      return
      end
