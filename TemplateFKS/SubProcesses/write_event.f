      subroutine finalize_event(xx,res_abs,lunlhe,plotEv,putonshell)
      implicit none
      include "genps.inc"
      integer            mapconfig(0:lmaxconfigs), this_config
      common/to_mconfigs/mapconfig, this_config
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer itmax,ncall
      common/citmax/itmax,ncall
      logical Hevents
      common/SHevents/Hevents
      integer i,j,lunlhe
      real*8 xx(ndim),res_abs,plot_wgt,evnt_wgt
      logical plotEv, putonshell
      double precision wgt
      double precision x(99),p(0:3,99),pp(0:3,nexternal)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      integer iplot_ev,iplot_cnt,iplot_born
      parameter (iplot_ev=11)
      parameter (iplot_cnt=12)
      parameter (iplot_born=20)
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      double precision p1_cnt(0:3,nexternal,-2:2)
      double precision wgt_cnt(-2:2)
      double precision pswgt_cnt(-2:2)
      double precision jac_cnt(-2:2)
      common/counterevnts/p1_cnt,wgt_cnt,pswgt_cnt,jac_cnt

      integer np,npart

      double precision jampsum,sumborn
      double complex wgt1(2)

      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born

      do i=1,99
        if(i.le.ndim)then
          x(i)=xx(i)
        else
          x(i)=0.d0
        endif
      enddo
      
      wgt=1d0
c Normalization to the number of requested events is done in subroutine
c topout (madfks_plot_mint.f), so multiply here to get # of events.
      plot_wgt=evtsgn*itmax*ncall
      evnt_wgt=evtsgn*res_abs/(itmax*ncall)
      call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ndim,wgt,x,p)
      do i=1,nexternal
         do j=0,3
            pp(j,i)=p(j,i)
         enddo
      enddo

c
c Get all the info we need for writing the events.
c      
      if (Hevents) then
         call set_cms_stuff(-100)
      else
         call set_cms_stuff(0)
      endif

      call add_write_info(p_born,pp,ybst_til_tolab,mincfig,Hevents,
     &     putonshell,ndim,ipole,x,jpart,npart,pb)

c Plot the events also on the fly
      if(plotEv) then
         if (Hevents) then
            call outfun(pp,ybst_til_tolab,plot_wgt,iplot_ev)
         else
            call outfun(p1_cnt(0,1,0),ybst_til_tolab,plot_wgt,iplot_cnt)
         endif
      endif

c Write-out the events
      call write_events_lhe(pb(0,1),evnt_wgt,jpart(1,1),npart,lunlhe)
      
      return
      end


      function sigintS(xx,w,ifl)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      real*8 sigintS,xx(ndim),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigS
      double precision x(99),p(0:3,99)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision result
      save result
c
      do i=1,99
        if(i.le.ndim)then
          x(i)=xx(i)
        else
          x(i)=0.d0
        endif
      enddo
      wgt=1.d0
      fold=ifl
      if (ifl.eq.0)then
         call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ndim,wgt,x,p)
         result = w*dsigS(p,wgt,w)
         sigintS = result
      elseif(ifl.eq.1) then
         call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ndim,wgt,x,p)
         result = result+w*dsigS(p,wgt,w)
         sigintS = result
      elseif(ifl.eq.2) then
         if (unwgt) then
            evtsgn=sign(1d0,result)
            sigintS = abs(result)
         else
            sigintS = result
         endif
      endif
      return
      end

     
      function sigintH(xx,w,ifl)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      real*8 sigintH,xx(ndim),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigH
      double precision x(99),p(0:3,99)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision result
      save result
c
      do i=1,99
        if(i.le.ndim)then
          x(i)=xx(i)
        else
          x(i)=0.d0
        endif
      enddo
      wgt=1.d0
      fold=ifl
      if (ifl.eq.0)then
         call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ndim,wgt,x,p)
         result = w*dsigH(p,wgt,w)
         sigintH = result
      elseif(ifl.eq.1) then
         call x_to_f_arg(ndim,ipole,mincfig,maxcfig,ndim,wgt,x,p)
         result = result+w*dsigH(p,wgt,w)
         sigintH = result
      elseif(ifl.eq.2) then
         if (unwgt) then
            evtsgn=sign(1d0,result)
            sigintH = abs(result)
         else
            sigintH = result
         endif
      endif
      return
      end

      subroutine write_header_init(lunlhe,nevents,res,err)
      implicit none
      integer lunlhe,nevents
      double precision res,err,res_abs
      character*120 string
      logical Hevents
      common/SHevents/Hevents
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo

c Les Houches init block (for the <init> info)
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)


      open(unit=58,file='res_1',status='old')
      read(58,'(a)')string
      read(string(index(string,':')+1:index(string,'+/-')-1),*) res_abs
      close(58)

c get info on beam and PDFs
      call setrun
      XSECUP(1)=res
      XERRUP(1)=err
      XMAXUP(1)=res_abs/nevents
      LPRUP(1)=66
      IDWTUP=-4
      NPRUP=1

      write(lunlhe,'(a)')'<LesHouchesEvents version="1.0">'
      write(lunlhe,'(a)')'  <!--'
      write(lunlhe,'(a)')MonteCarlo
      write(lunlhe,'(a)')'  -->'
      write(lunlhe,'(a)')'  <header>'
      write(lunlhe,250)  nevents
      write(lunlhe,'(a)')'  </header>'
      write(lunlhe,'(a)')'  <init>'
      write(lunlhe,501)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      write(lunlhe,502)XSECUP(1),XERRUP(1),XMAXUP(1),LPRUP(1)
      write(lunlhe,'(a)')'  </init>'
 250  format(1x,i8)
 501  format(2(1x,i6),2(1x,d14.8),2(1x,i2),2(1x,i6),1x,i2,1x,i3)
 502  format(3(1x,d14.8),1x,i6)

      return
      end

      subroutine write_events_lhe(p,wgt,ic,npart,lunlhe)
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "madfks_mcatnlo.inc"
      double precision p(0:4,2*nexternal-3),wgt
      integer ic(7,2*nexternal-3),npart,lunlhe

      double precision pi
      parameter (pi=3.1415926535897932385d0)
      integer ievent
      double precision aqcd,aqed,scale

      character*140 buff

      double precision SCALUP
      common /cshowerscale/SCALUP

      integer iSorH_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe
c
      ievent=66
      scale = SCALUP
      aqcd=g**2/(4d0*pi)
      aqed=gal(1)**2/(4d0*pi)

      if(AddInfoLHE)then
        write(buff,200)'#',iSorH_lhe,ifks_lhe,jfks_lhe,
     #                     fksfather_lhe,ipartner_lhe,
     #                     scale1_lhe,scale2_lhe
      else
        buff=' '
      endif

c********************************************************************
c     Writes one event from data file #lun according to LesHouches
c     ic(1,*) = Particle ID
c     ic(2.*) = Mothup(1)
c     ic(3,*) = Mothup(2)
c     ic(4,*) = ICOLUP(1)
c     ic(5,*) = ICOLUP(2)
c     ic(6,*) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(7,*) = Helicity
c********************************************************************

      call write_event
     &     (lunlhe,p,wgt,npart,ic,ievent,scale,aqcd,aqed,buff)

 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8))
      return
      end

