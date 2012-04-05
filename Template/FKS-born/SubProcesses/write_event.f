      subroutine finalize_event(xx,res_abs,lunlhe,plotEv,putonshell)
      implicit none
      include 'nexternal.inc'
      include "genps.inc"
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer itmax,ncall
      common/citmax/itmax,ncall
      logical Hevents
      common/SHevents/Hevents
      integer i,j,lunlhe
      include 'mint.inc'
      real*8 xx(ndimmax),res_abs,plot_wgt,evnt_wgt
      logical plotEv, putonshell
      double precision wgt,unwgtfun
      double precision x(99),p(0:3,nexternal)
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
          x(i)=-9d99
        endif
      enddo
      
      wgt=1d0
c Normalization to the number of requested events is done in subroutine
c topout (madfks_plot_mint.f), so multiply here to get # of events.
      plot_wgt=evtsgn*itmax*ncall
      evnt_wgt=evtsgn*res_abs/(itmax*ncall)
      call generate_momenta(ndim,iconfig,wgt,x,p)
c
c Get all the info we need for writing the events.
c      
      if (Hevents) then
         call set_cms_stuff(-100)
      else
         call set_cms_stuff(0)
      endif

      call set_shower_scale()

      call add_write_info(p_born,p,ybst_til_tolab,iconfig,Hevents,
     &     putonshell,ndim,ipole,x,jpart,npart,pb)

c Plot the events also on the fly
      if(plotEv) then
         if (Hevents) then
            call outfun(p,ybst_til_tolab,plot_wgt,iplot_ev)
         else
            call outfun(p1_cnt(0,1,0),ybst_til_tolab,plot_wgt,iplot_cnt)
         endif
      endif

      call unweight_function(p_born,unwgtfun)
      if (unwgtfun.ne.0d0) then
         evnt_wgt=evnt_wgt/unwgtfun
      else
         write (*,*) 'ERROR in finalize_event, unwgtfun=0',unwgtfun
         stop
      endif
c Write-out the events
      call write_events_lhe(pb(0,1),evnt_wgt,jpart(1,1),npart,lunlhe)
      
      return
      end


      function sigintF(xx,w,ifl,f_abs)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      include 'mint.inc'
      real*8 sigintF,xx(ndimmax),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigS,dsigH,f_abs
      include 'nexternal.inc'
      double precision x(99),p(0:3,nexternal)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      logical Hevents
      common/SHevents/Hevents
      double precision result,result1,result2,ran2,rnd
      external ran2
      double precision sigintF_save,f_abs_save
      save sigintF_save,f_abs_save
      include 'nFKSconfigs.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      character*4 abrv
      common /to_abrv/ abrv
      logical nbodyonly
      common/cnbodyonly/nbodyonly
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      logical sum,firsttime
      parameter (sum=.false.)
      data firsttime /.true./
      integer nFKSprocessBorn
      save nFKSprocessBorn
c
      do i=1,99
         if (i.le.ndim) then
            x(i)=xx(i)
         else
            x(i)=-9d99
         endif
      enddo

c Find the nFKSprocess for which we compute the Born-like contributions
      if (firsttime) then
         firsttime=.false.
         nFKSprocess=fks_configs
         call fks_inc_chooser()
         do while (particle_type(i_fks).ne.8)
            write (*,*) i_fks,particle_type(i_fks)
            nFKSprocess=nFKSprocess-1
            call fks_inc_chooser()
            if (nFKSprocess.eq.0) then
               write (*,*) 'ERROR in sigint'
               stop
            endif
         enddo
         nFKSprocessBorn=nFKSprocess
      endif

      sigintF=0d0
      f_abs=0d0
      fold=ifl
      if (ifl.eq.0)then
c
c Compute the Born-like contributions with nbodyonly=.true.
c THIS CAN BE OPTIMIZED
c     
         nFKSprocess=nFKSprocessBorn
         nbodyonly=.true.
         call fks_inc_chooser()
         call leshouche_inc_chooser()
         call setcuts
         call setfksfactor(iconfig)
         wgt=1d0
         call generate_momenta(ndim,iconfig,wgt,x,p)
         call dsigF(p,wgt,w,dsigS,dsigH)
         result1= w*dsigS
         result2= w*dsigH
         sigintF= sigintF+result1+result2
         f_abs = f_abs+abs(result1)+abs(result2)
c
c Compute the subtracted real-emission corrections either as an explicit
c sum or a Monte Carlo sum.
c      
         if (abrv.eq.'born' .or. abrv.eq.'grid' .or.
     &        abrv(1:2).eq.'vi') return
         nbodyonly=.false.
         if (sum) then
c THIS CAN BE OPTIMIZED
            do nFKSprocess=1,fks_configs
               call fks_inc_chooser()
               call leshouche_inc_chooser()
               call setcuts
               call setfksfactor(iconfig)
               wgt=1d0
               call generate_momenta(ndim,iconfig,wgt,x,p)
               call dsigF(p,wgt,w,dsigS,dsigH)
               result1= w*dsigS
               result2= w*dsigH
               sigintF= sigintF+result1+result2
               f_abs = f_abs+abs(result1)+abs(result2)
            enddo
         else                   ! Monte Carlo over nFKSprocess
            rnd=ran2()
            nFKSprocess=0
            do while (nFKSprocess.lt.rnd*fks_configs)
               nFKSprocess=nFKSprocess+1
            enddo
c THIS CAN BE OPTIMIZED
            call fks_inc_chooser()
            call leshouche_inc_chooser()
            call setcuts
            call setfksfactor(iconfig)
            wgt=1d0
            call generate_momenta(ndim,iconfig,wgt,x,p)
            call dsigF(p,wgt,w,dsigS,dsigH)
            result1= w*dsigS
            result2= w*dsigH
            sigintF= sigintF+(result1+result2)*fks_configs
            f_abs = f_abs+(abs(result1)+abs(result2))*fks_configs
         endif
         sigintF_save=sigintF
         f_abs_save=f_abs
      elseif(ifl.eq.1) then
         write (*,*) 'Folding not implemented'
         stop
      elseif(ifl.eq.2) then
         sigintF = sigintF_save
         f_abs = f_abs_save
c Determine if we need to write S or H events according to their
c relative weights
         if (f_abs.gt.0d0) then
            if (ran2().le.abs(result1)/f_abs) then
               Hevents=.false.
               evtsgn=sign(1d0,result1)
            else
               Hevents=.true.
               evtsgn=sign(1d0,result2)
            endif
         endif
      endif
      return
      end

     
      function sigintS(xx,w,ifl,f_abs)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      include 'mint.inc'
      real*8 sigintS,xx(ndimmax),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigS,f_abs
      include 'nexternal.inc'
      double precision x(99),p(0:3,nexternal)
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
          x(i)=-9d99
        endif
      enddo
      wgt=1.d0
      fold=ifl
      if (ifl.eq.0)then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = w*dsigS(p,wgt,w)
         sigintS = result
         f_abs=abs(sigintS)
      elseif(ifl.eq.1) then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = result+w*dsigS(p,wgt,w)
         sigintS = result
         f_abs=abs(sigintS)
      elseif(ifl.eq.2) then
         sigintS = result
         f_abs=abs(sigintS)
         evtsgn=sign(1d0,result)
      endif
      return
      end

     
      function sigintH(xx,w,ifl,f_abs)
c From dsample_fks
      implicit none
      integer ndim,ipole
      common/tosigint/ndim,ipole
      integer           iconfig
      common/to_configs/iconfig
      integer i
      integer ifl
      integer fold
      common /cfl/fold
      include 'mint.inc'
      real*8 sigintH,xx(ndimmax),w
      integer ione
      parameter (ione=1)
      double precision wgt,dsigH,f_abs
      include 'nexternal.inc'
      double precision x(99),p(0:3,nexternal)
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
          x(i)=-9d99
        endif
      enddo
      wgt=1.d0
      fold=ifl
      if (ifl.eq.0)then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = w*dsigH(p,wgt,w)
         sigintH = result
         f_abs=abs(sigintH)
      elseif(ifl.eq.1) then
         call generate_momenta(ndim,iconfig,wgt,x,p)
         result = result+w*dsigH(p,wgt,w)
         sigintH = result
         f_abs=abs(sigintH)
      elseif(ifl.eq.2) then
         sigintH = result
         f_abs=abs(sigintH)
         evtsgn=sign(1d0,result)
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
      include 'reweight.inc'
      double precision p(0:4,2*nexternal-3),wgt
      integer ic(7,2*nexternal-3),npart,lunlhe

      double precision pi,zero
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0.d0)
      integer ievent,izero
      parameter (izero=0)
      double precision aqcd,aqed,scale

      character*140 buff

      double precision SCALUP
      common /cshowerscale/SCALUP

      INTEGER MAXNUP,i
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)

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
        if(.not.doreweight)then
          write(buff,200)'#',iSorH_lhe,ifks_lhe,jfks_lhe,
     #                       fksfather_lhe,ipartner_lhe,
     #                       scale1_lhe,scale2_lhe,
     #                       izero,izero,izero,
     #                       zero,zero,zero,zero,zero
        else
          if(iwgtinfo.lt.1.or.iwgtinfo.gt.4)then
            write(*,*)'Error in write_events_lhe'
            write(*,*)'  Inconsistency in reweight parameters'
            write(*,*)doreweight,iwgtinfo
            stop
          endif
          write(buff,200)'#',iSorH_lhe,ifks_lhe,jfks_lhe,
     #                       fksfather_lhe,ipartner_lhe,
     #                       scale1_lhe,scale2_lhe,
     #                       iwgtinfo,nexternal,iwgtnumpartn,
     #                       zero,zero,zero,zero,zero
        endif
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

      NUP=npart
      IDPRUP=ievent
      XWGTUP=wgt
      AQEDUP=aqed
      AQCDUP=aqcd
      do i=1,NUP
        IDUP(i)=ic(1,i)
        ISTUP(i)=ic(6,i)
        MOTHUP(1,i)=ic(2,i)
        MOTHUP(2,i)=ic(3,i)
        ICOLUP(1,i)=ic(4,i)
        ICOLUP(2,i)=ic(5,i)
        PUP(1,i)=p(1,i)
        PUP(2,i)=p(2,i)
        PUP(3,i)=p(3,i)
        PUP(4,i)=p(0,i)
        PUP(5,i)=p(4,i)
        VTIMUP(i)=0.d0
        SPINUP(i)=dfloat(ic(7,i))
      enddo
      call write_lhef_event(lunlhe,
     #    NUP,IDPRUP,XWGTUP,scale,AQEDUP,AQCDUP,
     #    IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8),1x,i1,2(1x,i2),5(1x,d14.8))
      return
      end

