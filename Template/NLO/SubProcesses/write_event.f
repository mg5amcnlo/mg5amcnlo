      subroutine finalize_event(xx,weight,lunlhe,putonshell,p_label)
      use mint_module
      use process_module
      use scale_module
      implicit none
      include 'nexternal.inc'
      include "genps.inc"
      include "unlops.inc"
      include "run.inc"
      include 'timing_variables.inc'
      logical Hevents
      common/SHevents/Hevents
      integer i,j,lunlhe,p_label
      real*8 xx(ndimmax),weight,evnt_wgt
      logical putonshell
      double precision wgt
      double precision x(99),p(0:3,nexternal),p_lab(0:3,nexternal)
     $     ,p_cms(0:3,nexternal)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      integer npart
      double precision shower_scale_a(-nexternal+3:2
     $     *nexternal-3,-nexternal+3:2*nexternal-3)
      double precision p_born(0:3,nexternal-1)
      common/pborn/p_born
      call cpu_time(tBefore)

      do i=1,99
        if(i.le.ndim)then
          x(i)=xx(i)
        else
          x(i)=-9d99
        endif
      enddo
      
      wgt=1d0
      evnt_wgt=evtsgn*weight
      call generate_momenta(ndim,iconfig,wgt,x,p,p_lab,p_cms)
c
c Get all the info we need for writing the events.
c      
      if (Hevents) then
         call set_cms_stuff(-100)
      else
         call set_cms_stuff(0)
      endif

      if (ickkw.eq.4) putonshell=.false.

      if (ickkw.eq.4) then
         if (Hevents) then
            write (*,*) 'For ickkw=4, Hevents should be false',Hevents
            stop
         endif
         Hevents=.true.
         call add_write_info(p_born,p,ybst_til_tolab,iconfig,Hevents,
     &        .false.,ndim,x,jpart,npart,pb,shower_scale_a)
c Put the Hevent info in a common block
         NUP_H=npart
         do i=1,NUP_H
            IDUP_H(i)=jpart(1,i)
            ISTUP_H(i)=jpart(6,i)
            MOTHUP_H(1,i)=jpart(2,i)
            MOTHUP_H(2,i)=jpart(3,i)
            ICOLUP_H(1,i)=jpart(4,i)
            ICOLUP_H(2,i)=jpart(5,i)
            PUP_H(1,i)=pb(1,i)
            PUP_H(2,i)=pb(2,i)
            PUP_H(3,i)=pb(3,i)
            PUP_H(4,i)=pb(0,i)
            PUP_H(5,i)=pb(4,i)
            VTIMUP_H(i)=0.d0
            SPINUP_H(i)=dfloat(jpart(7,i))
         enddo
         Hevents=.false.
      endif
      
      call add_write_info(p_born,p,ybst_til_tolab,iconfig,Hevents,
     &     putonshell,ndim,x,jpart,npart,pb,shower_scale_a)
      
c Write-out the events
      call write_events_lhe(pb(0,1),evnt_wgt,jpart(1,1),npart,lunlhe
     $     ,ickkw,shower_scale_a,p_label)
      
      call cpu_time(tAfter)
      t_write=t_write+(tAfter-tBefore)
      return
      end

      subroutine write_header_init
      implicit none
      integer lunlhe
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
      integer ifile,ievents
      double precision inter,absint,uncer
      common /to_write_header_init/inter,absint,uncer,ifile,ievents

c Les Houches init block (for the <init> info)
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      character*7 event_norm
      common /event_normalisation/event_norm

      lunlhe=ifile
c get info on beam and PDFs
      call setrun
      call write_lhef_header(lunlhe,ievents,MonteCarlo)
      call write_lhef_init(lunlhe,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
 250  format(1x,i8)
 501  format(2(1x,i6),2(1x,d14.8),2(1x,i2),2(1x,i8),1x,i2,1x,i3)
 502  format(3(1x,d14.8),1x,i6)

      return
      end

      subroutine write_events_lhe(p,wgt,ic,npart,lunlhe ,ickkw
     $     ,shower_scale_a,p_label)
      use extra_weights
      use process_module
      use scale_module
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      double precision p(0:4,2*nexternal-3),wgt
      integer ic(7,2*nexternal-3),npart,lunlhe,kwgtinfo,ickkw,p_label
      double precision pi,zero
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0.d0)
      integer izero
      parameter (izero=0)
      double precision aqcd,aqed,scale
      character*1000 buff
      INTEGER MAXNUP,i,j,k
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,AQEDUP,AQCDUP,SCALUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP),
     # SCALUP_a(MAXNUP,MAXNUP)
      include 'nFKSconfigs.inc'
      INTEGER NFKSPROCESS
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      integer iSorH_lhe,ifks_lhe(fks_configs) ,jfks_lhe(fks_configs)
     &     ,fksfather_lhe(fks_configs) ,ipartner_lhe(fks_configs)
      double precision scale1_lhe(fks_configs),scale2_lhe(fks_configs)
      common/cto_LHE1/iSorH_lhe,ifks_lhe,jfks_lhe,
     #                fksfather_lhe,ipartner_lhe
      common/cto_LHE2/scale1_lhe,scale2_lhe
      double precision muR2_current,muF12_current,
     #                 muF22_current,QES2_current
      common/cscales_current_values/muR2_current,muF12_current,
     #                              muF22_current,QES2_current
      logical firsttime
      data firsttime/.true./
      double precision shower_scale_a(-nexternal+3:2
     $     *nexternal-3,-nexternal+3:2*nexternal-3)
c
      scalup_a=-1d0
      if (ickkw.eq.4) then
         SCALUP = sqrt(muF12_current)
         do j=1,2*nexternal-3
            do k=1,2*nexternal-3
               if(j.eq.k)cycle
               scalup_a(j,k)=sqrt(muF12_current)
            enddo
         enddo
      elseif (ickkw.eq.-1) then
         SCALUP = mu_r
         do j=1,2*nexternal-3
            do k=1,2*nexternal-3
               if(j.eq.k)cycle
               scalup_a(j,k)=mu_r
            enddo
         enddo
      else
         if (.not.mcatnlo_delta_mod) then
            if (iSorH_lhe.eq.1) then ! S-event
               SCALUP=showerscaleS(1,1)
            else
               SCALUP=showerscaleH(1,1)
            endif
            scalup_a(1:npart,1:npart)=-1d0
         else
            ! use array of scale for MC@NLO-Delta
            SCALUP=-1d0
            scalup_a(1:npart,1:npart)=shower_scale_a(1:npart,1:npart)
         endif
      endif
c
      aqcd=g**2/(4d0*pi)
      aqed=gal(1)**2/(4d0*pi)
c
c 'write_header_init' should be called after 'aqcd' has been set,
c because it includes a call to 'setrun', which resets the value of
c alpha_s to the one in the param_card.dat (without any running).
      if (firsttime) then
         call write_header_init
         firsttime=.false.
      endif
c
      if(.not.doreweight)then
         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe(nFKSprocess)
     &          ,jfks_lhe(nFKSprocess),fksfather_lhe(nFKSprocess)
     &          ,ipartner_lhe(nFKSprocess),scale1_lhe(nFKSprocess)
     &          ,scale2_lhe(nFKSprocess),izero,izero,izero,zero,zero
     &          ,zero,zero,zero
      else
         if(iwgtinfo.ne.-5)then
            write(*,*)'Error in write_events_lhe'
            write(*,*)'  Inconsistency in reweight parameters'
            write(*,*)doreweight,iwgtinfo
            stop
         endif
         kwgtinfo= iwgtinfo
         write(buff,201)'#aMCatNLO',iSorH_lhe,ifks_lhe(nFKSprocess)
     &         ,jfks_lhe(nFKSprocess),fksfather_lhe(nFKSprocess)
     &         ,ipartner_lhe(nFKSprocess),scale1_lhe(nFKSprocess)
     &         ,scale2_lhe(nFKSprocess),kwgtinfo,nexternal,iwgtnumpartn
     &         ,zero,zero,zero,zero,zero
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
      IDPRUP=p_label
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
     #    NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     #    IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff,SCALUP_a)
 201  format(a9,1x,i1,4(1x,i2),2(1x,d14.8),2x,i2,2(1x,i2),5(1x,d14.8))
      return
      end
