      subroutine finalize_event(xx,weight,lunlhe,putonshell)
      implicit none
      include 'nexternal.inc'
      include "genps.inc"
      include "unlops.inc"
      include "run.inc"
      include 'timing_variables.inc'
      include 'mint.inc'
C     To access Pythia8 control variables
      include 'pythia8_control.inc'
C     To access event streams to communicate with PY8
      include 'hep_event_streams.inc'
C     To access mu_r
      include 'coupl.inc'
C     START local variable for the example only. Can be removed when removing
C     example
      double precision p_read(0:4,2*nexternal-3), wgt_read
C     STOP local variables for the example.

      integer ndim
      common/tosigint/ndim
      logical Hevents
      common/SHevents/Hevents
      integer i,j,lunlhe
      real*8 xx(ndimmax),weight,evnt_wgt
      logical putonshell
      double precision wgt
c missing???
      double precision unwgtfun
      double precision x(99),p(0:3,nexternal)
      integer jpart(7,-nexternal+3:2*nexternal-3)
      double precision pb(0:4,-nexternal+3:2*nexternal-3)
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      double precision ybst_til_tolab,ybst_til_tocm,sqrtshat,shat
      common/parton_cms_stuff/ybst_til_tolab,ybst_til_tocm,
     #                        sqrtshat,shat
      integer npart
      double precision shower_scale
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
      call generate_momenta(ndim,iconfig,wgt,x,p)
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
     &        .false.,ndim,x,jpart,npart,pb,shower_scale)
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
     &     putonshell,ndim,x,jpart,npart,pb,shower_scale)

cC     ---------------------------------------------------------------
cC     START of example of a dynamic call to PY8 using current event
cC     ---------------------------------------------------------------
c      if (is_pythia_active.eq.-1) then
cC       Pythia8 was not available when the process was compiled!
c        continue
c      else
c        call fill_HEPEUP_event(pb(0,1),evnt_wgt,jpart(1,1),npart,mu_r)
cC       Check if Pythia8 needs to be initialized
c        if (is_pythia_active.eq.0) then
cC         By default, we now use an empty command file
cC          call pythia_init(pythia_cmd_file)
c          call pythia_init_default()
c        endif
cC       Send current event to Pythia8
c        call pythia_setevent()
cC       Ask Pythia8 to shower current event
c        call pythia_next()
cC       Ask Pythia8 to printout its internal record of the event
c        call pythia_stat()
cC       Read (i.e. simply access) the output HEPEUP event stream
c        call read_HEPEUP_event(p_read,wgt_read)
cC       And printout the corresponding event kinematics and weight
c        do j=1,2*nexternal-3
c          write(*,*) 'p_read(*,',j,')=',(p_read(i,j),i=0,4)
c        enddo
c        write(*,*) 'wgt_read=',wgt_read
c      endif
cC     ---------------------------------------------------------------
cC     END of example.
cC     ---------------------------------------------------------------

c missing function???
c      call unweight_function(p_born,unwgtfun)
c      if (unwgtfun.ne.0d0) then
c         evnt_wgt=evnt_wgt/unwgtfun
c      else
c         write (*,*) 'ERROR in finalize_event, unwgtfun=0',unwgtfun
c         stop
c      endif

c      if (abrv.ne.'grid') then
c  Write-out the events
      call write_events_lhe(pb(0,1),evnt_wgt,jpart(1,1),npart,lunlhe
     $     ,shower_scale,ickkw)
      
      call cpu_time(tAfter)
      t_write=t_write+(tAfter-tBefore)

c error???
c      endif

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
      XSECUP(1)=inter
      XERRUP(1)=uncer
      XMAXUP(1)=absint/ievents
      LPRUP(1)=66
      if (event_norm(1:5).eq.'unity'.or.event_norm(1:3).eq.'sum') then
         IDWTUP=-3
      else
         IDWTUP=-4
      endif
      NPRUP=1

      write(lunlhe,'(a)')'<LesHouchesEvents version="3.0">'
      write(lunlhe,'(a)')'  <!--'
      write(lunlhe,'(a)')'  <scalesfunctionalform>'
      write(lunlhe,'(2a)')'    muR  ',muR_id_str(1:len_trim(muR_id_str))
      write(lunlhe,'(2a)')'    muF1 ',muF1_id_str(1:len_trim(muF1_id_str))
      write(lunlhe,'(2a)')'    muF2 ',muF2_id_str(1:len_trim(muF2_id_str))
      write(lunlhe,'(2a)')'    QES  ',QES_id_str(1:len_trim(QES_id_str))
      write(lunlhe,'(a)')'  </scalesfunctionalform>'
      write(lunlhe,'(a)')MonteCarlo
      write(lunlhe,'(a)')'  -->'
      write(lunlhe,'(a)')'  <header>'
      write(lunlhe,250)ievents
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

      subroutine write_events_lhe(p,wgt,ic,npart,lunlhe,shower_scale
     $     ,ickkw)
      use extra_weights
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "madfks_mcatnlo.inc"
      double precision p(0:4,2*nexternal-3),wgt
      integer ic(7,2*nexternal-3),npart,lunlhe,kwgtinfo,ickkw
      double precision pi,zero
      parameter (pi=3.1415926535897932385d0)
      parameter (zero=0.d0)
      integer ievent,izero
      parameter (izero=0)
      double precision aqcd,aqed,scale
      character*140 buff
      double precision shower_scale
      INTEGER MAXNUP,i
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
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
c
      if (ickkw.eq.4) then
         scale = sqrt(muF12_current)
      elseif (ickkw.eq.-1) then
         scale = mu_r
      else
         scale = shower_scale
      endif

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
      ievent=66
c
      if(AddInfoLHE)then
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

 201  format(a9,1x,i1,4(1x,i2),2(1x,d14.8),2x,i2,2(1x,i2),5(1x,d14.8))
      return
      end

      subroutine write_random_numbers(lunlhe)
      implicit none
      integer lunlhe,i
      double precision x(99),sigintF_save,f_abs_save
      common /c_sigint/ x,sigintF_save,f_abs_save
      integer ndim
      common/tosigint/ndim
      write (lunlhe,'(a)')'  <event>'
      write (lunlhe,*) ndim,sigintF_save,f_abs_save
      write (lunlhe,*) (x(i),i=1,ndim)
      write (lunlhe,'(a)')'  </event>'
      return
      end

C     ---------------------------------------------------------------
C     Pythia8 accessibility subroutines
C     ---------------------------------------------------------------

      subroutine fill_HEPEUP_event(p,wgt,ic,npart,shower_scale)
      implicit none
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      include "nexternal.inc"
      include "coupl.inc"
      include 'hep_event_streams.inc'
      double precision shower_scale, aqcd, aqed

      double precision p(0:4,2*nexternal-3),wgt
      integer ic(7,2*nexternal-3),npart, i, proc_code
      logical firsttime
      data firsttime/.true./
c
      scalup_out = shower_scale
      scalup_out = 1d9

      aqcd=g**2/(4d0*pi)
      aqed=gal(1)**2/(4d0*pi)
c
c 'fill_HEPrup_block' should be called after 'aqcd' has been set,
c because it includes a call to 'setrun', which resets the value of
c alpha_s to the one in the param_card.dat (without any running).
      if (firsttime) then
         call fill_HEPRUP_init()
         firsttime=.false.
      endif

c

c********************************************************************
c     Fill in LesHouches event block according to conventions
c     ic(1,*) = Particle ID
c     ic(2.*) = Mothup(1)
c     ic(3,*) = Mothup(2)
c     ic(4,*) = ICOLUP(1)
c     ic(5,*) = ICOLUP(2)
c     ic(6,*) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(7,*) = Helicity
c********************************************************************
      proc_code = 66
      NUP_out=npart
      IDPRUP_out=proc_code
      XWGTUP_out=wgt
      AQEDUP_out=aqed
      AQCDUP_out=aqcd
      do i=1,NUP_out
        IDUP_out(i)=ic(1,i)
        ISTUP_out(i)=ic(6,i)
        MOTHUP_out(1,i)=ic(2,i)
        MOTHUP_out(2,i)=ic(3,i)
        ICOLUP_out(1,i)=ic(4,i)
        ICOLUP_out(2,i)=ic(5,i)
        PUP_out(1,i)=p(1,i)
        PUP_out(2,i)=p(2,i)
        PUP_out(3,i)=p(3,i)
        PUP_out(4,i)=p(0,i)
        PUP_out(5,i)=p(4,i)
        VTIMUP_out(i)=0.d0
        SPINUP_out(i)=dfloat(ic(7,i))
      enddo

      return
      end

      subroutine fill_HEPEUP_event_2(p, wgt, npart, id, status, mothers,
     &           cols, spin, scale)
      implicit none
      double precision pi
      parameter (pi=3.1415926535897932385d0)
      include "nexternal.inc"
      include "coupl.inc"
c      include "pmass.inc"
      include 'hep_event_streams.inc'
      double precision wgt, aqcd, aqed

      double precision p(0:3,nexternal)
      integer id(nexternal)
      integer mothers(2,nexternal)
      integer cols(2,nexternal)
      integer status(nexternal)
      integer spin(nexternal)
      double precision scale(2*nexternal)
      double precision pmass(nexternal)
      REAL*8 ZERO
      PARAMETER (ZERO=0D0)

      integer npart, i, proc_code
      logical firsttime
      data firsttime/.true./
c
      scalup_out = scale(1)
      scalup_out = 1d9

c     Read the particle masses.
      include "pmass.inc"

      aqcd=g**2/(4d0*pi)
      aqed=gal(1)**2/(4d0*pi)
c
c 'fill_HEPrup_block' should be called after 'aqcd' has been set,
c because it includes a call to 'setrun', which resets the value of
c alpha_s to the one in the param_card.dat (without any running).
      if (firsttime) then
         call fill_HEPRUP_init()
         firsttime=.false.
      endif

c

c********************************************************************
c     Fill in LesHouches event block according to conventions
c     ic(1,*) = Particle ID
c     ic(2.*) = Mothup(1)
c     ic(3,*) = Mothup(2)
c     ic(4,*) = ICOLUP(1)
c     ic(5,*) = ICOLUP(2)
c     ic(6,*) = ISTUP   -1=initial state +1=final  +2=decayed
c     ic(7,*) = Helicity
c********************************************************************
      proc_code = 66
      NUP_out=npart
      IDPRUP_out=proc_code
      XWGTUP_out=wgt
      AQEDUP_out=aqed
      AQCDUP_out=aqcd
      do i=1,NUP_out
        IDUP_out(i)=id(i)
        ISTUP_out(i)=status(i)
        MOTHUP_out(1,i)=mothers(1,i)
        MOTHUP_out(2,i)=mothers(2,i)
        ICOLUP_out(1,i)=cols(1,i)
        ICOLUP_out(2,i)=cols(2,i)
        PUP_out(1,i)=p(1,i)
        PUP_out(2,i)=p(2,i)
        PUP_out(3,i)=p(3,i)
        PUP_out(4,i)=p(0,i)
c        PUP_out(5,i)=dsqrt(max(0.0,p(0,i)*p(0,i) - p(1,i)*p(1,i) - p(2,i)*p(2,i) - p(3,i)*p(3,i)))
        PUP_out(5,i)=pmass(i)
        VTIMUP_out(i)=0.d0
        SPINUP_out(i)=dfloat(spin(i))
      enddo

      return
      end

      subroutine clear_HEPEUP_event()
      include 'hep_event_streams.inc'

      NUP_out=-1
      IDPRUP_out=0
      XWGTUP_out=0.0
      AQEDUP_out=0.0
      AQCDUP_out=0.0
      do i=1,maxpup_out
        IDUP_out(i)=0
        ISTUP_out(i)=0
        MOTHUP_out(1,i)=0
        MOTHUP_out(2,i)=0
        ICOLUP_out(1,i)=0
        ICOLUP_out(2,i)=0
        PUP_out(1,i)=0.0
        PUP_out(2,i)=0.0
        PUP_out(3,i)=0.0
        PUP_out(4,i)=0.0
        PUP_out(5,i)=0.0
        VTIMUP_out(i)=0.0
        SPINUP_out(i)=0.0
      enddo

      return
      end

      subroutine read_HEPEUP_event(p, wgt)
         include 'hep_event_streams.inc'
         include 'nexternal.inc'
         double precision p(0:4,2*nexternal-3),wgt
         integer i,j
         do i=1,2*nexternal-3
           p(1,i) = pup_in(1,i)
           p(2,i) = pup_in(2,i)
           p(3,i) = pup_in(3,i)
           p(0,i) = pup_in(4,i)
           p(4,i) = pup_in(5,i)
         enddo
         wgt = xwgtup_in

      end

      subroutine fill_HEPRUP_init()
        implicit none
C       This fills in the common block that has the necessary
C       information to initialize the shower
        include 'hep_event_streams.inc'

C       Retrieve information set by setrun()
        integer maxpup
        parameter(maxpup=100)
        integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
        double precision ebmup,xsecup,xerrup,xmaxup
        common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)

        integer ifile, i
c
        integer ievents
        double precision inter,absint,uncer
        common /to_write_header_init/inter,absint,uncer,ifile,ievents

        character*7 event_norm
        common /event_normalisation/event_norm

C       Retrieve information from the run parameters
        call setrun() 
        XSECUP_out(1)=inter
        XERRUP_out(1)=uncer
        XMAXUP_out(1)=absint/ievents
        LPRUP_out(1)=66
        if (event_norm(1:5).eq.'unity'.or.event_norm(1:3).eq.'sum') then
          IDWTUP_out=-3
        else
          IDWTUP_out=-4
        endif
        NPRUP_out=1

        do i=1,2
          idbmup_out(i)=idbmup(i)
          ebmup_out(i)=ebmup(i)
          pdfgup_out(i)=pdfgup(i)
          pdfsup_out(i)=pdfsup(i)
        enddo
      
      return

      end
