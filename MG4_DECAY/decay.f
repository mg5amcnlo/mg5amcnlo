      program decay
c***********************************************************************
c     1. reads events from file                                        *
c     2. asks  which particle to decay                                 *
c     3. asks  the decay mode                                          *
c     4. decays particle                                               *
c     5. write out decayed events.                                     *    
c----------------------------------------------------------------------*
c     FIRST VERSION 16-May-2003                                        *
c----------------------------------------------------------------------*
c     LAST UPDATE  26-Sep-2003:                                        *
c     - W->anything and t->b anything decays added.                    *
c     - rnd number generator seed changes in sequential runs.          *
c----------------------------------------------------------------------*
c     LAST UPDATE  07-Nov-2003:                                        *
c     - exactly turns unweighted evts in unweighted evts.              * 
c     - decaying identical particles in one event fixed.               * 
c     - bug in the Z->jets quark fractions corrected.                  *
c     - error trap routines added                                      *
c----------------------------------------------------------------------*
c     LAST UPDATE 25-Feb-2004:                                         *
c     - H->ZZ>leptons : wrong extra factor of two deleted              * 
c----------------------------------------------------------------------*
c     LAST UPDATE 12-Dec-2004:                                         *
c     - Merging it to the official web template                        *
c----------------------------------------------------------------------*
c     LAST UPDATE  June-10-2005:                                       *
c     - Updated for use with new rw_events.f format                    *
c     LAST UPDATE  August-18-2005:                                     *
c     - Fixed a bug in  the previous update                            * 
c----------------------------------------------------------------------*
c     LAST UPDATE  July-27-2006:                                        *
c     - Make it compliant with MadGraph v 4.0                          *
c***********************************************************************
      implicit none
      include 'decay.inc'
c
c     parameters
c
      real*8     smallmass
      parameter (smallmass=1d0) 
c     
c     Local
c
c
      integer nexternal, ic(7,MaxParticles),idecay(MaxParticles)
      double precision P(0:4,MaxParticles),wgt
      integer i,k,iten
      integer nevent,nfound(0:MaxParticles)
      data nevent/0/
c     
      real*8 sum(0:maxievent),maxwgt(0:maxievent)
      character*50 dec_name
      character*3 name
      character*132 buff
      character*140 buff2
      integer iseed
      real*8 aa
      real*8 scale,aqcd,aqed
      integer ievent,eventnr,eventnumber
      integer maxpup
      parameter(maxpup=100)
      integer idbmup(2),pdfgup(2),pdfsup(2),idwtup,nprup,lprup
      double precision ebmup(2),xsecup,xerrup,xmaxup
C      
      logical done,firsttime,fopenend,lwrite,newbanner
      LOGICAL DEBUG
      data firsttime/.true./
      data done/.false./
      DATA DEBUG/.TRUE./
c
c     external
c     
      real*8 dot
      real xran1
c
c     Global
c
      include 'coupl.inc'
      data id /201*' '/
c
      character*70 infile,outfile
      integer run_mode
      common/mode/run_mode,infile
c
      integer iunit
      common/to_lh/iunit

      integer sizeievent
      common/ievents/sizeievent
c
c     Date
c
c      CHARACTER*20 DATE
c      EXTERNAL DATE_Y2KBUG
      
c-----
c   Begin Code
c-----
c
      write(*,*) '*****************************************************'
      write(*,*) '*                    DECAY                          *'
      write(*,*) '*              a MadEvent program                   *'
      write(*,*) '*        for decaying unstable particles            *'
      write(*,*) '*             in the Standar Model                  *'  
      write(*,*) '*        ---------------------------------          *'
      write(*,*) '*        version compliant with MG_ME_V4.0          *'
      write(*,*) '*                                                   *'
      write(*,*) '*                27-July-2006                       *'
      write(*,*) '*****************************************************'
c
c     run mode: 0 calculates partial widths
c               1 decays event in file
c
      write(*,*)
      write(*,*) '  Input run mode:'
      write(*,*) '  ---------------'
      write(*,*)
      write(*,*) '  0 = calculates decay widths'
      write(*,*) '  1 = decay events in file'
      write(*,*)
            
      read(*,'(i1)')  run_mode

c
c     initialize rnd number for decay 
c
      call rnd_init(lunrnd,iseed)  
c
c
c     open scratch file which will contain the param_card.dat
c
      open(lunp,status='scratch')
c
      if(run_mode.eq.0) then
c
c     calculates the decay width at LO
c
       write(*,*) '*****************************************************'
       write(*,*) '*  run_mode=0 => calculting decay widths            *'
       write(*,*) '*****************************************************' 
       write(*,*)
       write(*,*) '  Input parameter file (e.g. ../Cards/param_card.dat)'
       write(*,*) '  ---------------------------------------------------'
       write(*,*)
       read(*,'(a)')  infile
       open(lunr,file=infile,status='old',err=102)
       write(*,*)
       write(*,*) '*****************************************************'
       write(*,*) '* Using file:                                       *'
       write(*,'(1x,a1,a25,a1)') ' ',infile,' '
       write(*,*) '* for the input params.                             *'
       write(*,*) '*                                                   *'
       write(*,*) '* >>>>>>Total widths are recalculated here<<<<<     *'
       write(*,*) '*****************************************************'
       write(*,*)

       goto 103 
 102   write(*,*) 'file', infile ,' not found : stopping here'
       stop
 103   continue

c
c      write the param information into the scratch file
c
      done=.false.
      do while(.not.done)
         read(lunr,'(a132)',err=101,end=101) buff
         write(lunp,'(a132)') buff
      enddo
c
 101  continue
       
      elseif(run_mode.eq.1) then

       write(*,*) '*****************************************************'
       write(*,*) '*     run_mode=1 => decaying events                 *'
       write(*,*) '*                                                   *'
       write(*,*) '* Using the param_card.dat in the banner for        *'
       write(*,*) '* the input params.                                 *'
       write(*,*) '*                                                   *'
       write(*,*) '* >>>>>>Total widths are recalculated here<<<<<     *'
       write(*,*) '*****************************************************'
c 
      write(*,*) 'input event file: (e.g. events.lhe)'
      read(*,'(a)')  infile
c
      write(*,*) 'name for output file: (e.g. dec-events.lhe)'
      read(*,'(a)')  outfile
c
      open(lunr,file=infile,status='old')
      open(lunw,status='scratch')
      open(luni,file=outfile,status='unknown')
c
c     copy old banner into new banner and the param_card.dat into the scratch file
c
      done=.false.
      lwrite=.false.
      newbanner=.false.
      read(lunr,'(a132)',err=99) buff
      do while(index(buff,'<init>') .eq. 0)
         if(index(buff,"<header>") .ne. 0) newbanner=.true.
         if (index(buff,"<slha>") .ne. 0 .or.
     $        index(buff,"Begin param_card.dat") .ne. 0) lwrite=.true.
         if (index(buff,"</slha>") .ne. 0 .or.
     $        index(buff,"End param_card.dat") .ne. 0) lwrite=.false.
c         
         if(index(buff,'</header>') .eq. 0 .and.
     $        (newbanner .or. index(buff,'-->') .eq. 0 ))
     $      write(luni,'(a)') buff(1:len_trim(buff))
         if(lwrite) write(lunp,'(a)') buff(1:len_trim(buff))
c         if(lwrite) write(*,'(a50)') 'found in param_card: ',buff
c         if(.not.lwrite) write(*,'(a50)') 'found in banner: ',buff
         read(lunr,'(a132)',err=99) buff
      enddo
c
      rewind(lunr)
c
      endif

      call setpara
      call printout
      call set_mass
      call set_id
c
c
c
      write(*,*) '*****************************************************'
         aa=xran1(iseed)
         write(*,'(a22,f8.6,a24)') ' *  first rnd number= ',aa    ,
     &        '                 *'
      write(*,*) '*****************************************************'
c
c     Input the particle to be decayed
c
      write(*,*)
      write(*,*) ' Implemented decays are for:'
      write(*,*) ' ---------------------------'
      write(*,*)
      write(*,*) ' Leptons: ta- ta+' 
      write(*,*) ' Quarks : t t~' 
      write(*,*) ' Bosons : z w+ w- h'        
      write(*,*) ' Input particle to be decayed (e.g. t~):'
      read(*,'(a3)',err=99)  name
      call case_trap(name,3)
c
c     identify the particle ip
c
      ip=0
      i=-25
      do while(i.le.25.and.ip.eq.0)
         if(name.eq.id(i)) ip=i
         i=i+1
      enddo
c
      if(ip.eq.0) goto 99  ! particle not existing
c
c     find out whether the decay requested is implemented.
c     In case it is, printout the possibilities and
c     ask the user to choose.
c
      call decay_modes(dec_name)
      if(imode.eq.0) goto 99  ! decay not implemented
c
c     setting up branching ratio, widths and so on..
c
      call init
c
c     write out decay information
c
      write(*,*) 
      write(*,*) '----------------------------------------------'
      write(*,*) '              Decay information               '
      write(*,*) '----------------------------------------------'
      write(*,*) ' '
      write(*,*) '  particle name   : ',id(ip)
      write(*,*) '  decay mode      : ',dec_name      
c
c     calculate the MC partial width for normalization purposes
c
      call tot_decay
c
      if(run_mode.eq.0) goto 98  !If only decay width is needed, I'm done 
c
c     first check that the particle to be decayed
c     is present in the events and gather some info
c     on the events
c

      write(*,*) ' '
      write(*,*) '  Wait....Reading In Event File '
      write(*,*) ' '      
      done=.false.
      nevent=0
      sizeievent=0
      do i=0,maxievent
         sum(i)=0d0
         maxwgt(i)=-1d0
      enddo
      do i=0,MaxParticles
         nfound(i)=0
      enddo
      do while (.not. done)
	 call read_event(lunr,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2,done)
	 if(.not.done) then 
            call find_particle(ip,nexternal,ic,k,idecay)         
            nfound(k)=nfound(k)+1
            eventnumber=eventnr(ievent)
            sum(eventnumber)=sum(eventnumber)+wgt
            maxwgt(eventnumber) = max(wgt,maxwgt(eventnumber))
            nevent=nevent+1
         endif
      enddo
      do i=1, sizeievent
         sum(0)=sum(0)+sum(i)
         maxwgt(0)=maxwgt(0)+maxwgt(i)
      enddo


      rewind(lunr)

      write(*,*) '----------------------------------------------'
      write(*,*) '        Input events information              '
      write(*,*) '----------------------------------------------'
      write(*,*) ' '
      write(*,*) '  Input Event file             :  ',infile
      write(*,*) '  Number of Events             :  ',nevent
      write(*,*) '  Integrated weight (pb)       :  ',sum(0)
      write(*,*) '  Max wgt                      :  ',maxwgt(0)
      write(*,*) '  Average wgt                  :  ',sum(0)/nevent
      write(*,*)

      if(nfound(0).eq.nevent) then
         write(*,*) '  There are no events with particle ',
     &        id(ip),' to decay !!'
         stop
      else
         do i=0,MaxParticles
            if(nfound(i).gt.0) then
      write(*,'(1x,a14,i2,a1,a2,a13,i10)') 
     & '  Events with ',i,' ',id(ip),'           :',nfound(i)
            endif   
         enddo                       
      endif

      
c
c     read each event, decay it, and write it out
c     note that it is NOT assumed that all the events
c     have the same number of particles in the final
c     state or the same type. Each event is treated
c     independently from the others.
c
      write(*,*) ' '
      write(*,*) '----------------------------------------------'
      write(*,*) '   Decay events in file : in progress         '      
      write(*,*) '----------------------------------------------'
      write(*,*) ' '
      done=.false.
      i   = 0
      do while (.not. done)
         call read_event(lunr,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2,done)
         if (.not. done) then
            i=i+1
            call decay_event(P,wgt,nexternal,ic)
            call write_event(lunw,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2)
         endif
         iten=int(real(nevent)/10e0)+1
         if(real(i)/real(iten).eq.real(i/iten)) then
         write(*,'(10x,i3,a)')int(real(i)/real(nevent)*100),' % done'
         endif
      enddo
      write(*,*) '           100% done'
c
c     decayed event information
c
c-- statistics of the decay_event routine
c
      write(*,*) 
      write(*,*) '----------------------------------------------'
      write(*,*) '   Statistics of the unweighting              '      
      write(*,*) '----------------------------------------------'
      write(*,*) ' '
      write(*,*) '  Events weighted   = ' ,n_wei
      write(*,*) '  Events unweighted = ' ,nevent
      write(*,*) '  Width  (wgt)      = ' ,s_wei/real(n_wei), ' GeV'
      write(*,*) '  Width  (unwgt)    = ' ,s_unw/real(n_wei), ' GeV'
      IF(DEBUG) then
       write(*,*) '  Events   over     = ' ,n_ove
       write(*,*) '  Integral over     = ' ,s_ove/real(n_wei), ' GeV'
      ENDIF
      write(*,*) ' '
c
      write(*,*) ' '
      write(*,*) '  Wait....Writing Out Decayed Event file '
      write(*,*) ' '      
      rewind(lunw)
      done=.false.
      nevent=0
      do i=0,sizeievent
         sum(i)=0d0
         maxwgt(i)=-1d0
      enddo
      do while (.not. done)
         call read_event(lunw,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2,done)
         if(.not.done) then
            eventnumber=eventnr(ievent)
            sum(eventnumber)=sum(eventnumber)+wgt
            maxwgt(eventnumber) = max(wgt,maxwgt(eventnumber))
            nevent=nevent+1
         endif
      enddo
      sum(0)=0d0
      maxwgt(0)=-1d0
      do i=1, sizeievent
         sum(0)=sum(0)+sum(i)
         maxwgt(0)=maxwgt(0)+maxwgt(i)
      enddo

c RF: To write the correct iseed in the banner, read it again form file iseed.dat
      open(lunrnd,file='./iseed.dat',status="old",err=300)
      read(lunrnd,err=300,end=300,fmt='(i10)') iseed
      iseed=iseed+1
      close(lunrnd)
 300  continue
cend RF


      write(*,*) '----------------------------------------------'
      write(*,*) '        Output events information             '
      write(*,*) '----------------------------------------------'
      write(*,*) ' '
      write(*,*) '  Output Event file            :  ',outfile
      write(*,*) '  Number of Events             :  ',nevent
      write(*,*) '  Integrated weight (pb)       :  ',sum(0)
      write(*,*) '  Max wgt                      :  ',maxwgt(0)
      write(*,*) '  Average wgt                  :  ',sum(0)/nevent
      write(*,*) ' '
c
      if(newbanner) then
         write(luni,'(a)') "<MGDecayInfo>"
      else
         write(luni,'(a)') "#********************************************************************"
         write(luni,'(a)') '#    particle decayed               '
         write(luni,'(a)') "#********************************************************************"
      endif
      write(luni,'(a,a3)')  '#  particle name   : ',id(ip)
      write(luni,'(a,a)')   '#  decay mode      : ',dec_name      
      write(luni,'(a,e10.5)')   '#  MC partial width: ',MC_width
      write(luni,'(a,i10  )') '#  Rnd seed              : ',iseed
      write(luni,'(a,i10  )') '#  Number of Events      : ',nevent
      write(luni,'(a,e10.5)') '#  Integrated weight (pb): ',sum(0)
      write(luni,'(a,e10.5)') '#  Max wgt               : ',maxwgt(0)
      write(luni,'(a,e10.5)') '#  Average wgt           : ',sum(0)/nevent
      if(newbanner) then
         write(luni,'(a)') '</MGDecayInfo>'
         write(luni,'(a)') '</header>'
      else
         write(luni,'(a)') '#********************************************************************' 
         write(luni,'(a)') '-->'
      endif


C   Write out compulsory init info
      write(luni,'(a)') '<init>'

      rewind(lunr)
      do while(index(buff,'</init') .eq. 0)
         read(lunr,'(a132)',err=99) buff
         if (index(buff,'<init') .eq. 0) cycle
         read(lunr,*) (idbmup(i),i=1,2),(ebmup(i),i=1,2),(pdfgup(i),i=1,2),
     $        (pdfsup(i),i=1,2),idwtup,nprup
         write(luni,90) (idbmup(i),i=1,2),(ebmup(i),i=1,2),(pdfgup(i),i=1,2),
     $        (pdfsup(i),i=1,2),idwtup,nprup
         do i=1,sizeievent
            read(lunr,*) xsecup,xerrup,xmaxup,lprup 
            write(luni,91) sum(i),xerrup*sum(i)/xsecup,maxwgt(i),lprup
         enddo
      enddo
      write(luni,'(a)') '</init>'

 90   FORMAT(2i6,2e19.11,2i2,2i6,i2,i3)
 91   FORMAT(3e19.11,i4)

      rewind(lunw)
      i=0
      done = .false.
      do while (.not. done)
         call read_event(lunw,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2,done)

      if (.not. done) then
         call write_event(luni,P,wgt,nexternal,ic,ievent,scale,aqcd,aqed,buff2)
         
      endif
         i=i+1
      enddo


 98   continue

      write(luni,'(a)')'</LesHouchesEvents>'

c
c     close datafiles
c
      close(luni)
      close(lunr)
      close(lunw)

      stop

 99   write(*,*) 'error'
      stop

      end 


      
      integer function eventnr(number)
c*************************************************************************
c     function that should be given a number and returns
c     the number:
c     1   for the first number (i.e. the first time this
c         function gets executed)
c     2   for the second number if the second number is
c         diffent from the first, otherwise it returns 1
c     3   for the third number, if the third number is
c         different from the first or second, otherwise
c         it returns 1, or 2 depending if the third number
c         is equal to the first or the second.
c     4   etc.
c     It updates also a common block with the total number
c     of different values used as the argument of this function
c
c     before the first call to this function sizeievent should
c     should be set to 0.
c*************************************************************************
      include "decay.inc"
      integer i,number,test(maxievent)
      logical exist
      
      integer sizeievent
      common/ievents/sizeievent

      save test

      exist = .false.
      if (sizeievent.lt.1)goto 188
      do i=1, sizeievent
         if (number.eq.test(i))then
            exist=.true.
            eventnr=i
         endif
      enddo

 188  continue
      if (.not.exist)then
         sizeievent=sizeievent+1
         eventnr=sizeievent
         test(sizeievent)=number
      endif

      if (sizeievent.ge.maxievent)then
         write (*,*)
     &   'Error, too many different event numbers in event file'
         return
      endif

      return
      end






      subroutine set_id
c****************************************
c     to set the particles ids
*****************************************
      implicit none
      include 'decay.inc'
c
c
c-Quarks: d u s c b t d~ u~ s~ c~ b~ t~ 
      id(1) ='d'
      id(2) ='u'
      id(3) ='s'
      id(4) ='c'
      id(5) ='b'
      id(6) ='t'
      id(-1)='d~'
      id(-2)='u~'
      id(-3)='s~'
      id(-4)='c~'
      id(-5)='b~'
      id(-6)='t~'
c-Leptons: e- mu- ta- ve vm vt e+ mu+ ta+ ve~ vm~ vt~ 
      id(11) ='e-'
      id(12) ='ve'
      id(13) ='mu-'
      id(14) ='vm'
      id(15) ='ta-'
      id(16) ='vt'
      id(-11)='e+'
      id(-12)='ve~'
      id(-13)='mu+'
      id(-14)='vm~'
      id(-15)='ta+'
      id(-16)='vt~'
c-Bosons: g a z w+ w- h        
      id(21)  ='g'
      id(22)  ='a'
      id(23)  ='z'
      id(24)  ='w+'
      id(-24) ='w-'
      id(25)  ='h'
      return
      end
 

      subroutine set_mass
c****************************************
c     to set the particles ids
*****************************************
      implicit none
      include 'decay.inc'
      include 'coupl.inc'
c
c
c-Quarks: d u s c b t d~ u~ s~ c~ b~ t~ 
      pdgmass(1) =0d0
      pdgmass(2) =0d0
      pdgmass(3) =0d0
      pdgmass(4) =cmass
      pdgmass(5) =bmass
      pdgmass(6) =tmass
      pdgmass(-1)=0d0
      pdgmass(-2)=0d0
      pdgmass(-3)=0d0
      pdgmass(-4)=cmass
      pdgmass(-5)=bmass
      pdgmass(-6)=tmass
c-Leptons: e- mu- ta- ve vm vt e+ mu+ ta+ ve~ vm~ vt~ 
      pdgmass(11) =0d0
      pdgmass(12) =0d0
      pdgmass(13) =0d0
      pdgmass(14) =0d0
      pdgmass(15) =lmass
      pdgmass(16) =0d0
      pdgmass(-11)=0d0
      pdgmass(-12)=0d0
      pdgmass(-13)=0d0
      pdgmass(-14)=0d0
      pdgmass(-15)=lmass
      pdgmass(-16)=0d0
c-Bosons: g a z w+ w- h        
      pdgmass(21)  =0d0
      pdgmass(22)  =0d0
      pdgmass(23)  =zmass
      pdgmass(24)  =wmass
      pdgmass(-24) =wmass
      pdgmass(25)  =hmass
      return
      end
      
      subroutine decay_modes(dec_name)
c*********************************************
c     to set the particles ids
c*********************************************
      implicit none
      include 'decay.inc'
c
c     argument
c
      character*50 dec_name
c
c     common
c
      include 'coupl.inc'
c
c     local
c
      character*50 string(30)
      integer i
c----------
c     START
c----------

      write(*,*) 'particle to decay :',id(ip)
      write(*,*) 
      write(*,*) 'Implemented decay modes:'
      write(*,*) '------------------------'
c
      if(ip.eq.6) then  !top
         string(1) = ' t -> b  w+ '
         string(2) = ' t -> b  ve e+'
         string(3) = ' t -> b  vm mu+'
         string(4) = ' t -> b  vt ta+'
         string(5) = ' t -> b  vl l+    (e+mu)'
         string(6) = ' t -> b  vl l+    (e+mu+ta)'
         string(7) = ' t -> b  j   j    (ud+cs)'
         string(8) = ' t -> b  anything (e+mu+ta+ud+cs)'

         write(*,'(i2,2x,a40)') (i,string(i),i=1,8)
         write(*,*) ' '
         write(*,*) 'your choice is:'   
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.8)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
         if(imode.eq.1) then !number of particles in the decay
            nd=2
         else
            nd=3
         endif
         dec_name=string(imode)
         m1=tmass
         return
      endif
c
      if(ip.eq.-6) then  !anti-top
         string(1) =  ' t~ -> b~  w- '
         string(2) =  ' t~ -> b~  e- ve~'
         string(3) =  ' t~ -> b~ mu- vm~'
         string(4) =  ' t~ -> b~ ta- vt~'
         string(5) =  ' t~ -> b~  l- vl~  (e+mu)'
         string(6) =  ' t~ -> b~  l- vl~  (e+mu+ta)'
         string(7) =  ' t~ -> b~  j   j   (ud+cs)'
         string(8) =  ' t~ -> b~ anything (e+mu+ta+ud+cs)'
         write(*,'(i2,2x,a40)') (i,string(i),i=1,8)
         write(*,*) ' '
         write(*,*) 'your choice is:'   
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.8)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
         if(imode.eq.1) then    !number of particles in the decay 
            nd=2
         else
            nd=3
         endif
         m1=tmass
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.15) then  ! tau-
         string(1) =  ' ta- -> vt   e- ve~  '
         string(2) =  ' ta- -> vt  mu- vm~  '
         string(3) =  ' ta- -> vt   l- vl~ (e+mu) '
         string(4) =  ' ta- -> vt   pi-     '
         string(5) =  ' ta- -> vt  rho(770)-'
         write(*,'(i2,2x,a30)') (i,string(i),i=1,5)
         write(*,*) ' '
         write(*,*) 'your choice is:'   
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.5)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
         if(imode.le.3) then    !number of particles in the decay
            nd=3
         else
            nd=2
         endif
         m1=lmass
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.-15) then  ! tau+
         string(1) =  ' ta+ -> vt~  ve e+   '
         string(2) =  ' ta+ -> vt~  vm mu+   '
         string(3) =  ' ta+ -> vt~  vl  l+  (e+mu)'
         string(4) =  ' ta+ -> vt~   pi+     '
         string(5) =  ' ta+ -> vt~  rho(770)+'
         write(*,'(i2,2x,a30)') (i,string(i),i=1,5)
         write(*,*) ' '
         write(*,*) 'your choice is:'   
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.5)) then
            write(*,*) 'choice not implemented'
            imode=0
         endif
         if(imode.le.3) then !number of particles in the decay
            nd=3
         else
            nd=2
         endif
         m1=lmass
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.23) then  ! z
         string(1) =  ' z -> e-   e+ '
         string(2) =  ' z -> mu- mu+ '
         string(3) =  ' z -> ta- ta+ '
         string(4) =  ' z -> l-   l+  (e+mu)'
         string(5) =  ' z -> l-   l+  (e+mu+ta )'
         string(6) =  ' z -> vl  vl~  (ve+vm+vt)'
         string(7) =  ' z -> b    b~            '
         string(8) =  ' z -> c    c~            '
         string(9) =  ' z -> j    j~  (u+d+c+s) '
         string(10)=  ' z -> j    j~  (u+d+c+s+b)'
         string(11)=  ' z -> visible  (e+mu+ta+u+d+c+s+b)'
         string(12)=  ' z -> anything'

         write(*,'(i2,2x,a40)') (i,string(i),i=1,10)
         write(*,*) ' '
         write(*,*) 'your choice is:'   
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.12)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
         m1=zmass
         nd=2                   !number of particles in the decay
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.24) then  ! w+
         string(1) =  ' w+ -> e+  ve'
         string(2) =  ' w+ -> mu+ vm'
         string(3) =  ' w+ -> ta+ vt '
         string(4) =  ' w+ -> l+  vl (e+mu)'
         string(5) =  ' w+ -> l+  vl (e+mu+ta)'
         string(6) =  ' w+ -> j   j  (ud+cs)'
         string(7) =  ' w+ -> anything (e+mu+ta+ud+cs)'

         write(*,'(i2,2x,a40)') (i,string(i),i=1,7)
         write(*,*) 'your choice is:'   
         write(*,*) ' '
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.7)) then
            write(*,*) 'choice not implemented'
            imode=0
         endif
         nd=2                   !number of particles in the decay
         m1=wmass
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.-24) then  ! w-
         string(1) =  ' w- -> e-  ve~'
         string(2) =  ' w- -> mu- vm~'
         string(3) =  ' w- -> ta- vt~ '
         string(4) =  ' w- -> l-  vl~ (e+mu)'
         string(5) =  ' w- -> l-  vl~ (e+mu+ta)'
         string(6) =  ' w- -> j   j   (ud+cs)'
         string(7) =  ' w- -> anything (e+mu+ta+ud+cs)'
         write(*,'(i2,2x,a40)') (i,string(i),i=1,7)
         write(*,*) 'your choice is:'   
         write(*,*) ' '
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.7)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
         m1=wmass
         nd=2                   !number of particles in the decay
         dec_name=string(imode)
         return
      endif
c
      if(ip.eq.25) then  ! h
        string(1) =' h -> b   b~ '
        string(2) =' h -> ta- ta+'
        string(3) =' h -> mu- mu+'
        string(4) =' h -> c   c~ '
        string(5) =' h -> t   t~ (when m_h>2*m_t)'
        string(6) =' h -> g   g  '
        string(7) =' h -> a   a  '
        string(8) =' h -> z   a  (when m_h>  m_z)'
        string(9) =' h -> w+  w- (when m_h>2*m_w)'
        string(10)=" h -> w*  w -> l  vl  l' vl' (l,l'=e,mu)" 
        string(11)=" h -> w*  w -> l  vl  l' vl' (l,l'=e,mu,ta)" 
        string(12)=" h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu)"
        string(13)=" h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu,ta)"
        string(14)=' h -> z   z  (when m_h>2*m_z)'
        string(15)=" h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu)"
        string(16)=" h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu,ta)"
        string(17)=" h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu )"
        string(18)=" h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu,ta)"
        string(19)=" h -> z*  z -> b  b~  l-  l+ (l=e,mu )"
        string(20)=" h -> z*  z -> b  b~  l-  l+ (l=e,mu,ta )"
        string(21)=" h -> z*  z -> vl vl~ l-' l+'(l=e,mu,ta;l'=e,mu) "
        string(22)=" h -> z*  z -> vl vl~ l-' l+'(l,l'=e,mu,ta) "

         write(*,'(i2,2x,a50)') (i,string(i),i=1,22)
         write(*,*) 'your choice is:'   
         write(*,*) ' '
         read (*,*) imode
         if(.not.(imode.ge.1.and.imode.le.22)) then
            write(*,*) 'choice not implemented'
            imode=0
            return
         endif
c--   number of decay products
         ND=4
         m1=hmass
         if(imode.le.9.or.imode.eq.14) ND=2
         dec_name=string(imode)
         return
      endif
      
      write(*,*) 'no decay mode implemented for particle ',id(ip)
      

      return
      end
      

      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)
      end


      subroutine init
c********************************************************
c     sets up : the branching ratios (best=HDECAY or PDG)
c               LO partial widths 
c               ND: the number of decay products
c********************************************************
      implicit none
c
c     Include
c
      include 'decay.inc'
      include 'coupl.inc'
      include 'calc_values.inc'
c
c-----
cstart
c-----
c      

c
c     PDG2002 values for the Branching ratios:
c

c--   tau
      br_ta_lv =0.175d0
      br_ta_pi =0.111d0
      br_ta_ro =0.254d0
c--   w
      br_w_lv =0.1068d0
      br_w_jj =1d0-3d0*br_w_lv
c--   z
      br_z_ll =0.0336d0
      br_z_vv =0.2000d0
      br_z_cc =0.1176d0
      br_z_bb =0.1514d0
      br_z_jj =0.6991d0-br_z_cc-br_z_bb
      
      write(*,*)
      write(*,*) '----------------------------------------------'
      write(*,*) '         Relevant branching ratios            '
      write(*,*) '----------------------------------------------'
      write(*,*)

c-------------------------------------------------------------------
      if(abs(ip).eq.6) then     !top


         write(*,16) 'BR (t -> b w    ) =  ', 1d0
         bratio=1d0
         if(imode.eq.2) then
         write(*,16) 'BR (w -> e ve   ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.3) then
         write(*,16) 'BR (w -> mu vm  ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.4) then
         write(*,16) 'BR (w -> tau vt ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.5) then
         write(*,15) 'BR (w -> l vl   ) =  ', 2d0*br_w_lv,' (l=e,mu)'
         bratio=2d0*br_w_lv
         elseif(imode.eq.6) then
         write(*,15) 'BR (w -> l vl   ) =  ', 3d0*br_w_lv,' (l=e,mu,ta)'
         bratio=3d0*br_w_lv
         elseif(imode.eq.7) then
         write(*,15) 'BR (w -> j j    ) =  ', br_w_jj   ,' (jj=ud+cs)'
         bratio=br_w_jj
         elseif(imode.eq.8) then
         write(*,15) 'BR (w ->anything) =  ',1d0
         bratio=1d0
         endif
         
         If(imode.eq.1) then    !t -> b w+
            calc_width=twidth
         elseif(imode.eq.2.or.imode.eq.3.or.imode.eq.5) then !  t -> b vl l+
            calc_width=twidth*(w_w_nl/wwidth)
            if(imode.eq.5) calc_width=calc_width*2d0
         elseif(imode.eq.4) then !     t -> b vt tau+
            calc_width=twidth*(w_w_tau/wwidth)
         elseif(imode.eq.6) then !     t -> b vl l+
            calc_width=twidth*((2d0*w_w_nl+w_w_tau)/wwidth)
         elseif(imode.eq.7) then !    t -> b j j 
            calc_width=twidth*(w_w_ud+w_w_cs)/wwidth
         elseif(imode.eq.8) then !    t -> b anything 
            calc_width=twidth
         endif
                  
      endif
c-------------------------------------------------------------------
      if(abs(ip).eq.15) then    !tau
         if(imode.eq.1)     then
            write(*,16) 'BR (ta -> vt  e ve) =  ', br_ta_lv
         bratio=br_ta_lv
         elseif(imode.eq.2)     then
            write(*,16) 'BR (ta -> vt mu vm) =  ', br_ta_lv
         bratio=br_ta_lv
         elseif(imode.eq.3)     then
        write(*,15) 'BR (ta -> vt  l vl) =  ', 2d0*br_ta_lv,' (l=e,mu)'
         bratio=2d0*br_ta_lv
         elseif(imode.eq.4) then
            write(*,16) 'BR (ta -> vt  pi  ) =  ', br_ta_pi
         bratio=br_ta_pi
         elseif(imode.eq.5) then
            write(*,16) 'BR (ta -> vt  rho ) =  ', br_ta_ro
         bratio=br_ta_ro
         endif
         
         if(imode.eq.1.or.imode.eq.2) then !ta -> vt vl l
            calc_width=lwidth*br_ta_lv
         elseIf(imode.eq.3) then
               calc_width=2d0*lwidth*br_ta_lv 
         elseIf(imode.eq.5) then !     tau -> vt rho
            calc_width=lwidth*br_ta_ro
         elseif(imode.eq.4) then !    tau -> vt pi
            calc_width=lwidth*br_ta_pi
         endif
         
      endif

c-------------------------------------------------------------------
      if(ip.eq.23) then         ! z
         if    (imode.eq.1) then
            write(*,16) 'BR (z -> e- e+  ) =  ', br_z_ll
         bratio=br_z_ll
         elseif(imode.eq.2) then
            write(*,16) 'BR (z -> mu- mu+) =  ', br_z_ll
         bratio=br_z_ll
         elseif(imode.eq.3) then
            write(*,16) 'BR (z -> ta- ta+) =  ', br_z_ll
         bratio=br_z_ll
         elseif(imode.eq.4) then
        write(*,15) 'BR (z -> l- l+  ) =  ', 2d0*br_z_ll,' (l=e,mu)'
         bratio=2d0*br_z_ll
         elseif(imode.eq.5) then
        write(*,15) 'BR (z -> l- l+  ) =  ', 3d0*br_z_ll,' (l=e,mu,ta)'
         bratio=3d0*br_z_ll
         elseif(imode.eq.6) then
            write(*,15) 'BR (z -> vl vl~ ) =  ', br_z_vv,' (l=e,mu,ta)'
         bratio=br_z_vv
         elseif(imode.eq.7) then
             write(*,16) 'BR (z -> b  b~  ) =  ', br_z_bb
         bratio=br_z_bb
         elseif(imode.eq.8)then
            write(*,16) 'BR (z -> c  c~  ) =  ', br_z_cc
         bratio=br_z_cc
         elseif(imode.eq.9)then
            write(*,15) 'BR (z -> j  j~  ) =  ', br_z_jj+br_z_cc,
     &           ' (j=u+d+c+s)'
         bratio=br_z_jj+br_z_cc
         elseif(imode.eq.10)then
            write(*,15) 'BR (z -> j  j~  ) =  ', 
     &           br_z_jj+br_z_bb+br_z_cc,' (j=u+d+c+s+b)'
         bratio=br_z_jj+br_z_bb+br_z_cc
         elseif(imode.eq.11)then
            write(*,15) 'BR (z -> visible) =  ', 
     &           br_z_jj+br_z_bb+br_z_cc+3d0*br_z_ll,
     &           ' (j=l+j)'
         bratio=br_z_jj+br_z_bb+br_z_cc+3d0*br_z_ll
         elseif(imode.eq.12)then
            write(*,15) 'BR (z -> anyth. ) =  ', 
     &           1d0,' (j=l+v+j)'
         bratio=1d0
         endif
         
         
         If(imode.eq.1.or.imode.eq.2) then !     z->e- e+
            calc_width=w_z_ll
         elseif(imode.eq.4) then !     z->e- e+,mu-mu+,ta-ta+
            calc_width=2d0*w_z_ll
         elseif(imode.eq.3) then !     z->ta- ta+
            calc_width=w_z_tau
         elseif(imode.eq.5) then !     z->e- e+,mu-mu+,ta-ta+
            calc_width=2d0*w_z_ll+w_z_tau
         elseif(imode.eq.6) then !    z-> vl  vl~
            calc_width=w_z_nn*3d0
         elseif(imode.eq.7) then !    z->b b~
            calc_width=w_z_bb
         elseif(imode.eq.8) then !     z->c c~
            calc_width=w_z_cc
         elseif(imode.eq.9) then !     z->jj (u+d+c+s)
            calc_width=Two*w_z_dd + w_z_uu + w_z_cc 
         elseif(imode.eq.10) then !*     z->jj (u+d+c+s+b)
            calc_width=Two*w_z_dd + w_z_uu + w_z_cc + w_z_bb 
         elseif(imode.eq.11) then !*     z->any (l+u+d+c+s+b)
            calc_width=zwidth-w_z_nn*3d0
         elseif(imode.eq.12) then !*     total width
            calc_width=zwidth
         endif
      endif
      
c-------------------------------------------------------------------
      if(abs(ip).eq.24) then    ! w
         if    (imode.eq.1) then
         write(*,16) 'BR (w -> e ve   ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.2) then
         write(*,16) 'BR (w -> mu vm  ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.3) then
         write(*,16) 'BR (w -> tau vt ) =  ', br_w_lv
         bratio=br_w_lv
         elseif(imode.eq.4) then
         write(*,15) 'BR (w -> l vl   ) =  ', 2d0*br_w_lv,' (l=e,mu)'
         bratio=2d0*br_w_lv
         elseif(imode.eq.5) then
         write(*,15) 'BR (w -> l vl   ) =  ', 3d0*br_w_lv,' (l=e,mu,ta)'
         bratio=3d0*br_w_lv
         elseif(imode.eq.6) then
         write(*,15) 'BR (w -> j j    ) =  ', br_w_jj   ,' (jj=ud+cs)'
         bratio=br_w_jj
         elseif(imode.eq.7) then
         write(*,15) 'BR (w ->anything) =  1  (=e+mu+ta+ud+cs)'
         bratio=1d0
         endif
         
         If(imode.eq.1.or.imode.eq.2) then !w-> l ve
            calc_width=w_w_nl
         elseif(imode.eq.3) then !     w -> ta vt
            calc_width=w_w_tau
         elseif(imode.eq.4) then !     w -> l vl (e,mu)
            calc_width=2d0*w_w_nl
         elseif(imode.eq.5) then !    w -> l vl (e,mu,tau)
            calc_width=2d0*w_w_nl+w_w_tau
         elseif(imode.eq.6) then !     w -> ud+cs
            calc_width=w_w_ud+w_w_cs
         elseif(imode.eq.7) then !     w -> anything
            calc_width=wwidth
         endif

      endif

c-------------------------------------------------------------------
      if(ip.eq.25) then         ! higgs
         
         write(*,15) 'Higgs mass        =  ', hmass,  ' GeV'
         write(*,15) 'Higgs tot width   =  ', SMWDTH, ' GeV'
         if(imode.eq.1) then
            write(*,16) 'BR (h -> b b~   ) =  ', SMBRB
            bratio=SMBRB
         elseif(imode.eq.2) then
            write(*,16) 'BR (h -> ta+ ta-) =  ', SMBRL
            bratio=SMBRL
         elseif(imode.eq.3) then
            write(*,16) 'BR (h -> mu+ mu-) =  ', SMBRM
            bratio=SMBRM
         elseif(imode.eq.4) then 
            write(*,16) 'BR (h -> c  c~  ) =  ', SMBRC
            bratio=SMBRC
         elseif(imode.eq.5) then 
            write(*,16) 'BR (h -> t  t~  ) =  ', SMBRT
            bratio=SMBRT
         elseif(imode.eq.6) then 
            write(*,16) 'BR (h -> g  g   ) =  ', SMBRG
            bratio=SMBRG
         elseif(imode.eq.7) then 
            write(*,16) 'BR (h -> a  a   ) =  ', SMBRGA
            bratio=SMBRGA
         elseif(imode.eq.8) then 
            write(*,16) 'BR (h -> z  a   ) =  ', SMBRZGA
            bratio=SMBRZGA
         elseif(imode.ge.9.and.imode.le.13) then 
            write(*,16) 'BR (h -> w+ w-  ) =  ', SMBRW
            bratio=SMBRW
         elseif(imode.ge.14.and.imode.le.22) then 
            write(*,16) 'BR (h -> z  z   ) =  ', SMBRZ    
            bratio=SMBRZ    
         endif

         if(imode.eq.10) then
         write(*,15) 'BR (w -> l vl   ) =  ', 2d0*br_w_lv,' (l=e,mu)'
            bratio=bratio*(2d0*br_w_lv)**2
         elseif(imode.eq.11) then
         write(*,15) 'BR (w -> l vl   ) =  ', 3d0*br_w_lv,' (l=e,mu,ta)'
            bratio=bratio*(3d0*br_w_lv)**2
         elseif(imode.eq.12) then
         write(*,15) 'BR (w -> j j    ) =  ', br_w_jj   ,' (jj=ud+cs)'
         write(*,15) 'BR (w -> l vl   ) =  ', 2d0*br_w_lv,' (l=e,mu)'
            bratio=bratio*2d0*br_w_jj*2d0*br_w_lv
         elseif(imode.eq.13) then
         write(*,15) 'BR (w -> j j    ) =  ', br_w_jj   ,' (jj=ud+cs)'
         write(*,15) 'BR (w -> l vl   ) =  ', 3d0*br_w_lv,' (l=e,mu,ta)'
            bratio=bratio*2d0*br_w_jj*3d0*br_w_lv
         endif

         if(imode.eq.15) then
        write(*,15) 'BR (z -> l- l+  ) =  ', 2d0*br_z_ll,' (l=e,mu)'
            bratio=bratio*(2d0*br_z_ll)**2
         elseif(imode.eq.16) then
        write(*,15) 'BR (z -> l- l+  ) =  ', 3d0*br_z_ll,' (l=e,mu,ta)'
            bratio=bratio*(3d0*br_z_ll)**2
         elseif(imode.eq.17) then
        write(*,15) 'BR (z -> j  j~  ) =  ', br_z_jj+br_z_cc,
     &           ' (j=u+d+c+s)'
        write(*,15) 'BR (z -> l- l+  ) =  ', 2d0*br_z_ll,' (l=e,mu)'
            bratio=bratio*(br_z_jj+br_z_cc)*2d0*br_z_ll*2d0
         elseif(imode.eq.18) then
        write(*,15) 'BR (z -> j  j~  ) =  ', br_z_jj+br_z_cc,
     &           ' (j=u+d+c+s)'
        write(*,15) 'BR (z -> l- l+  ) =  ', 3d0*br_z_ll,' (l=e,mu,ta)'
            bratio=bratio*(br_z_jj+br_z_cc)*3d0*br_z_ll*2d0
         elseif(imode.eq.19) then
        write(*,16) 'BR (z -> b  b~  ) =  ', br_z_bb
        write(*,15) 'BR (z -> l- l+  ) =  ', 2d0*br_z_ll,' (l=e,mu)'
            bratio=bratio*br_z_bb*2d0*br_z_ll*2d0
         elseif(imode.eq.20) then
        write(*,16) 'BR (z -> b  b~  ) =  ', br_z_bb
        write(*,15) 'BR (z -> l- l+  ) =  ', 3d0*br_z_ll,' (l=e,mu,ta)'
            bratio=bratio*br_z_bb*3d0*br_z_ll*2d0
         elseif(imode.eq.21) then
        write(*,15) 'BR (z -> vl vl~ ) =  ', br_z_vv,' (l=e,mu,ta)'
        write(*,15) 'BR (z -> l- l+  ) =  ', 2d0*br_z_ll,' (l=e,mu)'
            bratio=bratio*br_z_vv*2d0*br_z_ll*2d0
         elseif(imode.eq.22) then
        write(*,15) 'BR (z -> vl vl~ ) =  ', br_z_vv,' (l=e,mu,ta)'
        write(*,15) 'BR (z -> l- l+  ) =  ', 3d0*br_z_ll,' (l=e,mu,ta)'
            bratio=bratio*br_z_vv*3d0*br_z_ll*2d0
         endif


        write(*,16) 'Best tot Br Ratio =  ', bratio

         If(imode.eq.1) then    !    h->b b~
            calc_width=w_h_bb
         elseif(imode.eq.2) then ! h->ta- ta+
            calc_width=w_h_tau
         elseif(imode.eq.3) then ! h->mu- mu+
            calc_width=w_h_tau/(lmass/0.105658389d0)**2
         elseif(imode.eq.4) then !   h-> c c~
            calc_width=w_h_cc
         elseif(imode.eq.5) then !    h-> t t~
            if(hmass.le.2d0*tmass) then
               write(*,*) 
               write(*,*) 'Error: this decay mode assumes m_h>2 m_t'
               write(*,*) 
               stop
            endif
             calc_width=w_h_tt
         elseif(imode.eq.6) then ! h-> g g 
            calc_width=SMBRG*SMWDTH 
         elseif(imode.eq.7) then !   h-> a a 
            calc_width= SMBRGA*SMWDTH 
         elseif(imode.eq.8) then !     h-> z a   
            if(hmass.le.zmass) then
               write(*,*) 
               write(*,*) 'Error: this decay mode assumes m_h>m_Z'
               write(*,*) 
               stop
            endif
            calc_width= SMBRZGA*SMWDTH 
         elseif(imode.eq.9) then !    h-> w w
            if(hmass.le.2d0*wmass) then
               write(*,*) 
               write(*,*) 'Error: this decay mode assumes m_h>2*m_W'
               write(*,*) 
               stop
            endif
            calc_width=w_h_ww
         elseif(imode.eq.10) then !  h -> w*  w -> e  ve mu vmu 
            calc_width= SMBRW*SMWDTH*(2d0*w_w_nl/wwidth)**2
         elseif(imode.eq.11) then ! h -> w*  w -> l  vl  l' vl' (l,l'=e,mu,ta)
            calc_width= SMBRW*SMWDTH*((2d0*w_w_nl+w_w_tau)/wwidth)**2
         elseif(imode.eq.12) then !    h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu)
        calc_width= SMBRW*SMWDTH*2d0*(2d0*w_w_nl*(w_w_ud+w_w_cs))/wwidth**2
         elseif(imode.eq.13) then !h -> w*  w -> j  j   l  vl  (jj=ud,cs;l=e,mu,ta)
            calc_width= SMBRW*SMWDTH*2d0*
     &           ((2d0*w_w_nl+w_w_tau)*(w_w_ud+w_w_cs))/wwidth**2
         elseif(imode.eq.14) then ! h-> z z 
            if(hmass.le.2d0*zmass) then
               write(*,*) 
               write(*,*) 'Error: this decay mode assumes m_h>2*m_Z'
               write(*,*) 
               stop
            endif
            calc_width=w_h_zz
         elseif(imode.eq.15) then ! h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu)
            calc_width= SMBRZ*SMWDTH*(2d0*w_z_ll/zwidth)**2
         elseif(imode.eq.16) then ! h -> z*  z -> l- l+  l-' l+'(l,l'=e,mu,ta)
            calc_width= SMBRZ*SMWDTH*
     &                  ((2d0*w_z_ll+w_z_tau)/zwidth)**2
         elseif(imode.eq.17) then ! h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu )
            calc_width= SMBRZ*SMWDTH*2d0*
     &      (2d0*w_z_ll*(Two*w_z_dd + w_z_uu + w_z_cc))/zwidth**2
         elseif(imode.eq.18) then ! h -> z*  z -> j  j~  l-  l+ (j=u,d,c,s;l=e,mu,ta)
            calc_width= SMBRZ*SMWDTH*2d0*
     &  ((2d0*w_z_ll+w_z_tau)*
     &  (Two*w_z_dd + w_z_uu + w_z_cc))/zwidth**2
         elseif(imode.eq.19) then ! h -> z*  z -> b  b~  l-  l+ (l=e,mu )
            calc_width=SMBRZ*SMWDTH*2d0*
     &                 w_z_ll*w_z_bb*2d0/zwidth**2
         elseif(imode.eq.20) then ! h -> z*  z -> b  b~  l-  l+ (l=e,mu,ta )
            calc_width=SMBRZ*SMWDTH*2d0*
     &           (2d0*w_z_ll+w_z_tau)*w_z_bb/zwidth**2
         elseif(imode.eq.21) then ! h -> z*  z -> vl vl~ l-' l+'(l=e,mu,ta;l'=e,mu)
            calc_width=SMBRZ*SMWDTH*2d0*
     &                 3d0*w_z_nn*2d0*w_z_ll/zwidth**2
         elseif(imode.eq.22) then ! h -> z*  z -> vl vl~ l-' l+'(l,l'=e,mu,ta)
            calc_width=SMBRZ*SMWDTH*2d0*
     &                 3d0*w_z_nn*(2d0*w_z_ll+w_z_tau)/zwidth**2
         endif   
 
         calc_br=calc_width/SMWDTH
         write(*,16) 'Calc Br Ratio     =  ', calc_br
         
      endif






 15   format( 3x,a,f9.5,a )
 16   format( 3x,a,f9.5 )

   

      return
      end


      subroutine tot_decay
c**********************************************************    
c     vegas evaluation of the width
c**********************************************************
      implicit none
      include 'decay.inc'
c
c     LOCAL
c
      INTEGER init,itmx,ncall,nprn,ndim,i
      REAL*4 region(2*mxdim),fxn,tgral,chi2a,sd
      real*8 diff,eff,ave_wei
      LOGICAL WRITEOUT
      DATA WRITEOUT /.TRUE./
c
c     EXTERNAL
c
      EXTERNAL FXN 
c
c-----
c     Begin 
c-----     
c
c
c     write out that calculation is going on
c
      write(*,*) 
      write(*,*) '----------------------------------------------'
      write(*,*) '      MC Evaluation of the Partial Width      '
      write(*,*) '----------------------------------------------'
c
c-MC
c
      ndim  =3*ND-4  !number of dimensions of the phase space
c
c-warm up
c
      init  =0
      ncall =100000
      itmx  =5
      nprn  =0
      do i=1,ndim
         region(i)=0E0
         region(ndim+i)=1E0
      enddo
      call vegas(region,ndim,fxn,init,ncall,itmx,nprn,tgral,sd,chi2a)
c
c-final run
c
      mxwgt =0d0 !reset the maximum weight to zero
      ncall =maxpoints
      itmx  =1   !only one large iteration
      nprn  =0
      init  =1
      call vegas(region,ndim,fxn,init,ncall,itmx,nprn,tgral,sd,chi2a)
c
c-result
c
      MC_width=tgral
      mxwgt=mxwgt*real(ncall)
      eff=MC_width/mxwgt*100e0
C
C     write results on the screen
C      
      write(*,*) ' '
      write(*,*) '  LO partial width = ' ,calc_width, ' GeV'
      write(*,*) '  MC partial width = ' ,MC_width,' +-',sd,' GeV'
      write(*,*) '  Max. wgt         = ' ,mxwgt
      write(*,1) '  Unwgt. Eff.(%)   = ' ,eff
 1    format(1x,a21,1x,f4.1)
c
c     write the result on a log file when WRITEOUT=.true.
c      
      IF(WRITEOUT) THEN
         open(lunout,file='all_decay.dat',status='unknown')
         call toend(lunout)
         diff=0d0
         if(calc_width.gt.0d0) then
            diff=(calc_width-MC_width)/(calc_width+MC_width)*100d0
         endif
         write(lunout,'(a3,2x,i2,2x,e12.5,2x,e10.5,
     &              1x,f5.2,3x,e10.5,3x,f5.1)') 
     &        id(ip),imode,calc_width,MC_width,diff,mxwgt,eff
         close(lunout)
      ENDIF   
      

      return
      end


      real*4 function fxn(xx,wgt)
c**********************************************************    
c     vegas evaluation of the width
c**********************************************************
      implicit none
      include 'decay.inc'
c
c     arguments
c
      real*4 xx(mxdim),wgt
c                  
c     Local
c
      integer IDS(MaxDecPart),ICOL(2,MaxDecPart)
      integer jhel(MaxDecPart)
      double precision pd(0:3,MaxDecPart),px(0:3),weight
      integer i
c
c     External
c
      real xran1
      integer iseed
      data iseed/1/   !this value is irrelevant
      integer get_hel
c      
c     Global
c
      include 'coupl.inc'
      include 'calc_values.inc'
c
c-----
c     Begin 
c-----
c--   momentum-random if you want to check Lorentz Invariance
      PD(0,1)=M1
      PD(1,1)=0d0
      PD(2,1)=0d0
      PD(3,1)=0d0
      PD(1,1)=xran1(iseed)*10d0
      PD(2,1)=xran1(iseed)*10d0
      PD(3,1)=xran1(iseed)*10d0 
      PD(0,1)=dsqrt(M1**2+PD(1,1)**2+PD(2,1)**2+PD(3,1)**2)

c--   helicity
      if(abs(ip).eq.24.or.ip.eq.23) then
         jhel(1) = get_hel(xran1(iseed),3)
      else
         jhel(1) = get_hel(xran1(iseed),2)
      endif
c--   pass information to the common block
      do i=1,mxdim
         x(i)=xx(i)
      enddo
C      
      CALL EVENT(PD,JHEL,IDS,ICOL,WEIGHT)
C      
c      write(*,*) 'from fxn: weight=',weight
      mxwgt   = max(weight*wgt,mxwgt) !max weight = wgt * fxn
      fxn     = REAL(weight)
C     
      return
      end


      subroutine toend(iunit)
c**********************************************************    
c read a file until the end
c**********************************************************
      ios = 0
      dowhile(ios.eq.0)
         read(unit=iunit,fmt='(1x)',iostat=ios)
      enddo
      end

      subroutine find_particle(ipp,nexternal,ic,n,idecay)
c***********************************************************    
c    looks for particle ip in the event ic and
c    returns the n instances found which
c    are not decayed yet. idecay(i) gives the position
c    of the i-th particle to be decayed.
c***********************************************************
      implicit none
      include 'decay.inc'
c
c     Arguments
c
      integer n,ipp,nexternal,ic(7,MaxParticles),idecay(MaxParticles)
c
c     Local
c
      integer i

c-----
c  Begin Code
c-----
      
      n=0
      do i=1,nexternal
         idecay(i)=0
         if (ic(1,i) .eq. ipp .and. ic(6,i) .eq. 1 ) then
            n=n+1
            idecay(n)=i
         endif
      enddo
      
      return
      end
      


      subroutine rnd_init(iunit,iseed)
c***********************************************************    
c initialize rnd number generator xran1 by reading iseed.dat
c***********************************************************
      implicit none
c
c     Arguments
c
      integer iunit,iseed
c
c     External
c
      real xran1
c
c     Local
c
      real*8 aa
      character*50 adummy
      logical done
      character*(132) buff
c-----
c  Begin Code
c-----

c
c     calculating decay rates only
c
c- try to open the iseed.dat file 
      open(iunit,file='./iseed.dat',status="old",err=200)
      read(iunit,err=200,end=200,fmt='(i10)') iseed
      write(*,*) '*****************************************************'
      write(*,*) '*      reading seed from iseed.dat                  *'
      write(*,'(a22,i6,a26)')   
     &           ' *  rnd number seed = ',iseed ,'               *'

      rewind(iunit)
      write(iunit,fmt='(i10)') iseed-1
      close(iunit)
      write(*,*) '*****************************************************'
      goto 201
c- if iseed.dat does not exist then set it to -1.
 200  iseed=-1
      open(iunit,file='./iseed.dat',status="new")
      write(iunit,fmt='(i10)') iseed-1
      write(*,*) '*****************************************************'
      write(*,*) '*      no iseed.dat  => iseed=-1                    *'
      write(*,*) '*      iseed.dat  now written                       *'
      write(*,*) '*****************************************************'
 201  continue
      close(iunit)
      return

      return
 99   write (*,*) 'error'

      return
      end



