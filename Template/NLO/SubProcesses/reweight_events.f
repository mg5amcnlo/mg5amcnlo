C***************************************************************
c Program that reads an Les Houches event file (with unweighted
c Born events) and calculates the virtual corrections for each
c event. The weights in the event file should sum to the total
c cross section.
c
c It must be run from a P* directory, because nexternal.inc,
c coupl.inc, fks.inc, pmass.inc  and run.inc should all be there.
c
c Call to virtual is via the BinothLHA() interface subroutine
c (in the corresponding file.)
c
C***************************************************************
c The virt_wgt coming from BinothLHA() should have the same
c adjusted identical particle symmetry factor as the Born in
c the MadFKS framework (i.e. divided by dble(ngluons)).
c This is also needed for normal inclusion of the virtual, so
c in general, nothing needs to be changed in BinothLHA().
C***************************************************************
c
      program reweight_events
      implicit none
      include "nexternal.inc"
      include "coupl.inc"
      include "run.inc"
      integer MaxParticles
      parameter (MaxParticles=20)
      integer lun,lun2,lun3
      parameter (lun=99,lun2=98,lun3=97)
      integer npart, ic(7,MaxParticles),i,j,k,l
      logical done,firsttime(2)
      data firsttime/.true.,.true./
      double precision P(0:4,MaxParticles),wgt,aqcd,aqed,xscale,virt_wgt
      integer ievent,nevent,unwgt,mxevt,unw_eventp,unw_eventn
      character*140 buff2,filename_in, filename_out,filename_unw
      double precision sumw,sumw_abs,mxwgt,p_born(0:3,nexternal-1)
      double precision born_wgt,pdf_wgt
      external dlum
      integer ID_save(MaxParticles)
      double complex wgt1(2)
      integer iseed
      data iseed/1/
      double precision rnd,ran2
      external ran2
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      character*1 proceed

      write (*,*) 'Give Les Houches event file with unweighted'//
     &     ' Born events'     
      read (*,'(a)') filename_in
      write (*,*) 'Unweight the reweighted events? (0: no, 1: yes)'
      read (*,*) unwgt
      if (unwgt.eq.0) then
         write (*,*) 'No unweighting'
      elseif (unwgt.eq.1) then
         write (*,*) 'Also unweighting the events'
      else
         write (*,*) 'Error: did not understand input',unwgt
      endif

      filename_out="rwgt_events.lhe"
      filename_unw="unwgt_events.lhe"

      open (unit=lun,file='nbodyonly.fks',status='old',err=96)
      read (lun,'(a)') proceed
      close(lun)
      if (proceed.eq.'N') then
         write (*,*) 'Can only reweight in a "nbodyonly" directory'
         stop
      endif

      call setpara('param_card.dat')   !Sets up couplings and masses
      call setrun

      open (unit=lun,file=filename_in,status='old',err=99)
      
      done = .false.
      sumw=0d0
      mxwgt=0d0
      nevent=0
      write (*,*) ''
      write (*,*) 'Parsing events to do some simple tests...'
      do while (.not.done)
         call read_event(lun,P,wgt,npart,ic,ievent,
     &        xscale,aqcd,aqed,buff2,done)
         if (done) cycle
         nevent=nevent+1
         if(mod(nevent,5000).eq.0) print *,'At event ',nevent
         if(npart.gt.20) then
            write (*,*) 'Error: Too many particles in event: ',npart
            stop
         endif
         sumw=sumw+wgt
         mxwgt = max(wgt,mxwgt)
         if (mxwgt.ne.wgt) then
            write (*,*) 'Error: not all events have equal weights'
            stop
         endif
         j=0
         do i=1,npart
            if (abs(ic(6,i)).eq.1) then
               j=j+1
               if (nevent.eq.1) then
                  id_save(j)=ic(1,i)
               else
                  if (ic(1,i).ne.id_save(j) .and. firsttime(1)) then
                     firsttime(1)=.false.
                     write (*,*) '* Warning: not all events'// 
     &                    'have the same final state partons'
                  endif
               endif
            endif
            if (ic(7,i).ne.0 .and. firsttime(2)) then
               firsttime(2)=.false.
               write (*,*) '* Warning: helicity info in Born events'//
     &              ' will be ignored'
            endif
         enddo
      enddo

      write (*,*) '...',nevent,' events parsed'
      write (*,*) ''
      write (*,*) 'Reweighting events...'

      rewind(lun)

      open (unit=lun2,file=filename_out,status='unknown',err=98)
      
      sumw=0d0
      sumw_abs=0d0
      mxwgt=0d0
      call fill_common_blocks()
      
      do i=1,nevent
         do j=1,140
            buff2(j:j)=' '
         enddo
         if (nevent.gt.50000) then
            if(mod(i,10000).eq.0) print *,'At event ',i
         elseif (nevent.gt.5000) then
            if(mod(i,1000).eq.0) print *,'At event ',i
         else
            if(mod(i,100).eq.0) print *,'At event ',i
         endif
c Read event
         call read_event(lun,P,wgt,npart,ic,ievent,
     &        xscale,aqcd,aqed,buff2,done)
         if (done) then
            write (*,*) 'ERROR, some events are missing '//
     &           'from event file',i,nevent
            stop
         endif
         j=0
         do l=1,npart
            if (abs(ic(6,l)).eq.1) then
               j=j+1
               do k=0,3
                  p_born(k,j)=p(k,l)
               enddo
            endif
         enddo
         calculatedBorn=.false.
c Set scale and couplings (on an event-by-event basis)
         call set_alphaS(p_born)
         if (abs((aqcd-g**2/(16d0*atan(1d0)))/aqcd).gt.1d-5) then
            write (*,*) "Error: inconsistency in strong coupling"
            write (*,*) aqcd,g**2/(16d0*atan(1d0)),scale
            stop
         endif
         call get_helicity_simplified()
c Compute Born
         call sborn(p_born,wgt1)
         born_wgt=dble(wgt1(1))
c Compute Virtual
         call BinothLHA(p_born,born_wgt,virt_wgt)
c Reweight the events
         wgt=wgt * virt_wgt/born_wgt
c Write event
         call write_event(lun2,P,wgt,npart,ic,ievent,
     &        xscale,aqcd,aqed,buff2)
         sumw=sumw+wgt
         sumw_abs=sumw_abs+abs(wgt)
         if (abs(wgt).gt.abs(mxwgt)) then
            mxwgt = wgt
            mxevt=i
         endif
      enddo


      write (*,*)
     &     'Reweighted events are written to file ',filename_out
      write (*,*) 'Cross section:',sumw
      write (*,*) 'average abs. of weights:',sumw_abs/dble(nevent)
      write (*,*) 'max wgt:',mxwgt,' for event',mxevt
      write (lun2,'(a)')'</LesHouchesEvents>'
c Do the unweighting
      if (unwgt.eq.1) then
         write (*,*) ''
         write (*,*) 'Unweighting...'
         rewind(lun2)
         open(unit=lun3,file=filename_unw,status='unknown',err=97)

         unw_eventp=0
         unw_eventn=0
         do i=1,nevent
            call read_event(lun2,P,wgt,npart,ic,ievent,
     &           xscale,aqcd,aqed,buff2,done)
            if (done) then
               write (*,*) 'ERROR, some events are missing '//
     &              'from event file',i,nevent
               stop
            endif
            rnd=ran2()
            if (rnd.lt.abs(wgt/mxwgt)) then
               if (wgt.gt.0) then
                  wgt=1d0
                  unw_eventp=unw_eventp+1
               else
                  wgt=-1d0
                  unw_eventn=unw_eventn+1
               endif
               call write_event(lun3,P,wgt,npart,ic,ievent,
     &              xscale,aqcd,aqed,buff2)
            endif
         enddo
         
         write (*,*)
     &        'Unweighted events are written to file ',filename_unw
         write (*,*) 'Number of events written is',unw_eventp+unw_eventn
         write (*,*) 'positive:',unw_eventp
         write (*,*) 'negative:',unw_eventn
         write (*,*) 'Unweighting efficiency (in %):',
     &        dble(unw_eventp+unw_eventn)/dble(nevent)*100d0
         write(lun3,'(a)')'</LesHouchesEvents>'
         close(lun3)
      endif


      close(lun)
      close(lun2)
      return
 99   write(*,*) 'Event file not found ',filename_in
      stop
 98   write(*,*) 'Cannot open event file ',filename_out
      stop
 97   write(*,*) 'Cannot open event file ',filename_unw
      stop
 96   write(*,*) 'File "nbodyonly.fks" not found. Cannot proceed.'
      stop
      end





      subroutine fill_common_blocks()
      implicit none
      include "nexternal.inc"
c      include "fks.inc"
      integer fks_j_from_i(nexternal,0:nexternal)
     &     ,particle_type(nexternal),pdg_type(nexternal)
      common /c_fks_inc/fks_j_from_i,particle_type,pdg_type
      integer i,config_fks
      double precision fkssymmetryfactor,fkssymmetryfactorBorn,
     &     fkssymmetryfactorDeg
      integer ngluons,nquarks(-6:6)
      common/numberofparticles/fkssymmetryfactor,fkssymmetryfactorBorn,
     &                         fkssymmetryfactorDeg,ngluons,nquarks
      integer           isum_hel
      logical                   multi_channel
      common/to_matrix/isum_hel, multi_channel
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i_type,j_type,m_type
      common/cparticle_types/i_type,j_type,m_type

      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn


      get_hel=0
      isum_hel=0

      open (unit=19,file="config.fks",status="old",err=99)
      read (19,*) config_fks
      close (19)
      i_fks=fks_i(config_fks)
      j_fks=fks_j(config_fks)

c Set color types of i_fks, j_fks and fks_mother.
      i_type=particle_type(i_fks)
      j_type=particle_type(j_fks)
      if (abs(i_type).eq.abs(j_type)) then
         m_type=8
         if ( (j_fks.le.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.i_type) .or.
     &        (j_fks.gt.nincoming .and.
     &        abs(i_type).eq.3 .and. j_type.ne.-i_type)) then
            write(*,*)'Flavour mismatch #1 in fill_common_blocks',
     &           i_fks,j_fks,i_type,j_type
            stop
         endif
      elseif(abs(i_type).eq.3 .and. j_type.eq.8)then
         if(j_fks.le.nincoming)then
            m_type=-i_type
         else
            write (*,*) 'Error in fill_common_blocks: (i,j)=(q,g)'
            stop
         endif
      elseif(i_type.eq.8 .and. abs(j_type).eq.3)then
         if (j_fks.le.nincoming) then
            m_type=j_type
         else
            m_type=j_type
         endif
      else
         write(*,*)'Flavour mismatch #2 in fill_common_blocks',
     &        i_type,j_type,m_type
         stop
      endif


      fkssymmetryfactor=1d0
      fkssymmetryfactorBorn=1d0
      fkssymmetryfactorDeg=1d0
      ngluons=0
      do i=nincoming+1,nexternal
         if (pdg_type(i).eq.21) ngluons=ngluons+1
      enddo

      call BinothLHAInit("contract.file")

      return
 99   write (*,*) 'Error: file config.fks not found'
      stop
      end


      subroutine get_helicity_simplified()
      implicit none
      include "nexternal.inc"
      include "born_nhel.inc"
      integer NHEL(nexternal,max_bhel*2),IHEL
chel  include "helicities.inc"
      integer nhelborn(nexternal-1,2),goodhelborn(2)
      integer nhelbornall(nexternal-1,max_bhel),hel_wgt_born
      common /c_nhelborn/ nhelborn,nhelbornall,goodhelborn,hel_wgt_born
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
      integer i,j,k
c Do not change these two lines, because ./bin/compile_madfks.sh might
c need to change them automatically
      logical HelSum
      parameter (HelSum=.true.)

      if (HelSum) return

c Even if HelSum is false, we still want to sum over all helicities
      do i=1,nexternal-1
         k=0
         do j=1,max_bhel*2
            if (nhel(i_fks,j).eq.-1) then
               k=k+1
               if (i.lt.i_fks) then
                  nhelbornall(i,k)=nhel(i,j)                  
               elseif(i.gt.i_fks) then
                  nhelbornall(i,k)=nhel(i+1,j)
               endif
            endif
         enddo
      enddo

      return
      end
