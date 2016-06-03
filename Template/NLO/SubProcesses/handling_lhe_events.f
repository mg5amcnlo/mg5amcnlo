c Utility routines for LHEF. Originally taken from collect_events.f
c
c Note: the routines read_lhef_event and write_lhef_event use the common
c blocks in reweight0.inc, relevant to reweight information. This is
c independent of the process, and in particular of process-related
c parameters such as nexternal, which is replaced here by (its supposed)
c upper bound maxparticles. The arrays which have one dimension defined
c by maxparticles may have a correspondence with process-specific ones,
c and the dimensions of the latter are typically defined by nexternal.
c Hence, one may need an explicit copy of one onto the other
c

      block data
      integer event_id
      common /c_event_id/ event_id
      data event_id /-99/
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      data nattr,npNLO,npLO /0,-1,-1/
      end

      subroutine write_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      include 'run.inc'
      include 'reweight0.inc'
      integer idwgt,kk,ii,jj,nn,n
      integer ifile,nevents
      character*10 MonteCarlo
      character*13 temp
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
c
      write(ifile,'(a)')
     #     '<LesHouchesEvents version="3.0">'
      write(ifile,'(a)')
     #     '  <!--'
      if (do_rwgt_scale .or. do_rwgt_pdf) then
         write(ifile,'(a)') '  <initrwgt>'
         idwgt=1000
         if (do_rwgt_scale) then
            do kk=1,dyn_scale(0)
               write(ifile,'(a,i4,a)') 
     &              "    <weightgroup name='scale_variation ",
     &              dyn_scale(kk),"' combine='envelope'>"
               if (lscalevar(kk)) then
                  do ii=1,nint(scalevarF(0))
                     do jj=1,nint(scalevarR(0))
                        idwgt=idwgt+1
                        write(ifile,'(a,i4,a,i4,a,e11.5,a,e11.5,a)')
     $                       "      <weight id='",idwgt,"'> dyn=",
     $                       dyn_scale(kk)," muR=",scalevarR(jj)," muF="
     $                       ,scalevarF(ii)," </weight>"
                     enddo
                  enddo
               else
                  idwgt=idwgt+1
                  write(ifile,'(a,i4,a,i4,a,e11.5,a,e11.5,a)')
     $                 "      <weight id='",idwgt,"'> dyn=",
     $                 dyn_scale(kk)," muR=",1d0 ," muF=",1d0
     $                 ," </weight>"
               endif
               write(ifile,'(a)') "    </weightgroup>"
            enddo
         endif
         if (do_rwgt_pdf) then
            do nn=1,lhaPDFid(0)
               if (lpdfvar(nn)) then
                  write(ifile,'(a)') "    <weightgroup "/
     &                 /"name='PDF_variation "/
     &                 /trim(adjustl(lhaPDFsetname(nn)))/
     &                 /"' combine='unknown'>"
                  do n=0,nmemPDF(nn)
                     idwgt=idwgt+1
                     write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)+n
                     write(ifile,'(a,i4,a)') "      <weight id='" ,idwgt
     $                    ,"'> "//trim(adjustl(temp))//' '
     $                    //trim(adjustl(lhaPDFsetname(nn)))/
     $                    /" </weight>"
                  enddo
               else
                  write(ifile,'(a)') "    <weightgroup "/
     &                 /"name='PDF_variation' combine='none'>"
                  idwgt=idwgt+1
                  write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)
                  write(ifile,'(a,i4,a)') "      <weight id='" ,idwgt
     $                 ,"'> "//trim(adjustl(temp))//' '
     $                 //trim(adjustl(lhaPDFsetname(nn)))//" </weight>"
               endif
               write(ifile,'(a)') "    </weightgroup>"
            enddo
         endif
         write(ifile,'(a)') '  </initrwgt>'
      endif
      write(ifile,'(a)')'  <scalesfunctionalform>'
      write(ifile,'(a)')muR_id_str(1:len_trim(muR_id_str))
      write(ifile,'(a)')muF1_id_str(1:len_trim(muF1_id_str))
      write(ifile,'(a)')muF2_id_str(1:len_trim(muF2_id_str))
      write(ifile,'(a)')QES_id_str(1:len_trim(QES_id_str))
      write(ifile,'(a)')'  </scalesfunctionalform>'
      write(ifile,'(a)')
     #     MonteCarlo
      write(ifile,'(a)')
     #     '  -->'
      write(ifile,'(a)')
     #     '  <header>'
      write(ifile,250) nevents
      write(ifile,'(a)')
     #     '  </header>'
 250  format(1x,i8)
      return
      end


      subroutine write_lhef_header_banner(ifile,nevents,MonteCarlo,path)
      implicit none 
      integer ifile, i, idwgt, nevents,iseed,ii,jj,kk,nn,n
      double precision mcmass(-16:21)
c     parameter to allow to include run_card.inc 
      include './run.inc'
      include './cuts.inc'
      integer lhaid
      character*20 pdlabel
      integer iappl
      character*7 event_norm
      character*13 temp
c     other parameter
      integer nevents_old
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      character*10 MonteCarlo
      character*100 path
      character*72 buffer,buffer_lc,buffer2
      integer event_id
      common /c_event_id/ event_id
      include 'reweight_all.inc'

c     Set the event_id to 0. If 0 or positive, this value will be update
c     in write_lhe_event. It is set to -99 through a block data
c     statement.
      event_id=0
c
      write(ifile,'(a)') '<LesHouchesEvents version="3.0">'
      write(ifile,'(a)') '  <header>'
      write(ifile,'(a)') '  <MG5ProcCard>'
      open (unit=92,file=path(1:index(path," ")-1)//'proc_card_mg5.dat'
     &     ,err=99)
      do
         read(92,'(a)',err=89,end=89) buffer
         write(ifile,'(a)') buffer
      enddo
 89   close(92)
      write(ifile,'(a)') '  </MG5ProcCard>'
      write(ifile,'(a)') '  <slha>'
      open (unit=92,file=path(1:index(path," ")-1)//'param_card.dat'
     &     ,err=98)
      do
         read(92,'(a)',err=88,end=88) buffer
         write(ifile,'(a)') buffer
      enddo
 88   close(92)
      write(ifile,'(a)') '  </slha>'
      write(ifile,'(a)') '  <MGRunCard>'
c     import the parameter from the run_card 
      nevents_old = nevents
      include './run_card.inc'
      nevents = nevents_old
c Replace the random number seed with the one used...
      open (unit=93,file="randinit",status="old",err=96)
      read(93,'(a)') buffer2
      if (index(buffer2,'=').eq.0) goto 96
      buffer2=buffer2(index(buffer2,'=')+1:)
      read(buffer2,*) iseed
      close(93)
      goto 95
 96      write (*,*) '"randinit" file not found in write_lhef_header_'/
     &     /'banner: not overwriting iseed in event file header.'
c Scale variation
 95   continue

c     copy the run_card as part of the banner.
      open (unit=92,file=path(1:index(path," ")-1)//'run_card.dat'
     &     ,err=97)
      do
         read(92,'(a)',err=87,end=87) buffer
         buffer_lc=buffer
         call case_trap3(72,buffer_lc)
c Replace the random number seed with the one used...
         if(index(buffer_lc,'iseed').ne.0 .and. buffer(1:1).ne.'#')then
            write(buffer,'(i11,a)')iseed,' =  iseed'
c Update the number of events
         elseif (index(buffer_lc,'nevents').ne.0 .and.
     &           buffer(1:1).ne.'#' .and.
     &           ( index(buffer_lc,'!').eq.0 .or.
     &             index(buffer_lc,'!').gt.index(buffer_lc,'nevents')
     &           )) then
            write(buffer,'(i11,a)')nevents,' = nevents'
         endif
         write(ifile,'(a)') buffer
      enddo
 87   close(92)
      write(ifile,'(a)') '  </MGRunCard>'
c Functional form of the scales
      write(ifile,'(a)') '  <scalesfunctionalform>'
      write(ifile,'(a)')muR_id_str(1:len_trim(muR_id_str))
      write(ifile,'(a)')muF1_id_str(1:len_trim(muF1_id_str))
      write(ifile,'(a)')muF2_id_str(1:len_trim(muF2_id_str))
      write(ifile,'(a)')QES_id_str(1:len_trim(QES_id_str))
      write(ifile,'(a)') '  </scalesfunctionalform>'
c MonteCarlo Masses
      write(ifile,'(a)') '  <MonteCarloMasses>'
      call fill_MC_mshell_wrap(MonteCarlo,mcmass)
      do i=1,5
         write (ifile,'(2x,i6,3x,e12.6)')i,mcmass(i)
      enddo
      write (ifile,'(2x,i6,3x,e12.6)')11,mcmass(11)
      write (ifile,'(2x,i6,3x,e12.6)')13,mcmass(13)
      write (ifile,'(2x,i6,3x,e12.6)')15,mcmass(15)
      write (ifile,'(2x,i6,3x,e12.6)')21,mcmass(21)
      write(ifile,'(a)') '  </MonteCarloMasses>'
c Write here the reweight information if need be
      if (do_rwgt_scale .or. do_rwgt_pdf) then
         write(ifile,'(a)') '  <initrwgt>'
         idwgt=1000
         if (do_rwgt_scale) then
            do kk=1,dyn_scale(0)
               write(ifile,'(a,i4,a)') 
     &              "    <weightgroup name='scale_variation ",
     &              dyn_scale(kk),"' combine='envelope'>"
               if (lscalevar(kk)) then
                  do ii=1,nint(scalevarF(0))
                     do jj=1,nint(scalevarR(0))
                        idwgt=idwgt+1
                        write(ifile,'(a,i4,a,i4,a,e11.5,a,e11.5,a)')
     $                       "      <weight id='",idwgt,"'> dyn=",
     $                       dyn_scale(kk)," muR=",scalevarR(jj)," muF="
     $                       ,scalevarF(ii)," </weight>"
                     enddo
                  enddo
               else
                  idwgt=idwgt+1
                  write(ifile,'(a,i4,a,i4,a,e11.5,a,e11.5,a)')
     $                 "      <weight id='",idwgt,"'> dyn=",
     $                 dyn_scale(kk)," muR=",1d0 ," muF=",1d0
     $                 ," </weight>"
               endif
               write(ifile,'(a)') "    </weightgroup>"
            enddo
         endif
         if (do_rwgt_pdf) then
            do nn=1,lhaPDFid(0)
               if (lpdfvar(nn)) then
                  write(ifile,'(a)') "    <weightgroup "/
     &                 /"name='PDF_variation "/
     &                 /trim(adjustl(lhaPDFsetname(nn)))/
     &                 /"' combine='unknown'>"
                  do n=0,nmemPDF(nn)
                     idwgt=idwgt+1
                     write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)+n
                     write(ifile,'(a,i4,a)') "      <weight id='" ,idwgt
     $                    ,"'> "//trim(adjustl(temp))//' '
     $                    //trim(adjustl(lhaPDFsetname(nn)))/
     $                    /" </weight>"
                  enddo
               else
                  write(ifile,'(a)') "    <weightgroup "/
     &                 /"name='PDF_variation' combine='none'>"
                  idwgt=idwgt+1
                  write(temp,'(a4,i8)') "PDF=",lhaPDFid(nn)
                  write(ifile,'(a,i4,a)') "      <weight id='" ,idwgt
     $                 ,"'> "//trim(adjustl(temp))//' '
     $                 //trim(adjustl(lhaPDFsetname(nn)))//" </weight>"
               endif
               write(ifile,'(a)') "    </weightgroup>"
            enddo
         endif
         write(ifile,'(a)') '  </initrwgt>'
      endif
      write(ifile,'(a)') '  </header>'
 250  format(1x,i8)
      return
 99   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' proc_card_mg5.dat not found   :',path(1:index(path," ")-1)
     &     //'proc_card_mg5.dat'
      stop
 98   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' param_card.dat not found   :',path(1:index(path," ")-1)
     &     //'param_card.dat'
      stop
 97   write (*,*) 'ERROR in write_lhef_header_banner: '/
     &     /' run_card.dat not found   :',path(1:index(path," ")-1)
     &     //'run_card.dat'
      stop
      end


      subroutine read_lhef_header(ifile,nevents,MonteCarlo)
      implicit none 
      include 'reweight0.inc'
      include './run.inc'
      logical already_found
      integer ifile,nevents,i,ii,ii2,iistr,itemp
      double precision temp
      character*10 MonteCarlo
      character*80 string,string0
      character*3 event_norm
      common/cevtnorm/event_norm
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      nevents = -1
      MonteCarlo = ''
c
      string='  '
      do while(string.ne.'  -->')
        string0=string
        if (index(string,'</header>').ne.0) return
        read(ifile,'(a)')string
        if(index(string,'= nevents').ne.0) read(string,*)nevents,string0
        if(index(string,'parton_shower').ne.0)then
           ii=iistr(string)
           ii2=min(index(string,'=')-1,ii+9)
           MonteCarlo=string(ii:ii2)
           call case_trap4(ii2-ii+1,MonteCarlo)
        endif
        if(index(string,'event_norm').ne.0)then
           ii=iistr(string)
           event_norm=string(ii:ii+2)
        endif
        if(index(string,'<scalesfunctionalform>').ne.0) then
           read(ifile,'(a)') muR_id_str
           read(ifile,'(a)') muF1_id_str
           read(ifile,'(a)') muF2_id_str
           read(ifile,'(a)') QES_id_str
        endif
c Find the reweight information for scale and PDF uncertainties
        if (index(string,'<initrwgt>').ne.0) then
           dyn_scale(0)=0
           lhaPDFid(0)=0
           lscalevar(1)=.false.
           lpdfvar(1)=.false.
           do_rwgt_scale=.false.
           do_rwgt_pdf=.false.
           do
c     find the start of a weightgroup
              do
                 read(ifile,'(a)')string
                 if (index(string,'<weightgroup').ne.0) exit
                 if (index(string,'</initrwgt').ne.0) exit
              enddo
              if (index(string,"name='scale_variation").ne.0) then
                 do_rwgt_scale=.true.
                 dyn_scale(0)=dyn_scale(0)+1
                 read(ifile,'(a)')string
                 read(string(index(string,'dyn=')+4:),*)
     $                dyn_scale(dyn_scale(0))
                 read(string(index(string,'muR=')+4:),*) scalevarR(1)
                 read(string(index(string,'muF=')+4:),*) scalevarF(1)
                 scalevarR(0)=1d0
                 scalevarF(0)=1d0
                 do
                    read(ifile,'(a)')string
                    if (index(string,'</weightgroup>').ne.0) exit
                    read(string(index(string,'muR=')+4:),*) temp
                    already_found=.false.
                    do i=1,nint(scalevarR(0))
                       if (temp.eq.scalevarR(i)) already_found=.true.
                    enddo
                    if (.not.already_found) then
                       scalevarR(0)=scalevarR(0)+1d0
                       scalevarR(nint(scalevarR(0)))=temp
                    endif
                    read(string(index(string,'muF=')+4:),*) temp
                    already_found=.false.
                    do i=1,nint(scalevarF(0))
                       if (temp.eq.scalevarF(i)) already_found=.true.
                    enddo
                    if (.not.already_found) then
                       scalevarF(0)=scalevarF(0)+1d0
                       scalevarF(nint(scalevarF(0)))=temp
                    endif
                 enddo
                 if (scalevarR(0).gt.1d0 .or. scalevarF(0).gt.1d0) then
                    lscalevar(dyn_scale(0))=.true.
                 else
                    lscalevar(dyn_scale(0))=.false.
                 endif
              elseif (index(string,"name='PDF_variation").ne.0) then
                 do_rwgt_pdf=.true.
                 lhaPDFid(0)=lhaPDFid(0)+1
                 nmemPDF(lhaPDFid(0))=-1
                 do 
                    read(ifile,'(a)')string
                    if (index(string,'</weightgroup>').ne.0) exit
                    nmemPDF(lhaPDFid(0))=nmemPDF(lhaPDFid(0))+1
                    if (nmemPDF(lhaPDFid(0)).eq.0) then
                       read(string(index(string,'PDF=')+4:),*)
     $                      lhaPDFid(lhaPDFid(0))
     $                      ,lhaPDFsetname(lhaPDFid(0))
                       lhaPDFsetname(lhaPDFid(0))
     $                      =trim(adjustl(lhaPDFsetname(lhaPDFid(0))))
                    endif
                 enddo
                 if (nmemPDF(lhaPDFid(0)).gt.0) then
                    lpdfvar(lhaPDFid(0))=.true.
                 else
                    lpdfvar(lhaPDFid(0))=.false.
                 endif
              elseif (index(string,'</initrwgt').ne.0) then
                 exit
              endif
           enddo
        endif
      enddo

c Works only if the name of the MC is the last line of the comments
      MonteCarlo=string0(1:10)
      call case_trap4(10,MonteCarlo)
c Here we are at the end of (user-defined) comments. Now go to end
c of headers
      do while(index(string,'</header>').eq.0)
        string0=string
        read(ifile,'(a)')string
      enddo
c if the file is a partial file the header is non-standard   
      if (MonteCarlo .ne. '')read(string0,250) nevents
 250  format(1x,i8)
      return
      end


c Same as read_lhef_header, except that more parameters are read.
c Avoid overloading read_lhef_header, meant to be used in utilities
      subroutine read_lhef_header_full(ifile,nevents,itempsc,itempPDF,
     #                                 MonteCarlo)
      implicit none
      include 'reweight0.inc'
      include 'run.inc'
      logical already_found
      integer ifile,nevents,i,ii,ii2,iistr,ipart,itempsc,itempPDF
      character*10 MonteCarlo
      character*80 string,string0
      character*3 event_norm
      common/cevtnorm/event_norm
      double precision temp,remcmass(-16:21)
      common/cremcmass/remcmass
c Scales
      character*80 muR_id_str,muF1_id_str,muF2_id_str,QES_id_str
      common/cscales_id_string/muR_id_str,muF1_id_str,
     #                         muF2_id_str,QES_id_str
      ipart=-1000000
      nevents = -1
      MonteCarlo = ''
      itempsc=0
      itempPDF=0
c
      string='  '
      do while(string.ne.'  -->')
        string0=string
        if (index(string,'</header>').ne.0) return
        read(ifile,'(a)')string
        if(index(string,'= nevents').ne.0)
     #    read(string,*)nevents,string0
        if(index(string,'parton_shower').ne.0)then
           ii=iistr(string)
           ii2=min(index(string,'=')-1,ii+9)
           MonteCarlo=string(ii:ii2)
           call case_trap4(ii2-ii+1,MonteCarlo)
        endif
        if(index(string,'event_norm').ne.0)then
           ii=iistr(string)
           event_norm=string(ii:ii+2)
        endif
        if( index(string,'<montecarlomasses>').ne.0 .or.
     #      index(string,'<MonteCarloMasses>').ne.0 )then
          read(ifile,'(a)')string
          do while( index(string,'</montecarlomasses>').eq.0 .and.
     #             index(string,'</MonteCarloMasses>').eq.0 )
            read(string,*)ipart,temp
            if(ipart.lt.-16.or.ipart.gt.21)then
              write(*,*)'Error in read_lhef_header:'
              write(*,*)' incomprehensible list of parton masses',ipart
              stop
            endif
            remcmass(ipart)=temp
            read(ifile,'(a)')string
          enddo
        endif
c Find the reweight information for scale and PDF uncertainties
        if (index(string,'<initrwgt>').ne.0) then
           dyn_scale(0)=0
           lhaPDFid(0)=0
           lscalevar(1)=.false.
           lpdfvar(1)=.false.
           do_rwgt_scale=.false.
           do_rwgt_pdf=.false.
           do
c     find the start of a weightgroup
              do
                 read(ifile,'(a)')string
                 if (index(string,'<weightgroup').ne.0) exit
                 if (index(string,'</initrwgt').ne.0) exit
              enddo
              if (index(string,"name='scale_variation").ne.0) then
                 do_rwgt_scale=.true.
                 dyn_scale(0)=dyn_scale(0)+1
                 read(ifile,'(a)')string
                 read(string(index(string,'dyn=')+4:),*)
     $                dyn_scale(dyn_scale(0))
                 read(string(index(string,'muR=')+4:),*) scalevarR(1)
                 read(string(index(string,'muF=')+4:),*) scalevarF(1)
                 scalevarR(0)=1d0
                 scalevarF(0)=1d0
                 do
                    read(ifile,'(a)')string
                    if (index(string,'</weightgroup>').ne.0) exit
                    read(string(index(string,'muR=')+4:),*) temp
                    already_found=.false.
                    do i=1,nint(scalevarR(0))
                       if (temp.eq.scalevarR(i)) already_found=.true.
                    enddo
                    if (.not.already_found) then
                       scalevarR(0)=scalevarR(0)+1d0
                       scalevarR(nint(scalevarR(0)))=temp
                    endif
                    read(string(index(string,'muF=')+4:),*) temp
                    already_found=.false.
                    do i=1,nint(scalevarF(0))
                       if (temp.eq.scalevarF(i)) already_found=.true.
                    enddo
                    if (.not.already_found) then
                       scalevarF(0)=scalevarF(0)+1d0
                       scalevarF(nint(scalevarF(0)))=temp
                    endif
                 enddo
                 if (scalevarR(0).gt.1d0 .or. scalevarF(0).gt.1d0) then
                    lscalevar(dyn_scale(0))=.true.
                 else
                    lscalevar(dyn_scale(0))=.false.
                 endif
              elseif (index(string,"name='PDF_variation").ne.0) then
                 do_rwgt_pdf=.true.
                 lhaPDFid(0)=lhaPDFid(0)+1
                 nmemPDF(lhaPDFid(0))=-1
                 do 
                    read(ifile,'(a)')string
                    if (index(string,'</weightgroup>').ne.0) exit
                    nmemPDF(lhaPDFid(0))=nmemPDF(lhaPDFid(0))+1
                    if (nmemPDF(lhaPDFid(0)).eq.0) then
                       read(string(index(string,'PDF=')+4:),*)
     $                      lhaPDFid(lhaPDFid(0))
     $                      ,lhaPDFsetname(lhaPDFid(0))
                       lhaPDFsetname(lhaPDFid(0))
     $                      =trim(adjustl(lhaPDFsetname(lhaPDFid(0))))
                    endif
                 enddo
                 if (nmemPDF(lhaPDFid(0)).gt.0) then
                    lpdfvar(lhaPDFid(0))=.true.
                 else
                    lpdfvar(lhaPDFid(0))=.false.
                 endif
              elseif (index(string,'</initrwgt').ne.0) then
                 exit
              endif
           enddo
        endif
        if(index(string,'<scalesfunctionalform>').ne.0) then
           read(ifile,'(a)') muR_id_str
           read(ifile,'(a)') muF1_id_str
           read(ifile,'(a)') muF2_id_str
           read(ifile,'(a)') QES_id_str
        endif
      enddo
c Works only if the name of the MC is the last line of the comments
      MonteCarlo=string0(1:10)
      call case_trap4(10,MonteCarlo)
c Here we are at the end of (user-defined) comments. Now go to end
c of headers
      do while(index(string,'</header>').eq.0)
        string0=string
        read(ifile,'(a)')string
      enddo
c if the file is a partial file the header is non-standard   
      if (MonteCarlo .ne. '') read(string0,250) nevents
 250  format(1x,i8)
      return
      end


      subroutine write_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,i,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      double precision XSECUP2(100),XERRUP2(100),XMAXUP2(100)
      integer LPRUP2(100)
      common /lhef_init/XSECUP2,XERRUP2,XMAXUP2,LPRUP2
c
      write(ifile,'(a)')
     # '  <init>'
      write(ifile,501)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      write(ifile,502)XSECUP,XERRUP,XMAXUP,LPRUP
      if (NPRUP.gt.1) then
         do i=2,NPRUP
            write(ifile,502)XSECUP2(i),XERRUP2(i),XMAXUP2(i),LPRUP2(i)
         enddo
      endif
      write(ifile,'(a)')
     # '  </init>'
 501  format(2(1x,i6),2(1x,e14.8),2(1x,i2),2(1x,i6),1x,i2,1x,i3)
 502  format(3(1x,e14.8),1x,i6)
c
      return
      end


      subroutine read_lhef_init(ifile,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
      implicit none
      integer ifile,i,IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      double precision XSECUP2(100),XERRUP2(100),XMAXUP2(100)
      integer LPRUP2(100)
      common /lhef_init/XSECUP2,XERRUP2,XMAXUP2,LPRUP2
      character*80 string
c
      read(ifile,'(a)')string
      read(ifile,*)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #                PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #                IDWTUP,NPRUP
      read(ifile,*)XSECUP,XERRUP,XMAXUP,LPRUP
      XSECUP2(1)=XSECUP
      XERRUP2(1)=XERRUP
      XMAXUP2(1)=XMAXUP
      LPRUP2(1)=LPRUP
      if (NPRUP.gt.1) then
         do i=2,NPRUP
            read(ifile,*)XSECUP2(i),XERRUP2(i),XMAXUP2(i),LPRUP2(i)
         enddo
      endif
      read(ifile,'(a)')string
c
      return
      end

      subroutine write_lhef_event(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      character*140 buff
      integer ifile,i,kk
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      integer event_id
      common /c_event_id/ event_id
      integer i_process
      common/c_i_process/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include './run.inc'
      include 'unlops.inc'
c     if event_id is zero or positive (that means that there was a call
c     to write_lhef_header_banner) update it and write it
c RF: don't use the event_id:
      event_id = -99
c
      if (event_id.ge.0) then
         event_id=event_id+1
         if (event_id.le.9) then
            write(ifile,'(a,i1,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99) then
            write(ifile,'(a,i2,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999) then
            write(ifile,'(a,i3,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.9999) then
            write(ifile,'(a,i4,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99999) then
            write(ifile,'(a,i5,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999999) then
            write(ifile,'(a,i6,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.9999999) then
            write(ifile,'(a,i7,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.99999999) then
            write(ifile,'(a,i8,a)') "  <event id='",event_id,"'>"
         elseif(event_id.le.999999999) then
            write(ifile,'(a,i9,a)') "  <event id='",event_id,"'>"
         else
            write (ifile,*) "ERROR: EVENT ID TOO LARGE",event_id
            write (*,*) "ERROR: EVENT ID TOO LARGE",event_id
            stop
         endif
      elseif(nattr.eq.2) then
         if ( (npLO.ge.10.or.npLO.lt.0) .and.
     &        (npNLO.ge.10.or.npNLO.lt.0)) then
            write(ifile,'(a,i2,a,i2,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.lt.10.or.npLO.ge.0) .and.
     &        (npNLO.ge.10.or.npNLO.lt.0)) then
            write(ifile,'(a,i1,a,i2,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.ge.10.or.npLO.lt.0) .and.
     &        (npNLO.lt.10.or.npNLO.ge.0)) then
            write(ifile,'(a,i2,a,i1,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         elseif( (npLO.lt.10.or.npLO.ge.0) .and.
     &        (npNLO.lt.10.or.npNLO.ge.0)) then
            write(ifile,'(a,i1,a,i1,a)') "  <event npLO=' ",npLO
     $           ," ' npNLO=' ",npNLO," '>"
         endif
      else
         write(ifile,'(a)') '  <event>'
      endif
      write(ifile,503)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        write(ifile,504)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      if(buff(1:1).eq.'#' .and. (do_rwgt_scale .or. do_rwgt_pdf .or.
     &     jwgtinfo.lt.0)) then
         write(ifile,'(a)') buff(1:len_trim(buff))
         read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
         if(jwgtinfo.eq.-5.or.jwgtinfo.eq.-9) then
            write(ifile,'(a)')'  <mgrwgt>'
            write (ifile,'(1x,d16.10,3(1x,i4))') wgtref,n_ctr_found
     &           ,n_mom_conf, nint(wgtcpower)
            do i=1,n_mom_conf
               do j=1,mexternal
                  write (ifile,'(4(1x,d21.15))')
     &                 (momenta_str(ii,j,i),ii=0,3)
               enddo
            enddo
            do i=1,n_ctr_found
               write (ifile,'(a)') trim(adjustl(n_ctr_str(i)))
            enddo
            write(ifile,'(a)')'  </mgrwgt>'
         endif
         if(jwgtinfo.eq.15) then
            write(ifile,'(a)')'  <unlops>'
            write(ifile,*)NUP_H
            do i=1,NUP_H
               write(ifile,504)IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $              ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $              ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $              VTIMUP_H(I),SPINUP_H(I)
            enddo
            write(ifile,'(a)')'  </unlops>'
         endif
         if(abs(jwgtinfo).eq.9)then
            if (do_rwgt_scale .or. do_rwgt_pdf) then
               write(ifile,'(a)') '  <rwgt>'
               idwgt=1000
               if (do_rwgt_scale) then
                  do kk=1,dyn_scale(0)
                     if (lscalevar(kk)) then
                        do i=1,nint(scalevarF(0))
                           do j=1,nint(scalevarR(0))
                              idwgt=idwgt+1
                              write(ifile,601) "   <wgt id='",idwgt,"'>"
     $                             ,wgtxsecmu(j,i,kk)," </wgt>"
                           enddo
                        enddo
                     else
                        idwgt=idwgt+1
                        write(ifile,601) "   <wgt id='",idwgt,"'>"
     $                       ,wgtxsecmu(1,1,kk)," </wgt>"
                     endif
                  enddo
               endif
               if (do_rwgt_pdf) then
                  do j=1,lhaPDFid(0)
                     if (lpdfvar(j)) then
                        do i=0,nmemPDF(j)
                           idwgt=idwgt+1
                           write(ifile,601) "   <wgt id='",idwgt,"'>"
     $                          ,wgtxsecPDF(i,j)," </wgt>"
                        enddo
                     else
                        idwgt=idwgt+1
                        write(ifile,601) "   <wgt id='",idwgt,"'>"
     $                       ,wgtxsecPDF(0,j)," </wgt>"
                     endif
                  enddo
               endif
               write(ifile,'(a)') '  </rwgt>'
            endif
         endif
      endif
      write(ifile,'(a)') '  </event>'
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
 601  format(a12,i4,a2,1x,e11.5,a7)
c
      return
      end


      subroutine read_lhef_event(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      integer ifile,i,kk
      character*140 buff
      character*80 string
      character*12 dummy12
      character*2 dummy2
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      integer i_process
      common/c_i_process/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include 'unlops.inc'
      include 'run.inc'
c
      read(ifile,'(a)')string
      nattr=0
      npNLO=-1
      npLO=-1
      if (index(string,'npLO').ne.0) then
         nattr=2
         read(string(index(string,'npLO')+6:),*) npLO
      endif
      if (index(string,'npNLO').ne.0) then
         nattr=2
         read(string(index(string,'npNLO')+7:),*) npNLO
      endif
      read(ifile,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        read(ifile,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      read(ifile,'(a)')buff
      if(buff(1:1).eq.'#')then
         read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        
         if(jwgtinfo.eq.-5 .or. jwgtinfo.eq.-9) then
            read(ifile,'(a)')string
            read(ifile,*) wgtref,n_ctr_found,n_mom_conf,wgtcpower
            do i=1,n_mom_conf
               do j=1,mexternal
                  read (ifile,*) (momenta_str(ii,j,i),ii=0,3)
               enddo
            enddo
            do i=1,n_ctr_found
               read (ifile,'(a)') n_ctr_str(i)
            enddo
            read(ifile,'(a)')string
         endif
         if(jwgtinfo.eq.15) then
            read(ifile,'(a)') string
            read(ifile,*)NUP_H
            do i=1,NUP_H
               read(ifile,*) IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $              ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $              ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $              VTIMUP_H(I),SPINUP_H(I)
            enddo
            read(ifile,'(a)') string
         endif
         if(abs(jwgtinfo).eq.9)then
            if (do_rwgt_scale .or. do_rwgt_pdf) then
               read(ifile,'(a)')string
               wgtref=XWGTUP
               if (do_rwgt_scale) then
                  do kk=1,dyn_scale(0)
                     if (lscalevar(kk)) then
                        do i=1,nint(scalevarF(0))
                           do j=1,nint(scalevarR(0))
                              call read_rwgt_line(ifile,idwgt
     $                             ,wgtxsecmu(j,i,kk))
                           enddo
                        enddo
                     else
                        call read_rwgt_line(ifile,idwgt,wgtxsecmu(1,1
     $                       ,kk))
                     endif
                  enddo
               endif
               if (do_rwgt_pdf) then
                  do j=1,lhaPDFid(0)
                     if (lpdfvar(j)) then
                        do i=0,nmemPDF(j)
                           call read_rwgt_line(ifile,idwgt,wgtxsecPDF(i
     $                          ,j))
                        enddo
                     else
                        call read_rwgt_line(ifile,idwgt,wgtxsecPDF(0,j))
                     endif
                  enddo
               endif
               read(ifile,'(a)')string
            endif
         endif
         read(ifile,'(a)')string
      else
         string=buff(1:len_trim(buff))
         buff=' '
      endif
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
c
      return
      end


c Same as read_lhef_event, except for the end-of-file catch
      subroutine read_lhef_event_catch(ifile,
     # NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      implicit none
      INTEGER NUP,IDPRUP,IDUP(*),ISTUP(*),MOTHUP(2,*),ICOLUP(2,*)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,*),VTIMUP(*),SPINUP(*)
      integer ifile,i,kk
      character*140 buff
      character*80 string
      character*12 dummy12
      character*2 dummy2
      character*9 ch1
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      integer ii,j,nps,nng,iFKS,idwgt
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      integer i_process
      common/c_i_process/i_process
      integer nattr,npNLO,npLO
      common/event_attributes/nattr,npNLO,npLO
      include 'reweight_all.inc'
      include 'unlops.inc'
      include 'run.inc'
c
      read(ifile,'(a)')string
      if(index(string,'<event').eq.0)then
        if(index(string,'</LesHouchesEvents>').ne.0)then
          buff='endoffile'
          return
        else
          write(*,*)'Unknown structure in read_lhef_event_catch:'
          write(*,*)string(1:len_trim(string))
          stop
        endif
      endif
      nattr=0
      npNLO=-1
      npLO=-1
      if (index(string,'npLO').ne.0) then
         nattr=2
         read(string(index(string,'npLO')+6:),*) npLO
      endif
      if (index(string,'npNLO').ne.0) then
         nattr=2
         read(string(index(string,'npNLO')+7:),*) npNLO
      endif
      read(ifile,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
      do i=1,nup
        read(ifile,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                  ICOLUP(1,I),ICOLUP(2,I),
     #                  PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                  VTIMUP(I),SPINUP(I)
      enddo
      read(ifile,'(a)')buff
      if(buff(1:1).eq.'#')then
        read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.eq.-5 .or. jwgtinfo.eq.-9) then
           read(ifile,'(a)')string
           read(ifile,*) wgtref,n_ctr_found,n_mom_conf,wgtcpower
           do i=1,n_mom_conf
              do j=1,mexternal
                 read (ifile,*) (momenta_str(ii,j,i),ii=0,3)
              enddo
           enddo
           do i=1,n_ctr_found
              read (ifile,'(a)') n_ctr_str(i)
           enddo
           read(ifile,'(a)')string
         endif
         if(jwgtinfo.eq.15) then
            read(ifile,'(a)') string
            read(ifile,*)NUP_H
            do i=1,NUP_H
               read(ifile,*) IDUP_H(I),ISTUP_H(I),MOTHUP_H(1,I)
     $              ,MOTHUP_H(2,I),ICOLUP_H(1,I),ICOLUP_H(2,I),PUP_H(1
     $              ,I),PUP_H(2,I),PUP_H(3,I),PUP_H(4,I),PUP_H(5,I),
     $              VTIMUP_H(I),SPINUP_H(I)
            enddo
            read(ifile,'(a)') string
         endif
         if(abs(jwgtinfo).eq.9)then
            if (do_rwgt_scale .or. do_rwgt_pdf) then
               read(ifile,'(a)')string
               wgtref=XWGTUP
               if (do_rwgt_scale) then
                  do kk=1,dyn_scale(0)
                     if (lscalevar(kk)) then
                        do i=1,nint(scalevarF(0))
                           do j=1,nint(scalevarR(0))
                              call read_rwgt_line(ifile,idwgt
     $                             ,wgtxsecmu(j,i,kk))
                           enddo
                        enddo
                     else
                        call read_rwgt_line(ifile,idwgt,wgtxsecmu(1,1
     $                       ,kk))
                     endif
                  enddo
               endif
               if (do_rwgt_pdf) then
                  do j=1,lhaPDFid(0)
                     if (lpdfvar(j)) then
                        do i=0,nmemPDF(j)
                           call read_rwgt_line(ifile,idwgt,wgtxsecPDF(i
     $                          ,j))
                        enddo
                     else
                        call read_rwgt_line(ifile,idwgt,wgtxsecPDF(0,j))
                     endif
                  enddo
               endif
               read(ifile,'(a)')string
            endif
         endif
         read(ifile,'(a)')string
      else
         string=buff(1:len_trim(buff))
         buff=' '
      endif
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
 503  format(1x,i2,1x,i6,4(1x,e14.8))
 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,e14.8),2(1x,e10.4))
c
      return
      end



      subroutine copy_header(infile,outfile,nevts)
      implicit none
      character*200 buff2
      integer nevts,infile,outfile
c
      buff2=' '
      do while(.true.)
         read(infile,'(a)')buff2
         if(index(buff2,'= nevents').eq.0)
     &        write(outfile,'(a)') trim(buff2)
         if(index(buff2,'= nevents').ne.0) exit
      enddo
      write(outfile,*)
     &     nevts,' = nevents    ! Number of unweighted events requested'
      do while(index(buff2,'</header>').eq.0)
         read(infile,'(a)')buff2
         write(outfile,'(a)')trim(buff2)
      enddo
c
      return
      end


      subroutine fill_MC_mshell_wrap(MC,masses)
      double precision mcmass(-16:21),masses(-16:21)
      common/cmcmass/mcmass
      character*10 MonteCarlo,MC
      common/cMonteCarloType/MonteCarlo
      MonteCarlo=MC
      call case_trap4(10,MonteCarlo)
      call fill_MC_mshell()
      do i=-16,21
         masses(i)=mcmass(i)
      enddo
      return
      end


      function iistr(string)
c returns the position of the first non-blank character in string
      implicit none
      logical is_i
      character*(*) string
      integer i,iistr
c
      is_i=.false.
      iistr=0
      do i=1,len(string)
         if(string(i:i).ne.' '.and..not.is_i)then
            is_i=.true.
            iistr=i
         endif
      enddo

      return
      end


      subroutine case_trap3(ilength,name)
c**********************************************************    
c change the string to lowercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k,ilength

      do i=1,ilength
         k=ichar(name(i:i))
         if(k.ge.65.and.k.le.90) then  !upper case A-Z
            k=ichar(name(i:i))+32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine case_trap4(ilength,name)
c**********************************************************    
c change the string to uppercase if the input is not
c**********************************************************
      implicit none
c
c     ARGUMENT
c      
      character*(*) name
c
c     LOCAL
c
      integer i,k,ilength

      do i=1,ilength
         k=ichar(name(i:i))
         if(k.ge.97.and.k.le.122) then  !lower case A-Z
            k=ichar(name(i:i))-32   
            name(i:i)=char(k)        
         endif
      enddo

      return
      end


      subroutine read_rwgt_line(unit,id,wgt)
c read a line in the <rwgt> tag. The syntax should be
c  <wgt id='1001'> 0.1234567e+01 </wgt>
c The id should be exactly 4 digits long.
      implicit none
      integer unit,id,wgt_start,id_start
      double precision wgt
      character*100 buff
      read (unit,'(a)') buff
c Use char() to make sure that the non-standard characters are compiler
c independent (char(62)=">", char(61)="=", char(39)="'")
      wgt_start=index(buff,CHAR(39)//CHAR(62))+2
      id_start=index(buff,'id'//CHAR(61)//CHAR(39))+4
      read (buff(id_start:100),'(i4)') id
      read (buff(wgt_start:100),*) wgt
      return
      end

