      program collect_events
      implicit none
      character*120 string120,eventfile
      character*19 basicfile,nextbasicfile
      character*15 outputfile
      integer istep,i,numoffiles,nbunches,nevents,ievents,junit(80)
      double precision xtotal,absxsec,evwgt,xsecfrac
      integer i_orig
      common /c_i_orig/i_orig
      integer ioutput
      parameter(ioutput=99)
      integer nevents_file(80),proc_id(80)
      double precision xsecfrac_all(80)
      common /to_xsecfrac/xsecfrac_all
      common /to_nevents_file/nevents_file,proc_id
      integer proc_id_tot(0:100)
      double precision xsec(100),xsecABS,xerr(100)
      logical get_xsec_from_res1
      common/total_xsec/xsec,xerr,xsecABS,proc_id_tot,get_xsec_from_res1

      write (*,*) "Overwrite the event weights?"
      write (*,*) "give '0' to keep original weights;"
      write (*,*) "give '1' to overwrite the weights"/
     $     /" to sum to the Xsec;"
      write (*,*) "give '2' to overwrite the weights"/
     $     /" to average to the Xsec (=default)"
      write (*,*) "give '3' to overwrite the weights"/
     $     /" to either +/- 1."
      read (*,*) i_orig
      if (i_orig.ne.0 .and. i_orig.ne.1 .and. i_orig.ne.2 .and.
     $     i_orig.ne.3) stop
      write(*,*) i_orig


      istep=0
 1    continue
      write (*,*) 'step #',istep
      outputfile='allevents_X_000'
      if(istep.eq.0) then
         basicfile='nevents_unweighted'
         outputfile='allevents_0_000'
      else
         basicfile='nevents_unweighted0'
         if(istep.gt.8) then
            write (*,*) 'Error, istep too large',istep
            stop
         endif
         write(basicfile(19:19),'(i1)')istep
         write(outputfile(11:11),'(i1)')istep
      endif
      nextbasicfile='nevents_unweighted0'
      write(nextbasicfile(19:19),'(i1)')istep+1
      open(unit=10,file=basicfile,status='old')
      open(unit=98,file=nextbasicfile,status='unknown')

c
c First get the cross section from the res_1 files
c
      if (istep.eq.0) then
         call get_xsec(10)
      endif

      numoffiles=0
      nbunches=0
      nevents=0
      xtotal=0.d0
      do while (.true.)
         read(10,'(120a)',err=2,end=2) string120
         eventfile=string120(2:index(string120,'   '))
         read(string120(index(string120,'   '):120),*)
     $    ievents,absxsec,xsecfrac
         if (ievents.eq.0) cycle
         nevents=nevents+ievents
         numoffiles=numoffiles+1
         xsecfrac_all(numoffiles) = xsecfrac
c store here the proc_id as computed from directory name ("@XX" in
c process generation)
         if (eventfile(1:1).eq.'P') then
            if (eventfile(3:3).eq.'_') then
               read(eventfile(2:2),'(i1)') proc_id(numoffiles)
            elseif(eventfile(4:4).eq.'_') then
               read(eventfile(2:3),'(i2)') proc_id(numoffiles)
            elseif(eventfile(5:5).eq.'_') then 
               read(eventfile(2:4),'(i3)') proc_id(numoffiles)
            else
               write (*,*) 'ERROR in collect_events: '/
     $              /'cannot find process ID'
               stop
            endif
         else
            proc_id(numoffiles)=-1
         endif
c store here the number of events per file         
         nevents_file(numoffiles) = ievents
         xtotal=xtotal+absxsec
         junit(numoffiles)=numoffiles+10
         open(unit=junit(numoffiles),file=eventfile,status='old',
     &        err=999)
c Every time we find 80 files, collect the events
         if (numoffiles.eq.80) then
            nbunches=nbunches+1
            if (i_orig.eq.1) then
               if (.not.get_xsec_from_res1) then
                  evwgt=xtotal/dfloat(nevents)
               else
                  evwgt=xsecABS/dfloat(nevents)
               endif
            elseif(i_orig.eq.2) then
               if (.not.get_xsec_from_res1) then
                  evwgt=xtotal
               else
                  evwgt=xsecABS
               endif
            elseif(i_orig.eq.3) then
               evwgt=1d0
            endif
            write (*,*) 'found ',numoffiles,
     &           ' files, bunch number is',nbunches
            if(nbunches.le.9) then
               write(outputfile(15:15),'(i1)')nbunches
            elseif(nbunches.le.99) then
               write(outputfile(14:15),'(i2)')nbunches
            elseif(nbunches.le.999) then
               write(outputfile(13:15),'(i3)')nbunches
            else
               write (*,*) 'Error, too many bunches'
               stop
            endif
            open (unit=ioutput,file=outputfile,status='unknown')
            call collect_all_evfiles(ioutput,numoffiles,junit,
     #                               nevents,evwgt)
            do i=1,numoffiles
               if (istep.eq.0) then
                  close (junit(i))
               else
                  close (junit(i),status='delete')
               endif
            enddo
            close (ioutput)
         write(98,*) outputfile(1:15),'     ',nevents,'  ',xtotal,
     #       '   ', 1e0
            numoffiles=0
            nevents=0
            xtotal=0.d0
         endif
      enddo
 2    continue
      close(10)
c Also collect events from the rest files
      if(numoffiles.ne.0) then
         nbunches=nbunches+1
         if (i_orig.eq.1) then
            if (.not.get_xsec_from_res1) then
               evwgt=xtotal/dfloat(nevents)
            else
               evwgt=xsecABS/dfloat(nevents)
            endif
         elseif(i_orig.eq.2) then
            if (.not.get_xsec_from_res1) then
               evwgt=xtotal
            else
               evwgt=xsecABS
            endif
         elseif(i_orig.eq.3) then
            evwgt=1d0
         endif
         write (*,*) 'found ',numoffiles,
     &        ' files, bunch number is',nbunches
         if(nbunches.le.9) then
            write(outputfile(15:15),'(i1)')nbunches
         elseif(nbunches.le.99) then
            write(outputfile(14:15),'(i2)')nbunches
         elseif(nbunches.le.999) then
            write(outputfile(13:15),'(i3)')nbunches
         else
            write (*,*) 'Error, too many bunches'
            stop
         endif
         open (unit=ioutput,file=outputfile,status='unknown')
         call collect_all_evfiles(ioutput,numoffiles,junit,
     #                            nevents,evwgt)
         do i=1,numoffiles
            if (istep.eq.0) then
               close (junit(i))
            else
               close (junit(i),status='delete')
            endif
         enddo
         close(ioutput)
         write(98,*) outputfile(1:15),'     ',nevents,'  ',xtotal,
     #       '   ', 1e0
      endif
      close(98)
c
      if(nbunches.gt.1) then
         istep=istep+1
         write (*,*) 'More than 1 bunch, doing next step',istep
         goto 1
      else
         write (*,*) 'Done. Final event file (with',nevents,
     &        ' events) is:'
         write (*,*) outputfile(1:15)
      endif
      return
c
 999  continue
      write (*,*) 'Error, event file',eventfile,' not found'
      stop
      end


      subroutine collect_all_evfiles(ioutput,numoffiles,junit,
     #                               imaxevt,evwgt)
      implicit none
      integer i_orig
      common /c_i_orig/i_orig
      integer ioutput,junit(80)
      integer imaxevt,maxevt,ii,numoffiles,nevents,itot,iunit,
     # mx_of_evt(80),i0,i,j,jj,kk,n,nn
      double precision evwgt,evwgt_sign
      integer ione
      parameter (ione=1)
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      integer IDBMUP1(2),PDFGUP1(2),PDFSUP1(2),IDWTUP1,NPRUP1,LPRUP1
      double precision EBMUP1(2),XSECUP1,XERRUP1,XMAXUP1
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      character*140 buff
      character*10 MonteCarlo,MonteCarlo1, MonteCarlo0
      character*100 path
      integer iseed
      data iseed/1/
      double precision rnd,fk88random
      external fk88random
      logical debug
      parameter (debug=.false.)
      integer nevents_file(80),proc_id(80)
      common /to_nevents_file/nevents_file,proc_id
      double precision xsecfrac_all(80)
      common /to_xsecfrac/xsecfrac_all
      double precision XSECUP2(100),XERRUP2(100),XMAXUP2(100)
      integer LPRUP2(100)
      common /lhef_init/XSECUP2,XERRUP2,XMAXUP2,LPRUP2
      double precision xsecup_l(100),xerrup_l(100)
      integer lprup_l(100),nproc_l
      logical found_proc
      include 'reweight_all.inc'
      include 'run.inc'
      integer proc_id_tot(0:100)
      double precision xsec(100),xsecABS,xerr(100)
      logical get_xsec_from_res1
      common/total_xsec/xsec,xerr,xsecABS,proc_id_tot,get_xsec_from_res1
c
      if(debug) then
         write (*,*) ioutput,numoffiles,(junit(ii),ii=1,numoffiles)
         write(ioutput,*)'test test test'
         return
      endif
      maxevt=0
      if (.not. get_xsec_from_res1) then
         do i=1,100
            xsecup_l(i)=0.d0
            xerrup_l(i)=0.d0
         enddo
      else
         do i=1,100
            if (i.le.proc_id_tot(0)) then
               xsecup_l(i)=xsec(i)
               xerrup_l(i)=xerr(i)
               lprup_l(i)  =proc_id_tot(i)
            else
               xsecup_l(i)=0.d0
               xerrup_l(i)=0.d0
            endif
         enddo
      endif
         
      call read_lhef_header(junit(ione),maxevt,MonteCarlo)
      if (MonteCarlo .ne. '') MonteCarlo0 = MonteCarlo
      call read_lhef_init(junit(ione),
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,XMAXUP,LPRUP)
c if the number of the events is in the header (as for evt files in the
c     subchannel folders (P*/G*/), the number of events should be in the
c      header. Check consistency in this case
      if (maxevt .gt. 0) then
        mx_of_evt(1)=maxevt
        if (mx_of_evt(1) .ne. nevents_file(1)) then
           write(*,*) 'Inconsistent event file 1, unit=', junit(1)
           write(*,*) 'Expected # of events:', nevents_file(1)
           write(*,*) 'Found # of events:', mx_of_evt(1)
           stop
        endif
      else
        mx_of_evt(1)=nevents_file(1)
      endif
      maxevt=mx_of_evt(1)

      nproc_l=NPRUP
      if (.not. get_xsec_from_res1) then
         do i=1,nproc_l
            xerrup_l(i)=xerrup2(i)**2 * xsecfrac_all(ione)
            xsecup_l(i)=xsecup2(i) * xsecfrac_all(ione)
            if (proc_id(ione).ne.-1) then
               lprup_l(i)=proc_id(ione)
               if (nproc_l.gt.1) then
                  write (*,*)
     $                 'ERROR: inconsistent nproc in collect_event'
                  write (*,*) nproc_l,NPRUP
                  write (*,*) proc_id
                  stop
               endif
            else
               lprup_l(i)=lprup2(i)
            endif
         enddo
      endif

      do ii=2,numoffiles
        call read_lhef_header(junit(ii),nevents,MonteCarlo1)
        if (nevents .gt. 0) then
            mx_of_evt(ii)=nevents
            if (mx_of_evt(ii) .ne. nevents_file(ii)) then
               write(*,*) 'Inconsistent event file, unit=',junit(ii)
               write(*,*) 'Expected # of events:', nevents_file(ii)
               write(*,*) 'Found # of events:', mx_of_evt(ii)
               stop
            endif
        else
            mx_of_evt(ii)=nevents_file(ii)
        endif
        if(MonteCarlo.ne.MonteCarlo1)then
          write(*,*)'Error in collect_all_evfiles'
          write(*,*)'Files ',ione,' and ',ii,' are inconsistent'
          write(*,*)'Monte Carlo types are not the same'
          write(*,*)'1', MonteCarlo, '2', MonteCarlo1
          stop
        endif
        maxevt=maxevt+mx_of_evt(ii)
        call read_lhef_init(junit(ii),
     #    IDBMUP1,EBMUP1,PDFGUP1,PDFSUP1,IDWTUP1,NPRUP1,
     #    XSECUP1,XERRUP1,XMAXUP1,LPRUP1)
        if (.not.get_xsec_from_res1) then
           if(proc_id(ii).ne.-1) then
              lprup2(1)=proc_id(ii)
           endif
           do i=1,NPRUP1
              found_proc=.false.
              do j=1,nproc_l
                 if (lprup_l(j).eq.lprup2(i)) then
                    xerrup_l(j)=xerrup_l(j)+xerrup2(i)**2 *xsecfrac_all(ii)
                    xsecup_l(j)=xsecup_l(j)+xsecup2(i) *xsecfrac_all(ii)
                    found_proc=.true.
                    exit
                 endif
              enddo
              if (.not.found_proc) then
                 nproc_l=nproc_l+1
                 xerrup_l(nproc_l)=xerrup2(i)**2 *xsecfrac_all(ii)
                 xsecup_l(nproc_l)=xsecup2(i) *xsecfrac_all(ii)
                 lprup_l(nproc_l)=lprup2(i)
              endif
           enddo
        endif
        if(
     #     IDBMUP(1).ne.IDBMUP1(1) .or.
     #     IDBMUP(2).ne.IDBMUP1(2) .or.
     #     EBMUP(1) .ne.EBMUP1(1)  .or.
     #     EBMUP(2) .ne.EBMUP1(2)  .or.
     #     PDFGUP(1).ne.PDFGUP1(1) .or.
     #     PDFGUP(2).ne.PDFGUP1(2) .or.
     #     PDFSUP(1).ne.PDFSUP1(1) .or.
     #     PDFSUP(2).ne.PDFSUP1(2))then
          write(*,*)'Error in collect_all_evfiles'
          write(*,*)'Files ',ione,' and ',ii,' are inconsistent'
          write(*,*)'Run parameters are not the same'
          stop
        endif
      enddo
      if(maxevt.ne.imaxevt)then
        write(*,*)'Error in collect_all_evfiles'
        write(*,*)'Total number of events inconsistent with input'
        write(*,*)maxevt,imaxevt
        stop
      endif
      if (.not.get_xsec_from_res1) then
         do i=1,nproc_l
            xerrup_l(i)=sqrt(xerrup_l(i))
         enddo
      endif
      XSECUP=xsecup_l(ione)
      XERRUP=xerrup_l(ione)
      LPRUP=lprup_l(ione)
      if (.not.get_xsec_from_res1) then
         NPRUP=nproc_l
      else
         NPRUP=proc_id_tot(0)
      endif
      do i=1,NPRUP
         XSECUP2(i)=xsecup_l(i)
         xerrup2(i)=xerrup_l(i)
         lprup2(i)=lprup_l(i)
         xmaxup2(i)=abs(evwgt)
      enddo
      path="../Cards/"
      call write_lhef_header_banner(ioutput,maxevt,MonteCarlo0,path)
      call write_lhef_init(ioutput,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,abs(evwgt),LPRUP)
      itot=maxevt
      do i=1,maxevt
        rnd=fk88random(iseed)
        call whichone(rnd,numoffiles,itot,mx_of_evt,junit,iunit,i0)
        call read_lhef_event(iunit,
     #    NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     #    IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
        if (proc_id(i0).ne.-1) IDPRUP=proc_id(i0)
        if (i_orig.eq.0) then
           evwgt_sign=XWGTUP
        else
c Overwrite the weights. Also overwrite the weights used for PDF & scale
c reweighting
           evwgt_sign=dsign(evwgt,XWGTUP)
           if (do_rwgt_scale) then
              do kk=1,dyn_scale(0)
                 if (lscalevar(kk)) then
                    do ii=1,nint(scalevarF(0))
                       do jj=1,nint(scalevarR(0))
                          wgtxsecmu(jj,ii,kk)=wgtxsecmu(jj,ii,kk)
     $                         *evwgt_sign/XWGTUP
                       enddo
                    enddo
                 else
                    wgtxsecmu(1,1,kk)=wgtxsecmu(1,1,kk)
     $                   *evwgt_sign/XWGTUP
                 endif
              enddo
           endif
           if (do_rwgt_pdf) then
              do nn=1,lhaPDFid(0)
                 if (lpdfvar(nn)) then
                    do n=0,nmemPDF(nn)
                       wgtxsecPDF(n,nn)=wgtxsecPDF(n,nn)*evwgt_sign/XWGTUP
                    enddo
                 else
                    wgtxsecPDF(0,nn)=wgtxsecPDF(0,nn)*evwgt_sign/XWGTUP
                 endif
              enddo
           endif
        endif
        call write_lhef_event(ioutput,
     #    NUP,IDPRUP,evwgt_sign,SCALUP,AQEDUP,AQCDUP,
     #    IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
      enddo
      write(ioutput,'(a)')'</LesHouchesEvents>'
      return
      end


      subroutine whichone(rnd,numoffiles,itot,mx_of_evt,junit,iunit,i0)
      implicit none
      double precision rnd,tiny,one,xp(80),xsum,prob
      integer numoffiles,itot,mx_of_evt(80),junit(80),iunit,ifiles,i0
      logical flag
      parameter (tiny=1.d-4)
c
      if(itot.le.0)then
        write(6,*)'fatal error #1 in whichone'
        stop
      endif
      one=0.d0
      do ifiles=1,numoffiles
        xp(ifiles)=dfloat(mx_of_evt(ifiles))/dfloat(itot)
        one=one+xp(ifiles)
      enddo
      if(abs(one-1.d0).gt.tiny)then
        write(6,*)'whichone: probability not normalized'
        stop
      endif
c
      i0=0
      flag=.true.
      xsum=0.d0
      do while(flag)
        if(i0.gt.numoffiles)then
          write(6,*)'fatal error #2 in whichone'
          stop
        endif
        i0=i0+1
        prob=xp(i0)
        xsum=xsum+prob
        if(rnd.lt.xsum)then
          flag=.false.
          itot=itot-1
          mx_of_evt(i0)=mx_of_evt(i0)-1
          iunit=junit(i0)
        endif
      enddo
      return
      end


      FUNCTION FK88RANDOM(SEED)
*     -----------------
* Ref.: K. Park and K.W. Miller, Comm. of the ACM 31 (1988) p.1192
* Use seed = 1 as first value.
*
      IMPLICIT INTEGER(A-Z)
      REAL*8 MINV,FK88RANDOM
      SAVE
      PARAMETER(M=2147483647,A=16807,Q=127773,R=2836)
      PARAMETER(MINV=0.46566128752458d-09)
      HI = SEED/Q
      LO = MOD(SEED,Q)
      SEED = A*LO - R*HI
      IF(SEED.LE.0) SEED = SEED + M
      FK88RANDOM = SEED*MINV
      END









      subroutine get_xsec(unit10)
      implicit none
      integer unit10
      character*120 string120,eventfile,results_file,read_line
      integer proc_id_l,add_xsec_to,i,ievents
      double precision xsec_read,xerr_read,absxsec,xsecfrac,xsecABS_read
      integer proc_id_tot(0:100)
      double precision xsec(100),xsecABS,xerr(100)
      logical get_xsec_from_res1
      common/total_xsec/xsec,xerr,xsecABS,proc_id_tot,get_xsec_from_res1

      proc_id_tot(0)=0
      get_xsec_from_res1=.true.
      xsecABS=0d0
      do 
         read(unit10,'(120a)',end=22,err=22) string120
         eventfile=string120(2:index(string120,'   '))
         read(string120(index(string120,'   '):120),*)
     $    ievents,absxsec,xsecfrac
         if (eventfile(1:1).eq.'P') then
            if (eventfile(3:3).eq.'_') then
               read(eventfile(2:2),'(i1)') proc_id_l
            elseif(eventfile(4:4).eq.'_') then
               read(eventfile(2:3),'(i2)') proc_id_l
            elseif(eventfile(5:5).eq.'_') then 
               read(eventfile(2:4),'(i3)') proc_id_l
            else
               write (*,*) 'ERROR in collect_events: '/
     $              /'cannot find process ID'
               stop
            endif
         else
            proc_id_l=-1
            get_xsec_from_res1=.false.
            exit
         endif
         if (index(eventfile,'events.lhe').eq.0) then
            get_xsec_from_res1=.false.
            exit
         endif
         results_file=eventfile(1:index(eventfile,'events.lhe')-1)
     $        //'res_1'
         open (unit=11,file=results_file,status='old',err=998)
         read (11,'(120a)',err=998) read_line
         read(read_line(index(read_line,'Final result [ABS]:')+20:),*
     $        ,err=998)xsecABS_read
         read (11,'(120a)',err=998) read_line
         close (11)
         read(read_line(index(read_line,'Final result:')+14:),*,err=998)
     $        xsec_read
         read(read_line(index(read_line,'+/-')+4:),*,err=998) xerr_read
         add_xsec_to=-1
         if (proc_id_tot(0).ge.1) then
            do i=1,proc_id_tot(0)
               if (proc_id_l.eq.proc_id_tot(i)) then
                  add_xsec_to=i
                  exit
               endif
            enddo
         endif
         if (add_xsec_to.eq.-1) then
            proc_id_tot(0)=proc_id_tot(0)+1
            if (proc_id_tot(0).gt.100) then
               write (*,*) 'ERROR, too many separate processes'
     $              ,proc_id_tot(0)
               stop
            endif
            proc_id_tot(proc_id_tot(0))=proc_id_l
            xsec(proc_id_tot(0))=xsec_read*xsecfrac
            xerr(proc_id_tot(0))=xerr_read**2*xsecfrac
         else
            xsec(add_xsec_to)=xsec(add_xsec_to)+xsec_read*xsecfrac
            xerr(add_xsec_to)=xerr(add_xsec_to)+xerr_read**2*xsecfrac
         endif
         xsecABS=xsecABS + xsecABS_read*xsecfrac
      enddo
 22   continue
      do i=1,proc_id_tot(0)
         xerr(i)=sqrt(xerr(i))
      enddo
      rewind(unit10)
      return
 998  continue
      write (*,*) 'Error, results file',results_file
     $     ,' not found or not the correct format.'
      stop
      end
