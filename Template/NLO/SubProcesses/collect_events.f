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
      integer nevents_file(80)
      double precision xsecfrac_all(80)
      common /to_xsecfrac/xsecfrac_all
      common /to_nevents_file/nevents_file

      write (*,*) "Overwrite the event weights?"
      write (*,*) "give '0' to keep original weights;"
      write (*,*) "give '1' to overwrite the weights"/
     $     /" to sum to the Xsec;"
      write (*,*) "give '2' to overwrite the weights"/
     $     /" to average to the Xsec (=default)"
      read (*,*) i_orig
      if (i_orig.ne.0 .and. i_orig.ne.1 .and. i_orig.ne.2) stop
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
               evwgt=xtotal/dfloat(nevents)
            elseif(i_orig.eq.2) then
               evwgt=xtotal
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
            evwgt=xtotal/dfloat(nevents)
         elseif(i_orig.eq.2) then
            evwgt=xtotal
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
     # mx_of_evt(80),i0,i,j
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
      integer nevents_file(80)
      common /to_nevents_file/nevents_file
      double precision xsecfrac_all(80)
      common /to_xsecfrac/xsecfrac_all
      include 'reweight_all.inc'
c
      if(debug) then
         write (*,*) ioutput,numoffiles,(junit(ii),ii=1,numoffiles)
         write(ioutput,*)'test test test'
         return
      endif
      maxevt=0
      xsecup=0.d0
      xerrup=0.d0
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

      xerrup=xerrup**2 * xsecfrac_all(ione)
      xsecup = xsecup * xsecfrac_all(ione)
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
        xsecup=xsecup+xsecup1 * xsecfrac_all(ii)
        xerrup=xerrup+xerrup1**2 * xsecfrac_all(ii)
        if(
     #     IDBMUP(1).ne.IDBMUP1(1) .or.
     #     IDBMUP(2).ne.IDBMUP1(2) .or.
     #     EBMUP(1) .ne.EBMUP1(1)  .or.
     #     EBMUP(2) .ne.EBMUP1(2)  .or.
     #     PDFGUP(1).ne.PDFGUP1(1) .or.
     #     PDFGUP(2).ne.PDFGUP1(2) .or.
     #     PDFSUP(1).ne.PDFSUP1(1) .or.
     #     PDFSUP(2).ne.PDFSUP1(2) .or.
     #     LPRUP .ne.LPRUP1 )then
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
      xerrup=sqrt(xerrup)
      path="../Cards/"
      call write_lhef_header_banner(ioutput,maxevt,MonteCarlo0,path)
      call write_lhef_init(ioutput,
     #  IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     #  XSECUP,XERRUP,abs(evwgt),LPRUP)
      itot=maxevt
      do ii=1,maxevt
        rnd=fk88random(iseed)
        call whichone(rnd,numoffiles,itot,mx_of_evt,junit,iunit,i0)
        call read_lhef_event(iunit,
     #    NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     #    IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
c Sanity check on weights read and computed a posteriori (check only for
c event samples that are relatively large, otherwise it could be a
c statistical fluctuation because the number of events per channel is
c assigned randomly with a weight according to the contribution of that
c channel to the total cross section)
        if( ( abs(XWGTUP/evwgt).gt.2.d0.or.
     #        abs(XWGTUP/evwgt).lt.0.5d0 ) .and.
     #       mx_of_evt(i0).gt.50 .and. i_orig.eq.1)then
           write(*,*)'Error in collect_all_evfiles'
           write(*,*)'Events weights appear to be wrong'
           write(*,*)XWGTUP,evwgt
           stop
        endif
        if (i_orig.eq.0) then
           evwgt_sign=XWGTUP
        else
c Overwrite the weights. Also overwrite the weights used for PDF & scale
c reweighting
           evwgt_sign=dsign(evwgt,XWGTUP)
           do i=1,numscales
              do j=1,numscales
                 wgtxsecmu(i,j)=wgtxsecmu(i,j)*evwgt_sign/XWGTUP
              enddo
           enddo
           do i=1,numPDFpairs*2
              wgtxsecPDF(i)=wgtxsecPDF(i)*evwgt_sign/XWGTUP
          enddo
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

