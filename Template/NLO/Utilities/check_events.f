      program check_events
c Checks self-consistency of event files. Compile with
c gfortran -I../SubProcesses/P0_<anydir> -ffixed-line-length-132 
c   -o check_events check_events.f handling_lhe_events.f 
c                   fill_MC_mshell.f dbook.f
c With some work on finalizeprocesses(), it should work also for 
c LH files created by Herwig, assuming they are identified by a 
c negative number of events
      implicit none
      integer maxevt,ifile,efile,mfile,jfile,kfile,rfile,i,npart,
     # iuseres_1,iwhmass,ilepmass,idec,itempsc,itempPDF,isavesc,
     # isavePDF,itemp,ii
      double precision chtot,xint,xinterr,xinta,xintaerr,qtiny
      parameter (qtiny=1.d-4)
      double precision charges(-100:100),zmasses(1:100)
      double precision remcmass(-16:21)
      common/cremcmass/remcmass
      integer nevS_lhe,nevH_lhe,npartS_lhe,npartH_lhe,mtoterr,
     # itoterr,numproc,numconn,idup_eff(22),icolup_eff(2,22)
      logical wrong
      integer mxlproc,minnp,maxnp,idups_proc(10000,-1:22)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer idups_Sproc_HW6(401:499,-1:22),
     #        idups_Hproc_HW6(401:499,-1:22)
      common/cHW6processes/idups_Sproc_HW6,idups_Hproc_HW6
      integer icolups_proc(10000,0:500,0:2,22)
      common/ccolconn/icolups_proc
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      double precision sum_wgt,sum_abs_wgt,err_wgt,toterr,diff
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      double precision wgt1a,wgt1s
      double precision wgt2a,wgt2s
      double precision wgt3a,wgt3s
      double precision wgt4a,wgt4s
      double precision wgt5a,wgt5s
      double precision saved_weight,tmp
      character*80 event_file
      character*140 buff
      character*6 ch6
      character*5 ch5
      character*3 ch3,event_norm
      character*10 MonteCarlo
      character*2 ch2,pm
      character*9 ch1
      common/cevtnorm/event_norm

      logical AddInfoLHE,rwgtinfo,unweighted,keepevent,shower

      include "nexternal.inc"
      integer j,k
      real*8 ecm,xmass(3*nexternal),xmom(0:3,3*nexternal),xnorm

      include 'reweight0.inc'
      integer kr,kf,kpdf
      double precision sum_wgt_resc_scale(maxscales,maxscales),
     # sum_wgt_resc_pdf(0:maxPDFs),
     # xmax_wgt_resc_scale(maxscales,maxscales),
     # xmax_wgt_resc_pdf(0:maxPDFs),
     # xmin_wgt_resc_scale(maxscales,maxscales),
     # xmin_wgt_resc_pdf(0:maxPDFs)
      integer istep
      double precision percentage
      include 'dbook.inc'

      call setcharges(charges)
      call setmasses(zmasses)
      mxlproc=0
      minnp=10000
      maxnp=-1
      do i=1,10000
        icolups_proc(i,0,1,1)=0
      enddo
      do i=401,499
        idups_Sproc_HW6(i,-1)=0
        idups_Hproc_HW6(i,-1)=0
      enddo
      do i=-16,21
         remcmass(i)=-2.d0
      enddo

      write (*,*) 'Enter event file name'
      read (*,'(a)') event_file

      write (*,*) 'Enter 0 to get integrals from res_1.txt'
      write (*,*) '      1 otherwise'
      read (*,*) iuseres_1

c Mass of particles other than partons and leptons always to be
c read from events
      write (*,*) 'Enter 0 to use MC masses as written in the header'
      write (*,*) '      1 to use MC masses as written in events'
      write (*,*) '      2 to enter them'
      read (*,*) iwhmass

      if(iwhmass.eq.2)then
        write (*,*) 'Enter 0 to use physical lepton masses'
        write (*,*) '      2 to enter them'
        read (*,*) ilepmass
      endif

      write (*,*) 'Enter 0 to study decays'
      write (*,*) '      1 otherwise'
      read (*,*) idec
      if(idec.eq.0)call setdecmat()

      ifile=34
      open (unit=ifile,file=event_file,status='old')
      open (unit=51,file='res_wgt',status='unknown')
      if(iuseres_1.eq.0)then
        jfile=50
        open (unit=jfile,file='res_1.txt',status='old')
        do while(buff(1:6).ne.'Total:')
           read(jfile,*)buff
        enddo
        read(jfile,*)xinta,pm,xintaerr
        read(jfile,*)xint,pm,xinterr
        if(pm.ne.'+-')then
           write(*,*)'File res_1.txt has unexpected format'
           stop
        endif
      elseif(iuseres_1.eq.1)then
         continue
      else
        write(*,*)'No such option for iuseres_1'
        stop
      endif
      efile=44
      open (unit=efile,file='LHEF.errors',status='unknown')
      mfile=45
      open (unit=mfile,file='LHEF.mass_errors',status='unknown')
      kfile=54
      open (unit=kfile,file='LHEF.stats',status='unknown')
      AddInfoLHE=.false.
      rwgtinfo=.false.
      unweighted=.true.
      keepevent=.true.
      shower=.false.

      call read_lhef_header_full(ifile,maxevt,itempsc,itempPDF,
     #                           MonteCarlo)
      isavesc=itempsc
      isavePDF=itempPDF
      if(itempsc.ne.0)then
        numscales=int(sqrt(dble(itempsc)))
        if(numscales**2.ne.itempsc)then
          write(*,*)'Different number of muR and muF choices?',itempsc
          stop
        endif
      else
        write(*,*)'Enter numscales, <0 to ignore'
        read(*,*)itemp
        if(itemp.gt.0)numscales=itemp
      endif
      if(itempPDF.ne.0)then
        numPDFpairs=itempPDF/2
        if(2*numPDFpairs.ne.itempPDF)then
          write(*,*)'Number of PDF sets not even',itempPDF
          stop
        endif
      else
        write(*,*)'Enter numPDFpairs, <0 to ignore'
        read(*,*)itemp
        if(itemp.gt.0)numPDFpairs=itemp
      endif
c Showered LH files have maxevt<0; in that case, it is not the number of
c events, but its upper bound
      if(maxevt.lt.0)then
        write(*,*)'This appears to be a showered LH file'
        shower=.true.
      endif
      maxevt=abs(maxevt)
c Fill quark/antiquark and lepton/antilepton masses if not already in header
      do i=-16,16
        if(abs(i).le.5.or.abs(i).ge.11)then
          if(remcmass(i).eq.-2.d0)remcmass(i)=remcmass(-i)
        endif
      enddo
c
      if(iwhmass.eq.0)then
        do i=1,21
          if(remcmass(i).ne.-2.d0)zmasses(i)=remcmass(i)
        enddo
      elseif(iwhmass.eq.2)then
        do i=1,21
          if((i.ge.1.and.i.le.5).or.i.eq.21)then
            write(*,*)'Enter mass #',i
            read(*,*)zmasses(i)
          endif
        enddo
        if(ilepmass.eq.0)then
          zmasses(11)=0.510998928d-3
          zmasses(12)=0.d0
          zmasses(13)=0.1056583715d0
          zmasses(14)=0.d0
          zmasses(15)=1.77682d0
          zmasses(16)=0.d0
        elseif(ilepmass.eq.2)then
          do i=11,16
            if(mod(i,2).eq.1)then
              write(*,*)'Enter mass #',i
              read(*,*)zmasses(i)
            else
              zmasses(i)=0.d0
            endif
          enddo
        endif
      endif

      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

      
      do i=1,min(10,maxevt)
        call read_lhef_event_catch(ifile,
     &       NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &       IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

        if(i.eq.1.and.buff(1:1).eq.'#')AddInfoLHE=.true.
        if(AddInfoLHE)then
          if(buff(1:1).ne.'#')then
            write(*,*)'Inconsistency in event file',i,' ',buff
            stop
          endif
          read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                      fksfather_lhe,ipartner_lhe,
     #                      scale1_lhe,scale2_lhe,
     #                      jwgtinfo,mexternal,iwgtnumpartn,
     #           wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
          if(i.eq.1)then
            if( (jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #          jwgtinfo.eq.8.or.jwgtinfo.eq.9 )rwgtinfo=.true.
            saved_weight=abs(XWGTUP)
          else
            if( ((jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #           jwgtinfo.eq.8.or.jwgtinfo.eq.9) .and. 
     #          (.not.rwgtinfo) )then
              write(*,*)'Inconsistency #2 in event file',i,' ',buff
              stop
            endif
          unweighted=unweighted.and.
     #               abs(1.d0-abs(XWGTUP)/saved_weight).lt.1.d-5
          endif
        endif
        if(idec.eq.0)call findmothers(NUP,IDUP,ISTUP)

      enddo
      close(34)

      if(idec.eq.0)call checkmothers()

      write(*,*)'  '
      if(unweighted)then
        write(*,*)'The events appear to be unweighted'
      else
        write(*,*)'The events appear to be weighted'
      endif

      if(rwgtinfo)then
        wgt1a=0.d0
        wgt1s=0.d0
        wgt2a=0.d0
        wgt2s=0.d0
        wgt3a=0.d0
        wgt3s=0.d0
        wgt4a=0.d0
        wgt4s=0.d0
        wgt5a=0.d0
        wgt5s=0.d0

        rfile=64
        open (unit=rfile,file='LHEF.rwgt',status='unknown')
      endif

      open (unit=ifile,file=event_file,status='old')

      call read_lhef_header_full(ifile,maxevt,itempsc,itempPDF,
     #                           MonteCarlo)
      if(itempsc.ne.isavesc.or.itempPDF.ne.isavePDF)then
        write(*,*)'Error in check_events'
        write(*,*)itempsc,isavesc,itempPDF,isavePDF
      endif
      maxevt=abs(maxevt)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
      

      sum_wgt=0d0
      sum_abs_wgt=0d0
      nevS_lhe=0
      npartS_lhe=0
      nevH_lhe=0
      npartH_lhe=0
      itoterr=0
      mtoterr=0
      if(jwgtinfo.eq.8.or.jwgtinfo.eq.9)then
        do kr=1,maxscales
          do kf=1,maxscales
            sum_wgt_resc_scale(kr,kf)=0.d0
            xmax_wgt_resc_scale(kr,kf)=-1.d100
            xmin_wgt_resc_scale(kr,kf)=1.d100
          enddo
        enddo
        do kpdf=0,maxPDFs
          sum_wgt_resc_pdf(kpdf)=0.d0
          xmax_wgt_resc_pdf(kpdf)=-1.d100
          xmin_wgt_resc_pdf(kpdf)=1.d100
        enddo
      endif

      i=0
      call inihist
      call bookup(1,'scalup',2d0,0d0,200d0)
      call bookup(2,'scalup',8d0,0d0,800d0)
      call bookup(3,'log scalup',0.1d0,0d0,4d0)
      dowhile(i.lt.maxevt.and.keepevent)
         call read_lhef_event_catch(ifile,
     &        NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &        IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         call mfill(1,SCALUP,XWGTUP)
         call mfill(2,SCALUP,XWGTUP)
         if(SCALUP.gt.0d0)call mfill(3,log10(SCALUP),XWGTUP)
         if(index(buff,'endoffile').ne.0)then
           keepevent=.false.
           goto 111
         endif

         i=i+1
         sum_wgt=sum_wgt+XWGTUP
         sum_abs_wgt=sum_abs_wgt+abs(XWGTUP)

c Note: with pre-beta2 convention, the reweighting cross sections were
c normalized such that one needed to compute e.g. 
c XWGTUP*wgtxsecmu(kr,kf)/wgtref
         if(jwgtinfo.eq.8.or.jwgtinfo.eq.9)then
           do kr=1,numscales
             do kf=1,numscales
               sum_wgt_resc_scale(kr,kf)=sum_wgt_resc_scale(kr,kf)+
     #                                   wgtxsecmu(kr,kf)
               xmax_wgt_resc_scale(kr,kf)=
     #           max(xmax_wgt_resc_scale(kr,kf),
     #               wgtxsecmu(kr,kf))
               xmin_wgt_resc_scale(kr,kf)=
     #           min(xmin_wgt_resc_scale(kr,kf),
     #               wgtxsecmu(kr,kf))
             enddo
           enddo
           do kpdf=1,2*numPDFpairs
             sum_wgt_resc_pdf(kpdf)=sum_wgt_resc_pdf(kpdf)+
     #                              wgtxsecPDF(kpdf)
             xmax_wgt_resc_pdf(kpdf)=
     #         max(xmax_wgt_resc_pdf(kpdf),
     #             wgtxsecPDF(kpdf))
             xmin_wgt_resc_pdf(kpdf)=
     #         min(xmin_wgt_resc_pdf(kpdf),
     #             wgtxsecPDF(kpdf))
           enddo
         endif

         if(AddInfoLHE)then
           if(buff(1:1).ne.'#')then
             write(*,*)'Inconsistency in event file',i,' ',buff
             stop
           endif
           read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                       fksfather_lhe,ipartner_lhe,
     #                       scale1_lhe,scale2_lhe,
     #                       jwgtinfo,mexternal,iwgtnumpartn,
     #            wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
          if( ((jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #         jwgtinfo.eq.8.or.jwgtinfo.eq.9) .and. 
     #        (.not.rwgtinfo) )then
             write(*,*)'Inconsistency #2 in event file',i,' ',buff
             stop
           endif
         endif

         npart=0
         chtot=0.d0
         do k=1,nup
           if(abs(ISTUP(k)).eq.1)then
             npart=npart+1
             xmass(npart)=pup(5,k)
             do j=1,4
               xmom(mod(j,4),npart)=pup(j,k)
             enddo
             idup_eff(npart)=IDUP(k)
             icolup_eff(1,npart)=ICOLUP(1,k)
             icolup_eff(2,npart)=ICOLUP(2,k)
             chtot=chtot+ISTUP(k)*charges(IDUP(k))
             if( ( abs(IDUP(k)).eq.6.or.abs(IDUP(k)).gt.21.or.
     #             iwhmass.eq.1 ).and.
     #           zmasses(abs(IDUP(k))).eq.-1.d0 )then
               zmasses(abs(IDUP(k)))=xmass(npart)
             else
               if(abs(zmasses(abs(IDUP(k)))-xmass(npart)).gt.qtiny)then
                 write(45,*)'####event:',i
                 write(45,*)' Wrong mass shell',xmass(npart)
                 write(45,*)' for particle',k,
     #                      ' Must be:',zmasses(abs(IDUP(k)))
                 mtoterr=mtoterr+1
               endif
             endif
           endif
         enddo

         if(abs(chtot).gt.1.d-8)then
           write(44,*)'####event:',i
           write(44,*)' charge is not conserved',chtot
           itoterr=itoterr+1
         endif

         call storeprocesses(npart,idup_eff,numproc)
         call storecolconn(npart,numproc,icolup_eff,numconn)
         call checkcolconn(i,numproc,numconn,wrong)
         if(wrong)then
           write(44,*)'####event:',i
           write(44,*)' wrong colour connection',numproc,numconn
           itoterr=itoterr+1
         endif

         if(AddInfoLHE)then
           if(iSorH_lhe.eq.1)then
             nevS_lhe=nevS_lhe+1
             if(npartS_lhe.eq.0)then
               npartS_lhe=npart
             else
               if(npart.ne.npartS_lhe)then
                 write(44,*)'####event:',i
                 write(44,*)' wrong particle number [S]',npart
                 itoterr=itoterr+1
               endif
             endif
           elseif(iSorH_lhe.eq.2)then
             nevH_lhe=nevH_lhe+1
             if(npartH_lhe.eq.0)then
               npartH_lhe=npart
             else
               if(npart.ne.npartH_lhe)then
                 write(44,*)'####event:',i
                 write(44,*)' wrong particle number [H]',npart
                 itoterr=itoterr+1
               endif
             endif
           else
             write(44,*)'####event:',i
             write(44,*)' unknown iSorH',iSorH_lhe
             itoterr=itoterr+1
           endif
         endif


         if(rwgtinfo)then
           if(unweighted)then
             wgt1a=wgt1a+wgtcentral
             wgt1s=wgt1s+wgtcentral**2
             wgt2a=wgt2a+wgtmumin
             wgt2s=wgt2s+wgtmumin**2
             wgt3a=wgt3a+wgtmumax
             wgt3s=wgt3s+wgtmumax**2
             wgt4a=wgt4a+wgtpdfmin
             wgt4s=wgt4s+wgtpdfmin**2
             wgt5a=wgt5a+wgtpdfmax
             wgt5s=wgt5s+wgtpdfmax**2
           else
             tmp=wgtcentral/XWGTUP
             wgt1a=wgt1a+tmp
             wgt1s=wgt1s+tmp**2
             tmp=wgtmumin/wgtcentral
             wgt2a=wgt2a+tmp
             wgt2s=wgt2s+tmp**2
             tmp=wgtmumax/wgtcentral
             wgt3a=wgt3a+tmp
             wgt3s=wgt3s+tmp**2
             tmp=wgtpdfmin/wgtcentral
             wgt4a=wgt4a+tmp
             wgt4s=wgt4s+tmp**2
             tmp=wgtpdfmax/wgtcentral
             wgt5a=wgt5a+tmp
             wgt5s=wgt5s+tmp**2
           endif
         endif

         if(idec.eq.0)call finddaughters(NUP,IDUP,ISTUP,MOTHUP)

c Showered LH files only contain final-state particles.
c Don't check momentum conservation in that case
         if(.not.shower)call phspncheck_nocms2(i,npart,xmass,xmom)

         percentage=i*100d0/maxevt
         istep=maxevt/10
         if(mod(i,istep).eq.0.or.i.eq.maxevt)
     &        write(*,*)'Processed',int(percentage),'% of the file'

 111     continue
      enddo

      open(unit=99,file='SCALUP.top',status='unknown')
      call mclear
      xnorm=1d0
      do ii=1,nplots
         call mopera(ii,'+',ii,ii,xnorm,0.d0)
         call mfinal(ii)
      enddo
      call multitop(1,3,2,'scalup    ',' ','LOG')
      call multitop(2,3,2,'scalup    ',' ','LOG')
      call multitop(3,3,2,'log scalup',' ','LOG')
      close(99)

      if(event_norm.eq.'ave')sum_wgt=sum_wgt/maxevt
      if(event_norm.eq.'ave')sum_abs_wgt=sum_abs_wgt/maxevt
      err_wgt=sum_abs_wgt/sqrt(dfloat(maxevt))
      write(*,*)'  '
      write (*,*) 'The total number of events is:',i
      write (*,*) 'Sum of weights is    :',sum_wgt,' +-',err_wgt
      write (*,*) 'Sum of abs weights is:',sum_abs_wgt,' +-',err_wgt

      if(iuseres_1.eq.0)then
        toterr=sqrt(xinterr**2+err_wgt**2)
        diff=sum_wgt-xint
        if( (diff.le.0.d0.and.diff+toterr.lt.0.d0) .or.
     #      (diff.gt.0.d0.and.diff-toterr.gt.0.d0) )then
c Error if more that one sigma away
          itoterr=itoterr+1
          write(44,*)'WEIGHTS'
          write(44,*)'Integral:',xint,' +-',xinterr
          write(44,*)'Weights: ',sum_wgt,' +-',err_wgt
          write(44,*)' '
          write(44,*)'Sigmas:  ',abs(xint-sum_wgt)/
     #                           sqrt(xinterr**2+err_wgt**2)
        endif
        write (51,*)'Xsec     (check_events) = ',
     #              sum_wgt,' +-',err_wgt
        write (51,*)'Xsec     (res_1.txt)    = ',
     #               xint   ,' +-',xinterr
        write (51,*)' '
        write (51,*)'Xsec ABS (check_events) = ',
     #              sum_abs_wgt,' +-',err_wgt
        write (51,*)'Xsec ABS (res_1.txt)    = ',
     #              xinta,' +-',xintaerr
      elseif(iuseres_1.eq.1)then
        write (51,*)'Xsec     (check_events) = ',
     #              sum_wgt,' +-',err_wgt
        write (51,*)' '
        write (51,*)'Xsec ABS (check_events) = ',
     #              sum_abs_wgt,' +-',err_wgt
      else
        write(*,*)'No such option for iuseres_1'
        stop
      endif

      write (*,*) ' '
      write (*,*) 'Smallest and largest numbers of particles:',
     #            minnp,maxnp

      call finalizeprocesses(maxevt,kfile)

      if(rwgtinfo)then
        wgt1a=wgt1a/maxevt
        wgt1s=sqrt(abs(wgt1s/maxevt-wgt1a**2))
        wgt2a=wgt2a/maxevt
        wgt2s=sqrt(abs(wgt2s/maxevt-wgt2a**2))
        wgt3a=wgt3a/maxevt
        wgt3s=sqrt(abs(wgt3s/maxevt-wgt3a**2))
        wgt4a=wgt4a/maxevt
        wgt4s=sqrt(abs(wgt4s/maxevt-wgt4a**2))
        wgt5a=wgt5a/maxevt
        wgt5s=sqrt(abs(wgt5s/maxevt-wgt5a**2))

        write(64,*)'central:  ',wgt1a,' +-',wgt1s
        write(64,*)'mu lower: ',wgt2a,' +-',wgt2s
        write(64,*)'mu upper: ',wgt3a,' +-',wgt3s
        write(64,*)'PDF lower:',wgt4a,' +-',wgt4s
        write(64,*)'PDF upper:',wgt5a,' +-',wgt5s

        if(jwgtinfo.eq.8.or.jwgtinfo.eq.9)then
          write(64,*)'  '
          write(64,*)'Sums of rescaled weights'
          do kr=1,numscales
            do kf=1,numscales
              if(event_norm.eq.'ave')then
                write(64,300)'scales',kr,kf,' ->',
     #                     sum_wgt_resc_scale(kr,kf)/maxevt
              else
                write(64,300)'scales',kr,kf,' ->',
     #                     sum_wgt_resc_scale(kr,kf)
              endif
            enddo
          enddo
          if(event_norm.eq.'ave')then
            do kpdf=1,2*numPDFpairs
              write(64,301)'PDF',kpdf,' ->',
     #                   sum_wgt_resc_pdf(kpdf)/maxevt
            enddo
          else
            do kpdf=1,2*numPDFpairs
              write(64,301)'PDF',kpdf,' ->',
     #                   sum_wgt_resc_pdf(kpdf)
            enddo
          endif
        endif
        if(jwgtinfo.eq.8.or.jwgtinfo.eq.9)then
          write(64,*)'  '
          write(64,*)'Max and min of rescaled weights'
          do kr=1,numscales
            do kf=1,numscales
              if(event_norm.eq.'ave')then
                write(64,400)'scales',kr,kf,' ->',
     #                     xmax_wgt_resc_scale(kr,kf),
     #                     xmin_wgt_resc_scale(kr,kf)
              else
                write(64,400)'scales',kr,kf,' ->',
     #                     xmax_wgt_resc_scale(kr,kf),
     #                     xmin_wgt_resc_scale(kr,kf)
              endif
            enddo
          enddo
          if(event_norm.eq.'ave')then
            do kpdf=1,2*numPDFpairs
              write(64,401)'PDF',kpdf,' ->',
     #                   xmax_wgt_resc_pdf(kpdf),
     #                   xmin_wgt_resc_pdf(kpdf)
            enddo
          else
            do kpdf=1,2*numPDFpairs
              write(64,401)'PDF',kpdf,' ->',
     #                   xmax_wgt_resc_pdf(kpdf),
     #                   xmin_wgt_resc_pdf(kpdf)
            enddo
          endif
        endif

      endif

      write (*,*) ' '
      write (*,*) 'Total number of errors found:',itoterr+mtoterr
      write (*,*) '         Of which mass shell:',mtoterr

      close(34)
      close(44)
      close(50)
      close(51)
      close(54)
      if(rwgtinfo)close(64)

      if(idec.eq.0)call finalizedecays()

 300  format(a,2(1x,i2),a,(1x,f16.6))
 301  format(a,6x,i3,a,(1x,f16.6))
 400  format(a,2(1x,i2),a,2(1x,f16.6))
 401  format(a,6x,i3,a,2(1x,f16.6))

      end


      subroutine setcharges(charges)
      implicit none
      integer i
      double precision zup,zdown,charges(-100:100)
      parameter (zup=2/3.d0)
      parameter (zdown=-1/3.d0)
c
      do i=-100,100
        charges(i)=abs(i)*1.d6
      enddo
      charges(1)=zdown
      charges(2)=zup
      charges(3)=zdown
      charges(4)=zup
      charges(5)=zdown
      charges(6)=zup
      charges(11)=-1.d0
      charges(12)=0.d0
      charges(13)=-1.d0
      charges(14)=0.d0
      charges(15)=-1.d0
      charges(16)=0.d0
      charges(21)=0.d0
      charges(22)=0.d0
      charges(23)=0.d0
      charges(24)=1.d0
      charges(25)=0.d0
c
      do i=-100,-1
        if(abs(charges(-i)).le.1.d0)charges(i)=-charges(-i)
      enddo
c
      return
      end


      subroutine setmasses(zmasses)
      implicit none
      integer i
      double precision zmasses(1:100)
c
      do i=1,100
        zmasses(i)=-1.d0
      enddo
      return
      end


      subroutine storeprocesses(npart,idup_eff,numproc)
c Fills common block /cprocesses/ and return numproc, the number of the current
c process in the list of processes idups_proc
      implicit none
      integer npart,numproc,idup_eff(22)
      integer i,j
      logical exists,found
      integer mxlproc,minnp,maxnp,idups_proc(10000,-1:22)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
c mxlproc=current maximum number of different processes
c idups_proc(n,-1)=number of identical processes identified by n
c idups_proc(n,0)=number of particles in process n
c idups_proc(n,i)=ID of particle #i in process n; 1<=i<=idups_proc(n,0)
c
      if(npart.gt.22)then
        write(*,*)'Array idup_eff too small',npart
        stop
      endif
c
      exists=.false.
      i=1
      do while(.not.exists.and.i.le.mxlproc)
        found=npart.eq.idups_proc(i,0)
        if(found)then
          do j=1,npart
            found=found.and.idups_proc(i,j).eq.idup_eff(j)
          enddo
        endif
        exists=exists.or.found
        i=i+1
      enddo
c
      if(.not.exists)then
        mxlproc=mxlproc+1
        if(mxlproc.gt.10000)then
          write(*,*)'Error in storeprocesses: too many processes'
          stop
        endif
        numproc=mxlproc
        idups_proc(mxlproc,-1)=1
        idups_proc(mxlproc,0)=npart
        do i=1,npart
          idups_proc(mxlproc,i)=idup_eff(i)
        enddo
        if(npart.lt.minnp)minnp=npart
        if(npart.gt.maxnp)maxnp=npart
      else
        numproc=max(1,i-1)
        idups_proc(numproc,-1)=idups_proc(numproc,-1)+1
      endif
c
      return
      end


      subroutine finalizeprocesses(maxevt,iunit)
      implicit none
      integer iunit
      integer maxevt,iprocsum,iHW6procsum,nevS,nevH,i,id1,id2,ihpro
      logical isalquark,isagluon
      integer mxlproc,minnp,maxnp,idups_proc(10000,-1:22)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer idups_Sproc_HW6(401:499,-1:22),
     #        idups_Hproc_HW6(401:499,-1:22)
      common/cHW6processes/idups_Sproc_HW6,idups_Hproc_HW6
c Derived from conventions used by HW6 
C  401    q qbar -> X
C  402    q g    -> X
C  403    qbar q -> X
C  404    qbar g -> X
C  405    g q    -> X
C  406    g qbar -> X
C  407    g g    -> X
c Classify as 499 what is not explicitly written here
c
      iprocsum=0
      nevS=0
      nevH=0
      do i=1,mxlproc
        iprocsum=iprocsum+idups_proc(i,-1)
        id1=idups_proc(i,1)
        id2=idups_proc(i,2)
        if(isalquark(id1).and.isalquark(id2))then
          if(id1.gt.0.and.id2.lt.0)then
            ihpro=401
          elseif(id1.lt.0.and.id2.gt.0)then
            ihpro=403
          else
            ihpro=499
          endif
        elseif(isagluon(id1).and.isagluon(id2))then
          ihpro=407
        elseif(isagluon(id1))then
          if(.not.isalquark(id2))then
            write(*,*)'Error #1 in finalizeprocesses()',id1,id2
            stop
          endif
          if(id2.gt.0)then
            ihpro=405
          elseif(id2.lt.0)then
            ihpro=406
          endif
        elseif(isagluon(id2))then
          if(.not.isalquark(id1))then
            write(*,*)'Error #2 in finalizeprocesses()',id1,id2
            stop
          endif
          if(id1.gt.0)then
            ihpro=402
          elseif(id1.lt.0)then
            ihpro=404
          endif
        else
          write(*,*)'Unknown case #1 in finalizeprocesses()',id1,id2
          stop
        endif
        if(idups_proc(i,0).eq.minnp)then
          idups_Sproc_HW6(ihpro,-1)=idups_Sproc_HW6(ihpro,-1)+
     #                               idups_proc(i,-1)
          nevS=nevS+idups_proc(i,-1)
        elseif(idups_proc(i,0).eq.maxnp)then
          idups_Hproc_HW6(ihpro,-1)=idups_Hproc_HW6(ihpro,-1)+
     #                               idups_proc(i,-1)
          nevH=nevH+idups_proc(i,-1)
        else
          write(*,*)'Unknown case #2 in finalizeprocesses()',
     #      idups_proc(i,-1),minnp,maxnp
          stop
        endif
      enddo
      iHW6procsum=0
      do i=401,499
        iHW6procsum=iHW6procsum+idups_Sproc_HW6(i,-1)+
     #                          idups_Hproc_HW6(i,-1)
      enddo
      if(iprocsum.ne.iHW6procsum.or.iprocsum.ne.maxevt.or.
     #   maxevt.ne.(nevS+nevH))then
        write(*,*)'Counting is wrong in finalizeprocesses',
     #            iprocsum,iHW6procsum,nevS,nevH,maxevt
        stop
      endif
c
      write(iunit,*)'Statistics for processes'
      write(iunit,*)' S events:',nevS
      do i=401,499
        if(idups_Sproc_HW6(i,-1).ne.0)then
          write(iunit,111)i,idups_Sproc_HW6(i,-1),
     #                    idups_Sproc_HW6(i,-1)/dfloat(nevS),
     #                    idups_Sproc_HW6(i,-1)/dfloat(maxevt)
        endif
      enddo
      write(iunit,*)'   '
      write(iunit,*)' H events:',nevH
      do i=401,499
        if(idups_Hproc_HW6(i,-1).ne.0)then
          write(iunit,111)i,idups_Hproc_HW6(i,-1),
     #                    idups_Hproc_HW6(i,-1)/dfloat(nevH),
     #                    idups_Hproc_HW6(i,-1)/dfloat(maxevt)
        endif
      enddo
c
 111  format(1x,i3,1x,i8,2(2x,d14.8))
      return
      end


      function isalquark(id)
      implicit none
      logical isalquark
      integer id
c
      isalquark=abs(id).ge.1.and.abs(id).le.5
      return
      end


      function isagluon(id)
      implicit none
      logical isagluon
      integer id
c
      isagluon=id.eq.21
      return
      end


      subroutine storecolconn(npart,numproc,icolup_eff,numconn)
c Fills common block /ccolconn/ and return numconn, the number of the current
c colour connection in the list of connections icolups_proc.
c This routine works at fixed process number numproc
      implicit none
      integer npart,numproc,numconn,icolup_eff(2,22)
      integer i,j,ic,newline,jline(501:510),jcolup(2,22)
      logical exists,found
      integer mxlproc,minnp,maxnp,idups_proc(10000,-1:22)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer icolups_proc(10000,0:500,0:2,22)
      common/ccolconn/icolups_proc
c icolups_proc(numproc,0,1,1)=total number of colour connections
c icolups_proc(numproc,n,0,1)=number of identical connections identified by n
c icolups_proc(numproc,n,1,i)=ICOLUP(1,*) of particle #i
c icolups_proc(numproc,n,2,i)=ICOLUP(2,*) of particle #i
c
      if(npart.ne.idups_proc(numproc,0))then
        write(*,*)'Error #1 in storecolconn',
     #            npart,numproc,idups_proc(numproc,0)
        stop
      endif
c Write colour connections in a standard way: the n^th line has number 40n
c (or 4n if n.ge.10). Lines are counted starting from the colour of particle #1,
c then anticolour of particle #1, and so forth
      do i=501,510
        jline(i)=0
      enddo
      newline=401
      do i=1,npart
        do j=1,2
          if(icolup_eff(j,i).eq.0)then
            jcolup(j,i)=0
          else
            if(jline(icolup_eff(j,i)).eq.0)then
              jline(icolup_eff(j,i))=newline
              jcolup(j,i)=newline
              newline=newline+1
            else
              jcolup(j,i)=jline(icolup_eff(j,i))
            endif
          endif
        enddo
      enddo
      if(newline.lt.401.or.newline.gt.410)then
        write(*,*)'Error #2 in storecolconn',newline
        stop
      endif
c
      exists=.false.
      ic=1
      do while(.not.exists.and.ic.le.icolups_proc(numproc,0,1,1))
        found=.true.
        do i=1,npart
          do j=1,2
            found=found.and.icolups_proc(numproc,ic,j,i).eq.jcolup(j,i)
          enddo
        enddo
        exists=exists.or.found
        ic=ic+1
      enddo
c
      if(.not.exists)then
        icolups_proc(numproc,0,1,1)=icolups_proc(numproc,0,1,1)+1
        if(icolups_proc(numproc,0,1,1).gt.500)then
          write(*,*)'Error #3 in storecolconn: too many connections'
          stop
        endif
        numconn=icolups_proc(numproc,0,1,1)
        icolups_proc(numproc,numconn,0,1)=1
        do i=1,npart
          do j=1,2
            icolups_proc(numproc,numconn,j,i)=jcolup(j,i)
          enddo
        enddo
      else
        numconn=max(1,ic-1)
        icolups_proc(numproc,numconn,0,1)=
     #     icolups_proc(numproc,numconn,0,1)+1
      endif
c
      return
      end


      subroutine checkcolconn(iev,numproc,numconn,wrong)
      implicit none
      integer iev,numproc,numconn
      logical wrong
      integer npart,i,j,icol,iacl,iid,ncol1,ncol2,nacl1,nacl2,nneg
      integer mxlproc,minnp,maxnp,idups_proc(10000,-1:22)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer icolups_proc(10000,0:500,0:2,22)
      common/ccolconn/icolups_proc
c
      npart=idups_proc(numproc,0)
      wrong=.false.
      i=1
      do while(.not.wrong.and.i.le.npart)
        icol=icolups_proc(numproc,numconn,1,i)
        iacl=icolups_proc(numproc,numconn,2,i)
        iid=idups_proc(numproc,i)
        if( (iid.eq.21.and.(icol.eq.0.or.iacl.eq.0)) .or.
     #      (iid.ge.1.and.iid.le.6.and.(icol.eq.0.or.iacl.ne.0)) .or.
     #      (iid.le.-1.and.iid.ge.-6.and.(iacl.eq.0.or.icol.ne.0)) .or.
     #      (abs(iid).gt.6.and.iid.ne.21.and.
     #       (iacl.ne.0.or.icol.ne.0)) )wrong=.true.
        if(wrong)goto 100
c Find partner(s) of particle i. If the colour of particle i is attached to
c the colour (anticoulour) of particle n, then ncol1=n (ncol2=n).
c If the anticolour of particle i is attached to the colour (anticoulour) 
c of particle n, then nacl1=n (nacl2=n)
        ncol1=-1
        ncol2=-1
        nacl1=-1
        nacl2=-1
        do j=1,npart
          if(j.eq.i)goto 123
          if(icol.ne.0 .and.
     #       icolups_proc(numproc,numconn,1,j).eq.icol)then
            if(ncol1.gt.0)wrong=.true.
            ncol1=j
          endif
          if(icol.ne.0 .and.
     #       icolups_proc(numproc,numconn,2,j).eq.icol)then
            if(ncol2.gt.0)wrong=.true.
            ncol2=j
          endif
          if(iacl.ne.0 .and.
     #       icolups_proc(numproc,numconn,1,j).eq.iacl)then
            if(nacl1.gt.0)wrong=.true.
            nacl1=j
          endif
          if(iacl.ne.0 .and.
     #       icolups_proc(numproc,numconn,2,j).eq.iacl)then
            if(nacl2.gt.0)wrong=.true.
            nacl2=j
          endif
 123      continue
        enddo
        if(wrong)goto 100
        nneg=0
        if(ncol1.lt.0)nneg=nneg+ncol1
        if(ncol2.lt.0)nneg=nneg+ncol2
        if(nacl1.lt.0)nneg=nneg+nacl1
        if(nacl2.lt.0)nneg=nneg+nacl2
        if( (abs(iid).ge.1.and.abs(iid).le.6.and.nneg.ne.-3) .or.
     #      (iid.eq.21.and.nneg.ne.-2) .or.
     #      (ncol1.gt.0.and.ncol2.gt.0) .or.
     #      (nacl1.gt.0.and.nacl2.gt.0) )wrong=.true.
        if(wrong)goto 100
c Initial(final) state colour is connected to initial(final) state colour
        if( icol.gt.0 .and.
     #      ( (i.le.2.and.ncol1.ge.1.and.ncol1.le.2) .or.
     #        (i.ge.3.and.ncol1.ge.3) ) )wrong=.true.
c Initial(final) state colour is connected to final(initial) state anticolour
        if( icol.gt.0 .and.
     #      ( (i.le.2.and.ncol2.ge.3) .or.
     #        (i.ge.3.and.ncol2.ge.1.and.ncol2.le.2) ) )wrong=.true.
c Initial(final) state anticolour is connected to initial(final) state anticolour
        if( iacl.gt.0 .and.
     #      ( (i.le.2.and.nacl2.ge.1.and.nacl2.le.2) .or.
     #        (i.ge.3.and.nacl2.ge.3) ) )wrong=.true.
c Initial(final) state anticolour is connected to final(initial) state colour
        if( iacl.gt.0 .and.
     #      ( (i.le.2.and.nacl1.ge.3) .or.
     #        (i.ge.3.and.nacl1.ge.1.and.nacl1.le.2) ) )wrong=.true.
        i=i+1
 100    continue
      enddo
      return
      end


      subroutine setdecmat()
c Assume up to 20 intermediate states (called mothers) will be present in 
c the LH file, each with up to 40 decay channels, each with up to 5 decay
c products (called daughters). Maximum PDG code is 25
c  nmothers(0) = number of mothers (<=20)
c  nmothers(PDG) = number of mothers with PDG code = PDG
c  nch(i) = number of decay channels of mother #i (<=40)
c  ndec(j,i) = number of daughters in decay channel #j of mother #i (<=5)
c  jmo(i) = PDG code of mother #i
c  jda(k,j,i) = PDG code of daughter #k in decay channel #j of mother #i
c  ndecev(j,i) = number of events in decay channel #j of mother #i
c  ndecev(0,i) = total number of decays for mother #i
c  ndeccor(j,j1,i,i1) = number of events in decay channels #j and #j1
c                       of mothers #i and #i1 (for correlations)
      implicit none
      integer i,j,k,i1,j1
      integer nmothers(-25:25),nch(20),ndec(40,20)
      integer jmo(20),jda(5,40,20),ndecev(0:40,20)
      integer ndeccor(0:40,0:40,20,20)
      common/cdec/nmothers,nch,ndec,jmo,jda,ndecev,ndeccor
c
      nmothers(0)=0
      do i=1,20
        jmo(i)=0
        nch(i)=0
        do j=1,40
          ndec(j,i)=0
          ndecev(j,i)=0
          do k=-25,25
            if(i.eq.1.and.j.eq.1)nmothers(k)=0
            if(k.ge.1.and.k.le.5)jda(k,j,i)=0
          enddo
        enddo
      enddo
      do j=0,40
        do j1=0,40
          do i=1,20
            do i1=1,20
              ndeccor(j,j1,i,i1)=0
            enddo
          enddo
        enddo
      enddo
      return
      end


      subroutine findmothers(NUP,IDUP,ISTUP)
      implicit none
      integer i,k,itot,ipart,nmotmp(-25:25)
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDUP(MAXNUP),ISTUP(MAXNUP)
      integer nmothers(-25:25),nch(20),ndec(40,20)
      integer jmo(20),jda(5,40,20),ndecev(0:40,20)
      integer ndeccor(0:40,0:40,20,20)
      common/cdec/nmothers,nch,ndec,jmo,jda,ndecev,ndeccor
c
      itot=0
      do k=-25,25
        nmotmp(k)=0
      enddo
      do ipart=1,NUP
        if(ISTUP(ipart).eq.2)then
          if(nmothers(0).eq.0)then
c First event on record: save all those found
            itot=itot+1
            jmo(itot)=IDUP(ipart)
            nmothers(IDUP(ipart))=nmothers(IDUP(ipart))+1
          else
c Other events: save temporary variable, check later if overlap
            nmotmp(IDUP(ipart))=nmotmp(IDUP(ipart))+1
          endif
        endif
      enddo
c
      if(nmothers(0).eq.0)then
        nmothers(0)=itot
        return
      endif
c
      do k=-25,25
        if(k.ne.0.and.nmotmp(k).gt.nmothers(k))then
          nmothers(0)=nmothers(0)+(nmotmp(k)-nmothers(k))
          nmothers(k)=nmotmp(k)
          do i=nmothers(k)+1,nmotmp(k)
            jmo(i)=k
          enddo
        endif
      enddo
c
      return
      end


      subroutine finddaughters(NUP,IDUP,ISTUP,MOTHUP)
c Assumes decay products in an LH file have identical MOTHUP
      implicit none
      integer ipart,nmax,i,imo,jch,k,itmp,itot,imo1,imo2
      integer ipos0(50),ipos1(50),imotmp(50),idatmp(50),
     # jdatmp(5),nset(20,0:6),jchcor(20)
      logical found,kept(20)
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP)
      integer nmothers(-25:25),nch(20),ndec(40,20)
      integer jmo(20),jda(5,40,20),ndecev(0:40,20)
      integer ndeccor(0:40,0:40,20,20)
      common/cdec/nmothers,nch,ndec,jmo,jda,ndecev,ndeccor
c
      itot=0
c Assumes up to 50 particles in LH file
      do ipart=1,50
        ipos0(ipart)=0
        imotmp(ipart)=0
        ipos1(ipart)=0
        idatmp(ipart)=0
      enddo
      do ipart=1,NUP
        if( ISTUP(ipart).gt.0 .and.
     #      MOTHUP(1,ipart).eq.MOTHUP(2,ipart) )then
          if(ISTUP(MOTHUP(1,ipart)).ne.2)then
            write(*,*)'Error #1 in finddaughters'
            write(*,*)ipart,MOTHUP(1,ipart),MOTHUP(2,ipart),
     #                ISTUP(MOTHUP(1,ipart))
            stop
          endif
          itot=itot+1
          ipos0(itot)=MOTHUP(1,ipart)
          imotmp(itot)=IDUP(ipos0(itot))
          ipos1(itot)=ipart
          idatmp(itot)=IDUP(ipos1(itot))
        endif
      enddo
c Sanity check: all mothers types must have been found before
      if(itot.eq.0)then
        write(*,*)'Error #2 in finddaughters',itot
        stop
      endif
      do i=1,itot
        if(nmothers(imotmp(i)).eq.0)then
          write(*,*)'Error #3 in finddaughters'
          write(*,*)i,imotmp(i),nmothers(imotmp(i))
          stop
        endif
      enddo
c Collect sets of daughters with same mother. Conventions:
c  nset(*,0)   = # of decay products
c  nset(*,1)   = position of mother
c  nset(*,k+1) = position of daughter #k, 1<=k<=nset(*,0)
c Hence, each line of nset will describe a decay chain in the form
c  (#daughters,pos_of_mother,pos_of_daughter1,..,pos_of_daughtern)
c where the positions are those in the original event.
c At the end of the day, nmax will therefore be the total number
c of decays in the event under analysis
      nmax=1
      nset(nmax,0)=1
      nset(nmax,1)=ipos0(1)
      nset(nmax,2)=ipos1(1)
      do ipart=2,itot
        found=.false.
        imo=0
        dowhile(imo.lt.nmax.and.(.not.found))
          imo=imo+1
          if(nset(imo,1).eq.ipos0(ipart))found=.true.
        enddo
        if(found)then
          nset(imo,0)=nset(imo,0)+1
          nset(imo,nset(imo,0)+1)=ipos1(ipart)
        else
          nmax=nmax+1
          nset(nmax,0)=1
          nset(nmax,1)=ipos0(ipart)
          nset(nmax,2)=ipos1(ipart)
        endif
      enddo
c Sanity check: no more mothers than found before (can be more stringent,
c and impose equality if mothers are always written on file, and are
c always the same)
      if(nmax.gt.nmothers(0))then
        write(*,*)'Error #4 in finddaughters'
        write(*,*)nmax,nmothers(0)
        stop
      endif
c
      do i=1,nmothers(0)
        kept(i)=.false.
      enddo
      do imo=1,nmax
        i=0
        found=.false.
        dowhile( i.lt.nmothers(0).and.(.not.found) )
          i=i+1
          if(IDUP(nset(imo,1)).eq.jmo(i).and.(.not.kept(i)))then
            kept(i)=.true.
            found=.true.
          endif
        enddo
        if(.not.found)then
          write(*,*)'Error #5 in finddaughters'
          write(*,*)imo,i
          stop
        endif
c Now nset(imo,1) matches jmo(i). Fill nch, ndec, jda and ndecev, or 
c find match and fill what appropriate
        do k=1,nset(imo,0)
          jdatmp(k)=IDUP(nset(imo,k+1))
        enddo
        call intorder(jdatmp,nset(imo,0))
        if(nch(i).eq.0)then
          nch(i)=1
          ndec(nch(i),i)=nset(imo,0)
          do k=1,ndec(nch(i),i)
            jda(k,nch(i),i)=jdatmp(k)
          enddo
          ndecev(nch(i),i)=ndecev(nch(i),i)+1
          jchcor(imo)=nch(i)
        else
          jch=0
          found=.false.
          dowhile(jch.lt.nch(i).and.(.not.found))
            jch=jch+1
            if(ndec(jch,i).eq.nset(imo,0))then
              found=.true.
              do k=1,ndec(jch,i)
                found=found.and.jda(k,jch,i).eq.jdatmp(k)
              enddo
            endif
          enddo
          if(found)then
            ndecev(jch,i)=ndecev(jch,i)+1
            jchcor(imo)=jch
          else
            nch(i)=nch(i)+1
            ndec(nch(i),i)=nset(imo,0)
            do k=1,ndec(nch(i),i)
              jda(k,nch(i),i)=jdatmp(k)
            enddo
            ndecev(nch(i),i)=ndecev(nch(i),i)+1
            jchcor(imo)=nch(i)
          endif
        endif
      enddo
c
      do imo1=1,nmax
        do imo2=1,nmax
          ndeccor(jchcor(imo1),jchcor(imo2),imo1,imo2)=
     #      ndeccor(jchcor(imo1),jchcor(imo2),imo1,imo2)+1
        enddo
      enddo
c
      return
      end


      subroutine finalizedecays()
      implicit none
      integer dfile,i,j,k,i1,j1,isum
      integer j0,idecch(40),isorted0(40),isorted(40,20),jdatmp(5)
c Sort array of results: ismode<0 for integer, isway=0 for ascending order
      integer ismode,isway,izero
      parameter (ismode=-1)
      parameter (isway=0)
      parameter (izero=0)
      integer nmothers(-25:25),nch(20),ndec(40,20)
      integer jmo(20),jda(5,40,20),ndecev(0:40,20)
      integer ndeccor(0:40,0:40,20,20)
      common/cdec/nmothers,nch,ndec,jmo,jda,ndecev,ndeccor
c
      dfile=74
      open (unit=dfile,file='LHEF.decays',status='unknown')
c
      write(74,'(a)')' mother #      PDG code'
      do i=1,nmothers(0)
        write(74,100)i,jmo(i)
      enddo
c 
      do i=1,nmothers(0)
        ndecev(0,i)=0
        do j=1,nch(i)
          ndecev(0,i)=ndecev(0,i)+ndecev(j,i)
        enddo
c Use the following in order to avoid divisions by zero
        ndecev(0,i)=max(ndecev(0,i),1)
c Sort decay channels
        if(nch(i).gt.1)then
          do j=1,nch(i)
            idecch(j)=0
            do k=1,ndec(j,i)
              jdatmp(k)=abs(jda(k,j,i))
            enddo
            call intorder(jdatmp,ndec(j,i))
            do k=1,ndec(j,i)
              idecch(j)=idecch(j)+10**(2*(k-1))*jdatmp(k)
            enddo
          enddo
          call sortzv(idecch,isorted0,nch(i),ismode,isway,izero)
          do j=1,nch(i)
            isorted(j,i)=isorted0(j)
          enddo
        else
            isorted(1,i)=1
        endif
        write(74,'(a)')' '
        write(74,'(a)')'-------------------------------------------'
        write(74,101)' mother #',i,jmo(i)
        write(74,'(a)')' decay channel   decay products'
        do j=1,nch(i)
          j0=isorted(j,i)
          if(ndec(j,i).eq.2)then
            write(74,112)j,(jda(k,j0,i),k=1,ndec(j,i)),ndecev(j0,i),
     #                   dfloat(ndecev(j0,i))/dfloat(ndecev(0,i))
          elseif(ndec(j,i).eq.3)then
            write(74,113)j,(jda(k,j0,i),k=1,ndec(j,i)),ndecev(j0,i),
     #                   dfloat(ndecev(j0,i))/dfloat(ndecev(0,i))
          elseif(ndec(j,i).eq.4)then
            write(74,114)j,(jda(k,j0,i),k=1,ndec(j,i)),ndecev(j0,i),
     #                   dfloat(ndecev(j0,i))/dfloat(ndecev(0,i))
          elseif(ndec(j,i).eq.5)then
            write(74,115)j,(jda(k,j0,i),k=1,ndec(j,i)),ndecev(j0,i),
     #                   dfloat(ndecev(j0,i))/dfloat(ndecev(0,i))
          else
            write(*,*)'Error in #1 finalizedecays()',
     #                i,j,nch(i),ndec(j,i)
            stop
          endif
        enddo
      enddo
c Correlations
      write(74,'(a)')' '
      write(74,'(a)')' '
      write(74,'(a)')' '
      write(74,'(a)')'*******************************************'
      write(74,'(a)')'   CORRELATIONS'
      write(74,'(a)')'*******************************************'
      write(74,'(a)')' '
      do i=1,nmothers(0)
        do i1=1,nmothers(0)
          if(i.eq.i1)goto 999
c
          do j=1,nch(i)
            ndeccor(j,0,i,i1)=0
            do j1=1,nch(i1)
              ndeccor(j,0,i,i1)=ndeccor(j,0,i,i1)+ndeccor(j,j1,i,i1)
            enddo
          enddo
c Sanity check (perhaps fragile if some decays are missing)
          isum=0
          do j=1,nch(i)
            isum=isum+ndeccor(j,0,i,i1)
          enddo
          isum=max(isum,1)
          if(isum.ne.ndecev(0,i))then
            write(*,*)'Error in #2 finalizedecays()',
     #                i,i1,isum,ndecev(0,i)
            stop
          endif
c
          write(74,'(a)')' '
          write(74,'(a)')'-------------------------------------------'
          write(74,301)' mothers #',i,i1,jmo(i),jmo(i1)
          write(74,'(a)')' '
          write(74,302)(j1,j1=1,nch(i1))
          do j=1,nch(i)
            write(74,303)j,(ndeccor(isorted(j,i),isorted(j1,i1),i,i1),
     #                      j1=1,nch(i1))
          enddo
          write(74,'(a)')' '
          write(74,'(a)')' Normalized to total '
          write(74,302)(j1,j1=1,nch(i1))
          do j=1,nch(i)
            write(74,304)j,
     #        (dfloat(ndeccor(isorted(j,i),isorted(j1,i1),i,i1))/
     #         dfloat(isum),j1=1,nch(i1))
          enddo
          write(74,'(a)')' '
          write(74,'(a)')' Normalized to rows '
          write(74,302)(j1,j1=1,nch(i1))
          do j=1,nch(i)
            write(74,304)j,
     #        (dfloat(ndeccor(isorted(j,i),isorted(j1,i1),i,i1))/
     #         dfloat(ndeccor(isorted(j,i),0,i,i1)),j1=1,nch(i1))
          enddo
c
 999      continue
        enddo
      enddo


c
 100  format(3x,i2,12x,i3)
 101  format((a),i2,4x,i3)
 112  format(5x,i2,10x,2(1x,i3),10x,i9,2x,f10.6)
 113  format(5x,i2,10x,3(1x,i3),10x,i9,2x,f10.6)
 114  format(5x,i2,10x,4(1x,i3),10x,i9,2x,f10.6)
 115  format(5x,i2,10x,5(1x,i3),10x,i9,2x,f10.6)
 301  format((a),i2,1x,i2,4x,i3,1x,i3)
 302  format(1x,2x,20(2x,i9))
 303  format(1x,i2,20(2x,i9))
 304  format(1x,i2,20(2x,f9.5))
      return
      end


      subroutine intorder(iset,imax)
c Orders the first imax entries of iset
      implicit none
      integer imax,iset(5)
      integer i,j,itmp
c
      do i=imax,2,-1
        do j=1,i-1
          if(iset(j).gt.iset(j+1))then
            itmp=iset(j+1)
            iset(j+1)=iset(j)
            iset(j)=itmp
          endif
        enddo
      enddo
      return
      end


      subroutine checkmothers()
      implicit none
      integer k,itmp
      integer nmothers(-25:25),nch(20),ndec(40,20)
      integer jmo(20),jda(5,40,20),ndecev(0:40,20)
      integer ndeccor(0:40,0:40,20,20)
      common/cdec/nmothers,nch,ndec,jmo,jda,ndecev,ndeccor
c
      itmp=0
      do k=-25,25
        if(k.ne.0)itmp=itmp+nmothers(k)
      enddo
      if(nmothers(0).ne.itmp)then
        write(*,*)'Error in checkmothers:',nmothers(0),itmp
        stop
      endif
      return
      end



      subroutine phspncheck_nocms2(nev,npart,xmass,xmom)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer nev,npart,maxmom
      include "nexternal.inc"
      real*8 xmass(3*nexternal),xmom(0:3,3*nexternal)
      real*8 tiny,vtiny,xm,xlen4,den,xsum(0:3),xsuma(0:3),
     # xrat(0:3),ptmp(0:3)
      parameter (tiny=5.d-3)
      parameter (vtiny=1.d-6)
      integer jflag,i,j,jj
      double precision dot
      external dot
c
      jflag=0
      do i=0,3
        xsum(i)=-xmom(i,1)-xmom(i,2)
        xsuma(i)=abs(xmom(i,1))+abs(xmom(i,2))
        do j=3,npart
          xsum(i)=xsum(i)+xmom(i,j)
          xsuma(i)=xsuma(i)+abs(xmom(i,j))
        enddo
        if(xsuma(i).lt.1.d0)then
          xrat(i)=abs(xsum(i))
        else
          xrat(i)=abs(xsum(i))/xsuma(i)
        endif
        if(xrat(i).gt.tiny.and.jflag.eq.0)then
          write(*,*)'Momentum is not conserved [nocms]'
          write(*,*)'i=',i
          do j=1,npart
            write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          enddo
          jflag=1
        endif
      enddo
      if(jflag.eq.1)then
        write(*,'(4(d14.8,1x))') (xsum(jj),jj=0,3)
        write(*,'(4(d14.8,1x))') (xrat(jj),jj=0,3)
        write(*,*)'event #',nev
        stop
      endif
c
      do j=1,npart
        do i=0,3
          ptmp(i)=xmom(i,j)
        enddo
        xm=xlen4(ptmp)
        if(ptmp(0).ge.1.d0)then
          den=ptmp(0)
        else
          den=1.d0
        endif
        if(abs(xm-xmass(j))/den.gt.tiny .and.
     &       abs(xm-xmass(j)).gt.tiny)then
          write(*,*)'Mass shell violation [nocms]'
          write(*,*)'j=',j
          write(*,*)'mass=',xmass(j)
          write(*,*)'mass computed=',xm
          write(*,'(4(d14.8,1x))') (xmom(jj,j),jj=0,3)
          write(*,*)'event #',nev
          stop
        endif
      enddo

      return
      end


      double precision function dot(p1,p2)
C****************************************************************************
C     4-Vector Dot product
C****************************************************************************
      implicit none
      double precision p1(0:3),p2(0:3)
      dot=p1(0)*p2(0)-p1(1)*p2(1)-p1(2)*p2(2)-p1(3)*p2(3)

      if(dabs(dot).lt.1d-6)then ! solve numerical problem 
         dot=0d0
      endif

      end


      function xlen4(v)
      implicit none
      real*8 xlen4,tmp,v(0:3)
c
      tmp=v(0)**2-v(1)**2-v(2)**2-v(3)**2
      xlen4=sign(1.d0,tmp)*sqrt(abs(tmp))
      return
      end


*
* $Id: sortzv.F,v 1.1.1.1 1996/02/15 17:49:50 mclareni Exp $
*
* $Log: sortzv.F,v $
* Revision 1.1.1.1  1996/02/15 17:49:50  mclareni
* Kernlib
*
*
c$$$#include "kerngen/pilot.h"
      SUBROUTINE SORTZV (A,INDEX,N1,MODE,NWAY,NSORT)
C
C CERN PROGLIB# M101    SORTZV          .VERSION KERNFOR  3.15  820113
C ORIG. 02/10/75
C
      DIMENSION A(N1),INDEX(N1)
C
C
      N = N1
      IF (N.LE.0)            RETURN
      IF (NSORT.NE.0) GO TO 2
      DO 1 I=1,N
    1 INDEX(I)=I
C
    2 IF (N.EQ.1)            RETURN
      IF (MODE)    10,20,30
   10 CALL SORTTI (A,INDEX,N)
      GO TO 40
C
   20 CALL SORTTC(A,INDEX,N)
      GO TO 40
C
   30 CALL SORTTF (A,INDEX,N)
C
   40 IF (NWAY.EQ.0) GO TO 50
      N2 = N/2
      DO 41 I=1,N2
      ISWAP = INDEX(I)
      K = N+1-I
      INDEX(I) = INDEX(K)
   41 INDEX(K) = ISWAP
   50 RETURN
      END
*     ========================================
      SUBROUTINE SORTTF (A,INDEX,N1)
C
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTI (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF (AI.LE.A (I22)) GO TO 3
      INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (A(I22)-A(I222)) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (AI-A(I22)) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      SUBROUTINE SORTTC (A,INDEX,N1)
C
      INTEGER A,AI
      DIMENSION A(N1),INDEX(N1)
C
      N = N1
      DO 3 I1=2,N
      I3 = I1
      I33 = INDEX(I3)
      AI = A(I33)
    1 I2 = I3/2
      IF (I2) 3,3,2
    2 I22 = INDEX(I2)
      IF(ICMPCH(AI,A(I22)))3,3,21
   21 INDEX (I3) = I22
      I3 = I2
      GO TO 1
    3 INDEX (I3) = I33
    4 I3 = INDEX (N)
      INDEX (N) = INDEX (1)
      AI = A(I3)
      N = N-1
      IF (N-1) 12,12,5
    5 I1 = 1
    6 I2 = I1 + I1
      IF (I2.LE.N) I22= INDEX(I2)
      IF (I2-N) 7,9,11
    7 I222 = INDEX (I2+1)
      IF (ICMPCH(A(I22),A(I222))) 8,9,9
    8 I2 = I2+1
      I22 = I222
    9 IF (ICMPCH(AI,A(I22))) 10,11,11
   10 INDEX(I1) = I22
      I1 = I2
      GO TO 6
   11 INDEX (I1) = I3
      GO TO 4
   12 INDEX (1) = I3
      RETURN
      END
*     ========================================
      FUNCTION ICMPCH(IC1,IC2)
C     FUNCTION TO COMPARE TWO 4 CHARACTER EBCDIC STRINGS - IC1,IC2
C     ICMPCH=-1 IF HEX VALUE OF IC1 IS LESS THAN IC2
C     ICMPCH=0  IF HEX VALUES OF IC1 AND IC2 ARE THE SAME
C     ICMPCH=+1 IF HEX VALUES OF IC1 IS GREATER THAN IC2
      I1=IC1
      I2=IC2
      IF(I1.GE.0.AND.I2.GE.0)GOTO 40
      IF(I1.GE.0)GOTO 60
      IF(I2.GE.0)GOTO 80
      I1=-I1
      I2=-I2
      IF(I1-I2)80,70,60
 40   IF(I1-I2)60,70,80
 60   ICMPCH=-1
      RETURN
 70   ICMPCH=0
      RETURN
 80   ICMPCH=1
      RETURN
      END

c Dummy subroutine (normally used with vegas when resuming plots)
      subroutine resume()
      end
