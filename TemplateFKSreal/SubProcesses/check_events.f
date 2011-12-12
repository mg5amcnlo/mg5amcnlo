      program check_events
c Checks self-consistency of event files. Compile with
c g77 -o check_events check_events.f handling_lhe_events.f
c With some work on finalizeprocesses(), it should work also for 
c LH files created by Herwig, assuming they are identified by a 
c negative number of events
      implicit none
      integer maxevt,ifile,efile,jfile,kfile,rfile,i,npart,iuseres_1
      double precision chtot,xint,xinterr
      double precision charges(-100:100),zmasses(1:100)
      integer nevS_lhe,nevH_lhe,npartS_lhe,npartH_lhe,
     # itoterr,numproc,numconn,idup_eff(10),icolup_eff(2,10)
      logical wrong
      integer mxlproc,minnp,maxnp,idups_proc(1000,-1:10)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer idups_Sproc_HW6(401:499,-1:10),
     #        idups_Hproc_HW6(401:499,-1:10)
      common/cHW6processes/idups_Sproc_HW6,idups_Hproc_HW6
      integer icolups_proc(1000,0:500,0:2,10)
      common/ccolconn/icolups_proc
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      double precision sum_wgt,err_wgt,toterr,diff
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
      character*3 ch3
      character*10 MonteCarlo
      character*2 ch2
      character*1 ch1
      logical AddInfoLHE,rwgtinfo,unweighted,keepevent,shower

      include "genps.inc"
      integer j,k
      real*8 ecm,xmass(nexternal),xmom(0:3,nexternal)

      include 'reweight0.inc'
      integer kr,kf,kpdf
      double precision sum_wgt_resc_scale(maxscales,maxscales),
     # sum_wgt_resc_pdf(0:maxPDFs)


      call setcharges(charges)
      call setmasses(zmasses)
      mxlproc=0
      minnp=1000
      maxnp=-1
      do i=1,1000
        icolups_proc(i,0,1,1)=0
      enddo
      do i=401,499
        idups_Sproc_HW6(i,-1)=0
        idups_Hproc_HW6(i,-1)=0
      enddo

      write (*,*) 'Enter event file name'
      read (*,*) event_file

      write (*,*) 'Enter 0 to get integrals from res_1'
      write (*,*) '      1 otherwise'
      read (*,*) iuseres_1

      ifile=34
      open (unit=ifile,file=event_file,status='old')
      if(iuseres_1.eq.0)then
c Probably fragile. Assumes res_1 is in the form
c  string  tot_abs +/- err_abs
c  string  tot     +/- err
        jfile=50
        open (unit=jfile,file='res_1',status='old')
        read(jfile,'(a)')buff
        read(buff(index(buff,':')+1:index(buff,'+/-')-1),*) xint
        read(buff(index(buff,'+/-')+3:),*) xinterr
c Discard absolute values
        read(jfile,'(a)')buff
        read(buff(index(buff,':')+1:index(buff,'+/-')-1),*) xint
        read(buff(index(buff,'+/-')+3:),*) xinterr
      elseif(iuseres_1.eq.1)then
        jfile=50
        open (unit=jfile,file='res_wgt',status='unknown')
      else
        write(*,*)'No such option for iuseres_1'
        stop
      endif
      efile=44
      open (unit=efile,file='LHEF.errors',status='unknown')
      kfile=54
      open (unit=kfile,file='LHEF.stats',status='unknown')
      AddInfoLHE=.false.
      rwgtinfo=.false.
      unweighted=.true.
      keepevent=.true.
      shower=.false.

      call read_lhef_header(ifile,maxevt,MonteCarlo)
c Showered LH files have maxevt<0; in that case, it is not the number of
c events, but its upper bound
      if(maxevt.lt.0)then
        write(*,*)'This appears to be a showered LH file'
        shower=.true.
      endif
      maxevt=abs(maxevt)
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
          read(buff,200)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                      fksfather_lhe,ipartner_lhe,
     #                      scale1_lhe,scale2_lhe,
     #                      jwgtinfo,mexternal,iwgtnumpartn,
     #           wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
          if(i.eq.1)then
            if( (jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #          jwgtinfo.eq.8 )rwgtinfo=.true.
            saved_weight=abs(XWGTUP)
          else
            if( ((jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #           jwgtinfo.eq.8) .and. (.not.rwgtinfo) )then
              write(*,*)'Inconsistency #2 in event file',i,' ',buff
              stop
            endif
          unweighted=unweighted.and.
     #               abs(1.d0-abs(XWGTUP)/saved_weight).lt.1.d-5
          endif
        endif

      enddo
      close(34)

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

      call read_lhef_header(ifile,maxevt,MonteCarlo)
      maxevt=abs(maxevt)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
      

      sum_wgt=0d0
      nevS_lhe=0
      npartS_lhe=0
      nevH_lhe=0
      npartH_lhe=0
      itoterr=0
      if(jwgtinfo.eq.8)then
        do kr=1,maxscales
          do kf=1,maxscales
            sum_wgt_resc_scale(kr,kf)=0.d0
          enddo
        enddo
        do kpdf=0,maxPDFs
          sum_wgt_resc_pdf(kpdf)=0.d0
        enddo
      endif

      i=0
      dowhile(i.lt.maxevt.and.keepevent)
         call read_lhef_event_catch(ifile,
     &        NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &        IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)

         if(index(buff,'endoffile').ne.0)then
           keepevent=.false.
           goto 111
         endif

         i=i+1
         sum_wgt=sum_wgt+XWGTUP

         if(jwgtinfo.eq.8)then
           do kr=1,numscales
             do kf=1,numscales
               sum_wgt_resc_scale(kr,kf)=sum_wgt_resc_scale(kr,kf)+
     #                               XWGTUP*wgtxsecmu(kr,kf)/wgtref
             enddo
           enddo
           do kpdf=1,2*numPDFpairs
             sum_wgt_resc_pdf(kpdf)=sum_wgt_resc_pdf(kpdf)+
     #                       XWGTUP*wgtxsecPDF(kpdf)/wgtref
           enddo
         endif

         if(AddInfoLHE)then
           if(buff(1:1).ne.'#')then
             write(*,*)'Inconsistency in event file',i,' ',buff
             stop
           endif
           read(buff,200)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                       fksfather_lhe,ipartner_lhe,
     #                       scale1_lhe,scale2_lhe,
     #                       jwgtinfo,mexternal,iwgtnumpartn,
     #            wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
          if( ((jwgtinfo.eq.0.and.wgtcentral.ne.0.d0) .or.
     #         jwgtinfo.eq.8) .and. (.not.rwgtinfo) )then
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
             if(zmasses(abs(IDUP(k))).eq.-1.d0)then
               zmasses(abs(IDUP(k)))=xmass(npart)
             else
               if(zmasses(abs(IDUP(k))).ne.xmass(npart))then
                 write(44,*)'####event:',i
                 write(44,*)' wrong mass shell',xmass(npart)
                 write(44,*)' for particle',k,
     #                      ' Must be:',zmasses(abs(IDUP(k)))
                 itoterr=itoterr+1
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

c Showered LH files only contain final-state particles.
c Don't check momentum conservation in that case
         if(.not.shower)call phspncheck_nocms2(i,npart,xmass,xmom)

 111     continue

      enddo

      err_wgt=sum_wgt/sqrt(dfloat(maxevt))
      write(*,*)'  '
      write (*,*) 'The total number of events is:',i
      write (*,*) 'The sum of the weights is:',sum_wgt,' +-',err_wgt

      if(iuseres_1.eq.0)then
        toterr=sqrt(xinterr**2+err_wgt**2)
        diff=sum_wgt-xint
        if( (diff.le.0.d0.and.diff+toterr.lt.0.d0) .or.
     #      (diff.gt.0.d0.and.diff-toterr.gt.0.d0) )then
c Error if more that 1sigma away
          itoterr=itoterr+1
          write(44,*)'WEIGHTS'
          write(44,*)'Integral:',xint,' +-',xinterr
          write(44,*)'Weights: ',sum_wgt,' +-',err_wgt
        endif
      elseif(iuseres_1.eq.1)then
        write (50,*) 'The sum of the weights is:',sum_wgt,' +-',err_wgt
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

        if(jwgtinfo.eq.8)then
          write(64,*)'  '
          write(64,*)'Sums of rescaled weights'
          do kr=1,numscales
            do kf=1,numscales
              write(64,*)'scales',kr,kf,' ->',
     #                   sum_wgt_resc_scale(kr,kf)
            enddo
          enddo
          do kpdf=1,2*numPDFpairs
            write(64,*)'PDF',kpdf,' ->',
     #                 sum_wgt_resc_pdf(kpdf)
          enddo
        endif

      endif

      write (*,*) ' '
      write (*,*) 'Total number of errors found:',itoterr

      close(34)
      close(44)
      close(50)
      close(54)
      if(rwgtinfo)close(64)

 200  format(1a,1x,i1,4(1x,i2),2(1x,d14.8),1x,i1,2(1x,i2),5(1x,d14.8))

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
      charges(23)=0.d0
      charges(24)=1.d0
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
      integer npart,numproc,idup_eff(10)
      integer i,j
      logical exists,found
      integer mxlproc,minnp,maxnp,idups_proc(1000,-1:10)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
c mxlproc=current maximum number of different processes
c idups_proc(n,-1)=number of identical processes identified by n
c idups_proc(n,0)=number of particles in process n
c idups_proc(n,i)=ID of particle #i in process n; 1<=i<=idups_proc(n,0)
c
      if(npart.gt.10)then
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
        if(mxlproc.gt.1000)then
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
      integer mxlproc,minnp,maxnp,idups_proc(1000,-1:10)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer idups_Sproc_HW6(401:499,-1:10),
     #        idups_Hproc_HW6(401:499,-1:10)
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
      integer npart,numproc,numconn,icolup_eff(2,10)
      integer i,j,ic,newline,jline(501:510),jcolup(2,10)
      logical exists,found
      integer mxlproc,minnp,maxnp,idups_proc(1000,-1:10)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer icolups_proc(1000,0:500,0:2,10)
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
      integer mxlproc,minnp,maxnp,idups_proc(1000,-1:10)
      common/cprocesses/mxlproc,minnp,maxnp,idups_proc
      integer icolups_proc(1000,0:500,0:2,10)
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


      subroutine phspncheck_nocms2(nev,npart,xmass,xmom)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer nev,npart,maxmom
      include "genps.inc"
      real*8 xmass(nexternal),xmom(0:3,nexternal)
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
