c Same as plot_events.f, but relevant to LH showered files (ntuples).
c Compile with
c   f77 -g -ffixed-line-length-132 -fno-automatic 
c      -o plot_events_lhef plot_events_lhef.f madfks_plot*.f dbook.f 
c         handling_lhe_events.f any-dependencies-in-madfksplot
c Copy in this directory
c  reweight0.inc dbook.inc
      program plot_events
      implicit none
      integer maxevt,ifile,i,npart,itype
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      double precision sum_wgt,sum_wgt_resc,rat,www
      integer isorh_lhe,ifks_lhe,jfks_lhe,fksfather_lhe,ipartner_lhe
      double precision scale1_lhe,scale2_lhe
      double precision wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
      double precision zero
      parameter (zero=0.d0)
      integer ifk88seed
      real*8 rndec(10),fk88random
      common/crndec/rndec
      character*80 event_file
      character*140 buff
      character*9 ch1
      logical AddInfoLHE
      logical keepevent,rescale,doscale
      integer kr,kf,npdfset
      logical usexinteg,mint
      common/cusexinteg/usexinteg,mint
      integer itmax,ncall
      common/citmax/itmax,ncall
      logical unwgt
      double precision evtsgn
      common /c_unwgt/evtsgn,unwgt
      include 'reweight0.inc'

      integer nexternal
      parameter (nexternal=20)
      integer j,k
      real*8 ecm,xmass(3*nexternal),xmom(0:3,3*nexternal)
      character*10 MonteCarlo
      integer isc,ipdf

      usexinteg=.false.
      mint=.false.
      keepevent=.true.
      rescale=.true.
c evtsgn and unwgt used by plotting routines. Sort of fake values here
      evtsgn=0.d0
      unwgt=.false.
      itmax=1
      ncall=1

      ifk88seed=1
      do i=1,10
        rndec(i)=fk88random(ifk88seed)
      enddo

      write(*,*)'Enter event file name'
      read(*,*)event_file

      write(*,*)'Enter (kr,kf) -- central result is (1,1)'
      write(*,*)' Enter (0,0) to skip'
      read(*,*)kr,kf
      if(kr.eq.0.or.kf.eq.0)then
        write(*,*)'Enter number of PDF error set (e.g. 1,..,40)'
        write(*,*)' Enter 0 to skip'
        read(*,*)npdfset
      else
        npdfset=0
      endif
      if(kr.eq.0.and.kf.eq.0.and.npdfset.eq.0)then
        write(*,*)'Will not rescale weights'
        rescale=.false.
      else
        write(*,*)'Will rescale weights'
        if(npdfset.eq.0)then
          doscale=.true.
        else
          doscale=.false.
        endif
      endif


      ifile=34
      open (unit=ifile,file=event_file,status='old')
      AddInfoLHE=.false.

      call read_lhef_header_full(ifile,maxevt,isc,ipdf,MonteCarlo)
      numscales=int(sqrt(dble(isc)))
      numPDFpairs=ipdf/2
c Showered LH files have maxevt<0; in that case, it is not the number of
c events, but its upper bound
      if(maxevt.gt.0)then
        write(*,*)'This is unlikely to be a showered LH file'
        stop
      endif
      maxevt=abs(maxevt)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)

      itype=12
      sum_wgt=0d0
      sum_wgt_resc=0d0
      call initplot
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
         if(rescale)then
           if(doscale)then
             rat=wgtxsecmu(kr,kf)/wgtref
           else
             rat=wgtxsecPDF(npdfset)/wgtref
           endif
           www=XWGTUP*rat
         else
           www=XWGTUP
         endif

         sum_wgt_resc=sum_wgt_resc+www

         if(i.eq.1.and.buff(1:1).eq.'#')AddInfoLHE=.true.
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
         endif

         npart=0
         do k=1,nup
           if(abs(ISTUP(k)).eq.1)then
             npart=npart+1
             xmass(npart)=pup(5,k)
             do j=1,4
               xmom(mod(j,4),npart)=pup(j,k)
             enddo
           endif
         enddo

c Showered LH files only contain final-state particles.
c Don't check momentum conservation

         call outfun(xmom,zero,www,itype)

 111     continue

      enddo
      close(34)

      call mclear
      open(unit=99,file='events.top',status='unknown')
      call topout
      close(99)

      write (*,*) 'The total number of events is:',i
      write (*,*) 'The sum of weights is:',sum_wgt
      if(rescale)
     #  write (*,*) 'The sum of rescaled weights is:',sum_wgt_resc

      end


c Dummy subroutine (normally used with vegas when resuming plots)
      subroutine resume()
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


      subroutine boostwdir2(chybst,shybst,chybstmo,xd,xin,xout)
c chybstmo = chybst-1; if it can be computed analytically it improves
c the numerical accuracy
      implicit none
      real*8 chybst,shybst,chybstmo,xd(1:3),xin(0:3),xout(0:3)
      real*8 tmp,en,pz
      integer i
c
      if(abs(xd(1)**2+xd(2)**2+xd(3)**2-1).gt.1.d-6)then
        write(*,*)'Error #1 in boostwdir2',xd
        stop
      endif
c
      en=xin(0)
      pz=xin(1)*xd(1)+xin(2)*xd(2)+xin(3)*xd(3)
      xout(0)=en*chybst-pz*shybst
      do i=1,3
        xout(i)=xin(i)+xd(i)*(pz*chybstmo-en*shybst)
      enddo
c
      return
      end
