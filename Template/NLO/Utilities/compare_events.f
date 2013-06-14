c Compile with
c gfortran -ffixed-line-length-132 -fno-automatic -I../SubProcesses/P0_<anydir>
c -o compare_events compare_events.f handling_lhe_events.f fill_MC_mshell.f
      program compare_events
      implicit none
      integer maxevt,maxevt2,ifile,ifile2,i
      integer IDBMUP(2),PDFGUP(2),PDFSUP(2),IDWTUP,NPRUP,LPRUP
      double precision EBMUP(2),XSECUP,XERRUP,XMAXUP
      integer IDBMUP2(2),PDFGUP2(2),PDFSUP2(2),IDWTUP2,NPRUP2,LPRUP2
      double precision EBMUP2(2),XSECUP2,XERRUP2,XMAXUP2
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP(MAXNUP),ISTUP(MAXNUP),
     # MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP)
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,
     # PUP(5,MAXNUP),VTIMUP(MAXNUP),SPINUP(MAXNUP)
      INTEGER NUP2,IDPRUP2,IDUP2(MAXNUP),ISTUP2(MAXNUP),
     # MOTHUP2(2,MAXNUP),ICOLUP2(2,MAXNUP)
      DOUBLE PRECISION XWGTUP2,SCALUP2,AQEDUP2,AQCDUP2,
     # PUP2(5,MAXNUP),VTIMUP2(MAXNUP),SPINUP2(MAXNUP)
      double precision sum_wgt,sum_wgt2,xtmp,ytmp,xmomshifts(4)
      character*80 event_file,event_file2
      character*10 MonteCarlo,MonteCarlo2
      character*140 buff
      include "nexternal.inc"
      include "genps.inc"
      integer j,k
      real*8 ecm,xmass(3*nexternal),xmom(0:3,3*nexternal)

      write (*,*) 'Enter name of event file #1'
      read (*,*) event_file

      ifile=34
      open (unit=ifile,file=event_file,status='old')

      write (*,*) 'Enter name of event file #2'
      read (*,*) event_file2

      ifile2=35
      open (unit=ifile2,file=event_file2,status='old')

      call read_lhef_header(ifile,maxevt,MonteCarlo)
      call read_lhef_init(ifile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
      call read_lhef_header(ifile2,maxevt2,MonteCarlo2)
      call read_lhef_init(ifile2,
     &     IDBMUP2,EBMUP2,PDFGUP2,PDFSUP2,IDWTUP2,NPRUP2,
     &     XSECUP2,XERRUP2,XMAXUP2,LPRUP2)

      if(MonteCarlo.ne.MonteCarlo2)then
        write(*,*)'Files are not relevant to the same MC:'
        write(*,*)MonteCarlo,MonteCarlo2
        stop
      endif
      if(maxevt.ne.maxevt2)then
        write(*,*)'Files have different number of events',
     #             maxevt,maxevt2
        stop
      endif
      
      sum_wgt=0d0
      sum_wgt2=0d0
      do i=1,4
        xmomshifts(i)=0.d0
      enddo

      do i=1,maxevt
         call read_lhef_event(ifile,
     &        NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &        IDUP,ISTUP,MOTHUP,ICOLUP,PUP,VTIMUP,SPINUP,buff)
         sum_wgt=sum_wgt+XWGTUP

         do k=1,nup
           xmass(k)=pup(5,k)
           do j=1,4
             xmom(mod(j,4),k)=pup(j,k)
           enddo
         enddo
         call phspncheck_nocms2(nup,xmass,xmom)

         call read_lhef_event(ifile2,
     &        NUP2,IDPRUP2,XWGTUP2,SCALUP2,AQEDUP2,AQCDUP2,
     &        IDUP2,ISTUP2,MOTHUP2,ICOLUP2,PUP2,VTIMUP2,SPINUP2,buff)
         sum_wgt2=sum_wgt2+XWGTUP2

         if(nup.ne.nup2)then
           write(*,*)'Mismatch in event #',i,nup,nup2
           stop
         endif

         do k=1,nup2
           xmass(k)=pup2(5,k)
           do j=1,4
             xmom(mod(j,4),k)=pup2(j,k)
           enddo
         enddo
         call phspncheck_nocms2(nup2,xmass,xmom)

         do j=1,4
           ytmp=0.d0
           do k=1,nup
             if(abs(pup(j,k)).lt.1.d0)then
               xtmp=pup(j,k)-pup2(j,k)
             else
               xtmp=(pup(j,k)-pup2(j,k))/pup(j,k)
             endif
             ytmp=ytmp+abs(xtmp)/dfloat(nup)
           enddo
           xmomshifts(j)=(xmomshifts(j)*(i-1)+ytmp)/dfloat(i)
         enddo

      enddo

      write (*,*) 'The sum of weights #1 is:',sum_wgt
      write (*,*) 'The sum of weights #2 is:',sum_wgt2

      write(*,*)'   '
      write(*,*)'Average momentum shifts'
      do i=1,4
        write(*,*)'  ',i,': ',xmomshifts(i)
      enddo

      end


      subroutine phspncheck_nocms2(npart,xmass,xmom)
c Checks four-momentum conservation. Derived from phspncheck;
c works in any frame
      implicit none
      integer npart,maxmom
      include "nexternal.inc"
      include "genps.inc"
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
