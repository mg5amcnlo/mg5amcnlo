      program sum_html
c*****************************************************************************
c     Program to combine results from all of the different sub amplitudes 
c     and given total  cross section and error.
c*****************************************************************************
      implicit none
c
c     Constants
c
      character*(*) subfile
      parameter (subfile='subproc.mg')
      character*(*) symfile
      parameter (symfile='symfact.dat')
      character*(*) rfile
      parameter (rfile='results.dat')
      integer   max_amps      , max_iter
      parameter (max_amps=9999, max_iter=50)
c
c     local
c
      double precision xsec(max_amps), xerr(max_amps)
      double precision xerru(max_amps),xerrc(max_amps)
      double precision xmax(max_amps), eff(max_amps)
      double precision xlum(max_amps), xlum_min
      double precision ysec_iter(0:max_iter)
      double precision yerr_iter(0:max_iter)
      double precision tsec_iter(0:max_iter)
      double precision terr_iter(0:max_iter)
      double precision f(0:max_iter)
      double precision ysec, yerr, yeff, ymax
      double precision tsec, terr, teff, tmax
      double precision tmean, tsig, chi2
      integer nw(max_amps), nevents(max_amps), maxit
      integer icor(max_amps)
      integer nunwgt(max_amps)
      integer minit
      character*80 fname, pname
      character*80 linkname(max_amps)
      integer i,j,k,l
      double precision xtot,errtot,err_goal, xi
      double precision errtotc, errtotu
      logical correlated
      integer mfact(max_amps)
      integer ntevents, ntw 
      integer ilen
      logical errex

      logical sumproc
      common/to_sumproc/sumproc
c-----
c  Begin Code
c-----
      correlated = .true.
      minit = max_iter
      do i=0,max_iter
         tsec_iter(i)=0d0
         terr_iter(i)=0d0
      enddo
      sumproc = .true.
      open(unit=15,file=symfile,status='old',err=10)
      sumproc=.false.
 10   if (sumproc) then
         open(unit=15,file=subfile,status='old',err=999)
      endif
      errtot=0d0
      errtotu=0d0     !uncorrelated errors
      errtotc=0d0     !correlated errors
      xtot = 0d0
      xlum_min = 1d99
      ntevents = 0
      ntw = 0
      i = 0
      do while (.true.)
         if (sumproc) then
            j = 1
            read(15,*,err=99,end=99) pname
            i=i+1
            ilen = index(pname,' ')-1
c            write(*,*) 'found ilen',ilen,pname
            write(fname,'(a,a,a)') pname(1:ilen),'/',rfile
            write(linkname(i),'(a,a,a)')pname(1:ilen),'/','results.html'
c            write(*,*) i,'found ilen',ilen,fname(1:ilen+10)
         else
            read(15,*,err=99,end=99) xi,j
            if (j .gt. 0) then
            i=i+1
            if ( (xi-int(xi+.01)) .lt. 1d-5) then
               k = int(xi+.01)
            if (k .lt. 10) then
               write(fname,'(a,i1,a,a)') 'G',k,'/',rfile
               write(linkname(i),'(a,i1,a,a)') 'G',k,'/','log.txt'
            else if (k .lt. 100) then
               write(fname,'(a,i2,a,a)') 'G',k,'/',rfile
               write(linkname(i),'(a,i2,a,a)') 'G',k,'/','log.txt'
            else if (k .lt. 1000) then
               write(fname,'(a,i3,a,a)') 'G',k,'/',rfile
               write(linkname(i),'(a,i3,a,a)') 'G',k,'/','log.txt'
            else if (k .lt. 10000) then
               write(fname,'(a,i4,a,a)') 'G',k,'/',rfile
               write(linkname(i),'(a,i4,a,a)') 'G',k,'/','log.txt'
            endif
            else
            if (xi .lt. 10) then
               write(fname,'(a,f5.3,a,a)') 'G',xi,'/',rfile
               write(linkname(i),'(a,f5.3,a,a)') 'G',xi,'/','log.txt'
            else if (xi .lt. 100) then
               write(fname,'(a,f6.3,a,a)') 'G',xi,'/',rfile
               write(linkname(i),'(a,f6.3,a,a)') 'G',xi,'/','log.txt'
            else if (xi .lt. 1000) then
               write(fname,'(a,f7.3,a,a)') 'G',xi,'/',rfile
               write(linkname(i),'(a,f7.3,a,a)') 'G',xi,'/','log.txt'
            else if (xi .lt. 10000) then
               write(fname,'(a,f8.3,a,a)') 'G',xi,'/',rfile
               write(linkname(i),'(a,f8.3,a,a)') 'G',xi,'/','log.txt'
            endif
c            write(*,*) 'log name ',fname
            endif
            endif
         endif


         if (j .gt. 0) then
            nevents(i)=0d0
            xsec(i)=0d0
            xerr(i)=0d0
            xerru(i)=0d0
            xerrc(i)=0d0
            nw(i)  =0d0
            mfact(i)=j
c
c     Read in integration data from run
c
            open(unit=25,file=fname,status='old',err=95)
            read(25,*) xsec(i),xerru(i),xerrc(i),nevents(i),nw(i)
     $           ,maxit,nunwgt(i),xlum(i)
 11         xerr(i) = sqrt(xerru(i)**2+xerrc(i)**2)
            xlum(i)=xlum(i)/mfact(i)
            ntevents=ntevents+nevents(i)
            ntw=ntw+nw(i)
c            maxit = min(maxit,2)
            if (sumproc) then
               write(*,'(a20,e15.5)') pname(2:ilen), xsec(i)
            else
               write(*,*) fname,i,xsec(i),mfact(i)
            endif
            tmax = -1d0
            terr = 0d0
            teff = 0d0
            tsec = 0d0
            if (.true.) then
               k = 0
               ysec_iter(0) = xsec(i)
               do while ( k .le. maxit .and. ysec_iter(k) .gt. 0)
                  k=k+1
                  read(25,*,err=92,end=92) l,ysec_iter(k),yerr_iter(k)
c                  write(*,*) k,ysec_iter(k),yerr_iter(k)
               enddo
 92            maxit = k-1      !In case of error reading file
               if (maxit .gt. 0) then
c
c          Check to see if we need to reduce the number of iterations
c
               if (maxit .lt. minit) then  !need to reset minit
                  do k=1,minit-maxit       !and combine first iterations
                     tsec_iter(1)=tsec_iter(1)+tsec_iter(k+1)
                     terr_iter(1)=terr_iter(1)+terr_iter(k+1)
                  enddo
                  tsec_iter(1)=tsec_iter(1)/(minit-maxit+1)
                  terr_iter(1)=terr_iter(1)/(minit-maxit+1)
                  do k=2,maxit
                     tsec_iter(k) = tsec_iter(k+maxit-1)
                     terr_iter(k) = terr_iter(k+maxit-1)
                     tsec_iter(k+maxit-1) = 0d0
                     terr_iter(k+maxit-1) = 0d0
c                     write(*,*) k+1,k+maxit
                  enddo
                  minit = maxit
               endif
c
c          If this channel has more iterations, combine first few
c          together into 1 
c
               if (maxit .gt. minit) then
               do k=1,maxit-minit
                  ysec_iter(1)=ysec_iter(1)+ysec_iter(k+1)
                  yerr_iter(1)=sqrt(yerr_iter(1)**2+yerr_iter(k+1)**2)
               enddo
               ysec_iter(1)=ysec_iter(1)/(maxit-minit+1)
               yerr_iter(1)=yerr_iter(1)/(maxit-minit+1)
               do k=2,minit
                  ysec_iter(k) = ysec_iter(k+minit-1)
                  yerr_iter(k) = yerr_iter(k+minit-1)
                  ysec_iter(k+minit-1) = 0d0
                  yerr_iter(k+minit-1) = 0d0
               enddo
               endif
c
c              Now add these statistics to our totals for each iteration
c
               do k=1,minit
                  tsec_iter(k)=tsec_iter(k)+ysec_iter(k)*mfact(i)
                  terr_iter(k)=terr_iter(k)+(yerr_iter(k)*mfact(i))**2
c                  write(*,*) k,ysec_iter(k),yerr_iter(k)
               enddo
               endif
c               if (maxit .gt. 0) then
c                  xsec(i)=tsec/maxit
c                  xerr(i)=sqrt(terr)/maxit
c               else
c                  xsec(i)=0d0
c                  xerr(i)=0d0
c               endif
            endif
            if (xsec(i) .gt. 0d0) then
               xlum_min = min(xlum(i),xlum_min)
               xmax(i)=tmax/xsec(i)
               eff(i)= xerr(i)*sqrt(real(nevents(i)))/xsec(i)
            else
               eff(i) = 0d0
            endif
            xtot = xtot+ xsec(i)*mfact(i)
            errtot = errtot+(mfact(i)*xerr(i))**2
c
c     Combine error linearly if correlated, or in quadrature if not.
c
            errtotc = errtotc+xerrc(i)*mfact(i)
            errtotu = errtotu + (mfact(i)*xerru(i))**2
            write(*,*) i, sqrt(errtotu), errtotc

            if (xsec(i) .eq. 0d0) xsec(i)=1d-99 
 95         close(25)
c            do k=1,minit
c               write(*,*) i,k,tsec_iter(k),sqrt(terr_iter(k))
c            enddo
c            write(*,*) i,maxit,xsec(i), eff(i)
         endif         
      enddo
 99   close(15)
      errtot=sqrt(errtotu+errtotc**2)
c      if (sumproc) then
         open(unit=26,file=rfile,status='unknown')
         write(26,'(3e12.5,2i9,i5,i9,e10.3)') xtot,sqrt(errtotu),
     $        errtotc, ntw, minit,0,0,xlum_min
         if (xtot .gt. 0) then
            teff = sqrt(errtotc**2+errtotu**2)*sqrt(real(ntevents))/xtot
         else
            teff = 0d0
         endif
c         write(26,*) minit, xtot,errtot, ntevents, teff
c         write(*,*) minit
         tmean = 0d0
         tsig  = 0d0
         f(0)  = 0d0
         do j=1,minit
            f(j) = tsec_iter(j)**2/terr_iter(j)
c            f(j)=1
            tmean = tmean + tsec_iter(j)*f(j)
            tsig  = tsig + terr_iter(j)*f(j)
            f(0)=f(0)+f(j)
c            write(*,*) 'Iteration',j,tmean/f(0),sqrt(tsig)/f(0)
         enddo
         tmean=tmean/f(0)
         tsig = sqrt(tsig/f(0)/minit)
         chi2 = 0d0
         do j=1,minit
            chi2  = chi2+f(j)*minit*
     &           (tsec_iter(j)-tmean)**2/terr_iter(j)/(f(0)+1d-99)
            write(26,*)j,tsec_iter(j),sqrt(terr_iter(j)),chi2/max(1,j-1)
c            write(*,*) j,tsec_iter(j),sqrt(terr_iter(j)),chi2/max(1,j-1)
         enddo
         write(26,*) tmean,tsig,chi2/max((minit-1),1),
     &        tsig*sqrt(real(ntevents))/tmean
         write(*,*) 'Results', xtot,errtot, ntevents, teff
         close(26)
c      endif
      call write_html(xsec,xerr,eff,xmax,xtot,errtot,mfact,i,nevents,
     $     nunwgt,linkname,xlum)
      if(sumproc.and.xtot.eq.0d0)then
        open(unit=26,file='../error',status='unknown',err=999)
        write(*,'(a)') 'Cross section is 0, try loosening cuts'
        write(26,'(a)') 'Cross section is 0, try loosening cuts'
        close(26)
        stop
      else
        inquire(file="../error",exist=errex)
        if(errex) then
          call system ('rm -f ../error')
        endif
      endif
      stop
 999  write(*,*) 'error'
      end


      subroutine write_html(xsec,xerr,eff,xmax,xtot,errtot,mfact,ng
     $     ,nevents,nw,linkname,xlum)
c*****************************************************************************
c     Writes out HTML table of results for process
c*****************************************************************************
      implicit none
c
c     Constants
c
      character*(*) eventfile
      parameter (eventfile='events.txt>')
      character*(*) logfile
      parameter (logfile='log.txt>')
      character*(*) htmfile
      parameter (htmfile='results.html')
      integer   max_amps
      parameter (max_amps=9999)
c
c     Arguments
c
      double precision xsec(max_amps), xerr(max_amps)
      double precision xmax(max_amps), eff(max_amps),xlum(max_amps)
      integer ng, nevents(max_amps),nw(max_amps)
      double precision xtot,errtot
      integer mfact(max_amps)
      integer nsubproc          !Number of specific processes requested
      logical found
      integer ig
      character*80 linkname(max_amps)
      integer sname(256)
      integer gname
c
c     Local
c
      integer i,j,k, io(max_amps), ik
      integer ntot, ip,jp
      character*40 procname
      character*4 cpref
      character*20 fnamel, fnamee
      double precision scale,xt(max_amps), teff
      double precision subtotal

      logical sumproc
      common/to_sumproc/sumproc

c-----
c  Begin Code
c-----
c
c     Here we determine the appropriate units. Assuming the results 
c     were written in picobarns
c
      if (xtot .ge. 1e4) then         !Use nano barns
         scale=1e-3
         cpref='(nb)'
      elseif (xtot .ge. 1e1) then     !Use pico barns
         scale=1e0
         cpref='(pb)'
      elseif (xtot .ge. 1e-2) then    !Use fempto
         scale=1e+3
         cpref='(fb)'
      else                               !Use Attobarns
         scale=1e+6
         cpref='(ab)'
      endif
      ntot = 0
      do j=1,ng
         io(j) = j
         ntot = ntot+nevents(j)
         xt(j)=xsec(j)*mfact(j)                   !sort by xsec
c         xt(j)= xerr(j)*mfact(j)                   !sort by error
c         xt(j)=ng-j                               !sort by graph
c         write(*,*) j,xt(j),xsec(j)
      enddo
c      write(*,*) 'Number of channels',ng
      call sort2(xt,io,ng)

c      do i=1,ng
c         write(*,*) i,io(i),(xsec(1,io(i))+xsec(2,io(i)))/2d0
c      enddo

      if (xtot .gt. 0d0) then
         teff = errtot*sqrt(real(ntot))/xtot
      else
         teff = 0d0
      endif
cfax 12.04.2006
c      procname='Set caption in file input.dat'
      procname='Process results'
      ip = 1
      jp = 30
      open(unit=15, file='input.dat', status='old',err=11)
      read(15,'(a)',err=11,end=11) procname
 11   close(15)
      open(unit=15, file='dname.mg', status='old',err=12)
      read(15,'(a)',err=12,end=12) procname
cxx   tjs 3-20-2006
c      ip = index(procname,'P')+1  !Strip off first P
      ip = index(procname,'P')+1      !Strip off P
      ip = ip+index(procname(ip:),'_') !Strip off process number
      jp = ip+index(procname(ip:),' ')   !Strip off blanks
 12   close(15)
      open(unit=16,file=htmfile,status='unknown',err=999)      
      write(16,50) '<head><title>'//procname(ip:jp)//'</title></head>'
      write(16,50) '<body>'
      if (.not. sumproc) then
c         write(16,50) '<h2>Results for <a href=diagrams.html>'         
c         write(16,50) procname(ip:jp)//'</a></h2>'
         write(16,*) '<h2><a href=diagrams.html>'//
     &        procname(ip:jp)// '</a> <BR>'
         write(16,*) '<font face=symbol>s</font>='
         write(16,'(f8.3,a,f8.3,a)')xtot*scale,
     &        '<font face=symbol>&#177</font> ', errtot*scale,cpref
         write(16,*) '</center></h2>'
      else
         write(16,*) '<h2>', procname(ip:jp)// " <BR>"
         write(16,*) '<font face=symbol>s</font>='
         write(16,'(f8.3,a,f8.3,a)') xtot*scale,
     &        '<font face=symbol>&#177</font> ', errtot*scale,cpref
         write(16,*) '</center></h2>'
      endif
c
c     Now I want to write out information for each iteration 
c     of the run
c
cfax 12-04-2006
c      write(16,*) '<a href=results.dat> Iteration details </a>'
c      if (sumproc) write(16,*) '<p> <a href=../Events/plots.html> Plots </a>'

c      call gen_run_html(16)
c
c     Next we'll get information on the cuts. This requires linking to
c     cuts.o and also coupsm.o
c
c      call gen_cut_html(16)


      write(16,50) '<table border>'
c      write(16,50) '<Caption> Caption Results'
      write(16,49) '<tr><th>Graph</th>'
      write(16,48) '<th>Cross Sect',cpref,'</th><th>Error',cpref,'</th>' 
      write(16,49) '<th>Events (K)</th><th>Eff</th>'
      write(16,50) '<th>Unwgt</th><th>Luminosity</th></tr>'

      write(16,60) '<tr><th>Sum</th><th>',xtot*scale
     $     ,'</th><th>',errtot*scale,'</th><th align=right>',
     $     ntot/1000,'</th><th align=right>',teff,'</th></tr>'
c
c     Check number of requested processes
c
      if (sumproc) then
         nsubproc=0
         do i=1,ng
            procname = linkname(io(i))(:40)
            gname=0
            read(procname(2:index(procname,'_')-1),*,err=20) gname
 20         found = .false.
            j = 0
            do while(j < nsubproc .and. .not. found)
               j=j+1
               found = (gname .eq. sname(j))
            enddo
            if (.not. found) then
               nsubproc=nsubproc+1
               sname(nsubproc) = gname
c               write(*,*) i,nsubproc, " " , sname(nsubproc)
            endif
         enddo
cfax 12.05.2006
c          nsubproc=nsubproc-1
      else
         nsubproc = 1
      endif
c
c     Now loop through all the subprocesses
c
      subtotal=0d0

      do ig = 1, nsubproc
         if (nsubproc .gt. 1) then
cfax If added 12.05.2006
c          	  if(sname(ig).ne.0) then 
                      write(16,*) '<tr> <td colspan="7" align="center"> Sub Group ',sname(ig), '</td></tr>'
c                  endif
         endif
      do i=1,ng
c         write(*,*) i,io(i),xsec(io(i))
         if(sumproc)
     $        read(linkname(io(i))(2:index(linkname(io(i)),'_')-1),*,err=30) gname
 30      if ((.not. sumproc .and. xsec(io(i)) .ne. 0d0) .or.
     $        (sumproc .and. gname .eq. sname(ig))) then
c            write(*,*) ig," ",i," ",linkname(io(i))(:30)," ", sname(ig)
c
c     Create directory names using the linkname
c
c
            if (.false.) then
            if (io(i) .lt. 10) then
               write(fnamel,'(a,i1,a,a)') 'G',io(i),'/',logfile
               write(fnamee,'(a,i1,a,a)') 'G',io(i),'/',eventfile
            else if (io(i) .lt. 100) then
               write(fnamel,'(a,i2,a,a)') 'G',io(i),'/',logfile
               write(fnamee,'(a,i2,a,a)') 'G',io(i),'/',eventfile
            else if (io(i) .lt. 1000) then
               write(fnamel,'(a,i3,a,a)') 'G',io(i),'/',logfile
               write(fnamee,'(a,i3,a,a)') 'G',io(i),'/',eventfile
            else if (io(i) .lt. 10000) then
               write(fnamel,'(a,i4,a,a)') 'G',io(i),'/',logfile
               write(fnamee,'(a,i4,a,a)') 'G',io(i),'/',eventfile
            endif
            endif

            ik = index(linkname(io(i)),'log.txt')-1
            fnamel = linkname(io(i))(1:ik) // logfile
            fnamee = linkname(io(i))(1:ik) // eventfile
c            write(*,*) i,fnamel,fnamee
            if (.not. sumproc) then

c            write(16,65) '<tr><td align=right>',io(i),
              write(16,65) '<tr><td align=right>',linkname(io(i))(1:ik-1),
     $           ' </td><td align=right><a href='//linkname(io(i))//'>',
     $           xsec(io(i))*mfact(io(i))*scale
     $       ,'</a> </td><td align=right>',
     $           xerr(io(i))*
     $           mfact(io(i))*scale,'</td><td align=right>',
     $           nevents(io(i))/1000,'</td><td align=right>',
     $           eff(io(i)),'</td><td align=right>',
     $           nw(io(i)),'</td><td align=right>',
     $           xlum(io(i))/scale,'</td></tr>'
c            write(*,*) io(i),xmax(io(i))
            else
                procname = linkname(io(i))(:40)
cxx   tjs 3-20-2006  + cfax 12.05.2006
c                ip = index(procname,'P')+2 !Strip off first P_
                ip = index(procname,'P')+1 !Strip off first P
                jp = ip+index(procname(ip:),'/')-2 !Strip off blanks
c                write(*,*) 'Writing out ',sname(ig),"  ",procname(ip:jp)

                if (xsec(io(i)) .ne. 0) then
                subtotal = subtotal+xsec(io(i))*mfact(io(i))*scale
                   write(16,66) '<tr><td align=right> <a href=P'
     $               //procname(ip:jp)//'/diagrams.html >'
     $               //'P'//procname(ip:jp),
     $           ' </a></td><td align=right><a href='
     $               //linkname(io(i))//'>',
     $           xsec(io(i))*mfact(io(i))*scale
     $       ,'</a> </td><td align=right>',
     $           xerr(io(i))*
     $           mfact(io(i))*scale,'</td><td align=right>',
     $           nevents(io(i))/1000,'</td><td align=right>',
     $           eff(io(i)),'</td><td align=right>',
     $           nw(io(i)),'</td><td align=right>',
     $           xlum(io(i))/scale,'</td></tr>'

c                else
c                   write(16,66) '<tr><td align=right> <a href=P_'
c     $               //procname(ip:jp)//'/diagrams.html >'
c     $               //procname(ip:jp),
c     $           ' </a></td><td align=right><a href='
c     $               //linkname(io(i))//'> </td></tr>'
                endif
            endif
          else
c             write(*,*) 'Skipping process',i
         endif
      enddo     !Loop over different groups
      
      if (nsubproc .gt. 1) then
c          	  if(sname(ig).ne.0) then 
                      if(subtotal .ne. 0e0) then 
                         write(16,*) '<tr> <td colspan="7" align="center"> Sub Group total = ',subtotal, '</td></tr>'
                         subtotal=0d0
                      endif 
c                  endif
      endif

      enddo
      write(16,50) '</table></body>'
 48   format(a,a,a,a)
 49   format(a)
 50   format( a)
 60   format( a,f10.3,a,f10.3,a,i10,a,f8.1,a)
c 65   format( a,i4,a,f10.3,a,f10.3,a,i10,a,f8.1,a,i10.0,a,f10.0,a)
 65   format( a,a,a,f10.3,a,f10.3,a,i10,a,f8.1,a,i10.0,a,f10.2,a)
 66   format( a,a,f10.3,a,f10.3,a,i10,a,f8.1,a,i10.0,a,f10.2,a)
      write(*,*) 'Updated results in file ',htmfile
 999  close(16)
      end

      subroutine sort2(array,aux1,n)
      implicit none
! Arguments
      integer n
      integer aux1(n)
      double precision array(n)
!  Local Variables
      integer i,k
      double precision temp
      logical done

!-----------
! Begin Code
!-----------
      do i=n-1,1,-1
         done = .true.
         do k=1,i
            if (array(k) .lt. array(k+1)) then
               temp = array(k)
               array(k) = array(k+1)
               array(k+1) = temp
               temp = aux1(k)
               aux1(k) = aux1(k+1)
               aux1(k+1) = temp
               done = .false.
            end if
         end do
         if (done) return
      end do
      end 

      

      subroutine gen_run_html(lun)
c***************************************************************************
c     Writes out run information in html format
c
c***************************************************************************
      implicit none
c
c     Arguments
c      
      integer lun
c
c     local
c
c
c     Global
c            
c-----
c  Begin Code
c-----
      write(lun,*) 
      end

      subroutine gen_cut_html(lun)
c***************************************************************************
c     Writes out run information in html format
c
c***************************************************************************
      implicit none
c
c     Parameters
c
c      include 'genps.inc'
c
c     Arguments
c
      integer lun
c
c     Local
c
      integer i,j
      real stot
c
c     Global
c

c-----
c  Begin Code
c-----

c      call read_cuts()
c      call write_cuts()  !Writes cuts.dat in parent directory

c
c     Write out collider information table
c
c      stot = 2d0*sqrt(ebeam(1)*ebeam(2))
c      write(lun,*) '<p> <TABLE> <table border=1>'
c      write(lun,*) '<TD> <B> sqrt(s) </B> </TD>'
c      write(lun,*) '<TD> <B> Beam 1 </B> </TD>'
c      write(lun,*) '<TD> <B> Beam 2 </B> </TD>'

c      write(lun,*) '</TR><TR><TD><B>',stot,' GeV </B></TD>'
c      write(lun,*) '<TD> <B>',Ebeam(1),'  GeV </B></TD>'
c      write(lun,*) '<TD> <B>',Ebeam(2),'  GeV </B></TD>'
c      write(lun,*) '</TR> <TR>'
c      write(lun,*) '<TD> </TD>'
c      do i=1,2
c         write(lun,*) '<TD> <center> <B>'
c         if (lpp(i) .eq. 1) then
c            write(lun,*) 'Proton'
c         elseif(lpp(i) .eq. -1) then
c            write(lun,*) 'Antiproton'
c         else
c            write(lun,*) 'No pdf'
c         endif
c         write(lun,*) '</B></center></TD>'
c      enddo
c      write(lun,*) '</TR></table>'



c
c     Now write out the cuts information table
c
c      if (.false.) then
c      write(lun,*) '<p> <TABLE> <table border=1>'      
c
c     Header
c
c      write(lun,*) '<TR> <TD><B> Cuts </B> </TD>'
c      do i=3,nexternal
c         write(lun,*) '<TD> <B> <center>',i
c         write(lun,*) '</center> </B> </TD>'
c      enddo
c
c     PT
c
c      write(lun,*) '<TR> <TD><B> Et > </B> </TD>'
c      do i=3,nexternal
c         write(lun,'(a,f6.0)') '<TD> <B> <right>',etmin(i)
c         write(lun,*) '</right> </B> </TD>'
c      enddo
c
c     Rapidity
c
c      write(lun,*) '<TR> <TD><B> eta <  </B> </TD>'
c      do i=3,nexternal
c         write(lun,'(a,f6.0)') '<TD> <B> <right>',etamax(i)
c         write(lun,*) '</right> </B> </TD>'
c      enddo
c
c     Delta R
c
c      do j=4,nexternal
c         write(lun,*) '<TR> <TD><B> Delta R',j, ' </B> </TD>'
c         do i=3,nexternal
c            if ( i .lt. j) then
c               write(lun,'(a,f6.1)') '<TD> <B> <right>',r2min(j,i)
c            else
c               write(lun,'(a)') '<TD> <B> <right>'
c            endif
c            write(lun,*) '</right> </B> </TD>'
c         enddo
c         write(lun,*) '</TR>'
c      enddo
c      write(lun,*) '</table>'
c      write(lun,*) '<p>'
c      endif

c
c     Now write out the cuts information table
c
c      write(lun,*) '<p> <TABLE> <table border=1>'      
c
c     Header
c
c      write(lun,*) '<TD><B> Particle </B> </TD>'
c      write(lun,*) '<TD><B> Et > </B> </TD>'
c      write(lun,*) '<TD><B> eta < </B> </TD>'
c      do i=3,nexternal-1
c         write(lun,*) '<TD> <B> <center>Delta R',i
c         write(lun,*) '</center> </B> </TD>'
c      enddo
c      write(lun,*) '</TR>'
c
c     PT
c
c      write(lun,*) '<TR> <TD><B> Et > </B> </TD>'
c      do i=3,nexternal
c         write(lun,'(a,i3)') '<TR><TD> <B> <center>',i
c         write(lun,*) '</center> </B> </TD>'
c         write(lun,'(a,f6.0)') '<TD> <B> <right>',etmin(i)
c         write(lun,*) '</right> </B> </TD>'
c         write(lun,'(a,f6.1)') '<TD> <B> <right>',etamax(i)
c         write(lun,*) '</right> </B> </TD>'

c         do j=3,nexternal-1
c            if ( i .gt. j) then
c               write(lun,'(a,f6.1)') '<TD> <B> <right>',r2min(i,j)
c            else
c               write(lun,'(a)') '<TD> <B> <right>'
c            endif
c            write(lun,*) '</right> </B> </TD>'
c         enddo
c         write(lun,*) '</TR>'

c      enddo
c
c     Rapidity
c
c      write(lun,*) '<TR> <TD><B> eta <  </B> </TD>'
c      do i=3,nexternal
c      enddo
c
c     Delta R
c

c      write(lun,*) '</table>'
c      write(lun,*) '<p>'
      end




