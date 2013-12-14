      program test
c*****************************************************************
c     tests traversing directories to find all events
c****************************************************************
      implicit none
c
c     Constants
c
      include 'run_config.inc'
      integer    maxsubprocesses
      parameter (maxsubprocesses=9999)
      integer    cmax_events
      parameter (cmax_events=500000)
      integer    sfnum
      parameter (sfnum=17)   !Unit number for scratch file
      integer    maxexternal
      parameter (maxexternal=17)
      integer maxpara
      parameter (maxpara=1000)
c     
c     Local
c
      character*30 subname(maxsubprocesses)
      integer i,j,m,ns,nreq,ievent
      integer kevent,revent,iarray(cmax_events)
      double precision sum, xsec, xerr, goal_wgt,xarray(cmax_events)
      integer i4,r8,record_length
      integer iseed
      real xran1
      double precision wgt,maxwgt
      double precision p(0:4,maxexternal)
      integer ic(7,maxexternal),n
      double precision scale,aqcd,aqed
      character*20 param(maxpara),value(maxpara)
      integer npara,nunwgt
      double precision xtrunc, min_goal,max_goal
      logical keep(cmax_events),done
      integer ntry
      logical gridrun,gridpack

      character*140 buff

      data iseed/-1/
c-----
c  Begin Code
c-----
c
c     Get requested number of events
c
      call load_para(npara,param,value)
      call get_logical(npara,param,value," gridrun ",gridrun,.false.)
      call get_logical(npara,param,value," gridpack ",gridpack,.false.)
      if (gridrun.and.gridpack) then
         call get_integer(npara,param,value," gevents "  ,nreq  ,2000   )
      else
         call get_integer(npara,param,value," nevents "  ,nreq  ,10000   )
      endif

c   Get information for the <init> block
      call setrun

c      nreq = 10000
c
c     Get total cross section
c
      xsec = 0d0
      xerr = 0d0
      open(unit=15,file='results.dat',status='old',err=21)
      read(15,*,err=20) xsec,xerr
      write(*,*) "Results.dat xsec = ",xsec
 20   close(15)
 21   if (nreq .gt. 0 .and. xsec .gt. 0) then
         goal_wgt = xsec/nreq/4d0   !Extra factor of 4 for weighted events
      else
         goal_wgt = 0d0    !Write out everything
      endif
c
c     Get list of subprocesses
c
      call get_subprocess(subname,ns)

c
c     Create scratch file to hold events
c
      I4 = 4
      R8 = 8
      record_length = 4*I4+maxexternal*I4*7+maxexternal*5*R8+3*R8+
     &   140
      open(unit=sfnum,access='direct',file='scratch',err=999,
     &     recl=record_length)
c
c     Loop through subprocesses filling up scratch file with events
c
      sum=0d0
      kevent=0
      revent=0
      maxwgt=0d0
      write(*,*) 'SubProcess/Channel     kept   read   xsec '
      do i=1,ns
c         write(*,*) 'Subprocess: ',subname(ns)
         call read_channels(subname(i),sum,kevent,revent,goal_wgt,maxwgt)
      enddo 
c
c     Get Random order for events
c
      do i=1,kevent
         iarray(i)=i
         xarray(i)=xran1(iseed)
      enddo
      call sortO3(xarray,iarray,kevent)
c
c     Write out the events in iarray order
c
      open(unit=15,file='events.lhe',status='unknown',err=98)
      call writebanner(15,kevent,sum,maxwgt,xerr)
      do i=1,kevent
         read(sfnum,rec=iarray(i)) wgt,n,
     &        ((ic(m,j),j=1,maxexternal),m=1,7),ievent,
     &        ((p(m,j),m=0,4),j=1,maxexternal),scale,aqcd,aqed,
     &     buff
         call write_event(15,P,wgt,n,ic,ievent,scale,aqcd,aqed,buff)
      enddo
      close(15)
c
c     Now select unweighted events.
c
      goal_wgt = sum/(nreq*1.03)
      min_goal = goal_wgt/5d0
      max_goal = goal_wgt*5d0
      ntry = 1
c
c     Loop to refine guess for goal_wgt while keeping xtrunc<0.01
c
      done=.false.
      do while(.not. done)
         done=.true.
         nunwgt=0
         xtrunc=0d0
         do i=1,kevent
            read(sfnum,rec=iarray(i)) wgt,n,
     &           ((ic(m,j),j=1,maxexternal),m=1,7),ievent,
     &           ((p(m,j),m=0,4),j=1,maxexternal),scale,aqcd,aqed,
     &        buff
            if (wgt .gt. goal_wgt*xran1(iseed)) then
               keep(i) = .true.
               nunwgt=nunwgt+1
               if (wgt .gt. goal_wgt) then
                  xtrunc=xtrunc+wgt-goal_wgt
               endif
            else
               keep(i)=.false.
            endif
         enddo
         if (xtrunc .gt. 0.01d0*sum) then
            done=.false.
            min_goal = max(goal_wgt,min_goal)
            goal_wgt = goal_wgt*1.3d0            
            write(*,*) 'Iteration ',ntry, ' too large truncation ',xtrunc/sum,nunwgt
c            write(*,*) min_goal,goal_wgt,max_goal
         elseif (nunwgt .lt. nreq) then
            done=.false.
            max_goal = min(goal_wgt,max_goal)
            goal_wgt = goal_wgt*0.95d0
            write(*,*) 'Iteration ',ntry, ' too few events ',xtrunc/sum,nunwgt
c            write(*,*) min_goal,goal_wgt,max_goal
            if (goal_wgt .lt. min_goal) then
               done=.true.
               write(*,*) 'Failed to find requested number of unweighted events',nreq,nunwgt
            endif
         endif
         ntry=ntry+1
         if (ntry .gt. 20) done=.true.
      enddo
      if (nunwgt .lt. nreq) then
         write(*,*) 'Unable to get ',nreq,' events. Writing ',nunwgt
         nreq = nunwgt
      else
         write(*,*) 'Found ',nunwgt,' events writing first ',nreq
      endif
      write(*,*) 'Unweighting selected ',nreq, ' events.'
      write(*,'(a,f5.2,a)') 'Truncated ',xtrunc*100./sum, '% of cross section'
      open(unit=15,file='unweighted_events.lhe',status='unknown',err=99)
      call writebanner_u(15,nreq,xsec,xtrunc,xerr)
      ntry = 0
      do i=1,kevent
         if (keep(i) .and. ntry .lt. nreq) then
            read(sfnum,rec=iarray(i)) wgt,n,
     &           ((ic(m,j),j=1,maxexternal),m=1,7),ievent,
     &           ((p(m,j),m=0,4),j=1,maxexternal),scale,aqcd,aqed,
     $        buff
            call write_event(15,P,xsec/nreq,n,ic,ievent,scale,aqcd,aqed,buff)
            ntry=ntry+1
         endif
      enddo
      close(15)
      close(sfnum)
      return
 98   write(*,*) 'Error writing events.dat' 
      return
 99   write(*,*) 'Error writing unweighted_events.dat' 
      return
 999  write(*,*) 'Error opening scratch file'
      end


      subroutine writebanner(lunw,nevent,sum,maxwgt,xerr)
c**************************************************************************************
c     Writes out banner information at top of event file
c**************************************************************************************
      implicit none
c
c     Arguments
c     
      integer lunw,nevent
      double precision sum,maxwgt,xerr
c
c     Local
c
      integer i,j

c
c     Les Houches init block (for the <init> info)
c
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)

c
c     Global
c
c      double precision etmin(3:nexternal),etamax(3:nexternal)
c      double precision                    r2min(3:nexternal,3:nexternal)
c      double precision s_min(nexternal,nexternal)
c      common/to_cuts/  etmin     ,etamax     , r2min, s_min

c-----
c  Begin Code
c-----
c
c     gather the info
c
c      call setpara('param_card.dat')
c      call setcuts
c
c     write it out
c
c      call write_para(lunw)
c      write(lunw,'(a70)') '##                                                                    '
c      write(lunw,'(a70)') '##-------------------                                                 '
c      write(lunw,'(a70)') '## Run-time options                                                   '
c      write(lunw,'(a70)') '##-------------------                                                 '
c      write(lunw,'(a70)') '##                                                                    '
c     write(lunw,'(a70)') '##********************************************************************'     
c     write(lunw,'(a70)') '## Standard Cuts                                                     *'
c     write(lunw,'(a70)') '##********************************************************************'    
c      write(lunw,'(a13,8i8)')   '## Particle  ',(i,i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Et       >',(etmin(i),i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Eta      <',(etamax(i),i=3,nexternal)
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## d R #',j,'  >',(-0.0,i=3,j),
c     &        (r2min(i,j),i=j+1,nexternal)
c         do i=j+1,nexternal
c            r2min(i,j)=r2min(i,j)**2 !Since r2 returns distance squared
c         enddo
c      enddo
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## s min #',j,'>',
c     &        (s_min(i,j),i=3,nexternal)
c      enddo
c      write(lunw,'(a70)') '#********************************************************************'    
c
c     Now write out specific information on the event set
c
c
      write(lunw,'(a)') '<MGGenerationInfo>'
      write(lunw,'(a30,i10)')   '#  Number of Events        :  ',nevent
      write(lunw,'(a30,e10.5)') '#  Integrated weight (pb)  :  ',sum
      write(lunw,'(a30,e10.5)') '#  Max wgt                 :  ',maxwgt
      write(lunw,'(a30,e10.5)') '#  Average wgt             :  ',sum/nevent
      write(lunw,'(a)') '</MGGenerationInfo>'
   
    

C   Write out compulsory init info
      write(lunw,'(a)') '</header>'
      write(lunw,'(a)') '<init>'
      write(lunw,90) (idbmup(i),i=1,2),(ebmup(i),i=1,2),(pdfgup(i),i=1,2),
     $   (pdfsup(i),i=1,2),2,nprup
      do i=1,nprup
         write(lunw,91) xsecup(i),xerr*xsecup(i)/sum,maxwgt,lprup(i) ! FACTOR OF nevts for maxwgt and wgt? error?
      enddo
      write(lunw,'(a)') '</init>'
 90   FORMAT(2i6,2e19.11,2i2,2i6,i2,i3)
 91   FORMAT(3e19.11,i4)
      end


      subroutine writebanner_u(lunw,nevent,sum,maxwgt,xerr)
c**************************************************************************************
c     Writes out banner information at top of event file
c**************************************************************************************
      implicit none
c
c     Arguments
c     
      integer lunw,nevent
      double precision sum,maxwgt,xerr
c
c     Local
c
      integer i,j
c
c     Les Houches init block (for the <init> info)
c
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
c
c     Global
c
c      double precision etmin(3:nexternal),etamax(3:nexternal)
c      double precision                    r2min(3:nexternal,3:nexternal)
c      double precision s_min(nexternal,nexternal)
c      common/to_cuts/  etmin     ,etamax     , r2min, s_min

c-----
c  Begin Code
c-----
c
c     gather the info
c
c      call setpara('param_card.dat')
c      call setcuts
c
c     write it out
c
c      call write_para(lunw)
c      write(lunw,'(a70)') '##                                                                    '
c      write(lunw,'(a70)') '##-------------------                                                 '
c      write(lunw,'(a70)') '## Run-time options                                                   '
c      write(lunw,'(a70)') '##-------------------                                                 '
c      write(lunw,'(a70)') '##                                                                    '
c     write(lunw,'(a70)') '##********************************************************************'     
c     write(lunw,'(a70)') '## Standard Cuts                                                     *'
c     write(lunw,'(a70)') '##********************************************************************'    
c      write(lunw,'(a13,8i8)')   '## Particle  ',(i,i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Et       >',(etmin(i),i=3,nexternal)
c      write(lunw,'(a13,8f8.1)') '## Eta      <',(etamax(i),i=3,nexternal)
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## d R #',j,'  >',(-0.0,i=3,j),
c     &        (r2min(i,j),i=j+1,nexternal)
c         do i=j+1,nexternal
c            r2min(i,j)=r2min(i,j)**2 !Since r2 returns distance squared
c         enddo
c      enddo
c      do j=3,nexternal-1
c         write(lunw,'(a,i2,a,8f8.1)') '## s min #',j,'>',
c     &        (s_min(i,j),i=3,nexternal)
c      enddo
c      write(lunw,'(a70)') '##********************************************************************'    
c
c     Now write out specific information on the event set
c

      write(lunw,'(a)') '<MGGenerationInfo>'
      write(lunw,'(a30,i10)')   '#  Number of Events        :  ',nevent
      write(lunw,'(a30,e10.5)') '#  Integrated weight (pb)  :  ',sum
      write(lunw,'(a30,e10.5)') '#  Truncated wgt (pb)      :  ',maxwgt
      write(lunw,'(a30,e10.5)') '#  Unit wgt                :  ',sum/nevent
      write(lunw,'(a)') '</MGGenerationInfo>'

C   Write out compulsory init info
      write(lunw,'(a)') '</header>'
      write(lunw,'(a)') '<init>'
      write(lunw,90) (idbmup(i),i=1,2),(ebmup(i),i=1,2),(pdfgup(i),i=1,2),
     $   (pdfsup(i),i=1,2),3,nprup
      do i=1,nprup
         write(lunw,91) xsecup(i),xerr*xsecup(i)/sum,sum/nevent,lprup(i) ! FACTOR OF nevts for maxwgt and wgt? error?
      enddo
      write(lunw,'(a)') '</init>'
 90   FORMAT(2i6,2e19.11,2i2,2i6,i2,i3)
 91   FORMAT(3e19.11,i4)

      end


      subroutine read_channels(dir,sum,kevent,revent,goal_wgt,maxwgt)
c*****************************************************************
c     tests traversing directories to find all events
c****************************************************************
      implicit none
c
c     Constants
c
      character*(*) symfile
      parameter (symfile='symfact.dat')
c
c     Arguments
c
      character*(30) dir
      integer kevent,revent
      double precision sum,goal_wgt,maxwgt
c
c     Local
c
      integer i,j, k, ip
      double precision xi
      character*50 dirname,dname,channame
c-----
c  Begin Code
c-----
       i = index(dir," ")
      dname = dir(1:i-1)// "/" // symfile
      open(unit=35, file=dname ,status='old',err=59)
      do while (.true.)
         read(35,*,err=99,end=99) xi,j
         if (j .gt. 0) then
            if ( (xi-int(xi+.01)) .lt. 1d-5) then
               k = int(xi+.01)
               if (k .lt. 10) then
                  write(dirname,'(a,i1,a)') 'G',k,'/'
               else if (k .lt. 100) then
                  write(dirname,'(a,i2,a)') 'G',k,'/'
               else if (k .lt. 1000) then
                  write(dirname,'(a,i3,a)') 'G',k,'/'
               else if (k .lt. 10000) then
                  write(dirname,'(a,i4,a)') 'G',k,'/'
               endif
            else               !Handle B.W.
               if (xi .lt. 10) then
                  write(dirname,'(a,f5.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 100) then
                  write(dirname,'(a,f6.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 1000) then
                  write(dirname,'(a,f7.3,a,a)') 'G',xi,'/'
               else if (xi .lt. 10000) then
                  write(dirname,'(a,f8.3,a,a)') 'G',xi,'/'
               endif
            endif     
            ip = index(dirname,'/')
            channame = dname(1:i-1)// "/" //dirname(1:ip)
            call read_dir_events(channame(1:i+ip),j,kevent,revent,sum,goal_wgt,maxwgt)
            write(*,'(a,2i8,e10.3)') channame(1:i+ip),kevent,revent,sum
         endif
 98   enddo
 99   close(35)
      return
c
c     Come here if there isn't a symfact file. Means we will work on
c     this file alone
c
 59   dirname="./"
      j = 1
      ip = 2
      channame = dirname(1:ip)
      call read_dir_events(channame,j,kevent,revent,sum,goal_wgt,maxwgt)
      write(*,'(a30,i8,e10.3)') channame(1:i+ip),kevent,sum
      return
      end

      subroutine read_dir_events(channame,nj,kevent,revent,sum,goal_wgt,maxwgt)
c********************************************************************
c********************************************************************
      implicit none
c
c     parameters
c     
      integer    sfnum
      parameter (sfnum=17)   !Unit number for scratch file
      character*(*) scaled_file
      parameter (scaled_file='events.lhe')
      integer maxexternal
      parameter (maxexternal=17)
      include 'run_config.inc'
      integer    max_read
      parameter (max_read = 2000000)
c
c     Arguments
c
      character*(*) channame
      integer nj,kevent,revent
      double precision sum,goal_wgt,maxwgt
c
c     Local
c
      double precision wgt
      double precision p(0:4,maxexternal)
      double precision gsfact
      real xwgt(max_read),xtot
      integer i,j,k,m, ic(7,maxexternal),n
      double precision scale,aqcd,aqed
      integer ievent,iseed
      logical done,found
      character*140 buff
      character*60 fullname
c
c     Les Houches init block (for the <init> info)
c
      integer maxpup
      parameter(maxpup=100)
      integer idbmup,pdfgup,pdfsup,idwtup,nprup,lprup
      double precision ebmup,xsecup,xerrup,xmaxup
      common /heprup/ idbmup(2),ebmup(2),pdfgup(2),pdfsup(2),
     &     idwtup,nprup,xsecup(maxpup),xerrup(maxpup),
     &     xmaxup(maxpup),lprup(maxpup)
      data nprup/0/
c
c     external
c
      real xran1
c
c     data
c
      data iseed/-1/
c-----
c  Begin Code
c-----     
      fullname = channame // "gscalefact.dat"
      gsfact = 1d0
      open (unit=15,file=fullname,status='old',err=12)
      read(15,*) gsfact    !Scale factor for grid runs that only use some channels
 12   close(15)
      if (gsfact .eq. 0d0) return
      fullname = channame // scaled_file      
      open(unit=15,file=fullname, status='old',err=999)
      done=.false.
c
c     Start by initializing all event variables to zero (not really necessary)
c
      do j=1,maxexternal
         do i=1,7
            ic(i,j)=0
         enddo
         do i=0,4
            p(i,j) = 0d0
         enddo
      enddo
c
c     Now loop through events
c
      do while (.not. done)
         call read_event(15,P,wgt,n,ic,ievent,scale,aqcd,aqed,buff,done)
         if (.not. done) then
            revent = revent+1
            wgt = wgt*nj*gsfact                 !symmetry factor * grid factor
            if (wgt .gt. maxwgt) maxwgt=wgt
            if (wgt .ge. goal_wgt*xran1(iseed)) then
               kevent=kevent+1
               if (wgt .lt. goal_wgt) wgt = goal_wgt
               write(sfnum,rec=kevent) wgt,n,
     &              ((ic(i,j),j=1,maxexternal),i=1,7),ievent,
     &              ((p(i,j),i=0,4),j=1,maxexternal),scale,aqcd,aqed,buff
               sum=sum+wgt
               found=.false.
               do i=1,nprup
                  if(ievent.eq.lprup(i))then
                     xsecup(i)=xsecup(i)+wgt
                     found=.true.
                  endif
               enddo
               if(.not.found)then
                  nprup=nprup+1
                  lprup(nprup)=ievent
                  xsecup(nprup)=wgt
               endif
            endif
         endif
         if (kevent .ge. max_read) then
            write(*,*) 'Error too many events to read in select_events'
     $           , kevent
            write(*,*) 'Reset max_read in Source/select_events.f'
            stop
         endif
      enddo
 99   close(15)
 55   format(i3,4e19.11)         
c      write(*,*) 'Found ',kevent,' events'
c      write(*,*) 'Integrated weight',sum
      return
 999  write(*,*) 'Error opening file ',channame,scaled_file

      end



      subroutine get_subprocess(subname,ns)
c*****************************************************************
c     tests traversing directories to find all events
c****************************************************************
      implicit none
c
c     Constants
c
      character*(*) plist
      parameter (plist='subproc.mg')
      integer    maxsubprocesses
      parameter (maxsubprocesses=999)
c
c     Arguments
c
      character*30 subname(maxsubprocesses)
      integer ns
c-----
c  Begin Code
c-----
      ns = 1
      open(unit=15, file=plist,status='old',err=99)
      do while (.true.)
         read(15,*,err=999,end=999) subname(ns)
         ns=ns+1
      enddo
 99   subname(ns) = './'
      write(*,*) "Did not find ", plist
      return
 999  ns = ns-1
      write(*,*) "Found ", ns," subprocesses"
      close(15)
      end


      function xran1(idum)
      dimension r(97)
      parameter (m1=259200,ia1=7141,ic1=54773,rm1=3.8580247e-6)
      parameter (m2=134456,ia2=8121,ic2=28411,rm2=7.4373773e-6)
      parameter (m3=243000,ia3=4561,ic3=51349)
      data iff /0/
      save r, ix1,ix2,ix3
      if (idum.lt.0.or.iff.eq.0) then
        iff=1
        ix1=mod(ic1-idum,m1)
        ix1=mod(ia1*ix1+ic1,m1)
        ix2=mod(ix1,m2)
        ix1=mod(ia1*ix1+ic1,m1)
        ix3=mod(ix1,m3)
        do 11 j=1,97
          ix1=mod(ia1*ix1+ic1,m1)
          ix2=mod(ia2*ix2+ic2,m2)
          r(j)=(float(ix1)+float(ix2)*rm2)*rm1
11      continue
        idum=1
      endif
      ix1=mod(ia1*ix1+ic1,m1)
      ix2=mod(ia2*ix2+ic2,m2)
      ix3=mod(ia3*ix3+ic3,m3)
      j=1+(97*ix3)/m3
      if(j.gt.97.or.j.lt.1)then
         write(*,*) 'j is bad in ran1.f',j, 97d0*ix3/m3
         STOP
      endif
      xran1=r(j)
      r(j)=(float(ix1)+float(ix2)*rm2)*rm1
      return
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

      subroutine sortO3(array,aux1,n)

c   O-Sort Version 3, Sorting routine by Erik Oosterwal
c   http://www.geocities.com/oosterwal/computer/sortroutines.html

      implicit none

! Arguments
      integer n
      integer aux1(n)
      double precision array(n)
!  Local Variables
      integer step,i,itemp
      double precision SngPhi,SngFib

      SngPhi = 0.78             ! Define phi value
      SngFib = n * SngPhi       ! Set initial real step size
      step = int(SngFib)        ! set initial integer step size

      do while (step > 0)
        do i = 1,n-step         ! Set the range of the lower search cells
          if (array(aux1(i))<array(aux1(i+step))) then ! Compare cells
            itemp = aux1(i)     !                       \
            aux1(i) = aux1(i+step) !                     | Swap cells
            aux1(i+step) = itemp !                      /
          end if
        enddo
        
        SngFib = SngFib * SngPhi ! Decrease the Real step size
        Step = Int(SngFib)      ! Set the integer step value

      enddo

      end
