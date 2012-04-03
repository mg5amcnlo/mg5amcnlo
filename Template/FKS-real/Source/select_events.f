      program pick_events
c********************************************************************
c     Takes events from scaled.dat and plots distributions 
c********************************************************************
      implicit none
      integer nevents
      double precision mxwgt,mxwgt1, wgt_goal, eff,xsec
      integer gevents
c-----
c   Begin Code
c-----
      call read_all_events(nevents,mxwgt,mxwgt1,eff,xsec)
      write(*,*) 'Number of events',nevents,mxwgt,mxwgt1
      write(*,'($a,i7,a)') 'Enter number of events desired ( <',
     &     int(nevents*eff),') '
      read(*,*) gevents
      wgt_goal = mxwgt1*Max((nevents*eff)/gevents,1d0)
c      wgt_goal =  mxwgt1      
      call write_events(wgt_goal,xsec,gevents)
      end

      subroutine write_events(wgt_goal,xsec,gevents)
c********************************************************************
c     
c********************************************************************
      implicit none
c
c     parameters
c     
      character*(*) scaled_file
      parameter (scaled_file='events.lhe')
      character*(*) unwgt_file
      parameter (unwgt_file='unweighted_events.lhe')
      integer    max_write
      parameter (max_write=100000)
      integer    maxexternal
      parameter (maxexternal=15)
c
c     Arguments
c
      double precision wgt_goal,xsec
      integer gevents
c
c     Local
c
      double precision sum, wgt, s_over
      double precision p(0:4,maxexternal),r
      integer i,j,k,m, kevent, ic(7,maxexternal)
      integer iseed, n_over, n
      integer i4,r8,record_length,irc,ng,iol
      real xran1
      character*79 buff
      logical lwrite
      logical*1 l_store(max_write)
      logical done
      double precision scale, aqcd, aqed
      integer ievent

      external xran1
c-----
c  Begin Code
c-----     
      sum=0d0
      s_over = 0d0
      n_over = 0
      iseed = 0
c      mxwgt=-1d0
      kevent = 0
      rewind(37)        !This is comment block
      open(unit=15,file=scaled_file, status='old',err=999)
      I4 = 4
      R8 = 8
      record_length = 3*I4+maxexternal*I4*7+maxexternal*4*R8+3*R8
      open(unit=16,access='direct',status='scratch',err=999,
     &     recl=record_length)
      done=.false.
      do while (.not. done)
c         write(*,*) 'Reading Event'
         call read_event(15,P,wgt,n,ic,ievent,scale,aqcd,aqed,done)
c         write(*,*) 'read event',n,wgt,wgt_goal
         if (.not. done) then
            r = xran1(iseed)
            lwrite = (wgt / r .gt. wgt_goal)
c            write(*,*) 'lwrite',lwrite,r
            if (lwrite) then
               sum=sum+wgt_goal
               kevent = kevent+1
               l_store(kevent)=.false.
c               write(*,*) 'writing event',kevent
               write(16,rec=kevent) wgt_goal,n,
     &              ((ic(i,j),j=1,maxexternal),i=1,7),
     &              ((p(i,j),i=0,3),j=1,maxexternal),scale,aqcd,aqed 
               if (wgt .gt. wgt_goal) then
                  s_over = s_over-wgt_goal+wgt
                  n_over = n_over + 1
               endif
            endif
         endif
      enddo
 99   close(15)
c      rewind(16)
c      write(*,*) 'Done',kevent
      open(unit=26,file=unwgt_file, status='unknown',err=999)
      call write_comments(26)
      write(26,'(a2,75a1)') '##',('*', i=1,68)
      write(26,'(a)') '##'
      write(26,'(a)') '##-----------------------'
      write(26,'(a)') '## Unweighting Statistics'
      write(26,'(a)') '##-----------------------'
      write(26,'(a)') '##'
      write(26,'(a,i7)') '##   Number of Events Written   :' , kevent
      write(26,'(a,i7)') '##   Number of Events Truncated :' , n_over
      write(26,'(a,f7.2,a)') '##   Truncated Cross Section    :' ,
     $     s_over*100./(xsec),"%"

      write(26,'(a2,75a1)') '##',('*', i=1,68)

      k=0
      do while(k .lt. kevent)
         r = xran1(iseed)
         ng=0
         i=0
c         write(*,*) r,kevent,k
         do while (r .gt. real(ng)/(kevent-k) .and. i .lt. kevent)
c            write(*,*) r,kevent,k
            i=i+1
            if (.not. l_store(i)) ng=ng+1
         enddo
         k=k+1
         l_store(i)=.true.
c         write(*,*) 'Writing event ',k,i
         read(16,rec=i) wgt_goal,n,
     &        ((ic(i,j),j=1,maxexternal),i=1,7),
     &        ((p(i,j),i=0,3),j=1,maxexternal),scale,aqcd,aqed 

         wgt_goal = wgt_goal*xsec/sum
         call write_event(26,p,wgt_goal,n,ic,k,scale,aqcd,aqed)
      enddo
 92   close(16,status='delete')
      close(26)
 55   format(i3,4e19.11)         
      write(*,*) 'Wrote ',kevent,' events'
      write(*,*) 'Integrated weight',xsec
      write(*,*) 'Truncated events ',n_over,
     &     s_over, s_over/xsec*100,'%'      
      return
 999  write(*,*) 'Error opening file ',scaled_file
      end

      subroutine read_all_events(kevent, mxwgt,mxwgt1, eff, xsec)
c********************************************************************
c********************************************************************
      implicit none
c
c     parameters
c     
      character*(*) scaled_file
      parameter (scaled_file='events.lhe')
      integer maxexternal
      parameter (maxexternal=15)
      include 'run_config.inc'
      integer    max_read
      parameter (max_read = 2000000)
c
c     Arguments
c
      integer kevent
      double precision mxwgt,mxwgt1,eff,xsec
c
c     Local
c
      double precision sum, wgt
      double precision p(0:3,maxexternal)
      real xwgt(max_read),xtot
      integer i,j,k,m, ic(7,maxexternal),n
      double precision scale,aqcd,aqed
      integer ievent
      logical done
      character*79 buff
c-----
c  Begin Code
c-----     
      sum=0d0
      mxwgt=-1d0
      kevent = 0
      open(unit=15,file=scaled_file, status='old',err=999)
      done=.false.
      do while (.not. done)
         call read_event(15,P,wgt,n,ic,ievent,scale,aqcd,aqed,done)
c         call read_event(15,p,wgt,n,ic,done)
         if (.not. done) then
            sum=sum+wgt
            kevent = kevent+1
            xwgt(kevent) = wgt
            mxwgt = max(wgt,mxwgt)
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
      write(*,*) 'Sorting',kevent
      call sort(kevent,xwgt)
      xtot = 0d0
      i = kevent
      do while (xtot-xwgt(i)*(kevent-i) .lt. sum*trunc_max
     $     .and. i .gt. 2)      !Find out minimum target
         xtot = xtot + xwgt(i)
         i=i-1
      enddo
      eff = sum/kevent/xwgt(i)
      write(*,*) 'Found ',kevent,' events'
      write(*,*) 'Integrated weight',sum
      write(*,*) 'Maximum wgt',mxwgt, xwgt(i)
      write(*,*) 'Average wgt', sum/kevent
      write(*,*) 'Unweight Efficiency', eff
      mxwgt1=xwgt(i)
      xsec = sum
      return
 999  write(*,*) 'Error opening file ',scaled_file

      end



      subroutine sort(n,ra)
      real ra(n)
      l=n/2+1
      ir=n
10    continue
        if(l.gt.1)then
          l=l-1
          rra=ra(l)
        else
          rra=ra(ir)
          ra(ir)=ra(1)
          ir=ir-1
          if(ir.eq.1)then
            ra(1)=rra
            return
          endif
        endif
        i=l
        j=l+l
20      if(j.le.ir)then
          if(j.lt.ir)then
            if(ra(j).lt.ra(j+1))j=j+1
          endif
          if(rra.lt.ra(j))then
            ra(i)=ra(j)
            i=j
            j=j+j
          else
            j=ir+1
          endif
        go to 20
        endif
        ra(i)=rra
      go to 10
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
         pause
      endif
      xran1=r(j)
      r(j)=(float(ix1)+float(ix2)*rm2)*rm1
      return
      end


