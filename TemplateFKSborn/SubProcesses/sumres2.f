      PROGRAM SUMRES
      implicit none
      integer maxsize
      parameter(maxsize=50000)
      integer map(maxsize),size,i,j
      real*8 num1(maxsize),num2(maxsize),onum2(maxsize),sum, avg
      character*140 buffer,filename
      character*1 todo(maxsize)

      character*100 dir(maxsize),graph(maxsize),cor(maxsize)
      integer ldir(maxsize),lgraph(maxsize),lcor(maxsize),yes,no

      real*8 min_err_per_channel
      real*8 min_err_total
      integer max_improve
      parameter( min_err_per_channel=0.40 )
      parameter( min_err_total=0.001 )
      parameter( max_improve=100 )

      
      real*8 totalrest,rest(maxsize),restsum,xtarget,ran1
      integer nevents,totevts,nevts(maxsize),restevents,totalevents
      integer iseed
      data iseed/1/

     
      size=1
      open (unit=1,file="res.txt",status="old")
      do while (1.gt.0)
         read (1,'(a)',err=11,end=11) buffer
         dir(size) =  buffer(1:index(buffer,"/")-1)
         ldir(size)= len(dir(size)(1:index(dir(size)," ")-1))

         cor(size) = buffer(ldir(size)+2:
     &        ldir(size)+index(buffer(ldir(size)+2:100),"/"))
         lcor(size) = len(buffer(ldir(size)+2:
     &        ldir(size)+index(buffer(ldir(size)+2:100),"/")))

c$$$         graph(size) = buffer(ldir(size)+lcor(size)+3:
c$$$     &        ldir(size)+lcor(size)+1+
c$$$     &        index(buffer(ldir(size)+lcor(size)+3:100),"/"))
c$$$         lgraph(size) = len(buffer(ldir(size)+lcor(size)+3:
c$$$     &        ldir(size)+lcor(size)+1+
c$$$     &        index(buffer(ldir(size)+lcor(size)+3:100),"/")))
         read(buffer(index(buffer,"result")+14:
     &        index(buffer,"+/-")-1),*),num1(size)
         read(buffer(index(buffer,"+/-")+3:140),*),num2(size)
         size=size+1
         if (size.ge.maxsize) then
            write (*,*) 'Too many channels: enlarge maxsize in sumres.f'
            stop
         endif
      enddo
 11   continue
      size=size-1

      close(1)

      sum=0d0
      avg=0d0
      do i=1,size
         sum=sum+num1(i)
         avg=avg+num2(i)**2
      enddo

      call order_elements(size,num2,onum2,map)

      yes=0
      no=0

      open (unit=1,file="res.txt",status="old", access="append")
      write (1,*) ''
      do i=1,size
         j=map(i)
         buffer=""
         write (buffer(1:10),'(i5,i5)') i,j
         write (buffer(14:14+ldir(j)),'(a)') dir(j)
         write (buffer(35:35+lcor(j)),'(a)') cor(j)
c$$$         write (buffer(45:45+lgraph(j)),'(a)') graph(j)
         write (buffer(52:70),'(d15.8)') num1(j)
         write (buffer(71:89),'(d15.8)') num2(j)
         write (buffer(90:105),'(f10.5)')
     &        abs(num2(j)/num1(j))*100d0
         write (buffer(106:106),'(a)')"%"
         if ((abs(num2(j)/num1(j)).ge.min_err_per_channel .or.
     &        num2(j).gt.min_err_total*Sqrt(avg)).and.
     &        yes.le.max_improve-1) then
             write (buffer(110:110),'(a)') "Y"
             yes=yes+1
          else
             write (buffer(110:110),'(a)') "N"
             no=no+1
          endif
         write (1,'(a)') buffer(1:110)

      enddo
      write (1,*) ''
      write (1,'(a7,2d15.8,f10.5,a2)') "Total: ",sum, Sqrt(avg),
     &     abs(sqrt(avg)/sum)*100d0,' %'
c
      close(1)


c Determine the number of events for unweighting
      write (*,*) 'give number of unweighted events'
      read (*,*) nevents
      write (*,*) nevents
      totalrest=0d0
      totevts=0
      do j=1,size
         nevts(j)=int(num1(j)/sum*nevents)
         totevts=totevts+nevts(j)
         rest(j)=num1(j)/sum*nevents - int(num1(j)/sum*nevents)
         totalrest=totalrest+dsqrt(rest(j)+dble(nevts(j)))
      enddo
      restevents=nevents-totevts
      if (restevents.ge.size) then
         write (*,*) 'Error, more rest events than channels.'
         write (*,*) 'This is impossible.'
         stop
      endif

c Determine what to do with the few remaining 'rest events'.
c Put them to channels at random given by the sqrt(Nevents) 
c already in the channel (including the rest(j))
      do j=1,restevents
         i=1
         restsum=dsqrt(rest(1)+dble(nevts(1)))
         xtarget=ran1(iseed)*totalrest
         do while (restsum .lt. xtarget)
            i=i+1
            restsum=restsum+dsqrt(rest(i)+dble(nevts(i)))
         enddo
         totalrest=totalrest-dsqrt(rest(i)+dble(nevts(i)))+
     &        dsqrt(dble(nevts(i)+1))
         nevts(i)=nevts(i)+1
         rest(i)=0d0
      enddo

      totalevents=0
      do j=1,size
         totalevents=totalevents+nevts(j)
      enddo
      if (totalevents.ne.nevents) then
         write (*,*) 'Do not have the correct number of events',
     &        totalevents,nevents
         stop
      endif

c Write the number of events in each P*/G* directory in files
c called nevts
      do i=1,140
         buffer(i:i)=' '
         filename(i:i)=' '
      enddo
      open (unit=2,file='nevents_unweighted',status='unknown')
      do j=1,size
         filename(1:ldir(j))=dir(j)
         filename(ldir(j)+1:ldir(j)+1)='/'
         filename(ldir(j)+2:ldir(j)+lcor(j)+1)=cor(j)
         filename(ldir(j)+lcor(j)+2:ldir(j)+lcor(j)+2)='/'
         filename(ldir(j)+lcor(j)+3:ldir(j)+lcor(j)+7)='nevts'
         open (unit=1,
     &       file=filename(1:ldir(j)+lcor(j)+7),status='unknown')
         write (1,'(i8)') nevts(j)
         close(1)
         filename(ldir(j)+lcor(j)+3:ldir(j)+lcor(j)+12)='events.lhe'
         write (2,*) filename(1:ldir(j)+lcor(j)+12),'     ',nevts(j),
     &        '     ',num1(j)
      enddo
      close(2)

      return
 999  write (*,*) 'Could not determine how many events you want'
      end






      subroutine order_elements(size,unordered,ordered,map)
      implicit none
      integer size
      integer maxsize
      parameter(maxsize=50000)
      double precision unordered(maxsize),ordered(maxsize)
      integer map(maxsize)
      logical foundi(maxsize),foundj(maxsize)
      
      double precision temp
      integer i,j

      do i=1,size
         ordered(i)=unordered(i)
         map(i)=i
         foundi(i)=.false.
         foundj(i)=.false.
      enddo
      do i=1,size
         do j=i+1,size
            if (ordered(j).gt.ordered(i))then
               temp=ordered(j)
               ordered(j)=ordered(i)
               ordered(i)=temp
            endif
         enddo
      enddo
      do i=1,size
         j=1
         do while (.not.foundi(i))
            if(ordered(i).eq.unordered(j)) then
               if (.not.foundj(j)) then
                  map(i)=j
                  foundi(i)=.true.
                  foundj(j)=.true.
               endif
            endif
            j=j+1
         enddo
      enddo

      return
      end


      function ran1(idum)
      dimension r(97)
      parameter (m1=259200,ia1=7141,ic1=54773,rm1=3.8580247e-6)
      parameter (m2=134456,ia2=8121,ic2=28411,rm2=7.4373773e-6)
      parameter (m3=243000,ia3=4561,ic3=51349)
      data iff /0/
      save r, ix1, ix2, ix3
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
      if(j.gt.97.or.j.lt.1) stop
      ran1=r(j)
      r(j)=(float(ix1)+float(ix2)*rm2)*rm1
      return
      end
