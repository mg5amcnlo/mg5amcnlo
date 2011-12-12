      PROGRAM SUMRES
      implicit none
      integer maxsize
      parameter(maxsize=50000)
      integer map(maxsize),size,i,j
      real*8 num1(maxsize),num2(maxsize),onum2(maxsize),prob(maxsize)
      real*8 sum, avg
      character*140 buffer,filename
      character*1 todo(maxsize)

      character*100 dir(maxsize),graph(maxsize),cor(maxsize)
      integer ldir(maxsize),lgraph(maxsize),lcor(maxsize),yes,no

      real*8 min_err_per_channel,min_err_total
      integer max_improve,granularity
      parameter( min_err_per_channel=0.40 )
      parameter( min_err_total=0.001 )
      parameter( max_improve=100 )
      parameter( granularity=1 )

      
      real*8 totalrest,rest(maxsize),restsum,xtarget,fk88random
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
         write (buffer(14:14+ldir(j)),'(a)') dir(j)(1:ldir(j))
         write (buffer(35:35+lcor(j)),'(a)') cor(j)(1:lcor(j))
c$$$         write (buffer(45:45+lgraph(j)),'(a)') graph(j)(1:lgraph(j))
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


c***********************************************
c Determine the number of events for unweighting
c***********************************************
      write (*,*) 'give number of unweighted events'
      read (*,*) nevents
      write (*,*) nevents
c determine the relative, accumulative weights of the channels
      prob(1)=num1(1)/sum
      do j=2,size
         prob(j)=prob(j-1)+num1(j)/sum
      enddo
c final weight should be one
      if (abs(prob(size)-1d0).gt.1d-7) then
         write (*,*) 'error in determining accumulative probabilities',
     &        prob(size)+num1(size)/sum-1d0
         stop
      endif

c Assign events to the channels with a probability according
c to the relative weights of the channels.
      do i=1,nevents,granularity
         j=1
         xtarget=fk88random(iseed)
         do while (prob(j) .lt. xtarget)
            j=j+1
         enddo
         nevts(j)=nevts(j)+granularity
      enddo
c***********************************************



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
         if (ldir(j)+lcor(j)+13.ge.40) then
            write (*,*) 'error #1 in sumres2.f'
            write (*,*) 'increase line length',ldir(j)+lcor(j)+13
            stop
         endif
         do i=ldir(j)+lcor(j)+13,40
            filename(i:i)=' '
         enddo

         write (2,*) filename(1:40),'     ',nevts(j),'     ',num1(j)
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


