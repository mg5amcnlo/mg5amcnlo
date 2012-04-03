      program split_jobs
      implicit none
      integer maxjobs
      parameter(maxjobs=100000)
      character*140 joblines(maxjobs)
      character*40 filename(maxjobs),dir(maxjobs),cor(maxjobs),
     &     n_filename
      integer ijob,i,j,nevts(maxjobs),max_nevts_S,max_nevts_H,
     $     max_nevts_V,max_nevts_F,max_nevts_B,max_S,max_H,max_V,max_F
     $     ,max_B,yes,ldir(maxjobs),lcor(maxjobs),n_nevts
     $     ,n_sjob(maxjobs),n_ijob,n_ijob_H,n_ijob_S,n_ijob_V,n_ijob_F
     $     ,n_ijob_B,tmp
      double precision num1(maxjobs),n_num1
      logical done
      data done /.false./
      
      open (unit=11,file='nevents_unweighted',status='old',err=99)
      ijob=0
      do while (.not.done)
         ijob=ijob+1
         read(11,'(a)',err=98,end=98) joblines(ijob)
         if (ijob.ge.maxjobs) then
            write (*,*) 'ERROR too many jobs in split_jobs'
            stop
         endif
      enddo
 98   continue
      close(11)
      ijob=ijob-1
      write (*,*) 'nevents_unweighted read. Found ',ijob,' jobs'
      
      max_nevts_S=0
      max_nevts_H=0
      max_nevts_V=0
      max_nevts_F=0
      max_nevts_B=0

      i=0
      do j=1,ijob
         i=i+1
         read (joblines(j)(1:45),'(1X,a)') filename(i)
         read (joblines(j)(46:140),*) nevts(i),num1(i)
         if (nevts(i).eq.0) then
            i=i-1
            cycle
         endif
         dir(i) =  filename(i)(1:index(filename(i),"/")-1)
         ldir(i)= len(dir(i)(1:index(dir(i)," ")-1))
         cor(i) = filename(i)(ldir(i)+2:
     &        ldir(i)+index(filename(i)(ldir(i)+2:40),"/"))
         lcor(i) = len(filename(i)(ldir(i)+2:
     &        ldir(i)+index(filename(i)(ldir(i)+2:40),"/")))
         if (cor(i)(2:2).eq.'H') then
            max_nevts_H=max(max_nevts_H,nevts(i))
         elseif (cor(i)(2:2).eq.'S') then
            max_nevts_S=max(max_nevts_S,nevts(i))
         elseif (cor(i)(2:2).eq.'V') then
            max_nevts_V=max(max_nevts_V,nevts(i))
         elseif (cor(i)(2:2).eq.'F') then
            max_nevts_F=max(max_nevts_F,nevts(i))
         elseif (cor(i)(2:2).eq.'B') then
            max_nevts_B=max(max_nevts_B,nevts(i))
         else
            write (*,*) 'ERROR, cannot recognize run_mode ',cor(i)
         endif
      enddo
      ijob=i

      if (max_nevts_H.ne.0) then
         write (*,*) 'Max H-events per channel found is ',max_nevts_H
         max_H=max_nevts_H/20
         if(max_H.lt.100) max_H=100
         write (*,*) 'max number of events in splitted jobs is: ',max_H
      endif
      if (max_nevts_S.ne.0) then
         write (*,*) 'Max S-events per channel found is ',max_nevts_S
         max_S=max_nevts_S/20
         if(max_S.lt.100) max_S=100
         write (*,*) 'max number of events in splitted jobs is: ',max_S
      endif
      if (max_nevts_V.ne.0) then
         write (*,*) 'Max V-events per channel found is ',max_nevts_V
         max_V=max_nevts_V/20
         if(max_V.lt.100) max_V=100
         write (*,*) 'max number of events in splitted jobs is: ',max_V
      endif
      if (max_nevts_F.ne.0) then
         write (*,*) 'Max F-events per channel found is ',max_nevts_F
         max_F=max_nevts_F/20
         if(max_F.lt.100) max_F=100
         write (*,*) 'max number of events in splitted jobs is: ',max_F
      endif
      if (max_nevts_B.ne.0) then
         write (*,*) 'Max B-events per channel found is ',max_nevts_B
         max_B=max_nevts_B/20
         if(max_B.lt.100) max_B=100
         write (*,*) max_B
         write (*,*) 'max number of events in splitted jobs is: ',max_B
      endif
      goto 13

 12   continue
      if (max_nevts_H.ne.0) then
         write (*,*) 'found H-events'
         write (*,*) 'Max H-events per channel found is ',max_nevts_H
         write (*,*) 'Give new maximum if you want to split jobs '
         write (*,*) '(max number of splitted jobs is 99)'
         read (*,*) max_H
         write (*,*) max_H
      endif
      if (max_nevts_S.ne.0) then
         write (*,*) 'found S-events'
         write (*,*) 'Max S-events per channel found is ',max_nevts_S
         write (*,*) 'Give new maximum if you want to split jobs '
         write (*,*) '(max number of splitted jobs is 99)'
         read (*,*) max_S
         write (*,*) max_S
      endif
      if (max_nevts_V.ne.0) then
         write (*,*) 'found V-events'
         write (*,*) 'Max V-events per channel found is ',max_nevts_V
         write (*,*) 'Give new maximum if you want to split jobs '
         write (*,*) '(max number of splitted jobs is 99)'
         read (*,*) max_V
         write (*,*) max_V
      endif
      if (max_nevts_F.ne.0) then
         write (*,*) 'found F-events'
         write (*,*) 'Max F-events per channel found is ',max_nevts_F
         write (*,*) 'Give new maximum if you want to split jobs '
         write (*,*) '(max number of splitted jobs is 99)'
         read (*,*) max_F
         write (*,*) max_F
      endif
      if (max_nevts_B.ne.0) then
         write (*,*) 'found B-events'
         write (*,*) 'Max B-events per channel found is ',max_nevts_B
         write (*,*) 'Give new maximum if you want to split jobs '
         write (*,*) '(max number of splitted jobs is 99)'
         read (*,*) max_B
         write (*,*) max_B
      endif
 13   continue

      n_ijob_H=0
      n_ijob_S=0
      n_ijob_V=0
      n_ijob_F=0
      n_ijob_B=0
      do i=1,ijob
         if (cor(i)(2:2).eq.'H'.and.max_H.ne.0)then
            n_sjob(i)=nevts(i)/max_H +1
            n_ijob_H=n_ijob_H+n_sjob(i)
         elseif(cor(i)(2:2).eq.'H'.and.max_H.eq.0)then
            n_sjob(i)=1
            n_ijob_H=n_ijob_H+1
         endif
         if (cor(i)(2:2).eq.'S'.and.max_S.ne.0)then
            n_sjob(i)=nevts(i)/max_S +1
            n_ijob_S=n_ijob_S+n_sjob(i)
         elseif(cor(i)(2:2).eq.'S'.and.max_S.eq.0)then
            n_sjob(i)=1
            n_ijob_S=n_ijob_S+1
         endif
         if (cor(i)(2:2).eq.'V'.and.max_V.ne.0)then
            n_sjob(i)=nevts(i)/max_V +1
            n_ijob_V=n_ijob_V+n_sjob(i)
         elseif(cor(i)(2:2).eq.'V'.and.max_V.eq.0)then
            n_sjob(i)=1
            n_ijob_V=n_ijob_V+1
         endif
         if (cor(i)(2:2).eq.'F'.and.max_F.ne.0)then
            n_sjob(i)=nevts(i)/max_F +1
            n_ijob_F=n_ijob_F+n_sjob(i)
         elseif(cor(i)(2:2).eq.'F'.and.max_F.eq.0)then
            n_sjob(i)=1
            n_ijob_F=n_ijob_F+1
         endif
         if (cor(i)(2:2).eq.'B'.and.max_B.ne.0)then
            n_sjob(i)=nevts(i)/max_B +1
            n_ijob_B=n_ijob_B+n_sjob(i)
         elseif(cor(i)(2:2).eq.'B'.and.max_B.eq.0)then
            n_sjob(i)=1
            n_ijob_B=n_ijob_B+1
         endif
      enddo
      n_ijob=n_ijob_H+n_ijob_S+n_ijob_V+n_ijob_F+n_ijob_B
      write (*,*) 'This will give:'
      if(n_ijob_H.gt.0) write (*,*) '  ',n_ijob_H,' H-events jobs'
      if(n_ijob_S.gt.0) write (*,*) '  ',n_ijob_S,' S-events jobs'
      if(n_ijob_V.gt.0) write (*,*) '  ',n_ijob_V,' V-events jobs'
      if(n_ijob_F.gt.0) write (*,*) '  ',n_ijob_F,' F-events jobs'
      if(n_ijob_B.gt.0) write (*,*) '  ',n_ijob_B,' B-events jobs'
      write (*,*) 'Is this acceptable? ("0" for no, "1" for yes)'
      read (*,*) yes
      write (*,*) yes
      if (yes.eq.0) goto 12

      open (unit=11,file='nevents_unweighted_splitted',status='unknown')
      do i=1,ijob
         do j=1,40
            n_filename(j:j)=' '
         enddo
         n_filename(1:ldir(i))=dir(i)
         n_filename(ldir(i)+1:ldir(i)+1)='/'
         n_filename(ldir(i)+2:ldir(i)+lcor(i)+1)=cor(i)
         n_filename(ldir(i)+lcor(i)+2:ldir(i)+lcor(i)+2)='/'
         do j=1,n_sjob(i)
            n_filename(ldir(i)+lcor(i)+3:ldir(i)+lcor(i)+9)=
     &           'events_'
            if (j.le.9) then
               write (n_filename(ldir(i)+lcor(i)+10:ldir(i)+lcor(i)+14),
     &              '(i1,a4)') j,'.lhe'
            elseif (j.le.99) then
               write (n_filename(ldir(i)+lcor(i)+10:ldir(i)+lcor(i)+15),
     &              '(i2,a4)') j,'.lhe'
            else
               write (*,*)'ERROR, too many splittings',j
            endif
            if (j.ne.n_sjob(i)) then
               n_nevts=nevts(i)/n_sjob(i)
               n_num1=num1(i)*dble(n_nevts)/dble(nevts(i))
            else
               tmp=nevts(i)/n_sjob(i)
               n_nevts=nevts(i)-(n_sjob(i)-1)*tmp
               n_num1=num1(i)*dble(n_nevts)/dble(nevts(i))
            endif
            write (11,*) n_filename(1:40),'     ',n_nevts,'     ',n_num1
            n_filename(ldir(i)+lcor(i)+3:ldir(i)+lcor(i)+9)='nevts__'
            if (j.le.9) then
               open (unit=1,file=n_filename(1:ldir(i)+lcor(i)+10),
     &              status='unknown')
            else
               open (unit=1,file=n_filename(1:ldir(i)+lcor(i)+11),
     &              status='unknown')
            endif
            write (1,'(i8)') n_nevts
            close(1)
         enddo
      enddo
      close(11)

      return
 99   write (*,*) 'Error: nevents_unweighted not found'
      stop
      end
