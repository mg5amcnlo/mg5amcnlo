
      subroutine open_bash_file(lun,fname,lname,mname)
c***********************************************************************
c     Opens bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
      integer lun
      integer ic
      data ic/0/
      logical found_tag
      character*120 buff
      integer lname
      character*30 mname
      character*30 fname
      integer run_cluster
      common/c_run_mode/run_cluster
c-----
c  Begin Code
c-----
      ic=ic+1
      if (ic .lt. 10) then
         write(fname(5:5),'(i1)') ic
         write(mname(3:7),'(i1,a4)') ic,".cmd"
         lname=lname+1
      elseif (ic .lt. 100) then
         write(fname(5:6),'(i2)') ic
         write(mname(3:8),'(i2,a4)') ic,".cmd"
         lname=lname+2
      elseif (ic .lt. 1000) then
         write(fname(5:7),'(i3)') ic
         write(mname(3:9),'(i3,a4)') ic,".cmd"
         lname=lname+3
      endif
      if (run_cluster.eq.1) then
         open (unit=lun+1,file=mname,status='unknown')
         write(lun+1,15) 'universe = vanilla'
         write(lun+1,15) 'executable = '//fname
         write(lun+1,15) 'output = /dev/null'
         write(lun+1,15) 'error = /dev/null'
         write(lun+1,15) 'requirements = (MADGRAPH == True)'
         write(lun+1,15) 'log = /dev/null'
         write(lun+1,15) ''
         write(lun+1,15) 'queue'
         write(lun+1,15) ''
         close(lun+1)
      endif
      open (unit=lun, file = fname, status='unknown')
      open (unit=lun+1,file="../ajob_template",status="old")
      found_tag=.false.
      do while (.true.) 
         read(lun+1,15,err=99,end=99) buff
         if (index(buff,'TAGTAGTAGTAGTAG').ne.0) exit
         write(lun,15) buff
      enddo
      write(lun,'(a$)') 'for i in $channel '
      return
 99   write (*,*) 'ajob_template or ajob_template_cluster '/
     &     /'does not have the correct format'
      stop
 15   format(a)
      end

      subroutine close_bash_file(lun)
c***********************************************************************
c     Closes bash file for looping including standard header info
c     which can be used with pbs, or stand alone
c***********************************************************************
      implicit none
      integer lun
      character*120 buff
      write(lun,'(a)') '; do'
      do while (.true.) 
         read(lun+1,15,err=99,end=99) buff
         write(lun,15) buff
      enddo
 99   continue
      close(lun+1)
      close(lun)

 15   format(a)
      end

