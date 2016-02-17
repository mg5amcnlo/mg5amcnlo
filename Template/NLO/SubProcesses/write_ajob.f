
      subroutine open_bash_file(lun,fname,lname)
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
      character*30 fname
      integer run_cluster
      common/c_run_mode/run_cluster
c-----
c  Begin Code
c-----
      ic=ic+1
      if (ic .lt. 10) then
         write(fname(5:5),'(i1)') ic
         lname=lname+1
      elseif (ic .lt. 100) then
         write(fname(5:6),'(i2)') ic
         lname=lname+2
      elseif (ic .lt. 1000) then
         write(fname(5:7),'(i3)') ic
         lname=lname+3
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

