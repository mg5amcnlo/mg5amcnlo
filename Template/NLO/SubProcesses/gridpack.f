            subroutine open_file_local(lun,filename,fopened)
c***********************************************************************
c     opens file input-card.dat in current directory or above
c***********************************************************************
      implicit none
      include 'nexternal.inc'
      include 'mint.inc'
c
c     Arguments
c
      integer lun
      logical fopened
      character*(*) filename
      character*300  tempname
      character*300  tempname2
      character*300 path ! path of the executable
      character*30  upname ! sequence of ../
      character*30 buffer,buffer2
      integer fine,fine2
      integer i, pos

c-----
c  Begin Code
c-----
c
c     first check that we will end in the main directory
c

c
cv    check local file
c
      fopened=.false.
      tempname=filename 	 
      fine=index(tempname,' ') 	 
      fine2=index(path,' ')-1	 
      if(fine.eq.0) fine=len(tempname)
      open(unit=lun,file=tempname,status='old',ERR=20)
      fopened=.true.
      write(*,*) "FOUND LOCALLY"
      return

c      
c     getting the path of the executable
c
 20   call getarg(0,path) !path is the PATH to the madevent executable (either global or from launching directory)
      pos = index(path,'/', .true.)
      path = path(:pos)
      fine2 = index(path, ' ')-1
c
c     getting the name of the directory
c
c      if (lbw(0).eq.0)then
         ! No BW separation
         write(buffer,*)  iconfig
         path = path(:fine2)//'GF'//adjustl(buffer)
         fine2 = index(path, ' ') -1

         write(*,*) "search path", path, fine2, filename
c      else
c         ! BW separation
c         call Encode(jconfig,lbw(1),3,nexternal)
c         write(buffer,*) mincfig
c         buffer = adjustl(buffer)
c         fine = index(buffer, ' ')-1
c         write(buffer2,*) jconfig
c         buffer2=adjustl(buffer2)
c         path = path(:fine2)//'G'//buffer(:fine)//'.'//buffer2
c         fine2 = index(path, ' ')-1
c      endif
         tempname = path(:fine2)//'/'//filename
         write(*,*) 'search tempname', tempname
      open(unit=lun,file=tempname,status='old',ERR=30)
      fopened = .true.
      
 30    return
       end


