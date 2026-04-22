      subroutine open_file(lun,filename,fopened)
c***********************************************************************
c     opens file input-card.dat in current directory or above
c***********************************************************************
      implicit none
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
      integer fine,fine2
      integer i, pos
      
c-----
c  Begin Code
c-----
c      
c     getting the path of the executable
c
      call getarg(0,path) !path is the PATH to the madevent executable (either global or from launching directory)
      pos = index(path,'/',.true.)
      path = path(:pos)
c
c     first check that we will end in the main directory
c

c
c 	  if I have to read a card
c

      tempname=filename 	 
      fine=index(tempname,' ') 	 
      fine2=index(path,' ')-1	 
      if(fine.eq.0) fine=len(tempname)
      open(unit=lun,file=tempname,status='old',ERR=20)
      fopened=.true.
      return
c
c     check path from the executable
c
 20   if(index(filename,"_card").gt.0) then
         tempname='Cards/'//tempname(1:fine)
         fine=fine+6
      endif
      tempname2 = path//tempname

      fopened=.false.
      upname='../../../../../../../'
      do i=0,6
         open(unit=lun,file=tempname2,status='old',ERR=30)
         fopened=.true.
         exit
 30      tempname2=path(:fine2)//upname(:3*i)//tempname
         if (i.eq.6)then
            write(*,*) 'Warning: file ',filename,' not found'
         endif
      enddo
      end


