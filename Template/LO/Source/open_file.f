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
      character*90  tempname
      integer fine
      integer i
      
c-----
c  Begin Code
c-----
c
c     first check that we will end in the main directory
c

c
c 	  if I have to read a card
c

      tempname=filename 	 
      fine=index(tempname,' ') 	 
      if(fine.eq.0) fine=len(tempname)
      open(unit=lun,file=tempname,status='old',ERR=20)
      fopened=.true.
      return

 20   if(index(filename,"_card").gt.0) then
         tempname='Cards/'//tempname(1:fine)
         fine=fine+6
      endif
      
      fopened=.false.
      do i=0,5
         open(unit=lun,file=tempname,status='old',ERR=30)
         fopened=.true.
         exit
 30      tempname='../'//tempname
         if (i.eq.5)then
            write(*,*) 'Warning: file ',filename,' not found'
         endif
      enddo
      end
