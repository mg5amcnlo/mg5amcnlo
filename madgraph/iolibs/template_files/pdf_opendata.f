      Integer Function NextUnopen()
c********************************************************************
C     Returns an unallocated FORTRAN i/o unit.
c********************************************************************

      Logical EX
C
      Do 10 N = 10, 300
         INQUIRE (UNIT=N, OPENED=EX)
         If (.NOT. EX) then
            NextUnopen = N
            Return
         Endif
 10   Continue
      Stop ' There is no available I/O unit. '
C               *************************
      End



      subroutine OpenData(Tablefile)
c********************************************************************
c generic subroutine to open the table files in the right directories
c********************************************************************
      implicit none
c
      Character Tablefile*(*),up*3,lib*4,dir*8,tempname*100
      data up,lib,dir/'../','lib/','Pdfdata/'/
      Integer IU,NextUnopen,i
      External NextUnopen
      common/IU/IU
      character*300 tempname2, path
      character*25 upbuff
      integer pos, fine2
c
c--   start
c
      IU=NextUnopen()

c     First try system wide (for cluster if define)
      %(pdf_systemwide)s

c     Then try in the current directory (for cluster use)
 5    tempname=Tablefile
      open(IU,file=tempname,status='old',ERR=10)
      return

 10   tempname=up//Tablefile
      open(IU,file=tempname,status='old',ERR=20)
      return

c     then try PdfData directory
 20   tempname=dir//Tablefile
      open(IU,file=tempname,status='old',ERR=30)
      return

 30   tempname=lib//tempname
      open(IU,file=tempname,status='old',ERR=40)

 40   continue
      do i=0,6
         open(IU,file=tempname,status='old',ERR=50)
         return
 50      tempname=up//tempname
      enddo

c     try to find the path from the executable
c                                                                                                                     
      call getarg(0,path)       !path is the PATH to the madevent executable (either global or from launching directory)
      pos = index(path,'/', .true.)                                                                                   
      path = path(:pos)                                                                                               
      fine2 = index(path, ' ')-1
      upbuff = '../../../../../../../'
      tempname = Tablefile
      do i=0,6
         tempname2= path(:fine2)//upbuff(:3*i)//dir//tempname
         open(IU,file=tempname2,status='old',ERR=60)
         return
 60      tempname2= path(:fine2)//upbuff(:3*i)//lib//dir//tempname
         open(IU,file=tempname2,status='old',ERR=70)
         return
 70      if (i.eq.6)then
            write(*,*) 'Error: PDF file ',Tablefile,' not found'
            stop
         endif
      enddo


      print*,'table for the pdf NOT found!!!'
      
      return
      end

