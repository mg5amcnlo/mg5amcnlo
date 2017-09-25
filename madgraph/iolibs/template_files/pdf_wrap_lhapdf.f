      subroutine pdfwrap
      implicit none
C
C     INCLUDE
C
      include 'pdf.inc'
      include '../alfas.inc'
      real*8 zmass
      data zmass/91.188d0/
      Character*150 LHAPath
      character*20 parm(20)
      double precision value(20)
      real*8 alphasPDF
      external alphasPDF


c-------------------
c     START THE CODE
c-------------------      

c     initialize the pdf set
      call FindPDFPath(LHAPath)
      CALL SetPDFPath(LHAPath)
      value(1)=lhaid
      parm(1)='DEFAULT'
      call pdfset(parm,value)
      call GetOrderAs(nloop)
      nloop=nloop+1  
      asmz=alphasPDF(zmass)
      
      return
      end
 

      subroutine FindPDFPath(LHAPath)
c********************************************************************
c generic subroutine to open the table files in the right directories
c********************************************************************
      implicit none
c
      Character LHAPath*150,up*3
      data up/'../'/
      logical exists
      integer i, pos
      character*300  tempname2
      character*300 path ! path of the executable
      integer fine2
      character*30  upname ! sequence of ../

c     first try in the current directory
      LHAPath='./PDFsets'
      Inquire(File=LHAPath, exist=exists)
      if(exists)return
      %(cluster_specific_path)s
      do i=1,6
         LHAPath=up//LHAPath
         Inquire(File=LHAPath, exist=exists)
         if(exists)return
      enddo

c      
c     getting the path of the executable
c
      call getarg(0,path) !path is the PATH to the madevent executable (either global or from launching directory)
      pos = index(path,'/',.true.)
      path = path(:pos)
      fine2=index(path,' ')-1	 


c
c     check path from the executable
c
      LHAPath='lib/PDFsets'
      Inquire(File=LHAPath, exist=exists)
      if(exists)return
      upname='../../../../../../../'
      do i=1,6
          tempname2=path(:fine2)//upname(:3*i)//LHAPath
c         LHAPath=up//LHAPath
          Inquire(File=tempname2, exist=exists)
         if(exists)then
            LHAPath = tempname2
            return
         endif
      enddo
      print*,'Could not find PDFsets directory, quitting'
      stop
      
      return
      end

