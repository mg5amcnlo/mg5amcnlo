      subroutine pdfwrap
      implicit none
C
C     INCLUDE
C
      include 'pdf.inc'
      include '../alfas.inc'
## if(LO) {
      include '../vector.inc'
## }
      include '../coupl.inc'
      real*8 zmass
      data zmass/91.188d0/
      Character*150 LHAPath
      character*20 parm(20)
c RR(2025_0401: value is not used/known 
c                 outside this scope/subroutine)
      double precision value(20)
      integer tmpnloop(2)
      double precision tmpasmz(2)
      real*8 alphasPDF,alphasPDFM
      external alphasPDF,alphasPDFM

c integer nloop
c      double precision asmz

c-------------------
c     START THE CODE
c-------------------      

c     initialize the pdf set
      call FindPDFPath(LHAPath)
      CALL SetPDFPath(LHAPath)

c RR(2025_0401): unsure about user interface since there 
c are now three lhaids floating around.
c consult with OM/others on "best" user strategy
c for now, call/initialize three PDF set.
c idea for runcard_check:
c set subid(1:2) = -1 by default. 
c if unchanged, then set subid(1:2) = lhaid
c in other words, always call two PDF sets)
c alternative idea:
c run always run two lhaids
      value(1)=lhaid
      value(2)=-1d0 ! lhasubid(1)
      value(3)=-1d0 ! lhasubid(2)
      parm(1)='DEFAULT'

      if((pdsublabel(1).eq.'lhapdf').and.(pdsublabel(2).eq.'lhapdf')) then
         value(2)=lhasubid(1)
         value(3)=lhasubid(2)
         
         call pdfset(parm,value) ! <- initialize PDFs via lhapdf62.cc
         call GetOrderAsM(1,tmpnloop(1))
         call GetOrderAsM(2,tmpnloop(2))

         tmpasmz(1) = alphasPDFM(1,zmass)
         tmpasmz(2) = alphasPDFM(2,zmass)
         nloop = maxval(tmpnloop)+1
         asmz  = minval(tmpasmz)
         if(lhasubid(1).eq.lhasubid(2)) then
            nset = 1 ! guide calls to alphasPDFM
         else if(tmpasmz(1).le.tmpasmz(2)) then
            nset = 1 ! guide calls to alphasPDFM
         else
            nset = 2 ! guide calls to alphasPDFM
         endif

      else if(pdsublabel(1).eq.'lhapdf') then
         value(2)=lhasubid(1)
         call pdfset(parm,value) ! <- initialize PDFs via lhapdf62.cc
         call GetOrderAsM(1,nloop)
         asmz = alphasPDFM(1,zmass)
         nset = 1

      else if(pdsublabel(2).eq.'lhapdf') then
         value(3)=lhasubid(2)
         call pdfset(parm,value) ! <- initialize PDFs via lhapdf62.cc
         call GetOrderAsM(2,nloop)
         asmz = alphasPDFM(2,zmass)
         nset = 2

      else 
c RR(2025_0407): inspect, check, and delete the following
c     commented block
c     if (pdlabel.eq.'lhapdf') then
c     call pdfset(parm,value) !!! here PDFs are initialized in lhapdf62.cc!!!
c     if(lhasubid(1).lt.0) then
c     call GetOrderAs(nloop)
c     nloop=nloop+1  
c     asmz=alphasPDF(zmass)
c     else
c     call GetOrderAsM(1,tmpnloop(1))
c     call GetOrderAsM(2,tmpnloop(2))
c     call GetOrderAs(nloop) 
c     ! gen_ximprove.py cares about path. need to investigate
c     tmpasmz(1) = alphasPDFM(1,zmass)
c     tmpasmz(2) = alphasPDFM(2,zmass)
c     nloop = maxval(tmpnloop)+1
c     asmz  = minval(tmpasmz)
c     endif         
c     else
         write(*,*) 'Unknown PDLABEL', pdlabel
         stop 1
      endif

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

