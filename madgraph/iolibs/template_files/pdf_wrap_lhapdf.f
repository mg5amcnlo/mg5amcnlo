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
      character*20 parm(0:19) ! align indices with c++ value[20] for clarity
      double precision value(0:19) ! align indices with c++ value[20] for clarity
      integer tmpnloop(2)
      double precision tmpasmz(2)
      real*8 alphasPDFM
      external alphasPDFM ! defined in lhapdf62.cc


c-------------------
c     START THE CODE
c-------------------      

c     initialize the pdf set
      call FindPDFPath(LHAPath)
      CALL SetPDFPath(LHAPath)
      value(0)=-1
      value(1)=lhasubid(1)
      value(2)=lhasubid(2)
      value(4)=multi_lhaid_alphas_scheme ! = 0,1,2
      nsetBeam(1) = value(1) ! set for alpha_functions_lhapdf.f
      nsetBeam(2) = value(2) ! set for alpha_functions_lhapdf.f

      parm(0)='DEFAULT'

      if (pdsublabel(1).eq.'lhapdf'.or.pdsublabel(2).eq.'lhapdf') then
         call pdfset(parm,value) ! initialize PDFs via lhapdf62.cc

         select case(multi_lhaid_alphas_scheme) ! initialize alpha_s, etc

         case (1,2)     ! pull from PDF1 or PDF2
         call GetOrderAsM(nsetBeam(multi_lhaid_alphas_scheme),nloop)    ! set nloop
         asmz=alphasPDFM(nsetBeam(multi_lhaid_alphas_scheme),zmass)     ! set asmz

         case default   ! pull from PDF1 and PDF2
         call GetOrderAsM(lhasubid(1),tmpnloop(1))
         call GetOrderAsM(lhasubid(2),tmpnloop(2))
         nloop=minval(tmpnloop)     ! go with lower precision
         tmpasmz(1)=alphasPDFM(lhasubid(1),zmass)
         tmpasmz(2)=alphasPDFM(lhasubid(2),zmass)
         asmz=sqrt(tmpasmz(1)*tmpasmz(2))

         end select
         nloop=nloop+1

      else
          write(*,*) 'Unknown PDLABEL  ', pdlabel
          write(*,*) 'Unknown PDLABEL1 ', pdsublabel(1)
          write(*,*) 'Unknown PDLABEL2 ', pdsublabel(2)
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

