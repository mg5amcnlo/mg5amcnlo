      program dumpMadGraph
c
c	read a MadGraph ascii file and write an ascii dump 
c

      implicit none

#include "stdlun.inc"

c...local variables
      character*80 inname,outname
      integer nev,lok,istr,iev

c...get input and output file names and max number of HEPEUP events
      OPEN(55,FILE='dumpMadGraph.in',status='old')
      READ(55,*) inname,outname,nev

C initialize HEP logical units
      lnhwrt=0
      lnhrd=0
      lnhdcy=0
      lnhpdf=77
      lnhout=22
      lnhdmp=23

      open(unit=lnhout,file='dumpMadGraph.lpt',status='new')
      open(unit=lnhpdf,file=inname,status='old')

C.....Opening ascii dump file
      open(unit=lnhdmp,file=outname,status='new')

c... should be filling and writing HEPRUP, but that info is not available
c... in the ascii file and must be hand coded at this time
c... use setrup for now
      call setrup
      call dumpheprup

c.. now read and write an event
       lok = 0
       iev = 0
       do while( lok.eq.0 .and. (nev.le.0 .or. iev.lt.nev) )
	   if(mod(iev,10).eq.0) write(lnhout,*) 'begin event ',iev
           call readmadgraph(1.9,lok)
           if(lok.eq.0) then
	       iev = iev + 1
	       call dumphepeup(iev)
	   endif
       enddo
       write(lnhout,*) ' Processed ',iev,' HEPEUP events'


c...close event file
      close(lnhdmp)
      close(lnhpdf)
      end
      
      subroutine setrup

#include "heprup.inc"
#include "stdlun.inc"

C...Set incoming beams: Tevatron Run II.
      IDBMUP(1)=2212
      IDBMUP(2)=-2212
      EBMUP(1)=980D0
      EBMUP(2)=980D0

C...Set PDF's of incoming beams: CTEQ 5L.
      PDFGUP(1)=4
      PDFSUP(1)=46
      PDFGUP(2)=PDFGUP(1)
      PDFSUP(2)=PDFSUP(1)

C...Number of external processes. 
      NPRUP=1

      XSECUP(1)=0.
      XMAXUP(1)=0.
      LPRUP(1)=661

      return
      end
