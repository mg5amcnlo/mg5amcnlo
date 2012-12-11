      program madToXDR
c
c	read a MadGraph ascii file and write mcfio output
c

      implicit none

#include "stdlun.inc"

c...local variables
      character*80 inname,outname
      integer nev,lok,istr,iev

c...get input and output file names and max number of HEPEUP events
      OPEN(55,FILE='madToXDR.in',status='old')
      READ(55,*) inname,outname,nev

C initialize HEP logical units
      lnhwrt=23
      lnhrd=0
      lnhdcy=0
      lnhpdf=77
      lnhout=22
      lnhdmp=0

      open(unit=lnhout,file='madToXDR.lpt',status='new')
      open(unit=lnhpdf,file=inname,status='old')

C.....Opening stdhep file for writing
      call stdxwinit(outname,'MadGraph to MCFio',nev,istr,lok)
      if(lok.ne.0) write(lnhout,*) ' Problem opening file '

C          Write Stdhep begin-run record   
      call stdxwrt(100,istr,lok)
      if(lok.ne.0) write(lnhout,*)
     1        ' Problem writing stdhep begin run record'

c... should be filling and writing HEPRUP, but that info is not available
c... in the ascii file and must be hand coded at this time

c.. now read and write an event
       lok = 0
       iev = 0
       do while( lok.eq.0 .and. (nev.le.0 .or. iev.lt.nev) )
	   if(mod(iev,10).eq.0) write(lnhout,*) 'begin event ',iev
           call readmadgraph(1.9,lok)
           if(lok.eq.0) then
	       call stdxwrt(11,istr,lok)
	       iev = iev + 1
	   endif
       enddo
       write(lnhout,*) ' Processed ',iev,' HEPEUP events'

C          Write end-of-run record  
      call stdxwrt(200,istr,lok)
      if(lok.ne.0) write(lnhout,*) ' Problem writing end run record'

c...close event file
      call stdxend(istr)
      close(lnhpdf)
      end
