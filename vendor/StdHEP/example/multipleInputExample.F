      program multipleInputExample
C
C      read in multiple events to the common block
C
#include "stdlun.inc"
#include "stdhep.inc"
      integer lok,ifl,i,ntries,istr,istrout
      integer lstr(3)
      character*20 filin(3),filrd
      logical lrdit(3)

      data lrdit/.true.,.true.,.true./
      data filin/'herwigExample.io','pythiaExample.io',
     1           'isajetExample.io'/
      save lrdit

C initialize HEP logical units
      lnhwrt=0
      lnhdcy=0
      lnhout=22
      lnhpdf=0
      lnhdmp=0
C...open output files
      open(unit=lnhout,file='multipleInputExample.lpt',status='new')
c...stdxwinit calls call mcfio_init()
      call stdxwinit('multipleInputExample.io',
     1       'StdHep multiple interaction example',100,istrout,LOK)
c      call mcfio_init()		! initialize MCFio once only
C...open input files
      do 110 i = 1,3
        filrd=filin(i)
        lnhrd=lnhrdm(i)
        call stdxropen(filrd,ntries,istr,lok)
        if(lok.ne.0) then
            lrdit(i)=.false.
            write(lnhout,1001) filrd
        else
            lstr(i) = istr
            write(lnhout,1002) filrd,istr
        endif
 110  continue
C...book histograms
      call tstbkiom
C...now read events
      do while (lrdit(1).or.lrdit(2).or.lrdit(3))
        call stdzero
        do i=1,3
          if(lrdit(i))then
 200        call stdxrdm(ifl,lstr(i),lok)
            if(lok.ne.0)then
              call stdxend(lstr(i))
              lrdit(i)=.false.
            else
	      if(ifl.eq.100) call stdxwrt(100,istrout,lok)
	      if(ifl.eq.200) call stdxwrt(200,istrout,lok)
C... keep looping if this is not a HEPEVT record
              if(ifl.ne.1 .and. ifl.ne.2) go to 200
            endif
          endif
        enddo
	if (lrdit(1).or.lrdit(2).or.lrdit(3)) then
C... fill histograms
          write(lnhout,1003) nevhep
          call tstfliom
	  call stdxwrt(2,istrout,lok)
	endif
      enddo
C...print histograms
      call houtpu(lnhout)
      call histdo
1001  format(' could not open file ',a20)
1002  format(' file ',a20,' is input stream ',i2)
1003  format(' at event ',i5)
      end
