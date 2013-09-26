      program tstpyt
C
C      test program for stdhep/pythia interface
C      compare event list
C      write events and cross-section
C
#include "stdlun.inc"

      integer istr,mstat
      integer nevt,i,nprnt,lok
C
      nevt=100
      nprnt=1
      call upyinit(nevt,istr)

C...
      call mcfio_FPrintDictionary(lnhout)
      call cntcode(1)
      CALL hptrlstj
C          Fill Stdhep common block 1 (only valid for Pythia)
      call stdflpyxsec(nevt)
C          Write Stdhep begin-run record   
      call stdxwrt(100,istr,lok)
      if(lok.ne.0) write(lnhout,*)
     1        ' Problem writing stdhep begin run record'
c
      do 100 i=1,nevt
        call pyevnt
        call lunhep(1)
        if(i.le.nprnt)then
          call pylist(2)
          call heplst(1)
        endif
        call tstflem(20)
	call cntcode(2)
        call stdxwrt(1,istr,lok)
C          Fill Stdhep common block 1 (only valid for Pythia)
        call stdflpyxsec(nevt)
        call tstflxsec(20)
 100  continue
C
C          Fill Stdhep common block 1
      call stdflpyxsec(nevt)
C          Write end-of-run record  
      call stdxwrt(200,istr,lok)
      if(lok.ne.0) write(lnhout,*) ' Problem writing end run record'

C...   Print end of run statistics.
      mstat=1
      call pystat(mstat)

c...close event file
      call stdxend(istr)
      call cntcode(3)
C...print histograms
      call houtpu(lnhout)
      call histdo
c...close output file
      close(unit=lnhout)
      stop
      end

      subroutine upyinit(nevt,istr)
C
C   Pythia initialization
C
#include "pydat1.inc"
#include "stdlun.inc"

C...   PYTHIA initialization variables.
      character*4 frame
      character*5 beam,target
      integer nevt,istr
      double precision win
C
C initialize HEP logical units
      lnhwrt=23
      lnhrd=0
      lnhdcy=0
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      mstu(11)=lnhout
      open(unit=lnhout,file='pythiaExample.lpt',status='new')
      call stdxwinit('pythiaExample.io','StdHep/Pythia example',
     1               nevt,istr,lok)
C
      call tstbkem(20)
C
C...   Initialize PYTHIA
      frame  = 'cms'
      beam   = 'pbar'
      target = 'p'
      win    = 1800.0d0
      call pygive('msel=5')
      call pygive('mstj(21)=0')
      call pyinit(frame,beam,target,win)

      return
      end

      subroutine cntcode(ii)
c
c  make a table of isthep values
c
#include "stdhep.inc"
#include "stdlun.inc"
      integer ii,i,icnt(200)
      save icnt

      if(ii.eq.1)then
         do i=1,200
	   icnt(i)=0
	 enddo
      elseif(ii.eq.2)then
         do i=1,nhep
	   ist = isthep(i)
	   if(ist.le.200 .and. ist.gt.0) icnt(ist) = icnt(ist) + 1
	 enddo
      elseif(ii.eq.3)then
         write(lnhout,1001) 
	 do i=1,200
            write(lnhout,1002) i,icnt(i)
	 enddo
      else
      endif

      return
 1001 format(10x,'List of used isthep values')
 1002 format(' isthep ',i3,2x,i10)
      end
