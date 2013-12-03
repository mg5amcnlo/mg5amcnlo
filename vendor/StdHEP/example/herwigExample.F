      PROGRAM TSTHWG
C
C      test program
C      compare event list in herwig and standard structure
#include "herwig.inc"
#include "stdlun.inc"

      external hwudat
      integer n,lok
      integer istr
C
C initialize HEP logical units
      lnhwrt=23
      lnhrd=0
      lnhdcy=0
      lnhout=22
      lnhpdf=0
      lnhdmp=0
      lhwout=lnhout
      open(unit=lnhout,file='herwigExample.lpt',status='new')
      call tstbkem(0)
      call cntcode(1)
C
      call hptrlsth
C---max number of events this run
      maxev=100
C open XDR output file
      call stdxwinit('herwigExample.io','StdHep/Herwig example',
     1               maxev,istr,lok)
C---process
      iproc=1500
C---beams
      pbeam1=900.
      pbeam2=900.
      part1='PBAR    '
      part2='P       '
C---initialise other common blocks
      call hwigin
C---user can reset parameters at this point,
C   otherwise values set in HWIGIN will be used.
      lrsud=0
      lwsud=0
      lwdec=0
      maxpr=1
      nrn(1)=106645411
      nrn(2)=135135175
      maxer=100
      tlout=5.
      ptmin=10.
      ptmax=80.
      emmin=40.
      emmax=80.
      q2min=1600.
      q2max=3600.
C---HIGGS mass
      rmass(201)=400.
C---compute parameter-dependent constants
      iprint = 2
      call hwuinc
      iprint = 1
C---call HWUSTA to make any particle stable
      call hwusta('PI0     ')
C---user's initial calculations
      call hwabeg
C---initialise elementary process
      call hweini
C          Fill Stdhep common block 1
        call stdflhwxsec
C          Write Stdhep begin-run record   
        call stdxwrt(100,istr,lok)
        if(lok.ne.0) write(lnhout,*)
     1        ' Problem writing stdhep begin run record'
C---loop over events
      do 100 n=1,maxev
C---initialise event
        call hwuine
C---generate hard subprocess
        call hwepro
C---generate parton cascades
        call hwbgen
C---do heavy quark decays  - NOTE new routine name as of HERWIG 6.1
        call hwdhob
C---do cluster hadronization
        call hwcfor
C---do cluster decay
        call hwcdec
C---do unstable particle decays
        call hwdhad
C---do heavy flavour decays
        call hwdhvy
C---add soft underlying event if needed
        call hwmevt
C---finalise event
        call hwufne
c---user's event analysis
        call hwghep(1)
        call hwanal
        if(nevhep.le.maxpr) then
          call heplst(1)
          call heptree
        endif
        call tstflem(0)
	call cntcode(2)
        call stdxwrt(1,istr,lok)
C          Fill Stdhep common block 1
        call stdflhwxsec
        call tstflxsec(0)
  100 continue
C---terminate elementary process
      call hwefin
C---user's terminal calculations
      call hwaend
C          Fill Stdhep common block 1
        call stdflhwxsec
C          Write Stdhep end-run record   
        call stdxwrt(200,istr,lok)
        if(lok.ne.0) write(lnhout,*)
     1        ' Problem writing StdHep end run record'
c...close event file
      call stdxend(istr)
      call cntcode(3)
C...print histograms
      call houtpu(lnhout)
      call histdo
      close(unit=lnhout)
      close(unit=lnhwrt)
      stop
      end
C----------------------------------------------------------------------
      subroutine hwabeg
C... user's routine for initialization
      end
      subroutine hwaend
C... user's routine for terminal calculations, histogram output, etc
      end
      subroutine hwanal
C... user's routine to analyse data from event
      end
C----------------------------------------------------------------------

      subroutine cntcode(ii)
c
c  make a table of isthep values
c
#include "stdhep.inc"
#include "stdlun.inc"
      integer ii,i,ist,icnt(200)
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

