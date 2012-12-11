
      subroutine pdgrdcsvtb
C
C   This routine reads the new, complete, mass and width table supplied by PDG
C   fill two common blocks 
C	PDGTBL - complete list of particles in PDG table
C	STDTBL - list of particles in PDG table defined by STDHEP
C
      implicit none
C
#include "stdlun.inc"
#include "stdcsv.inc"
C
      integer indx,itr,pdgtran,icnt
      character*100 pdgmasst
      character*100 stdhepdir

      integer hepcmp
      external hepcmp
C
      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if (lfirst) then
        call stdversn
        lfirst=.FALSE.
      endif

C  point to standard decay tables
      call getenv( 'PDG_MASS_TBL', pdgmasst )
      call getenv( 'STDHEP_DIR', stdhepdir )
      if ( pdgmasst .eq. ' ' ) then
         if( stdhepdir .eq. ' ' ) then
             write( lnhout, 113 )
	     go to 900
	 else
	     write(pdgmasst, fmt='(a,"mass_width_2006.csv")') stdhepdir
         endif
      endif
      write( lnhout, 111 ) pdgmasst
      open(unit=lnhdcy, file=pdgmasst, status='old')

      icnt = 0

 100  continue
      read (lnhdcy, fmt = '(a)', end = 600) cline
c make sure this is not an old style file.
      if (cline (1 : 1) .eq. 'M' .or. cline (1 : 1) .eq. 'W') then
           write(lnhout,112) 
	   return
      endif

      if (cline (1 : 1) .ne. '*') then
        call parsecsv

C... fill stdtbl 
        if( pdgid.ne.0)then
C... translate to STDHEP numbering
          itr = pdgtran(pdgid,1)
          if(itr.ne.0)then
C... get stdhep compressed index
            indx = hepcmp(itr)
            if(indx.gt.0)then
	        call fillstdtbl(itr,indx)
c  is there an antiparticle?
              if(antip.ne.' ')then
                itr = pdgtran(-pdgid,1)
                if(itr.ne.0)then
        	  indx = hepcmp(itr)
        	  if(indx.gt.0)then
	            call fillastdtbl(itr,indx)
        	  endif
        	endif
	      endif 
	    else
	      write(lnhout,104) itr, pdgid
            endif
C  use for debugging only - too many messages otherwise
C	  else
C	    write(lnhout,103) pdgid
          endif
        endif

C... fill pdgtbl 
        if(pdgid.ne.0)then
            icnt = icnt + 1
	    call fillpdgtbl(icnt)
        endif
c... finished processing cline
      endif
      go to 100

 600  close(unit=lnhdcy)
 900  continue
      return

 101  format(' Cannot compress particle  ',
     1     a1, 1x, a21, 1x, 4i8, 1x, e25.16, 1x, 2e9.1)
 102  format(' Cannot translate particle ',
     1     a1, 1x, a21, 1x, 4i8, 1x, e25.16, 1x, 2e9.1)
 103  format('PDGRDCSVTB: cannot translate PDG particle ',i6)
 104  format('PDGRDCSVTB: cannot compress StdHep ID ',i6,
     1          ' from PDG particle ',i6)
 111  format('PDGRDCSVTB: Reading PDG table: ', a )
 112  format('PDGRDCSVTB ERROR: cannot use this routine to read the ',
     1       'old style table'/' -------> exiting from PDGRDCSVTB')
 113  format('PDGRDCSVTB:  Error - PDG_MASS_TBL is undefined')
      end

      subroutine fillstdtbl(itr,indx)
C
C   add an entry to the STDTBL common block 
C
      implicit none
C
#include "stdtbl.inc"
#include "stdcsv.inc" 
#include "stdlun.inc"
C
      character*20 chepn
      integer indx,itr,ichg
C
      integer hepchg
      external hepchg
C
      call hepnam(itr,chepn)
      idt(indx) = itr
      stmass(indx) = mval
      stmerr(1,indx) = mpos
      stmerr(2,indx) = mneg
      stname(indx) = chepn
      stwidth(indx) = wval
      stwerr(1,indx) = wpos
      stwerr(2,indx) = wneg
      ichg = hepchg(itr)
      if(ichg.ne.chg)then
         write(lnhout,101) chepn,ichg,chg
         istchg(indx) = ichg
      else
         istchg(indx) = chg
      endif
C
      return
 101  format('FILLSTDTBL ERROR: charge mismatch for ',A20,2I10)
      end

      subroutine fillastdtbl(itr,indx)
C
C   add an antiparticle entry to the STDTBL common block 
C
      implicit none
C
#include "stdtbl.inc"
#include "stdcsv.inc" 
#include "stdlun.inc"
C
      character*20 chepn
      integer indx,itr,ichg,achg
C
      integer hepchg
      external hepchg
C
      call hepnam(itr,chepn)
      idt(indx) = itr
      stmass(indx) = mval
      stmerr(1,indx) = mpos
      stmerr(2,indx) = mneg
      stname(indx) = chepn
      stwidth(indx) = wval
      stwerr(1,indx) = wpos
      stwerr(2,indx) = wneg
      ichg = hepchg(itr)
      achg = -chg
      if(ichg.ne.achg)then
         write(lnhout,101) chepn,ichg,achg
         istchg(indx) = ichg
      else
         istchg(indx) = achg
      endif
C
      return
 101  format('FILLASTDTBL ERROR: charge mismatch for ',A20,2I10)
      end

      subroutine fillpdgtbl(indx)
C
C   add an entry to the PDGTBL common block 
C
      implicit none
C
#include "pdgtbl.inc"
#include "stdcsv.inc"
#include "stdlun.inc"
C
      integer indx,pdgtran,ichg,id

      integer hepchg
      external hepchg
C
      idpdg(indx) = pdgid
      pdmass(indx) = mval
      pdmerr(1,indx) = mpos
      pdmerr(2,indx) = mneg
      pdwidth(indx) = wval
      pdwerr(1,indx) = wpos
      pdwerr(2,indx) = wneg
      write(pdname(indx),fmt='(A17,A4)') name,cchg
      id = pdgtran(pdgid,1)
      if(id.ne.0)then
	ichg = hepchg(id)
	if(ichg.ne.chg)then
           write(lnhout,101) pdname(indx),id,ichg,chg
           ipdchg(indx) = ichg
	else
           ipdchg(indx) = chg
	endif
      else
c not really an error
c        write(lnhout,102) pdname(indx),pdgid
        ipdchg(indx) = ichg
      endif
C
      return
 101  format('FILLPDGTBL ERROR: charge mismatch for ',A18,I11,2I10)
 102  format('FILLPDGTBL ERROR: cannot translate ',A18,I11)
      end
