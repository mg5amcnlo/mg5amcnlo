
      subroutine PDGRDTB
C
C   read mass and width table supplied by PDG
C   fill two common blocks 
C	PDGTBL - complete list of particles in PDG table
C	STDTBL - list of particles in PDG table defined by STDHEP
C
#include "pdgtbl.inc"
#include "stdtbl.inc"
#include "stdlun.inc"
C
      character cline*130
      character ctype*1
      character cpartic*21,chepn*20
      integer i,j,iidnum,indx,itr,pdgtran,icnt,icnt2
      integer aidnum(4),itmp(nmxln2)
      real rpos,rneg,tmpl(nmxln2),tmph(nmxln2)
      double precision rval,tmpw(nmxln2)
      character*100 pdgmasst
      character*100 stdhepdir

      integer hepcmp
      external hepcmp
C
      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
C  point to standard decay tables
      call getenv( 'PDG_MASS_TBL', pdgmasst )
      call getenv( 'STDHEP_DIR', stdhepdir )
      if ( pdgmasst .eq. ' ' ) then
         if( stdhepdir .eq. ' ' ) then
C             pdgmasst = 'mass_width_2006.mc'
             write( lnhout, 113 )
	     go to 900
	 else
	     write(pdgmasst, fmt='(a,"mass_width_2006.mc")') stdhepdir
         endif
      endif
      write( lnhout, 112 ) pdgmasst
      open(unit=lnhdcy, file=pdgmasst, status='old')
      icnt = 0
      icnt2 = 0
 100  continue
      read (lnhdcy, fmt = '(a)', end = 600) cline
      if (cline (1 : 1) .eq. 'M' .or. cline (1 : 1) .eq. 'W') then
          read (cline,
     &    fmt = '(bn, a1, 4i8, 1x, e15.0, 1x,e8.0, 1x,e8.0, 1x, a21)')
     &    ctype, (aidnum(iidnum), iidnum = 1, 4), rval, rpos, rneg,
     &    cpartic
C... fill stdtbl here
        do j=1,4
          if(aidnum(j).ne.0)then
C... translate to STDHEP numbering
            itr = pdgtran(aidnum(j),1)
            if(itr.ne.0)then
C... get stdhep compressed index
              indx = hepcmp(itr)
              if(indx.gt.0)then
                if(ctype.eq.'M')then
C... this is a mass definition
                  call hepnam(itr,chepn)
                  idt(indx) = itr
                  stmass(indx) = rval
                  stmerr(1,indx) = rpos
                  stmerr(2,indx) = rneg
                  stname(indx) = chepn
                elseif(ctype.eq.'W')then
C... this is a width definition
                  stwidth(indx) = rval
                  stwerr(1,indx) = rpos
                  stwerr(2,indx) = rneg
                endif
	      else
	        write(lnhout,104) itr, aidnum(j)
              endif
C  use for debugging only - too many messages otherwise
C	    else
C	      write(lnhout,103) aidnum(j)
            endif
          endif
        enddo
C... fill pdgtbl here
        do j=1,4
          if(aidnum(j).ne.0)then
            if(ctype.eq.'M')then
C... this is a mass definition
              icnt = icnt + 1
              idpdg(icnt) = aidnum(j)
              pdmass(icnt) = rval
              pdmerr(1,icnt) = rpos
              pdmerr(2,icnt) = rneg
              pdname(icnt)=cpartic
            elseif(ctype.eq.'W')then
C... this is a width definition
              icnt2 = icnt2 + 1
              itmp(icnt2) = aidnum(j)
              tmpw(icnt2) = rval
              tmph(icnt2) = rpos
              tmpl(icnt2) = rneg
            endif
          endif
        enddo
      endif
      go to 100
 600  close(unit=lnhdcy)
C... match widths to masses for PDG list
      do i=1,icnt2
        do j=1,icnt
          if(itmp(i).eq.idpdg(j))then
            pdwidth(j) = tmpw(i)
            pdwerr(1,j) = tmph(i)
            pdwerr(2,j) = tmpl(i)
            go to 610
          endif
        enddo
C... no match??
        write(lnhout,111) itmp(i)
 610    continue
      enddo
 900  continue
      return
 101  format(' Cannot compress particle  ',
     1     a1, 1x, a21, 1x, 4i8, 1x, e25.16, 1x, 2e9.1)
 102  format(' Cannot translate particle ',
     1     a1, 1x, a21, 1x, 4i8, 1x, e25.16, 1x, 2e9.1)
 103  format('PDGRDTB: cannot translate PDG particle ',i6)
 104  format('PDGRDTB: cannot compress StdHep ID ',i6,
     1          ' from PDG particle ',i6)
 111  format(' Failed to find match of width for ',i8)
 112  format('PDGRDTB: Reading PDG table: ', a )
 113  format('PDGRDTB:  Error - PDG_MASS_TBL is undefined')
      end
