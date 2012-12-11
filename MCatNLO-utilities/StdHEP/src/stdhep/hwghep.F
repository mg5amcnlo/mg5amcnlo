
      subroutine HWGHEP(mconv)

C...Purpose: to convert HERWIG event record contents to or from
C...the standard event record common block.
C...convert (mconv=1) from HERWIG numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to HERWIG numbering scheme

C...Only IDHEP needs to be changed
C   1/28/98 - Overwrite particle ID as used in Herwig

#include "herwig.inc"

      integer itmp,hwtran,mconv,I,J

      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
C...Conversion from HERWIG to standard.
      if(mconv.EQ.1) then
        do 140 I=1,NHEP
          if(IDHEP(I).NE.0)then
            itmp=IDHEP(I)
            IDHEP(I) = hwtran(itmp,1)
          elseif(IDHW(I).EQ.20)then
C...????
            IDHEP(I)=89
          elseif(IDHW(I).LE.18)then
C...the oddball intermediate states
            IDHEP(I)=IDHW(I)+67
          else
C...undefined
            IDHEP(I)=0
          endif
 140    CONTINUE
C...Conversion from standard to HERWIG.
      elseif(mconv.EQ.2)then
        do 180 I=1,NHEP
          itmp=IDHEP(I)
          IDHEP(I) = hwtran(itmp,2)
 180    CONTINUE
      endif
      return
      end
