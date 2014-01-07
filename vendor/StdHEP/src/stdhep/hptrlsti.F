      subroutine HPTRLSTI

C...writes a table of corresponding ISAJET and STDHEP particle ID's

#include "stdlun.inc"

      integer I,ID1

      write(lnhout,101)
      write(lnhout,102)
c special cases
      do  I=1,99
        ID1 = I
	call printinfo(id1)
      enddo
c deal with IDTAUL and IDTAUR (saved in isajet as parameters)
c these "particles" are used to preserve polarization info when NOTAU is set
      call printinfo(10016)
      call printinfo(20016)
c everything else
      do  I=100,100000
        if(i.ne.10016 .and. i.ne.20016)then
	  ID1 = I
	  if(I.EQ.100) write(lnhout,103)
	  if(I.EQ.1000) write(lnhout,104)
	  if(I.EQ.10000) write(lnhout,105)
	  call printinfo(id1)
	endif
      enddo
      return
 101  format(//10X,'ISAJET particle translation'/)
 102  format(//5X,'Special Cases'/4X,'IISA',4X,'ISTD  HNAM',15X,
     1     'IISA  INAME',11X,'IISA',4X,'ISTD  HNAM',15X,'IISA  INAME')
 103  format(//5X,'Mesons'/4X,'IISA',4X,'ISTD  HNAM',15X,'IISA  INAME',
     1     11X,'IISA',4X,'ISTD  HNAM',15X,'IISA  INAME')
 104  format(/5X,'Baryons and Diquarks'/4X,'IISA',4X,'IPDG  HNAM',15X,
     1     'IISA  INAME',11X,'IISA',4X,'IPDG  HNAM',15X,'IISA  INAME')
 105  format(/5X,'Excited Particles'/4X,'IISA',4X,'IPDG  HNAM',15X,
     1     'IISA  INAME',11X,'IISA',4X,'IPDG  HNAM',15X,'IISA  INAME')
      end

      subroutine printinfo(id1)

#include "stdlun.inc"

      integer ifL1,ifL2,ifL3,JSPIN,IDEX
      integer ID1,ID2,IT1,IT2,ITT1,ITT2,istran
      character*8 LB1,LB2,LABEL
      character*20 HNAM1,HNAM2

        ID2=-ID1
        call FLAVOR(ID1,ifL1,ifL2,ifL3,JSPIN,IDEX)
        if(IDEX.GT.0)then
          LB1 = LABEL(ID1)
          if(LB1.NE.'LERR' .AND. LB1.NE.'ERR')then
            IT1=istran(ID1,1)
            IT2=istran(ID2,1)
            ITT1=istran(IT1,2)
            LB2=LABEL(ID2)
            call HEPNAM(IT1,HNAM1)
            if(LB2.NE.'LERR' .AND. LB2.NE.'ERR' .AND. IT2.NE.0)then
              ITT2=istran(IT2,2)
              call HEPNAM(IT2,HNAM2)
              write(lnhout,1102) ID1,IT1,HNAM1,ITT1,LB1,
     1              ID2,IT2,HNAM2,ITT2,LB2
            else
              write(lnhout,1101) ID1,IT1,HNAM1,ITT1,LB1
            endif
          endif
        endif
      return
1101  format(1X,I7,1X,I8,2X,a16,1X,I7,2X,A8)
1102  format(1X,I7,1X,I8,2X,a16,1X,I7,2X,A8,
     1       5X,I7,1X,I8,2X,a16,1X,I7,2X,A8)
      end

