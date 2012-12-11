      subroutine hptrlstj
C
C    compare particle ID for pythia and standard
C
#include "stdlun.inc"
      integer id1,id2,it1,it2,itt1,itt2,itmp
      integer i,j,k,l,ll
      integer lutran,pycomp
      character*16 cnam1,cnam2
      character*20 hnam1,hnam2

C...special cases
      write(lnhout,1001)
      do 100 i=1,100
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
	  itmp=pycomp(id2)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0 .or. itmp.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
 100  continue
      do 150 j=11,300
        i=j*10
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
 150  continue
      do i=1000001,1000039
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
      enddo
      do i=2000001,2000020
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
      enddo
      do i=3000001,3500000
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
      enddo
      do i=4000001,5000050
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
      enddo
      do i=9900001,9911000
        itmp=pycomp(i)
        if(itmp.gt.0)then
          id1=i
          it1=lutran(id1,1)
          itt1=lutran(it1,2)
          call pyname(id1,cnam1)
          call hepnam(it1,hnam1)
          id2=-i
          it2=lutran(id2,1)
          if(it1.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          elseif(it2.eq.0)then
            write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
          else
            itt2=lutran(it2,2)
            call pyname(id2,cnam2)
            call hepnam(it2,hnam2)
            write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1               id2,it2,hnam2,itt2,cnam2
          endif
        endif
      enddo
C...diquarks
      write(lnhout,1002)
      do 200 i=11,99
        do 190 j=1,10
          id1=100*i+j-1
          itmp=pycomp(id1)
          if(itmp.gt.0)then
            id2=-id1
            it1=lutran(id1,1)
            itt1=lutran(it1,2)
            call pyname(id1,cnam1)
            call hepnam(it1,hnam1)
            it2=lutran(id2,1)
            if(it1.eq.0)then
              write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
            elseif(it2.eq.0)then
              write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
            else
              itt2=lutran(it2,2)
              call pyname(id2,cnam2)
              call hepnam(it2,hnam2)
              write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,id2,
     1                it2,hnam2,itt2,cnam2
            endif
          endif
 190    continue
 200  continue
C...mesons
      write(lnhout,1003)
      do 300 i=1,9
        do 290 j=1,9
          do 280 k=1,9
            do 270 l=1,10
	      do ll=1,2
        	id1=100000*(ll-1)+10000*(l-1)+100*i+10*j+k
        	itmp=pycomp(id1)
        	if(itmp.gt.0)then
                  id2=-id1
                  it1=lutran(id1,1)
                  itt1=lutran(it1,2)
                  call pyname(id1,cnam1)
                  call hepnam(it1,hnam1)
                  it2=lutran(id2,1)
                  if(it1.eq.0)then
                    write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
                  elseif(it2.eq.0)then
                    write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
                  else
                    itt2=lutran(it2,2)
                    call pyname(id2,cnam2)
                    call hepnam(it2,hnam2)
                    write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1                   id2,it2,hnam2,itt2,cnam2
                  endif
        	endif
	      enddo
 270        continue
 280      continue
 290    continue
 300  continue
C...baryons
      write(lnhout,1004)
      do 400 i=1,9
        do 390 j=1,9
          do 380 k=1,9
            do 370 l=1,9
              id1=1000*i+100*j+10*k+l
              itmp=pycomp(id1)
              if(itmp.gt.0)then
                id2=-id1
                it1=lutran(id1,1)
                itt1=lutran(it1,2)
                call pyname(id1,cnam1)
                call hepnam(it1,hnam1)
                it2=lutran(id2,1)
                if(it1.eq.0)then
                  write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
                elseif(it2.eq.0)then
                  write(lnhout,1101) id1,it1,hnam1,itt1,cnam1
                else
                  itt2=lutran(it2,2)
                  call pyname(id2,cnam2)
                  call hepnam(it2,hnam2)
                  write(lnhout,1102) id1,it1,hnam1,itt1,cnam1,
     1                      id2,it2,hnam2,itt2,cnam2
                endif
              endif
 370        continue
 380      continue
 390    continue
 400  continue
      return
 1001 format(//5X,'Special Cases'/5X,'IJET',5X,'ISTD  HNAM',16x,
     1     'IJET  JNAME',13X,'IJET',5X,'ISTD  HNAM',16x,'IJET  JNAME')
 1002 format(//5X,'Diquarks'/5X,'IJET',5X,'ISTD  HNAM',16x,
     1     'IJET  JNAME',13X,'IJET',5X,'ISTD  HNAM',16x,'IJET  JNAME')
 1003 format(//5X,'Mesons'/5X,'IJET',5X,'ISTD  HNAM',16x,'IJET  JNAME',
     1     13X,'IJET',4X,'ISTD  HNAM',16x,'IJET  JNAME')
 1004 format(//5X,'Baryons'/5X,'IJET',5X,'ISTD  HNAM',16x,'IJET  JNAME',
     1     13X,'IJET',5X,'ISTD  HNAM',16x,'IJET  JNAME')
 1101 format(1X,I8,1X,I8,2X,a16,1X,I8,2X,a16)
 1102 format(1X,I8,1X,I8,2X,a16,1X,I8,2X,a16,
     1          I8,1X,I8,1X,a16,1X,I8,1X,a16)
      end
