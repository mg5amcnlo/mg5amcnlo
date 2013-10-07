      program testpdgtran
c
c  test pdgtonew translation routine
c
      implicit none

#include "stdlun.inc"
#include "pdgtbl.inc"

      integer idold1,idold2,idnew1,idnew2,itmp1,itmp2
      integer i,j,k,l,m

      integer pdgtonew
      external pdgtonew
c
      lnhout = 22
      lnhdcy = 24

      open(unit=lnhout,file='testpdgtran.lpt',status='new')

c...special cases
      write(lnhout,1001)
      do i=1,100
        idold1=i
        idold2=-i
        idnew1=pdgtonew(idold1,1)
        if(idnew1.gt.0)then
          idnew2=pdgtonew(idold2,1)
          itmp1=pdgtonew(idnew1,2)
          if(idnew2.eq.0)then
            write(lnhout,1101) idold1,idnew1,itmp1
          else
            itmp2=pdgtonew(idnew2,2)
            write(lnhout,1102) idold1,idnew1,itmp1,idold2,idnew2,itmp2
          endif
        endif
      enddo
      do j=11,33
        i=j*10
        idold1=i
        idold2=-i
        idnew1=pdgtonew(idold1,1)
        if(idnew1.gt.0)then
          idnew2=pdgtonew(idold2,1)
          itmp1=pdgtonew(idnew1,2)
          if(idnew2.eq.0)then
            write(lnhout,1101) idold1,idnew1,itmp1
          else
            itmp2=pdgtonew(idnew2,2)
            write(lnhout,1102) idold1,idnew1,itmp1,idold2,idnew2,itmp2
          endif
        endif
      enddo
c...diquarks
      do i=1,3
        do j=1,2
          do k=1,3,2
              idold1 = i*1000 + j*100 + k
              idold2 = -idold1
              idnew1 = pdgtonew(idold1,1)
              if(idnew1.gt.0)then
        	idnew2=pdgtonew(idold2,1)
        	itmp1=pdgtonew(idnew1,2)
        	if(idnew2.eq.0)then
        	  write(lnhout,1101) idold1,idnew1,itmp1
        	else
        	  itmp2=pdgtonew(idnew2,2)
        	  write(lnhout,1102) 
     1        	        idold1,idnew1,itmp1,idold2,idnew2,itmp2
        	endif
              endif
          enddo
        enddo
      enddo
c...mesons
      do i=1,5
        do j=1,i
	  do k=0,4
	    do l=1,8
	      idold1 = 10000*(l-1)+100*i+10*j+2*k+1
              idold2=-idold1
              idnew1=pdgtonew(idold1,1)
              if(idnew1.gt.0)then
        	idnew2=pdgtonew(idold2,1)
        	itmp1=pdgtonew(idnew1,2)
        	if(idnew2.eq.0)then
        	  write(lnhout,1101) idold1,idnew1,itmp1
        	else
        	  itmp2=pdgtonew(idnew2,2)
        	  write(lnhout,1102) idold1,idnew1,itmp1,idold2,idnew2,itmp2
        	endif
              endif
	    enddo
	  enddo
	enddo
      enddo
c...baryons
      do i=1,5
        do j=1,5
	  do k=1,5
	    do l=1,4
	      do m=1,6
		idold1 = 10000*(m-1)+1000*i+100*j+10*k+2*l
        	idold2=-idold1
        	idnew1=pdgtonew(idold1,1)
        	if(idnew1.gt.0)then
        	  idnew2=pdgtonew(idold2,1)
        	  itmp1=pdgtonew(idnew1,2)
        	  if(idnew2.eq.0)then
        	    write(lnhout,1101) idold1,idnew1,itmp1
        	  else
        	    itmp2=pdgtonew(idnew2,2)
        	    write(lnhout,1102) idold1,idnew1,itmp1,idold2,idnew2,itmp2
        	  endif
        	endif
	      enddo
	    enddo
	  enddo
	enddo
      enddo

      close(unit=lnhout)

 1001 format(//4x,'PDG1996   PDG2000',3x,'PDG1996',
     1         6x,'PDG1996   PDG2000',3x,'PDG1996')
 1101 format(1x,3i10)
 1102 format(1x,3i10,2x,3i10)
      end
