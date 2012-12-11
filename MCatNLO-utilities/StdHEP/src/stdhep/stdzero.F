
      subroutine STDZERO

C...Purpose: to zero the standard common block.
C
#include "stdhep.inc"
#include "hepev4.inc"

      integer J,K

C...set everything to zero
      NHEP = 0
      nmulti = 0
      do 120 J=1,NMXHEP
        ISTHEP(J)=0
        IDHEP(J)=0
        jmulti(J)=0
        do 100 K=1,2
          JMOHEP(K,J)=0
          JDAHEP(K,J)=0
  100     icolorflowlh(K,J)=0
        do 105 K=1,5
  105     PHEP(K,J)=0.
        do 110 K=1,4
  110     VHEP(K,J)=0.
        do K=1,3
	   spinlh(K,J) = 0.
	enddo
  120 CONTINUE
      do j=1,NMXMLT
         nevmulti(j)=0
	 itrkmulti(j)=0
	 mltstr(j)=0
	 eventweightmulti(j)=0.
	 alphaqedmulti(j)=0.
	 alphaqcdmulti(j)=0.
	 do k=1,5
	     scalemulti(k,j)=0.
	 enddo
	 idrupmulti(j)=0
      enddo
      eventweightlh = 0.
      alphaqedlh = 0.
      alphaqcdlh = 0.
      do j=1,5
          scalelh(j) = 0.
      enddo
      idruplh = 0
      return
      end
