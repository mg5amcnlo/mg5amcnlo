      program testcsvpdg
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
C
C initialize HEP logical units
      lnhwrt=0
      lnhrd=0
      lnhdcy=24
      lnhout=22
      lnhpdf=0
      lnhdmp=0

      open(unit=lnhout,file='testcsvpdg.lpt',status='new')

c read the table
      call pdgrdcsvtb
c print the table
      call pdgprtb(1,lnhout)
      call pdgprtb(2,lnhout)
      call pdgprtb(3,lnhout)

      close(unit=lnhout)

 1001 format(//' --- calling readpdg --- ')
      end
