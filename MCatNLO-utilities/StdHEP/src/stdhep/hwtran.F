      integer function hwtran(id,mconv)
C...convert (mconv=1) from HERWIG numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to HERWIG numbering scheme
C
C           ID = particle identification number

#include "stdlun.inc"
C... ITABJ(I) converts miscellaneous HERWIG particle ID's to a standard scheme
      integer itabj(99,2), noant(7)
      data itabj/1,2,3,4,5,6,7,8,0,0,
     1           11,12,13,14,15,16,0,0,0,0,
     2           21,22,23,24,25,51,0,0,0,0,
     3           0,32,0,0,35,36,37,0,39,0,
     4           0,0,0,0,0,0,0,0,0,0,
     5           0,0,0,0,0,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,9920022,9922212,
     *           1,2,3,4,5,6,7,8,0,0,
     1           11,12,13,14,15,16,0,0,0,0,
     2           21,22,23,24,25,0,0,0,0,0,
     3           0,32,0,0,35,36,37,0,39,0,
     4           0,0,0,0,0,0,0,0,0,0,
     5           26,0,0,0,0,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,0,0/
      data noant/-21,-22,-23,-25,-51,-35,-36/
      save itabj,noant

      hwtran=id
      ida=iabs(id)
      j1=mod(ida,10)
      i1=mod(ida/10,10)
      i2=mod(ida/100,10)
      i3=mod(ida/1000,10)
      i4=mod(ida/10000,10)
      i5=mod(ida/100000,10)
      k99 = mod(ida/100000,100)
      ksusy = mod(ida/1000000,10)
      ku = mod(ida/10000000,10)
      kqn=mod(ida/1000000000,10)

      if(ida.eq.0)then
C..        write(lnhout,*) ' HWTRAN 1: particle ID is zero'
      elseif(mconv.lt.1 .or. mconv.gt.2)then
        hwtran=0
        write(lnhout,*) ' HWTRAN 2: unallowed conversion option'
      elseif(kqn.eq.1) then
C... ions not recognized
        hwtran=0
        write(lnhout,*) ' HWTRAN 3: nonallowed ion'
      elseif(ida.lt.100)then
C...Higgs, etc.
        hwtran=isign(itabj(ida,mconv),id)
C...check for illegal antiparticles
        if(id.lt.0)then
          if(hwtran.ge.-99 .and. hwtran.le.-81) hwtran=0
          do 101 j=1,7
            if(hwtran.eq.noant(j)) hwtran=0
 101      continue
        endif
      elseif(k99.eq.99 .and. mconv.eq.2) then
C... special case remnant particles - used internally
        hwtran = 0
        if(id.eq.9920022) hwtran=isign(98,id)
        if(ida.eq.9922212) hwtran=isign(99,id)
      elseif(ksusy.eq.1 .or. ksusy.eq.2) then
C... SUSY
      elseif(i1.ne.0 .and. i3.ne.0 .and. j1.eq.2)then
C...spin 1/2 baryons
      elseif(i1.ne.0 .and. i3.ne.0 .and. j1.eq.4)then
C...spin 3/2 baryons
      elseif(i1.ne.0 .and. i2.ne.0 .and. i3.eq.0)then
C...mesons 
C...check for illegal antiparticles
        if(i1.eq.i2 .and. id.lt.0) hwtran=0
      elseif(i2.ne.0 .and. i3.ne.0 .and. i1.eq.0)then
C...diquarks
      else
C...undefined
        hwtran=0
      endif
C...check for illegal anti KS, KL
      if(id.eq.-130 .or. id.eq.-310) hwtran=0

      if(hwtran.eq.0 .and. ida.ne.0)then
        if(mconv.eq.1) write(lnhout,111) id
        if(mconv.eq.2) write(lnhout,112) id
      endif
      return
 111  format('  HWTRAN 3: HERWIG particle ',I8,' translates to zero')
 112  format('  HWTRAN 3: PDG particle ',I8,' translates to zero')
      end
