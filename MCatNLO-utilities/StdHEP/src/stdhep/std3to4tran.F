      integer function std3to4tran(id,mconv)
C...convert (mconv=1) from StdHep 3.04 numbering scheme to StdHep 4.10/5.01 numbering scheme
C...     or (mconv=2) from StdHep 4.10/5.01 numbering scheme to StdHep 3.04 numbering scheme
C
C           ID = particle identification number

#include "stdlun.inc"
C... ITABJ(I) converts miscellaneous PDG92 particle ID's to a standard scheme
      integer itabj(100,2), noant(6,2)

      data itabj/1,2,3,4,5,6,7,8,0,0,
     1           11,12,13,14,15,16,17,18,110,990,
     2           21,22,23,24,25,0,3000221,42,41,32,
     3           34,35,37,52,33,36,53,54,55,0,
     4           1000001,1000002,1000003,1000004,1000005,1000006,
     4           1000021,1000022,0,0,
     5           1000011,1000012,1000013,1000014,1000015,1000016,
     5           51,35,36,0,
     6           2000001,2000002,2000003,2000004,2000005,2000006,
     6           1000022,1000023,1000025,1000035,
     7           2000011,2000012,2000013,2000014,2000015,2000016,
     7           1000024,1000037,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,98,99,0,
     *           1,2,3,4,5,6,7,8,0,0,
     1           11,12,13,14,15,16,17,18,0,0,
     2           21,22,23,24,25,0,0,0,0,0,
     3           0,30,35,31,32,36,33,0,0,0,
     4           29,28,0,0,0,0,0,0,0,0,
     5           57,34,37,38,39,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,98,99,0/
      data noant/-21,-22,-23,-25,-35,-36,-21,-22,-23,-25,-35,-36/
      save itabj,noant

      std3to4tran=id
      ida=iabs(id)
      j1=mod(ida,10)
      i1=mod(ida/10,10)
      i2=mod(ida/100,10)
      i3=mod(ida/1000,10)
      kl=mod(ida/10000,10)
      ksusy = mod(ida/1000000,10)
      k99 = mod(ida/100000,100)
      kqn=mod(ida/1000000000,10)

      if(ida.eq.0)then
#if HEPDBG
        write(lnhout,*) ' STD3TO4TRAN 1: particle ID is zero'
#endif
      elseif(mconv.lt.1 .or. mconv.gt.2)then
        std3to4tran=0
        write(lnhout,*) ' STD3TO4TRAN 2: unallowed conversion option'
C... ions are unchanged
      elseif(kqn.eq.1) then
C...Higgs, etc.
      elseif(ida.le.100)then
        std3to4tran=isign(itabj(ida,mconv),id)
C...check for illegal antiparticles
        if(id.lt.0)then
          do 101 j=1,6
            if(std3to4tran.eq.noant(j,mconv)) std3to4tran = 0
 101      continue
        endif
C...SUSY
      elseif(ksusy.eq.1 .or. ksusy.eq.2)then
        std3to4tran=0
        if(mconv.eq.2)then
          if(ida.eq.1000001) std3to4tran=isign(41,id)
          if(ida.eq.1000002) std3to4tran=isign(42,id)
          if(ida.eq.1000003) std3to4tran=isign(43,id)
          if(ida.eq.1000004) std3to4tran=isign(44,id)
          if(ida.eq.1000005) std3to4tran=isign(45,id)
          if(ida.eq.1000006) std3to4tran=isign(46,id)
          if(ida.eq.1000011) std3to4tran=isign(51,id)
          if(ida.eq.1000012) std3to4tran=isign(52,id)
          if(ida.eq.1000013) std3to4tran=isign(53,id)
          if(ida.eq.1000014) std3to4tran=isign(54,id)
          if(ida.eq.1000015) std3to4tran=isign(55,id)
          if(ida.eq.1000016) std3to4tran=isign(56,id)
          if(ida.eq.1000021) std3to4tran=isign(47,id)
          if(ida.eq.1000022) std3to4tran=isign(67,id)
          if(ida.eq.1000023) std3to4tran=isign(68,id)
          if(ida.eq.1000024) std3to4tran=isign(77,id)
          if(ida.eq.1000025) std3to4tran=isign(69,id)
          if(ida.eq.1000035) std3to4tran=isign(70,id)
          if(ida.eq.1000037) std3to4tran=isign(78,id)
          if(ida.eq.2000001) std3to4tran=isign(61,id)
          if(ida.eq.2000002) std3to4tran=isign(62,id)
          if(ida.eq.2000003) std3to4tran=isign(63,id)
          if(ida.eq.2000004) std3to4tran=isign(64,id)
          if(ida.eq.2000005) std3to4tran=isign(65,id)
          if(ida.eq.2000006) std3to4tran=isign(66,id)
          if(ida.eq.2000011) std3to4tran=isign(71,id)
          if(ida.eq.2000012) std3to4tran=isign(72,id)
          if(ida.eq.2000013) std3to4tran=isign(73,id)
          if(ida.eq.2000014) std3to4tran=isign(74,id)
          if(ida.eq.2000015) std3to4tran=isign(75,id)
          if(ida.eq.2000016) std3to4tran=isign(76,id)
	endif
C...Technicolor
      elseif(ksusy.eq.3)then
        std3to4tran=0
        if(mconv.eq.2)then
	  if(id.eq.3000221) std3to4tran=27
	endif
C...excited particles
      elseif(ksusy.eq.4)then
        std3to4tran=0
C...illegal
      elseif(ksusy.gt.4 .and. ksusy.ne.9)then
        std3to4tran=0
C...miscellaneous
      elseif(k99.eq.99)then
        std3to4tran=0
        if(mconv.eq.2)then
          if(id.eq.9910113) std3to4tran=isign(110,id)
          if(ida.eq.9910211) std3to4tran=isign(210,id)
          if(id.eq.9910223) std3to4tran=isign(220,id)
          if(id.eq.9910333) std3to4tran=isign(330,id)
          if(id.eq.9910443) std3to4tran=isign(440,id)
          if(ida.eq.9912112) std3to4tran=isign(2110,id)
          if(ida.eq.9912212) std3to4tran=isign(2210,id)
          if(id.eq.9920022) std3to4tran=isign(9998,id)
          if(ida.eq.9922212) std3to4tran=isign(9999,id)
	endif
C...miscellaneous
      elseif(j1.eq.0)then
        std3to4tran=0
        if(id.eq.130 .or. id.eq.310) std3to4tran=id
        if(mconv.eq.1)then
          if(id.eq.110) std3to4tran=isign(9910113,id)
          if(ida.eq.210) std3to4tran=isign(9910211,id)
          if(id.eq.220) std3to4tran=isign(9910223,id)
          if(id.eq.330) std3to4tran=isign(9910333,id)
          if(id.eq.440) std3to4tran=isign(9910443,id)
          if(ida.eq.2110) std3to4tran=isign(9912112,id)
          if(ida.eq.2210) std3to4tran=isign(9912212,id)
	elseif(mconv.eq.2)then
          if(id.eq.110) std3to4tran=isign(19,id)
          if(id.eq.990) std3to4tran=isign(20,id)
	endif
C...more miscellaneous
      elseif(j1.ge.8 .and. i1.eq.9 .and. i2.eq.9 .and. i3.eq.9)then
        std3to4tran=0
        if(mconv.eq.1)then
	  if(id.eq.9998) std3to4tran=isign(9920022,id)
	  if(ida.eq.9999) std3to4tran=isign(9922212,id)
	endif
C... baryons
      elseif(i1.ne.0 .and. i2.ne.0 .and. i3.ne.0)then
        if(ida.gt.9999)then
          std3to4tran=0
C...only spin 1/2 and 3/2 allowed
        elseif(j1.ne.2 .and. j1.ne.4)then
          std3to4tran=0
	endif
C...mesons 
      elseif(i1.ne.0 .and. i2.ne.0 .and. i3.eq.0)then
        if(mconv.eq.1)then
          if(ida.eq.10111) std3to4tran=isign(9000111,id)
          if(ida.eq.10211) std3to4tran=isign(9000211,id)
          if(ida.eq.20111) std3to4tran=isign(100111,id)
          if(ida.eq.20211) std3to4tran=isign(100211,id)
          if(ida.eq.30113) std3to4tran=isign(100113,id)
          if(ida.eq.30213) std3to4tran=isign(100213,id)
          if(ida.eq.40113) std3to4tran=isign(30113,id)
          if(ida.eq.40213) std3to4tran=isign(30213,id)
          if(ida.eq.10221) std3to4tran=isign(9010221,id)
          if(ida.eq.20221) std3to4tran=isign(100221,id)
          if(ida.eq.30221) std3to4tran=isign(10221,id)
          if(ida.eq.10331) std3to4tran=isign(10221,id)
          if(ida.eq.40221) std3to4tran=isign(100331,id)
          if(ida.eq.50221) std3to4tran=isign(9020221,id)
          if(ida.eq.60221) std3to4tran=isign(10331,id)
          if(ida.eq.30223) std3to4tran=isign(100223,id)
          if(ida.eq.40223) std3to4tran=isign(20333,id)
          if(ida.eq.50223) std3to4tran=isign(9000223,id)
          if(ida.eq.60223) std3to4tran=isign(30223,id)
          if(ida.eq.30333) std3to4tran=isign(100333,id)
          if(ida.eq.40333) std3to4tran=isign(9000223,id)
          if(ida.eq.20225) std3to4tran=isign(100335,id)
          if(ida.eq.30225) std3to4tran=isign(9050225,id)
          if(ida.eq.40225) std3to4tran=isign(9060225,id)
          if(ida.eq.30313) std3to4tran=isign(100313,id)
          if(ida.eq.30323) std3to4tran=isign(100323,id)
          if(ida.eq.40313) std3to4tran=isign(30313,id)
          if(ida.eq.40323) std3to4tran=isign(30323,id)
          if(ida.eq.20441) std3to4tran=isign(100441,id)
          if(ida.eq.30443) std3to4tran=isign(100443,id)
          if(ida.eq.40443) std3to4tran=isign(30443,id)
          if(ida.eq.50443) std3to4tran=isign(9000443,id)
          if(ida.eq.60443) std3to4tran=isign(9010443,id)
          if(ida.eq.70443) std3to4tran=isign(9020443,id)
          if(ida.eq.20551) std3to4tran=isign(100551,id)
          if(ida.eq.30551) std3to4tran=isign(110551,id)
          if(ida.eq.40551) std3to4tran=isign(200551,id)
          if(ida.eq.30553) std3to4tran=isign(100553,id)
          if(ida.eq.40553) std3to4tran=isign(110553,id)
          if(ida.eq.50553) std3to4tran=isign(120553,id)
          if(ida.eq.60553) std3to4tran=isign(200553,id)
          if(ida.eq.70553) std3to4tran=isign(300553,id)
          if(ida.eq.80553) std3to4tran=isign(9000553,id)
          if(ida.eq.90553) std3to4tran=isign(9010553,id)
          if(ida.eq.10555) std3to4tran=isign(100555,id)
C...look at t, b', and t'
          if(i3.ge.6)then
	     if(kl.gt.0) std3to4tran=0
	     if(kl.eq.0 .and. j1.gt.3) std3to4tran=0
	  endif
        elseif(mconv.eq.2) then
          if(ida.eq.10111) std3to4tran=0
          if(ida.eq.10211) std3to4tran=0
          if(ida.eq.9000111) std3to4tran=isign(10111,id)
          if(ida.eq.9000211) std3to4tran=isign(10211,id)
          if(ida.eq.100111) std3to4tran=isign(20111,id)
          if(ida.eq.100211) std3to4tran=isign(20211,id)
          if(ida.eq.100113) std3to4tran=isign(30113,id)
          if(ida.eq.100213) std3to4tran=isign(30213,id)
          if(ida.eq.30113) std3to4tran=isign(40113,id)
          if(ida.eq.30213) std3to4tran=isign(40213,id)
          if(ida.eq.119) std3to4tran=0
          if(ida.eq.219) std3to4tran=0
          if(ida.eq.10221) std3to4tran=isign(10331,id)
          if(ida.eq.100221) std3to4tran=isign(20221,id)
          if(ida.eq.9010221) std3to4tran=isign(10221,id)
          if(ida.eq.9020221) std3to4tran=isign(50221,id)
          if(ida.eq.10331) std3to4tran=isign(60221,id)
          if(ida.eq.9000223) std3to4tran=isign(40333,id)
          if(ida.eq.100331) std3to4tran=isign(40221,id)
          if(ida.eq.20333) std3to4tran=isign(40223,id)
          if(ida.eq.100223) std3to4tran=isign(30223,id)
          if(ida.eq.30223) std3to4tran=isign(60223,id)
          if(ida.eq.100333) std3to4tran=isign(30333,id)
          if(ida.eq.100335) std3to4tran=isign(20225,id)
          if(ida.eq.9050225) std3to4tran=isign(30225,id)
          if(ida.eq.9060225) std3to4tran=isign(40225,id)
          if(ida.eq.100313) std3to4tran=isign(30313,id)
          if(ida.eq.100323) std3to4tran=isign(30323,id)
          if(ida.eq.30313) std3to4tran=isign(40313,id)
          if(ida.eq.30323) std3to4tran=isign(40323,id)
          if(ida.eq.100441) std3to4tran=isign(20441,id)
          if(ida.eq.100443) std3to4tran=isign(30443,id)
          if(ida.eq.30443) std3to4tran=isign(40443,id)
          if(ida.eq.9000443) std3to4tran=isign(50443,id)
          if(ida.eq.9010443) std3to4tran=isign(60443,id)
          if(ida.eq.9020443) std3to4tran=isign(70443,id)
          if(ida.eq.100551) std3to4tran=isign(20551,id)
          if(ida.eq.110551) std3to4tran=isign(30551,id)
          if(ida.eq.200551) std3to4tran=isign(40551,id)
          if(ida.eq.30553) std3to4tran=0
          if(ida.eq.100553) std3to4tran=isign(30553,id)
          if(ida.eq.110553) std3to4tran=isign(40553,id)
          if(ida.eq.120553) std3to4tran=isign(50553,id)
          if(ida.eq.200553) std3to4tran=isign(60553,id)
          if(ida.eq.300553) std3to4tran=isign(70553,id)
          if(ida.eq.9000553) std3to4tran=isign(80553,id)
          if(ida.eq.9010553) std3to4tran=isign(90553,id)
          if(ida.eq.20555) std3to4tran=0
          if(ida.eq.100555) std3to4tran=isign(10555,id)
          if(ida.eq.557) std3to4tran=0
C...check for numbers unknown to StdHep 3.04
          kxt = mod(std3to4tran/100000,100)
	  if(kxt.gt.0) std3to4tran=0
        endif
C...check for illegal antiparticles
        if(i1.eq.i2 .and. id.lt.0) std3to4tran=0
C...diquarks
      elseif(i2.ne.0 .and. i3.ne.0 .and. i1.eq.0)then
        if(mconv.eq.1 .and. i2.eq.i3 .and. j1.eq.1)then
	  itmp = i3*1000+i2*100+3
	  std3to4tran=isign(itmp,id)
	endif
      else
C...undefined
        std3to4tran=0
      endif
#if HEPDBG
C...there are too many warnings - use only for debugging purposes
      if(std3to4tran.eq.0 .and. ida.ne.0)then
        if(mconv.eq.1) write(lnhout,111) id
        if(mconv.eq.2) write(lnhout,112) id
      endif
#endif
      return
 111  format('  STD3TO4TRAN 3: PDG92 particle ',I8,
     1        ' translates to zero')
 112  format('  STD3TO4TRAN 4: STD particle ',I8,' translates to zero')
      end
