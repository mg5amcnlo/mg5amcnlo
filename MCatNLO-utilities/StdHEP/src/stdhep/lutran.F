      integer function lutran(id,mconv)
C...convert (mconv=1) from PYTHIA numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to PYTHIA numbering scheme
C
C           ID = particle identification number

      implicit none
      integer id, mconv
      integer ida, j1, i1, i2, i3, i4, i5, ksusy, ku, kqn, j

C... ITABJ(I) converts miscellaneous PYTHIA particle ID's to a standard scheme
C...      only 31-40 change
      integer itabj(99,2), noant(11)
      data itabj/1,2,3,4,5,6,7,8,0,0,
     1           11,12,13,14,15,16,17,18,0,0,
     2           21,22,23,24,25,0,0,0,0,0,
     3           0,32,33,34,35,36,37,0,39,0,
     4           41,42,0,0,0,0,0,0,0,0,
     5           0,0,0,0,0,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,98,99,
     *           1,2,3,4,5,6,7,8,0,51,
     1           11,12,13,14,15,16,17,18,0,0,
     2           21,22,23,24,25,0,0,0,0,0,
     3           0,32,33,34,35,36,37,0,39,0,
     4           41,42,0,0,0,0,0,0,0,0,
     5           0,0,0,0,0,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           81,82,83,84,85,86,87,88,89,90,
     9           91,92,93,94,95,96,97,98,99/
      data noant/-10,-21,-22,-23,-25,-32,-33,-35,-36,-81,-83/
      save itabj,noant

      lutran=id
      ida=iabs(id)
      j1    = mod(ida,10)
      i1    = mod(ida/10,10)
      i2    = mod(ida/100,10)
      i3    = mod(ida/1000,10)
      i4    = mod(ida/10000,10)
      i5    = mod(ida/100000,10)
      ksusy = mod(ida/1000000,10)
      ku    = mod(ida/10000000,10)
      kqn   = mod(ida/1000000000,10)

      if(ida.eq.0)then
        call pyerrm(8,'(LUTRAN 1:) particle ID is zero')
      elseif(mconv.LT.1 .OR. mconv.GT.2)then
        lutran=0
        call pyerrm(8,'(LUTRAN 2:) unallowed conversion option')
C... ions not allowed
      elseif(kqn.eq.1) then
        lutran=0
        call pyerrm(8,'(LUTRAN 3:) unallowed ion')
C...SUSY 
      elseif(ksusy.eq.1 .or. ksusy.eq.2) then
C...technicolor
      elseif(ksusy.eq.3) then
        if(mconv.eq.1) then
	    if(ida.eq.3000331) lutran = isign(3100221,id)
	    if(ida.eq.3100111) lutran = isign(3060111,id)
	    if(ida.eq.3200111) lutran = isign(3160111,id)
	    if(ida.eq.3100113) lutran = isign(3130113,id)
	    if(ida.eq.3200113) lutran = isign(3140113,id)
	    if(ida.eq.3300113) lutran = isign(3150113,id)
	    if(ida.eq.3400113) lutran = isign(3160113,id)
        elseif(mconv.eq.2)then
	    if(ida.eq.3100221) lutran = isign(3000331,id)
	    if(ida.eq.3060111) lutran = isign(3100111,id)
	    if(ida.eq.3160111) lutran = isign(3200111,id)
	    if(ida.eq.3130113) lutran = isign(3100113,id)
	    if(ida.eq.3140113) lutran = isign(3200113,id)
	    if(ida.eq.3150113) lutran = isign(3300113,id)
	    if(ida.eq.3160113) lutran = isign(3400113,id)
	endif
C...excited quarks and leptons 
      elseif(ksusy.eq.4 .or. ksusy.eq.5) then
        if(mconv.eq.1) then
	    if(ida.eq.5000039) lutran = isign(4000039,id)
        elseif(mconv.eq.2)then
	    if(ida.eq.4000039) lutran = isign(5000039,id)
	endif
C...other exotics
      elseif(ksusy.eq.9 .and. i5.eq.9) then
        if(mconv.eq.1) then
	    if(ida.eq.9900110) lutran = isign(9910113,id)
	    if(ida.eq.9900210) lutran = isign(9910211,id)
	    if(ida.eq.9900220) lutran = isign(9910223,id)
	    if(ida.eq.9900330) lutran = isign(9910333,id)
	    if(ida.eq.9900440) lutran = isign(9910443,id)
	    if(ida.eq.9902110) lutran = isign(9912112,id)
	    if(ida.eq.9902210) lutran = isign(9912212,id)
        elseif(mconv.eq.2)then
	    if(ida.eq.9910113) lutran = isign(9900110,id)
	    if(ida.eq.9910211) lutran = isign(9900210,id)
	    if(ida.eq.9910223) lutran = isign(9900220,id)
	    if(ida.eq.9910333) lutran = isign(9900330,id)
	    if(ida.eq.9910443) lutran = isign(9900440,id)
	    if(ida.eq.9912112) lutran = isign(9902110,id)
	    if(ida.eq.9912212) lutran = isign(9902210,id)
	endif
C...check for illegal antiparticles
        if(i1.eq.i2 .and. i3.eq.0 .and. id.lt.0) lutran=0
C...Higgs, etc.
      elseif(ida.lt.100)then
        lutran=isign(itabj(ida,mconv),id)
C...check for illegal antiparticles
        if(id.lt.0)then
          if(lutran.ge.-99 .and. lutran.le.-91) lutran=0
          do 101 j=1,11
            if(lutran.eq.noant(j)) lutran=0
 101      continue
        endif
C...special particles
      elseif(j1.eq.0)then
C...check for illegal anti KS, KL
          if(id.eq.-130 .or. id.eq.-310) lutran=0
C...spin 1/2 baryons
      elseif(i1.ne.0 .and. i3.ne.0 .and. j1.eq.2)then
C...spin 3/2 baryons
      elseif(i1.ne.0 .and. i3.ne.0 .and. j1.eq.4)then
C...mesons 
      elseif(i1.ne.0 .and. i2.ne.0 .and. i3.eq.0)then
        if(mconv.eq.1)then
          if(ida.eq.10111) lutran=isign(9000111,id)
          if(ida.eq.10211) lutran=isign(9000211,id)
          if(ida.eq.10221) lutran=isign(9010221,id)
          if(ida.eq.10331) lutran=isign(10221,id)
	elseif(mconv.eq.2)then
          if(ida.eq.10111) lutran=0
          if(ida.eq.10211) lutran=0
          if(ida.eq.9000111) lutran=isign(10111,id)
          if(ida.eq.9000211) lutran=isign(20211,id)
          if(ida.eq.9010221) lutran=isign(10221,id)
          if(ida.eq.10221) lutran=isign(10331,id)
          if(ida.eq.9000223) lutran=0
	endif
C...check for illegal antiparticles
        if(i1.eq.i2 .and. id.lt.0) lutran=0
C...diquarks
      elseif(i2.ne.0 .and. i3.ne.0 .and. i1.eq.0)then
C...undefined
      else
        lutran=0
      endif

      if(lutran.eq.0 .and. ida.ne.0)
     1   call pyerrm(8,'(LUTRAN 5:) particle ID translation is zero')
      return
      end
