      integer function hepcmp_2004(id)

C...Purpose: to compress the standard ID codes for use in mass and decay
C...arrays; also to check whether a given code actually is defined.

C      ID     = particle ID
C      hepcmp_2004 = compressed index

      integer id
      integer cmplist(2210)

      integer hepchg_2004
      external hepchg_2004

      data (cmplist(i), i=1,100) /
     1 1, 2, 3, 4, 5, 6, 7, 8, 9,
     2 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     3 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     4 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
     5 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
     6 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
     7 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
     8 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
     9 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
     * 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100/
      data (cmplist(i), i=101,110) /
     1 310, 130, 0, 0, 0, 0, 110, 990, 9990, 0/
      data (cmplist(i), i=111,170) /
     1 1000001, 1000002, 1000003, 1000004, 1000005, 1000006, 
     1 1000007, 1000008, 1000009, 0,
     2 1000011, 1000012, 1000013, 1000014, 1000015, 1000016, 
     2 1000017, 1000018, 1000019, 0,
     3 1000021, 1000022, 1000023, 1000024, 1000025, 1000026, 
     3 1000027, 1000028, 1000029, 0,
     4 1000031, 1000032, 1000033, 1000034, 1000035, 1000036, 
     4 1000037, 1000038, 1000039, 0,
     5 1000041, 1000042, 1000043, 1000044, 1000045, 1000046, 
     5 1000047, 1000048, 1000049, 0,
     6 1000051, 1000052, 1000053, 1000054, 1000055, 1000056, 
     6 1000057, 1000058, 1000059, 0/
      data (cmplist(i), i=171,190) /
     1 2000001, 2000002, 2000003, 2000004, 2000005, 2000006, 
     1 2000007, 2000008, 2000009, 0,
     2 2000011, 2000012, 2000013, 2000014, 2000015, 2000016, 
     2 2000017, 2000018, 2000019, 0/
      data (cmplist(i), i=191,210) /
     1 3000111, 3000211, 3000221, 3000113, 3000213, 3100221, 
     1 3000223, 0, 0, 0,
     2 3100021, 3060111, 3160111, 3130113, 3140113, 3150113, 
     2 3160113, 0, 0, 0/
      data (cmplist(i), i=211,230) /
     1 4000001, 4000002, 4000003, 4000004, 4000005, 4000006, 
     1 4000007, 4000008, 4000009, 0,
     2 4000011, 4000012, 4000013, 4000014, 4000015, 4000016, 
     2 4000017, 4000018, 4000019, 4000039/
      data (cmplist(i), i=231,240) /
     1 9900061, 9900062, 9900024, 9900012, 9900014, 9900016, 
     1 9900023, 9221132, 9331122, 0/
      data (cmplist(i), i=241,259) /
     1 9910113, 9910211, 9910223, 9910333, 9910443, 9912112, 
     1 9912212, 9900443, 9900441, 9910441,
     2 9920022, 9922212, 10022, 20022, 
     3 9900553, 9900551, 9910551, 0, 0 /
      data (cmplist(i), i=260,2200) / 1941*0/
      data (cmplist(i), i=2201,2210) /
     1 1001001002, 1002001002, 1003001002, 1003002001, 1004002001,
     1 5*0/

      save cmplist

C...Subdivide standard ID code into constituent pieces.
      hepcmp_2004=0
      kqa=iabs(id)
      kqn=mod(kqa/1000000000,10)
      kqna=mod(kqa/1000000,1000)
      kqnz=mod(kqa/1000,1000)
      kqx=mod(kqa/1000000,10)
      kqr=mod(kqa/100000,10)
      kql=mod(kqa/10000,10)
      kq3=mod(kqa/1000,10)
      kq2=mod(kqa/100,10)
      kq1=mod(kqa/10,10)
      kqj=mod(kqa,10)
      kshort=mod(kqa,100)
      kqsm=(kqj-1)/2
      irt = mod(kqa,10000)
      k99=mod(kqa/100000,100)
      is = kqsm
      if(kqx.eq.9) is = kqsm + 5

      if(kqa.eq.0)  then
      elseif(kqa.ge.10000000) then
C... ions 
	do i=2201,2210
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        if(id.lt.0) hepcmp_2004=0
      elseif(kqa.le.100) then
C... need to check antiparticle legality
        hepcmp_2004=kqa
        ich=hepchg_2004(id)
        if(id.lt.0.and.ich.eq.0)then
          if(kqa.ge.1.and.kqa.le.8)then
C... quarks
          elseif(kqa.ge.11.and.kqa.le.18)then
C... leptons
          elseif(kqa.eq.41 .or. kqa.eq.42)then
C... R0 or LQ
          elseif(kqa.ge.81)then
C... internal definitions
          else
            hepcmp_2004=0
          endif
        endif
      elseif(kqj.eq.0) then
	do i=101,110
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        if(id.lt.0) hepcmp_2004=0
      elseif(kshort.eq.22 .and. kql.gt.0 .and. kq2.eq.0) then
C virtual gamma and Cerenkov
	do i=241,260
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        if(id.lt.0) hepcmp_2004=0
      elseif(kqx.eq.1 .or. kqx.eq.2)then
C...SUSY
	do i=111,190
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        ich=hepchg_2004(id)
        if(id.lt.0.and.ich.eq.0)then
          if(irt.ge.1.and.irt.le.8)then
C... quarks
          elseif(irt.ge.11.and.irt.le.18)then
C... leptons
          else
            hepcmp_2004=0
          endif
        endif
      elseif(kqx.ge.3 .and. kqx.le.4)then
	do i=191,240
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        if(kq3.eq.0 .and. id.lt.0) then
          if(kq1.gt.0 .and. kq2.eq.kq1) hepcmp_2004=0
	endif
      elseif(k99.eq.99)then
	do i=231,260
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
        if(kq3.eq.0 .and. id.lt.0) then
          if(kq1.gt.0 .and. kq2.eq.kq1) hepcmp_2004=0
	endif
	if(id.lt.0 .and. kshort.eq.22)  hepcmp_2004=0
      elseif(kqx.eq.9 .and. kqr.ge.1)then
C...possible pentaquark
	do i=231,240
            if(cmplist(i).eq.kqa) hepcmp_2004 = i
	enddo
      elseif(kqx.gt.4 .and. kqx.lt.9)then
C...undefined

      elseif(kq1.eq.0) then
C...Diquarks.
C... 103 
        if(kqx.ne.0 .and. kqr.ne.0 .and. kql.ne.0)then
C... no excited states
        elseif(kqj.ne.1.and.kqj.ne.3) then
C... only 2 spin states
        elseif(kq3.eq.0.or.kq3.eq.9.or.kq2.eq.0.or.kq2.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq3.lt.kq2) then
C... illegal quark order
        elseif(kq3.eq.kq2 .and. kqj.eq.1) then
C... illegal spin
        else
C... all diquarks are the same...
          hepcmp_2004=103
        endif

      elseif(kq3.eq.0) then
C...Mesons.
C... 260 - ...
        if(kq2.eq.0.or.kq2.eq.9.or.kq1.eq.0.or.kq1.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq2.lt.kq1) then
C... illegal quark order
        elseif(id.lt.0.and.(kq2.eq.kq1)) then
C... illegal antiparticle
        elseif(mod(kqj,2).eq.0)then
C... illegal spin
        elseif(kq2.le.2 .and. kq1.eq.1) then
C... 260 - 559 [ kqr = 0,1,2  kql = 0-4 ]
	  hepcmp_2004 = 260 + (kq2-1)*150 + kqr*50 + kql*10 + is
	  if(kql.gt.4 .or. kqr.gt.2) hepcmp_2004 = 0
	elseif(kq2.eq.2 .and. kq1.eq.2) then
C... 560 - 799 [ kqr = 0,1,2  kql = 0-9 ]
	  hepcmp_2004 = 560 + kqr*100 + kql*10 + is
	  if(kqr.gt.2) hepcmp_2004 = 0
	elseif(kq2.eq.3 .and. kq1.eq.3) then
C... 800 - 859 [ kqr = 0,1  kql = 0-2 ]
	  hepcmp_2004 = 860 + kqr*30 + kql*10 + is
	  if(kql.gt.2 .or. kqr.gt.1) hepcmp_2004 = 0
	elseif(kq2.eq.3 .and. kq1.le.2) then
C... 860 - 1099 [ kqr = 0,1,2  kql = 0-3 ]
	  hepcmp_2004 = 920 + (kq1-1)*120 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.2) hepcmp_2004 = 0
	elseif(kq2.eq.4 .and. kq1.le.3) then
C... 1100 - 1189 [ kqr = 0  kql = 0-2 ]
	  hepcmp_2004 = 1160 + kqr*90 + (kq1-1)*30 + kql*10 + is
cc	  if(kql.gt.2 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp_2004 = 0
	  if(kql.gt.2 .or. kqr.gt.2 .or. kqx.gt.0) hepcmp_2004 = 0
	elseif(kq2.eq.4 .and. kq1.eq.4) then
C... 1190 - 1269 [ kqr = 0,1  kql = 0-3 ]
	  hepcmp_2004 = 1430 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.1) hepcmp_2004 = 0
	elseif(kq2.eq.5 .and. kq1.le.4) then
C... 1270 - 1389 [ kqr = 0  kql = 0-2 ]
	  hepcmp_2004 = 1510 + (kq1-1)*30 + kql*10 + is
	  if(kql.gt.2 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp_2004 = 0
	elseif(kq2.eq.5 .and. kq1.eq.5) then
C... 1390 - 1549 [ kqr = 0,1,2,3  kql = 0-3 ]
	  hepcmp_2004 = 1630 + kqr*40 + kql*10 + is
	  if(kql.gt.3 .or. kqr.gt.3) hepcmp_2004 = 0
        elseif(kq2.eq.6)then
C... 1550 - 1609 [ kqr = 0  kql = 0 ]
	  hepcmp_2004 = 1790 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp_2004 = 0
        elseif(kq2.eq.7)then
C... 1610 - 1679 [ kqr = 0  kql = 0 ]
	  hepcmp_2004 = 1850 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp_2004 = 0
        elseif(kq2.eq.8)then
C... 1680 - 1759 [ kqr = 0  kql = 0 ]
	  hepcmp_2004 = 1920 + (kq1-1)*10 + is
	  if(kql.gt.0 .or. kqr.gt.0 .or. kqx.gt.0) hepcmp_2004 = 0
        endif

      elseif(kqx.eq.0.and.(kqj.eq.2.or.kqj.eq.4)) then
C...Baryons.
C... no excited states and only spins 1/2 and 3/2
        if(kq3.eq.0.or.kq3.eq.9.or.kq2.eq.0.or.kq2.eq.9.or.
     1        kq1.eq.0.or.kq1.eq.9) then
C... illegal (no gluons allowed)
        elseif(kq3.lt.kq1.or.kq3.lt.kq2) then
C... illegal quark order
        elseif(kq3.ge.7) then
C... l and h baryons are all the same
C... 104, 105
          hepcmp_2004=97+kq3
          if(kq3.eq.kq1.and.kq3.eq.kq2.and.kqj.eq.2) hepcmp_2004=0
          if(kq2.lt.kq1 .and. kq3.eq.kq1) hepcmp_2004=0
          if(kq2.lt.kq1 .and. kqj.eq.4) hepcmp_2004=0
          if(kq3.gt.8) hepcmp_2004=0
        elseif(kqj.eq.2) then
C...Spin 1/2 baryons.
          if(kq3.eq.kq1 .and. kq3.eq.kq2)then
          elseif(kq2.lt.kq1 .and. kq3.eq.kq1)then
          elseif(kq2.lt.kq1) then
C...antisymmtric (Lambda, etc.)
C... 1760 - 1780
            hepcmp_2004=2000+((kq3-1)*(kq3-2)*(kq3-3))/6+
     1                  ((kq1-1)*(kq1-2))/2+kq2
          else
C...symmtric
C... 1780 - 1791
            hepcmp_2004=2020+((kq3+1)*kq3*(kq3-1))/6+(kq2*(kq2-1))/2+kq1
          endif
        elseif(kqj.eq.4) then
C...Spin 3/2 baryons.
          if(kq2.lt.kq1 .and. kq3.eq.kq1)then
          elseif(kq2.lt.kq1) then
C...no excited states for antisymmetric baryons
          else
C...symmetric
C... 1840 - 1896
            hepcmp_2004=2080+((kq3+1)*kq3*(kq3-1))/6+(kq2*(kq2-1))/2+kq1
          endif
        endif

      endif

      return
      end
