      integer function hepcmp_old(ID)

C...Purpose: to compress the standard ID codes for use in mass and decay
C...arrays; also to check whether a given code actually is defined.

C      ID     = particle ID
C      hepcmp_old = compressed index

      integer ID

      integer hepchg_old
      external hepchg_old

C...Subdivide standard ID code into constituent pieces.
      hepcmp_old=0
      KQA=IABS(ID)
      KQN=MOD(KQA/1000000000,10)
      KQNA=MOD(KQA/1000000,1000)
      KQNZ=MOD(KQA/1000,1000)
      KQX=MOD(KQA/10000,10)
      KQ3=MOD(KQA/1000,10)
      KQ2=MOD(KQA/100,10)
      KQ1=MOD(KQA/10,10)
      KQJ=MOD(KQA,10)
      KQSM=(KQJ-1)/2

C...Simple cases: direct translation or special codes.
      if(KQA.EQ.0.OR.KQA.GE.100000) then
C... look for ions here
        if(KQN.eq.1 .and. KQNA.gt.0 .and. ID.gt.0) then
C... use stupid compression for now
          if(KQNZ.eq.1 .and. KQNA.le.3)then
            hepcmp_old = 1600 + 1*(KQNZ-1) + KQNA
            if(MOD(KQJ,2).ne.0) hepcmp_old = 0
          elseif(KQNZ.eq.2 .and. (KQNA.eq.3 .or. KQNA.eq.4))then
            hepcmp_old = 1600 + 1*(KQNZ-1) + KQNA
            if(MOD(KQJ,2).eq.0) hepcmp_old = 0
          endif
        endif
C... illegal if not an ion
      elseif(KQA.LE.99) then
        hepcmp_old=KQA
C... need to check antiparticle legality
        ICH=hepchg_old(ID)
        if(ID.LT.0.AND.ICH.EQ.0)then
          if(KQA.GE.1.AND.KQA.LE.8)then
C... quarks
          elseif(KQA.GE.11.AND.KQA.LE.18)then
C... leptons
          elseif(KQA.EQ.28 .OR. KQA.EQ.29 .OR. KQA.EQ.40)then
          elseif(KQA.GE.41.AND.KQA.LE.46)then
C... left squarks
          elseif(KQA.GE.51.AND.KQA.LE.56)then
C... left sleptons
          elseif(KQA.GE.61.AND.KQA.LE.66)then
C... right squarks
          elseif(KQA.GE.71.AND.KQA.LE.76)then
C... right sleptons
          elseif(KQA.GE.81.AND.KQA.LE.99)then
C... internal definitions
          else
            hepcmp_old=0
          endif
        endif
      elseif(KQJ.EQ.0) then
C... KS and KL
        if(ID.EQ.130) hepcmp_old=102
        if(ID.EQ.310) hepcmp_old=101
C... jetset diffractive particles
        if(ID.EQ.110) hepcmp_old=107
        if(KQA.EQ.210) hepcmp_old=108
        if(ID.EQ.220) hepcmp_old=109
        if(ID.EQ.330) hepcmp_old=110
        if(ID.EQ.440) hepcmp_old=111
        if(KQA.EQ.2110) hepcmp_old=112
        if(KQA.EQ.2210) hepcmp_old=113
C...Herwig remnant patrticles
      elseif(KQX.EQ.0.AND.KQ1.EQ.9.AND.KQ2.EQ.9.AND.KQ3.EQ.9)then
        if(KQJ.EQ.8 .and. ID.gt.0) hepcmp_old=114
        if(KQJ.EQ.9 .and. ID.gt.0) hepcmp_old=115
C...Mesons.
      elseif(KQ3.EQ.0) then
        if(KQ2.EQ.0.OR.KQ2.EQ.9.OR.KQ1.EQ.0.OR.KQ1.EQ.9) then
C... illegal (no gluons allowed)
        elseif(KQ2.LT.KQ1) then
C... illegal quark order
        elseif(ID.LT.0.AND.(KQ2.EQ.KQ1)) then
C... illegal antiparticle
        elseif(MOD(KQJ,2).EQ.0)then
C... illegal spin
        elseif(KQ2.EQ.KQ1) then
          if(KQ2.LE.6)then
            hepcmp_old = 120 + (KQ2-1)*50 + KQSM*10 + KQX
          elseif(KQX.LE.2)then
            hepcmp_old = 420 + (KQ2-7)*15 + KQSM*3 + KQX
          endif
        elseif(KQ2.EQ.3 .OR. KQ2.EQ.2) then
          hepcmp_old = 450 + (KQ2+KQ1-3)*50 + KQSM*10 + KQX
        elseif(KQ2.EQ.4)then
          hepcmp_old = 600 + (KQ1-1)*50 + KQSM*10 + KQX
        elseif(KQ2.EQ.5)then
          hepcmp_old = 750 + (KQ1-1)*50 + KQSM*10 + KQX
        elseif(KQ2.EQ.6)then
          hepcmp_old = 950 + (KQ1-1)*50 + KQSM*10 + KQX
        elseif(KQ2.EQ.7.AND.KQX.LE.2)then
          hepcmp_old = 1200 + (KQ1-1)*15 + KQSM*3 + KQX
        elseif(KQ2.EQ.8.AND.KQX.LE.2)then
          hepcmp_old = 1290 + (KQ1-1)*15 + KQSM*3 + KQX
        endif

C...Diquarks.
      elseif(KQ1.EQ.0) then
        if(KQX.NE.0)then
C... no excited states
        elseif(KQJ.NE.1.AND.KQJ.NE.3) then
C... only 2 spin states
        elseif(KQ3.EQ.0.OR.KQ3.EQ.9.OR.KQ2.EQ.0.OR.KQ2.EQ.9) then
C... illegal (no gluons allowed)
        elseif(KQ3.LT.KQ2) then
C... illegal quark order
        else
C... all diquarks are the same...
          hepcmp_old=103
        endif

C...Baryons.
      elseif(KQX.EQ.0.AND.(KQJ.EQ.2.OR.KQJ.EQ.4)) then
C... no excited states and only spins 1/2 and 3/2
        if(KQ3.EQ.0.OR.KQ3.EQ.9.OR.KQ2.EQ.0.OR.KQ2.EQ.9.OR.
     1        KQ1.EQ.0.OR.KQ1.EQ.9) then
C... illegal (no gluons allowed)
        elseif(KQ3.LT.KQ1.OR.KQ3.LT.KQ2) then
C... illegal quark order
        elseif(KQ3.GE.7) then
C... l and h baryons are all the same
          hepcmp_old=97+KQ3
          if(KQ3.EQ.KQ1.AND.KQ3.EQ.KQ2.AND.KQJ.EQ.2) hepcmp_old=0
          if(KQ2.LT.KQ1 .AND. KQ3.EQ.KQ1) hepcmp_old=0
          if(KQ2.LT.KQ1 .AND. KQJ.EQ.4) hepcmp_old=0
          if(KQ3.GT.8) hepcmp_old=0
        elseif(KQJ.EQ.2) then
C...Spin 1/2 baryons.
          if(KQ3.EQ.KQ1 .AND. KQ3.EQ.KQ2)then
          elseif(KQ2.LT.KQ1 .AND. KQ3.EQ.KQ1)then
          elseif(KQ2.LT.KQ1) then
C...antisymmtric (Lambda, etc.)
            hepcmp_old=1395+((KQ3-1)*(KQ3-2)*(KQ3-3))/6+
     1                  ((KQ1-1)*(KQ1-2))/2+KQ2
          else
C...symmtric
            hepcmp_old=1415+((KQ3+1)*KQ3*(KQ3-1))/6+(KQ2*(KQ2-1))/2+KQ1
          endif
        elseif(KQJ.EQ.4) then
C...Spin 3/2 baryons.
          if(KQ2.LT.KQ1 .AND. KQ3.EQ.KQ1)then
          elseif(KQ2.LT.KQ1) then
C...no excited states for antisymmetric baryons
          else
C...symmtric
            hepcmp_old=1475+((KQ3+1)*KQ3*(KQ3-1))/6+(KQ2*(KQ2-1))/2+KQ1
          endif
        endif

      endif

      return
      end
