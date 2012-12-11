
      integer function hepchg_old(ID)

C...Purpose: to give three times the charge for a particle/parton.

C      ID     = particle ID
C      hepchg_old = particle charge times 3

      integer ID,KQA,KQX,KQ1,KQ2,KQ3,KQJ

      integer ICHG(109)
      data ICHG/-1,2,-1,2,-1,2,-1,2,2*0,-3,0,-3,0,-3,0,-3,6*0,
     1          3,6*0,3,0,3,6,2*0,3,6,2*0,-1,2,-1,2,-1,2,4*0,
     2          -3,0,-3,0,-3,5*0,-1,2,-1,2,-1,2,4*0,
     3		-3,0,-3,0,-3,0,2*3,2*0,29*0/
      save ICHG

C...Initial values. Simple case of direct readout.
      hepchg_old=0

      KQA=IABS(ID)
      KQN=MOD(KQA/1000000000,10)
      KQX=MOD(KQA/10000,10)
      KQ3=MOD(KQA/1000,10)
      KQ2=MOD(KQA/100,10)
      KQ1=MOD(KQA/10,10)
      KQJ=MOD(KQA,10)

C...Simple cases: direct translation or special codes.
      if(KQA.EQ.0.OR.KQA.GE.100000) then
C...set ion charge to zero - not enough information
        if(KQN.eq.1) hepchg_old = 0
C... illegal
      elseif(KQA.LE.100) then
        hepchg_old = ICHG(KQA)
      elseif(KQJ.EQ.0) then
C... KS and KL (and undefined)
        hepchg_old = 0
C... jetset diffr particles
        if(KQA.EQ.210) hepchg_old=3
        if(KQA.EQ.2210) hepchg_old=3

C...Construction from quark content for heavy meson, diquark, baryon.
      elseif(KQ3.EQ.0) then
C...Mesons.
        hepchg_old = ICHG(KQ2) - ICHG(KQ1)
C...Strange mesons.
        if(KQ2.EQ.3) hepchg_old = ICHG(KQ1) - ICHG(KQ2)
C...Beauty mesons.
        if(KQ2.EQ.5) hepchg_old = ICHG(KQ1) - ICHG(KQ2)
      elseif(KQ1.EQ.0) then
C...Diquarks.
        hepchg_old = ICHG(KQ3) + ICHG(KQ2)
      else
C...Baryons
        hepchg_old = ICHG(KQ3) + ICHG(KQ2) + ICHG(KQ1)
      endif

C... fix sign of charge
      if(ID.LT.0 .AND. hepchg_old.NE.0) hepchg_old = -hepchg_old

      return
      end
