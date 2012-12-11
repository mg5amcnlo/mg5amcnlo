
      integer function hepchg_2004(id)

C...Purpose: to give three times the charge for a particle/parton.

C      ID     = particle ID
C      hepchg_2004 = particle charge times 3

      integer id,kqa,kq1,kq2,kq3,kqj,irt,kqx

      integer ichg(109)
      data ichg/-1,2,-1,2,-1,2,-1,2,2*0, 
     1 -3,0,-3,0,-3,0,-3,3*0,
     2 3*0,3,6*0,
     3 3*0,3,0,0,3,3*0,
     4 10*0,
     5 0,6,3,6,6*0,
     6 49*0/
      save ichg

C...Initial values. Simple case of direct readout.
      hepchg_2004=0

      kqa=iabs(id)
      kqn=mod(kqa/1000000000,10)
      kqx=mod(kqa/1000000,10)
      kq3=mod(kqa/1000,10)
      kq2=mod(kqa/100,10)
      kq1=mod(kqa/10,10)
      kqj=mod(kqa,10)
      irt = mod(kqa,10000)

C...illegal or ion
      if(kqa.eq.0.or.kqa.ge.10000000) then
C...set ion charge to zero - not enough information
        if(kqn.eq.1) hepchg_2004 = 0
C... direct translation
      elseif(kqa.le.100) then
        hepchg_2004 = ichg(kqa)
C... KS and KL (and undefined)
      elseif(kqj.eq.0) then
        hepchg_2004 = 0
C... direct translation
      elseif(kqx.gt.0 .and. irt.le.100) then
        hepchg_2004 = ichg(irt)
C... not everything maps, therefore deal with special cases
        if(kqa.eq.1000017 .or. kqa.eq.1000018) hepchg_2004 = 0
        if(kqa.eq.1000034 .or. kqa.eq.1000052) hepchg_2004 = 0
        if(kqa.eq.1000053 .or. kqa.eq.1000054) hepchg_2004 = 0
        if(kqa.eq.9900061 .or. kqa.eq.9900062) hepchg_2004 = 6
C... pentaquarks
      elseif(kqa.eq.9221132) then
        hepchg_2004 = 3
      elseif(kqa.eq.9331122) then
        hepchg_2004 = -6

C...Construction from quark content for heavy meson, diquark, baryon.
      elseif(kq3.eq.0) then
C...Mesons.
        hepchg_2004 = ichg(kq2) - ichg(kq1)
C...Strange or beauty mesons.
        if(kq2.eq.3 .or. kq2.eq.5) hepchg_2004 = ichg(kq1) - ichg(kq2)
      elseif(kq1.eq.0) then
C...Diquarks.
        hepchg_2004 = ichg(kq3) + ichg(kq2)
      else
C...Baryons
        hepchg_2004 = ichg(kq3) + ichg(kq2) + ichg(kq1)
      endif

C... fix sign of charge
      if(id.lt.0 .and. hepchg_2004.ne.0) hepchg_2004 = -hepchg_2004

      return
      end
