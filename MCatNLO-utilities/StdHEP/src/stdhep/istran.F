
      integer function istran(ID,mconv)
C...convert (mconv=1) from ISAJET numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to ISAJET numbering scheme
C...called by ISAHEP
C
C           ID = particle identification number

#include "itapes.inc"
      integer ID,mconv,ida,i4,if1,if2,if3,js,ind

C... ITABI(I) converts miscellaneous ISAJET particle ID's to a standard scheme
      integer ITABI(99,2), NOANT(11)
      data ITABI/2,1,3,4,5,6,7,8,21,22,
     1           12,11,14,13,16,15,0,0,0,310,
     2           1000002,1000001,1000003,1000004,1000005,1000006,
     2           0,0,1000021,1000022,
     3           1000012,1000011,1000014,1000013,1000016,1000015,
     3           0,0,1000024,1000023,
     4           2000002,2000001,2000003,2000004,2000005,2000006,
     4           0,0,1000037,1000025,
     5           2000012,2000011,2000014,2000013,2000016,2000015,
     5           0,0,0,1000035,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,24,
     8           25,51,35,36,55,37,53,52,54,23,
     9           1000039,39,0,0,0,0,0,0,0,
     *           2,1,3,4,5,6,7,8,0,0,
     1           12,11,14,13,16,15,0,0,0,0,
     2           9,10,90,80,81,0,0,0,0,0,
     3           0,0,0,0,83,84,86,0,92,0,
     4           0,0,0,0,0,0,0,0,0,0,
     5           82,88,87,89,85,0,0,0,0,0,
     6           0,0,0,0,0,0,0,0,0,0,
     7           0,0,0,0,0,0,0,0,0,0,
     8           0,0,0,0,0,0,0,0,0,0,
     9           0,0,10016,20016,0,0,0,0,0/
      data NOANT/-21,-22,-23,-25,-32,-33,-35,-36,-51,-55,-56/
      save ITABI,NOANT

      ida=IABS(ID)
      I4=MOD(ida/10000,10)
      istran=0
      if(mconv.ne.1) GO TO 200
C...ISAJET algorithm routine
      call FLAVOR(ID,if1,if2,if3,JS,IND)
      if(ida.EQ.0)then
        write(ITLIS,*) ' ISTRAN: particle ID is zero'
      elseif(ida.LT.100)then
        istran=isign(ITABI(ida,1),ID)
        if(ID.EQ.-20) istran=130
C...check for illegal antiparticles
        ITMP=istran
        if(ITMP.LT.0)then
          do 101 J=1,11
            if(ITMP.EQ.NOANT(J)) istran=0
 101      CONTINUE
        endif
      elseif(ida.eq.10016)then
        istran=isign(93,ID)
      elseif(ida.eq.20016)then
        istran=isign(94,ID)
      elseif(IND.NE.0)then
        IS1=IABS(if1)
        IS2=IABS(if2)
        IS3=IABS(if3)
C...mesons
        if(IS1.EQ.0)then
          if(IS2.LE.2 .AND. IS3.LE.2)then
C...         don't change
          else
C...         u and d have opposite definitions
            if(IS2.LE.2) IS2=ITABI(IS2,1)
            if(IS3.LE.2) IS3=ITABI(IS3,1)
          endif
          istran=IS3*100 + IS2*10 + 2*JS+1 + I4*10000
          istran=isign(istran,ID)
C...         charmed and top mesons have wrong sign
          if(IS3.EQ.4 .AND. IS2.NE.4) istran=-istran
          if(IS3.EQ.6 .AND. IS2.NE.6 .AND. IS2.NE.4) istran=-istran
C...    
	  if(ida.eq.112) istran=isign(225,ID)
	  if(ida.eq.10110) istran=isign(9010221,ID)
	  if(ida.eq.10111) istran=isign(20113,ID)
	  if(ida.eq.10121) istran=isign(20213,ID)
	  if(ida.eq.30231) istran=isign(100313,ID)
	  if(ida.eq.30131) istran=isign(100323,ID)
	  if(ida.eq.10441) istran=isign(100443,ID)
	  if(ida.eq.20440) istran=isign(10441,ID)
	  if(ida.eq.20442) istran=isign(445,ID)
C...check for illegal antiparticles
          if(IS2.EQ.IS3 .AND. ID.LT.0) istran=0
C...diquarks
        elseif(IS3.EQ.0)then
C...         u and d have opposite definitions
          if(IS1.le.2) IS1=ITABI(IS1,1)
          if(IS2.le.2) IS2=ITABI(IS2,1)
          if(IS2.lt.IS1)then
            istran=IS1*1000 + IS2*100 + 2*JS+1
          elseif(IS2.eq.IS1)then
            istran=IS2*1000 + IS1*100 + 2*JS+3
          else
            istran=IS2*1000 + IS1*100 + 2*JS+1
          endif
          istran=isign(istran,ID)
C...         charmed and top quarks have wrong sign
          if(IS2.EQ.4 .AND. IS1.NE.4) istran=-istran
          if(IS2.EQ.6 .AND. IS1.NE.6 .AND. IS1.NE.4) istran=-istran
C...baryons
        else
C...         u and d have opposite definitions
          if(IS1.LE.2) IS1=ITABI(IS1,1)
          if(IS2.LE.2) IS2=ITABI(IS2,1)
          if(IS3.LE.2) IS3=ITABI(IS3,1)
          if(IS3.LE.2)then
            istran=IS1*1000 + IS2*100 + IS3*10 + 2*JS+2
          elseif(IS1.LE.2 .AND. IS2.LE.2)then
            istran=IS3*1000 + IS1*100 + IS2*10 + 2*JS+2
          else
            istran=IS3*1000 + IS2*100 + IS1*10 + 2*JS+2
          endif
          istran=isign(istran,ID)
        endif
      endif
      GO TO 300

 200  if(mconv.NE.2) GO TO 300
      J1=MOD(IDA,10)
      I1=MOD(IDA/10,10)
      I2=MOD(IDA/100,10)
      I3=MOD(IDA/1000,10)
      I4=MOD(IDA/10000,10)
      KSUSY = MOD(IDA/1000000,10)
      KU = MOD(IDA/10000000,10)
      KQN=MOD(IDA/1000000000,10)
      if(IDA.EQ.0)then
        write(ITLIS,*) ' ISTRAN: particle ID is zero'
C... ion
      elseif(KQN.eq.1) then
        write(ITLIS,*) ' ISTRAN 2: ion is not allowed'
C...SUSY 
      elseif(KSUSY.eq.1 .or. KSUSY.eq.2) then
	if(ida.eq.1000001) istran=isign(22,ID)
	if(ida.eq.1000002) istran=isign(21,ID)
	if(ida.eq.1000003) istran=isign(23,ID)
	if(ida.eq.1000004) istran=isign(24,ID)
	if(ida.eq.1000005) istran=isign(25,ID)
	if(ida.eq.1000006) istran=isign(26,ID)
	if(ida.eq.1000011) istran=isign(32,ID)
	if(ida.eq.1000012) istran=isign(31,ID)
	if(ida.eq.1000013) istran=isign(34,ID)
	if(ida.eq.1000014) istran=isign(33,ID)
	if(ida.eq.1000015) istran=isign(36,ID)
	if(ida.eq.1000016) istran=isign(35,ID)
	if(ida.eq.2000001) istran=isign(42,ID)
	if(ida.eq.2000002) istran=isign(41,ID)
	if(ida.eq.2000003) istran=isign(43,ID)
	if(ida.eq.2000004) istran=isign(44,ID)
	if(ida.eq.2000005) istran=isign(45,ID)
	if(ida.eq.2000006) istran=isign(46,ID)
	if(ida.eq.2000011) istran=isign(52,ID)
	if(ida.eq.2000012) istran=isign(51,ID)
	if(ida.eq.2000013) istran=isign(54,ID)
	if(ida.eq.2000014) istran=isign(53,ID)
	if(ida.eq.2000015) istran=isign(56,ID)
	if(ida.eq.2000016) istran=isign(55,ID)
	if(ida.eq.1000021) istran=isign(29,ID)
	if(ida.eq.1000022) istran=isign(30,ID)
	if(ida.eq.1000023) istran=isign(40,ID)
	if(ida.eq.1000025) istran=isign(50,ID)
	if(ida.eq.1000035) istran=isign(60,ID)
	if(ida.eq.1000024) istran=isign(39,ID)
	if(ida.eq.1000037) istran=isign(49,ID)
	if(ida.eq.1000039) istran=isign(91,ID)
C...check for illegal antiparticles
        if(ID.LT.0)then
	  irt = -MOD(IDA,100)
          do J=1,11
            if(irt.EQ.NOANT(J)) istran=0
          enddo
        endif
C...elementary particles
      elseif(IDA.LT.100)then
        istran=isign(ITABI(IDA,2),ID)
C...check for illegal antiparticles
        if(ID.LT.0)then
          do 201 J=1,11
            if(ID.EQ.NOANT(J)) istran=0
 201      CONTINUE
        endif
C...K short and K long
      elseif(ID.EQ.130)then
        istran=-20
      elseif(ID.EQ.310)then
        istran=20
C...mesons
      elseif(I3.EQ.0)then
          if(I1.LE.2 .AND. I2.LE.2)then
C...         don't change
          else
C...         u and d have opposite definitions
            if(I1.LE.2) I1=ITABI(I1,2)
            if(I2.LE.2) I2=ITABI(I2,2)
          endif
          istran=I1*100 + I2*10 + (J1-1)/2 + I4*10000
          istran=isign(istran,ID)
C...         charmed and top mesons have wrong sign
          if(I2.EQ.4 .AND. I1.NE.4) istran=-istran
          if(I2.EQ.6 .AND. I1.NE.6 .AND. I1.NE.4) istran=-istran
C...    
	  if(ida.eq.225) istran=isign(112,ID)
	  if(ida.eq.10221) istran=0
	  if(ida.eq.9010221) istran=isign(10110,ID)
	  if(ida.eq.20113) istran=isign(10111,ID)
	  if(ida.eq.20213) istran=isign(10121,ID)
	  if(ida.eq.100313) istran=isign(30231,ID)
	  if(ida.eq.100323) istran=isign(30131,ID)
	  if(ida.eq.100443) istran=isign(10441,ID)
	  if(ida.eq.10441) istran=isign(20440,ID)
	  if(ida.eq.445) istran=isign(20442,ID)
C...check for illegal antiparticles
          if(I2.EQ.I1 .AND. ID.LT.0) istran=0
C...diquarks
      elseif(I1.EQ.0)then
C...         u and d have opposite definitions
          if(I3.LE.2) I3=ITABI(I3,2)
          if(I2.LE.2) I2=ITABI(I2,2)
          if(I3.LT.I2)then
            istran=I3*1000 + I2*100 + (J1-1)/2
          else
            istran=I2*1000 + I3*100 + (J1-1)/2
          endif
          istran=isign(istran,ID)
C...         charmed and top mesons have wrong sign
          if(I2.EQ.4 .AND. I3.NE.4) istran=-istran
          if(I2.EQ.6 .AND. I3.NE.6 .AND. I3.NE.4) istran=-istran
C...baryons
      else
C...         u and d have opposite definitions
          if(I3.LE.2) I3=ITABI(I3,2)
          if(I2.LE.2) I2=ITABI(I2,2)
          if(I1.LE.2) I1=ITABI(I1,2)
          if(I3.LE.2)then
            istran=I3*1000 + I2*100 + I1*10 + (J1-2)/2
          elseif(I1.LE.2 .AND. I2.LE.2)then
            istran=I2*1000 + I1*100 + I3*10 + (J1-2)/2
          else
            istran=I1*1000 + I2*100 + I3*10 + (J1-2)/2
          endif
          istran=isign(istran,ID)
      endif

 300  return
      end
