
      subroutine luntmp(mconv)

C...Purpose: to convert JETSET event record contents to or from
C...the standard event record common block.
C...convert (mconv=1) from JETSET numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to JETSET numbering scheme

C... need stdhep.inc for NMXHEP
#include "stdhep.inc"
#include "stdtmp.inc"
#include "pyjets.inc"
#include "pydat1.inc"
#include "pydat2.inc"

      integer pycomp
      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        if(mconv.EQ.1) nevhept = 0
        lfirst=.FALSE.
      endif
C...Conversion from JETSET to standard, the easy part.
      if(mconv.EQ.1) then
        nevhept=nevhept + 1
        if(N.GT.NMXHEP) call pyerrm(8,
     &  '(LUNTMP:) no more space in /HEPEVT/')
        nhept=MIN(N,NMXHEP)
        do 140 I=1,nhept
        isthept(I)=0
        if(K(I,1).GE.1.AND.K(I,1).LE.10) isthept(I)=1
        if(K(I,1).GE.11.AND.K(I,1).LE.20) isthept(I)=2
        if(K(I,1).GE.21.AND.K(I,1).LE.30) isthept(I)=3
        if(K(I,1).GE.31.AND.K(I,1).LE.100) isthept(I)=K(I,1)
        idhept(I)=lutran(K(I,2),1)
        jmohept(1,I)=K(I,3)
        jmohept(2,I)=0
        if(K(I,1).NE.3.AND.K(I,1).NE.13.AND.K(I,1).NE.14) then
          jdahept(1,I)=K(I,4)
          jdahept(2,I)=K(I,5)
        else
          jdahept(1,I)=0
          jdahept(2,I)=0
        endif
        do 100 J=1,5
  100   phept(J,I)=P(I,J)
        do 110 J=1,4
  110   vhept(J,I)=V(I,J)

C...Fill in missing mother information.
        if(I.GE.3.AND.K(I,1).EQ.21.AND.K(I,3).EQ.0) then
          IMO1=I-2
          if(I.GE.4.AND.K(I-1,1).EQ.21.AND.K(I-1,3).EQ.0) IMO1=IMO1-1
          jmohept(1,I)=IMO1
          jmohept(2,I)=IMO1+1
        elseif(K(I,2).GE.91.AND.K(I,2).LE.93) then
          I1=K(I,3)-1
  120     I1=I1+1
          if(I1.GE.I) call pyerrm(8,
     &    '(LUNTMP:) translation of inconsistent event history')
          if(I1.LT.I.AND.K(I1,1).NE.1.AND.K(I1,1).NE.11) GOTO 120
          KC=pycomp(K(I1,2))
          if(I1.LT.I.AND.KC.EQ.0) GOTO 120
          if(I1.LT.I.AND.KCHG(KC,2).EQ.0) GOTO 120
          jmohept(2,I)=I1
        elseif(K(I,2).EQ.94) then
          NJET=2
          if(nhept.GE.I+3.AND.K(I+3,3).LE.I) NJET=3
          if(nhept.GE.I+4.AND.K(I+4,3).LE.I) NJET=4
          jmohept(2,I)=MOD(K(I+NJET,4)/MSTU(5),MSTU(5))
          if(jmohept(2,I).EQ.jmohept(1,I)) jmohept(2,I)=
     &    MOD(K(I+1,4)/MSTU(5),MSTU(5))
        endif

C...Fill in missing daughter information.
        if(K(I,2).EQ.94.AND.MSTU(16).NE.2) then
          do 130 I1=jdahept(1,I),jdahept(2,I)
          I2=MOD(K(I1,4)/MSTU(5),MSTU(5))
  130     jdahept(1,I2)=I
        endif
        if(K(I,2).GE.91.AND.K(I,2).LE.94) GOTO 140
        I1=jmohept(1,I)
        if(I1.LE.0.OR.I1.GT.nhept) GOTO 140
        if(K(I1,1).NE.13.AND.K(I1,1).NE.14) GOTO 140
        if(jdahept(1,I1).EQ.0) then
          jdahept(1,I1)=I
        else
          jdahept(2,I1)=I
        endif
  140   CONTINUE
        do 150 I=1,nhept
        if(K(I,1).NE.13.AND.K(I,1).NE.14) GOTO 150
        if(jdahept(2,I).EQ.0) jdahept(2,I)=jdahept(1,I)
  150   CONTINUE

C...Conversion from standard to JETSET, the easy part.
      else
        if(nhept.GT.MSTU(4)) call pyerrm(8,
     &  '(LUNTMP:) insufficient space in /LUJETS/')
        N=MIN(nhept,MSTU(4))
        NKQ=0
        KQSUM=0
        do 180 I=1,N
        K(I,1)=0
        if(isthept(I).EQ.1) K(I,1)=1
        if(isthept(I).EQ.2) K(I,1)=11
        if(isthept(I).EQ.3) K(I,1)=21
        K(I,2)=lutran(idhept(I),2)
        K(I,3)=jmohept(1,I)
        K(I,4)=jdahept(1,I)
        K(I,5)=jdahept(2,I)
        do 160 J=1,5
  160   P(I,J)=phept(J,I)
        do 170 J=1,4
  170   V(I,J)=vhept(J,I)
        V(I,5)=0.
        if(isthept(I).EQ.2.AND.phept(4,I).GT.phept(5,I)) then
          I1=jdahept(1,I)
          if(I1.GT.0.AND.I1.LE.nhept) V(I,5)=(vhept(4,I1)-vhept(4,I))*
     &    phept(5,I)/phept(4,I)
        endif

C...Fill in missing information on colour connection in jet systems.
        if(isthept(I).EQ.1) then
          KC=pycomp(K(I,2))
          KQ=0
          if(KC.NE.0) KQ=KCHG(KC,2)*isign(1,K(I,2))
          if(KQ.NE.0) NKQ=NKQ+1
          if(KQ.NE.2) KQSUM=KQSUM+KQ
          if(KQ.NE.0.AND.KQSUM.NE.0) then
            K(I,1)=2
          elseif(KQ.EQ.2.AND.I.LT.N) then
            if(K(I+1,2).EQ.21) K(I,1)=2
          endif
        endif
  180   CONTINUE
        if(NKQ.EQ.1.OR.KQSUM.NE.0) call pyerrm(8,
     &  '(LUNTMP:) input parton configuration not colour singlet')
      endif
      return
      end
