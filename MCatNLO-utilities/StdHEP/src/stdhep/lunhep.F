
      subroutine lunhep(mconv)

C...Purpose: to convert Pythia event record contents to or from
C...the standard event record common block.
C...convert (mconv=1) from Pythia numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to Pythia numbering scheme

#include "stdhep.inc"
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
        if(mconv.EQ.1) NEVHEP = 0
        lfirst=.FALSE.
      endif
C...Conversion from Pythia to standard, the easy part.
      if(mconv.EQ.1) then
        NEVHEP=NEVHEP + 1
        if(N.GT.NMXHEP) call pyerrm(8,
     &  '(LUNHEP:) no more space in /HEPEVT/')
        NHEP=MIN(N,NMXHEP)
        do 140 I=1,NHEP
        ISTHEP(I)=0
        if(K(I,1).GE.1.AND.K(I,1).LE.10) ISTHEP(I)=1
        if(K(I,1).GE.11.AND.K(I,1).LE.20) ISTHEP(I)=2
        if(K(I,1).GE.21.AND.K(I,1).LE.30) ISTHEP(I)=3
        if(K(I,1).GE.31.AND.K(I,1).LE.100) ISTHEP(I)=K(I,1)
        IDHEP(I)=lutran(K(I,2),1)
        JMOHEP(1,I)=K(I,3)
        JMOHEP(2,I)=0
        if(K(I,1).NE.3.AND.K(I,1).NE.13.AND.K(I,1).NE.14) then
          JDAHEP(1,I)=K(I,4)
          JDAHEP(2,I)=K(I,5)
        else
          JDAHEP(1,I)=0
          JDAHEP(2,I)=0
        endif
        do 100 J=1,5
  100   PHEP(J,I)=P(I,J)
        do 110 J=1,4
  110   VHEP(J,I)=V(I,J)

C...Fill in missing mother information.
        if(I.GE.3.AND.K(I,1).EQ.21.AND.K(I,3).EQ.0) then
          IMO1=I-2
          if(I.GE.4.AND.K(I-1,1).EQ.21.AND.K(I-1,3).EQ.0) IMO1=IMO1-1
          JMOHEP(1,I)=IMO1
          JMOHEP(2,I)=IMO1+1
        elseif(K(I,2).GE.91.AND.K(I,2).LE.93) then
          I1=K(I,3)-1
  120     I1=I1+1
          if(I1.GE.I) call pyerrm(8,
     &    '(LUNHEP:) translation of inconsistent event history')
          if(I1.LT.I.AND.K(I1,1).NE.1.AND.K(I1,1).NE.11) GOTO 120
          KC=pycomp(K(I1,2))
          if(I1.LT.I.AND.KC.EQ.0) GOTO 120
          if(I1.LT.I.AND.KCHG(KC,2).EQ.0) GOTO 120
          JMOHEP(2,I)=I1
        elseif(K(I,2).EQ.94) then
          NJET=2
          if(NHEP.GE.I+3.AND.K(I+3,3).LE.I) NJET=3
          if(NHEP.GE.I+4.AND.K(I+4,3).LE.I) NJET=4
          JMOHEP(2,I)=MOD(K(I+NJET,4)/MSTU(5),MSTU(5))
          if(JMOHEP(2,I).EQ.JMOHEP(1,I)) JMOHEP(2,I)=
     &    MOD(K(I+1,4)/MSTU(5),MSTU(5))
        endif

C...Fill in missing daughter information.
        if(K(I,2).EQ.94.AND.MSTU(16).NE.2) then
          do 130 I1=JDAHEP(1,I),JDAHEP(2,I)
          I2=MOD(K(I1,4)/MSTU(5),MSTU(5))
  130     JDAHEP(1,I2)=I
        endif
        if(K(I,2).GE.91.AND.K(I,2).LE.94) GOTO 140
        I1=JMOHEP(1,I)
        if(I1.LE.0.OR.I1.GT.NHEP) GOTO 140
        if(K(I1,1).NE.13.AND.K(I1,1).NE.14) GOTO 140
        if(JDAHEP(1,I1).EQ.0) then
          JDAHEP(1,I1)=I
        else
          JDAHEP(2,I1)=I
        endif
  140   CONTINUE
        do 150 I=1,NHEP
        if(K(I,1).NE.13.AND.K(I,1).NE.14) GOTO 150
        if(JDAHEP(2,I).EQ.0) JDAHEP(2,I)=JDAHEP(1,I)
  150   CONTINUE

C...Conversion from standard to Pythia, the easy part.
      else
        if(NHEP.GT.MSTU(4)) call pyerrm(8,
     &  '(LUNHEP:) insufficient space in /PYJETS/')
        N=MIN(NHEP,MSTU(4))
        NKQ=0
        KQSUM=0
        do 180 I=1,N
        K(I,1)=0
        if(ISTHEP(I).EQ.1) K(I,1)=1
        if(ISTHEP(I).EQ.2) K(I,1)=11
        if(ISTHEP(I).EQ.3) K(I,1)=21
        K(I,2)=lutran(IDHEP(I),2)
        K(I,3)=JMOHEP(1,I)
        K(I,4)=JDAHEP(1,I)
        K(I,5)=JDAHEP(2,I)
        do 160 J=1,5
  160   P(I,J)=PHEP(J,I)
        do 170 J=1,4
  170   V(I,J)=VHEP(J,I)
        V(I,5)=0.
        if(ISTHEP(I).EQ.2.AND.PHEP(4,I).GT.PHEP(5,I)) then
          I1=JDAHEP(1,I)
          if(I1.GT.0.AND.I1.LE.NHEP) V(I,5)=(VHEP(4,I1)-VHEP(4,I))*
     &    PHEP(5,I)/PHEP(4,I)
        endif

C...Fill in missing information on colour connection in jet systems.
        if(ISTHEP(I).EQ.1) then
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
     &  '(LUNHEP:) input parton configuration not colour singlet')
      endif

      if(mconv.eq.1) call pythia2ev4

      return
      end
