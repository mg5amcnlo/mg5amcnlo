
      subroutine ISAHEP(mconv)

C...Purpose: to convert ISAJET event record contents to or from
C...the standard event record common block.
C...convert (mconv=1) from ISAJET numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to ISAJET numbering scheme

#include "stdhep.inc"

C...for event number
#include "idrun.inc"
C...initial partons
#include "pjets.inc"
#include "primar.inc"
C...partons created during decay
#include "jetset.inc"
C...particles created in the decay, including final state particles
#include "partcl.inc"

      integer I1,IHP,MPART,NPOFF,NJHEP,NWHEP,IMO,IJT
      integer JPMO(2,MXJSET),JPDA(2,MXJSET),JMX(MXJSET),JMN(MXJSET)
      integer JTMO(2,MXPTCL),JTDA(2,MXPTCL)

      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        lfirst=.FALSE.
      endif
C...Conversion from ISAJET to standard
      if(mconv.EQ.1) then
        NEVHEP = IEVT
C...initial jets
        NHEP = 0
C... W or Z
        if(IDENTW.NE.0)then
          NHEP = NHEP + 1
          ISTHEP(NHEP)=12
          JMOHEP(1,NHEP)=0
          JMOHEP(2,NHEP)=0
          JDAHEP(1,NHEP)= 2
          JDAHEP(2,NHEP)= NJET + 1
          IDHEP(NHEP) = istran(IDENTW,1)
          do 100 J=1,5
 100        PHEP(J,NHEP) = QWJET(J)
        endif
        NWHEP = NHEP
C... jets
        if(NJET.GT.0)then
          do 120 I=1,NJET
            NHEP = NHEP + 1
            ISTHEP(NHEP)=11
            JMOHEP(1,NHEP)=0
            if(IDENTW.NE.0) JMOHEP(1,NHEP) = 1
            JMOHEP(2,NHEP)= I
            JDAHEP(1,NHEP)=0
            JDAHEP(2,NHEP)=0
            IDHEP(NHEP) = istran(IDJETS(I),1)
            do 110 J=1,5
 110          PHEP(J,NHEP) = PJETS(J,I)
 120      CONTINUE
        endif
        NJHEP = NHEP
C... pairs
        if(NPAIR.GT.0)then
          do 150 I=1,NPAIR
            NHEP = NHEP + 1
            ISTHEP(NHEP)=13
            JMOHEP(1,NHEP)= JPAIR(I) + NWHEP
            JMOHEP(2,NHEP)= JPAIR(I)
            JDAHEP(1,NHEP)=0
            JDAHEP(2,NHEP)=0
            IDHEP(NHEP) = istran(IDPAIR(I),1)
            do 140 J=1,5
 140          PHEP(J,NHEP) = PPAIR(J,I)
 150      CONTINUE
        endif
        do 160 I=1,NHEP
          do 160 J=1,4
 160        VHEP(J,I) = 0.
C...save offset into hep list
        NPOFF = NHEP
C...partons
        do 200 I=1,NJSET
          IHP = NHEP + I
C...use JMX and JMN to find daughters in hadron list
          JMX(I) = 0
          JMN(I) = NHEP + NPTCL + 1
          IDHEP(IHP) = istran(JTYPE(I),1)
          MPART=MOD(JORIG(I),JPACK)
          JMOHEP(1,IHP)=0
          IJT = JORIG(I)/JPACK
          if(MPART.NE.0)then
            JMOHEP(1,IHP)=MPART+NHEP
          elseif(MPART.EQ.0 .AND. IJT.LT.10)then
C...find mother in jet/pair list
            IMO = IJT + NWHEP
            if(NJHEP.LT.NPOFF)then
              KST = NJHEP + 1
              do 170 K=KST,NPOFF
                if(IDHEP(K).EQ.IDHEP(IHP)) IMO=K
 170          CONTINUE
            endif
            JMOHEP(1,IHP)= IMO
            if(JDAHEP(1,IMO).EQ.0) JDAHEP(1,IMO)=IHP
            JDAHEP(1,IMO) = MIN(IHP,JDAHEP(1,IMO))
            JDAHEP(2,IMO) = MAX(IHP,JDAHEP(2,IMO))
C...amend information if a parton thinks this is it's daughter
            KND = IHP-1
            do 175 K=NPOFF,KND
              if(IHP.GE.JDAHEP(1,K) .AND. IHP.LE.JDAHEP(2,K))
     1             JMOHEP(1,IHP)=K
 175        CONTINUE
          endif
          JMOHEP(2,IHP)= IJT
          if(JDCAY(I).EQ.0)then
            ISTHEP(IHP) = 21
            JDAHEP(1,IHP)=0
            JDAHEP(2,IHP)=0
          else
	    JD1 = JDCAY(I)/JPACK
	    JD2 = MOD(JDCAY(I),JPACK)
	    if( JD1.eq.I .and. JD2.eq.I ) then
              ISTHEP(IHP) = 23
	    else
              ISTHEP(IHP) = 22
	    endif
            JDAHEP(1,IHP) = JD1 + NHEP
            JDAHEP(2,IHP) = JD2 + NHEP
          endif
          do 180 J=1,5
 180        PHEP(J,IHP) = PJSET(J,I)
          do 190 J=1,4
 190        VHEP(J,IHP) = 0.
 200    CONTINUE
        NHEP = NHEP + NJSET
C...hadrons
        do 250 I=1,NPTCL
          IHP = NHEP + I
          IDHEP(IHP) = istran(IDENT(I),1)
          I1 = MOD(IABS(IORIG(I)),IPACK)
          JMOHEP(1,IHP)=0
          JMOHEP(2,IHP)=IABS(IORIG(I))/IPACK
C...mother is pomeron
          if(I1.EQ.0)then
C...mother is in parton list
          elseif(IORIG(I).LT.0)then
            JMOHEP(1,IHP) = I1 + NPOFF
            JMN(I1) = MIN(JMN(I1),I)
            JMX(I1) = MAX(JMX(I1),I)
C...mother is in hadron list
          elseif(IORIG(I).GT.0)then
            JMOHEP(1,IHP) = I1 + NHEP
          endif
          if(IDCAY(I).EQ.0)then
            ISTHEP(IHP) = 1
            JDAHEP(1,IHP)=0
            JDAHEP(2,IHP)=0
          else
            ISTHEP(IHP) = 2
            JDAHEP(1,IHP) = IDCAY(I)/IPACK + NHEP
            JDAHEP(2,IHP) = MOD(IDCAY(I),IPACK) + NHEP
          endif
          do 210 J=1,5
 210        PHEP(J,IHP) = PPTCL(J,I)
          do 220 J=1,4
 220        VHEP(J,IHP) = 0.
 250    CONTINUE
        NHEP = NHEP + NPTCL
C...fill in missing daughter info for partons
        do 270 I=1,NJSET
          if(JMX(I).NE.0)then
            JDAHEP(1,I+NPOFF) = JMN(I) + NPOFF + NJSET
            JDAHEP(2,I+NPOFF) = JMX(I) + NPOFF + NJSET
          endif
 270    CONTINUE
C...Conversion from standard to ISAJET
      elseif(mconv.EQ.2)then
        IEVT = NEVHEP
C...        missing information
        IDENTW = 0
        NPAIR = 0
        do 330 I=1,5
          QWJET(I) = 0.
          do 330 J=1,4
            PPAIR(I,J) = 0.
 330    CONTINUE
        do 340 I=1,4
          IDPAIR(I) = 0
 340      JPAIR(I) = 0
C...zero counters
        IJ1 = 0
        IJ2 = 0
        IJ3 = 0
        IP = 0
        IPT = 0
        do 500 I=1,NHEP
C...initial jets
C... jets
          if(ISTHEP(I).EQ.11)then
            IJ1 = IJ1 + 1
            IDJETS(IJ1) = istran(IDHEP(I),2)
            do 410 J=1,5
 410          PJETS(J,IJ1) = PHEP(J,I)
C... W
          elseif(ISTHEP(I).EQ.12)then
            IJ2 = IJ2 + 1
            IDENTW = istran(IDHEP(I),2)
            do 420 J=1,5
 420          QWJET(J) = PHEP(J,I)
C... pairs
          elseif(ISTHEP(I).EQ.13)then
            IJ3 = IJ3 + 1
            IDPAIR(IJ3) = istran(IDHEP(I),2)
            JPAIR(IJ3) = JMOHEP(2,I)
            do 430 J=1,5
 430          PPAIR(J,IJ3) = PHEP(J,I)
C...partons
          elseif(ISTHEP(I).EQ.21 .OR. ISTHEP(I).EQ.22)then
            IP = IP + 1
            JTYPE(IP) = istran(IDHEP(I),2)
            do 440 J=1,5
 440          PJSET(J,IP) = PHEP(J,I)
C...      temporary storage until have counts
            JPMO(1,IP) = JMOHEP(1,I)
            JPMO(2,IP) = JMOHEP(2,I)
            JPDA(1,IP) = JDAHEP(1,I)
            JPDA(2,IP) = JDAHEP(2,I)
C...hadrons
          else
            IPT = IPT + 1
            IDENT(IPT) = istran(IDHEP(I),2)
            do 450 J=1,5
 450          PPTCL(J,IPT) = PHEP(J,I)
C...      temporary storage until have counts
            JTMO(1,IPT) = JMOHEP(1,I)
            JTMO(2,IPT) = JMOHEP(2,I)
            JTDA(1,IPT) = JDAHEP(1,I)
            JTDA(2,IPT) = JDAHEP(2,I)
          endif
 500    CONTINUE
C...completed counts
        NJET = IJ1
        NPAIR = IJ3
        NJSET = IP
        NPTCL = IPT
C...get mother/daughter information
        NSUM1 = NJET + IJ2 + NPAIR
        NSUM2 = NSUM1 + NJSET
        do 520 I=1,NJSET
          if(JPDA(1,I).EQ.0)then
            JDCAY(I) = 0
          elseif(JPDA(1,I).GT.NSUM2)then
            JDCAY(I) = 0
          else
            JDCAY(I) = JPACK*(JPDA(1,I)-NSUM1) + JPDA(2,I)-NSUM1
          endif
          if(JPMO(1,I).LE.NSUM1)then
            JORIG(I) = JPACK*JPMO(2,I)
          else
            JORIG(I) = JPACK*JPMO(2,I) + JPMO(1,I)-NSUM1
          endif
 520    CONTINUE
        do 550 I=1,NPTCL
          if(JTDA(1,I).EQ.0)then
            IDCAY(I) = 0
          else
            IDCAY(I) = IPACK*(JTDA(1,I)-NSUM2) + JTDA(2,I)-NSUM2
          endif
          if(JTMO(1,I).LE.NSUM1)then
            IORIG(I) = JTMO(2,I)*IPACK + 0
          elseif(JTMO(1,I).LE.NSUM2)then
            IORIG(I) = -(JTMO(2,I)*IPACK + JTMO(1,I)-NSUM1)
          else
            IORIG(I) = JTMO(2,I)*IPACK + JTMO(1,I)-NSUM2
          endif
 550    CONTINUE
      endif
      return
      end
