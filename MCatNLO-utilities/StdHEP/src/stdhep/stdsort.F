
      subroutine STDSORT

C... sort HEPEVT by daughter list

#include "stdhep.inc"
#include "stdlun.inc"

      integer IST(NMXHEP),ID(NMXHEP),JMO(2,NMXHEP),JDA(2,NMXHEP)
      double precision P1(5,NMXHEP),V1(4,NMXHEP)
      integer I,IN,J,K,L,LIST(NMXHEP),IREV(NMXHEP)
      integer LSIZE
      PARAMETER (LSIZE=100)
      integer NDAU,LDTR(LSIZE),LD,IP,JN1,JN2,JTMP

C...zero the temporary arrays
      do I=1,NMXHEP
        LIST(I)=0
        IREV(I)=0
        IST(I)=0
        ID(I)=0
        do J=1,2
          JMO(J,I)=0
          JDA(J,I)=0
        enddo
        do J=1,5
          P1(J,I)=0.
        enddo
        do J=1,4
          V1(J,I)=0.
        enddo
      enddo

      IN = 0
C... start by listing all particles with no parent
C...       or otherwise from initial state
      do I=1,NHEP
        if((JMOHEP(1,I).EQ.0 .AND. JMOHEP(2,I).EQ.0) .OR.
     1           (ISTHEP(I).EQ.3))then
          IN = IN+1
          LIST(I) = IN
          IREV(IN) = I
          IST(IN) = ISTHEP(I)
          ID(IN) = IDHEP(I)
          do J=1,5
            P1(J,IN) = PHEP(J,I)
          enddo
          do J=1,4
            V1(J,IN) = VHEP(J,I)
          enddo
        endif
      enddo
C... now start adding the remaining particles
      JN1 = 1
      JN2 = IN
      do WHILE (JN2.GE.JN1)
        do I=JN1,JN2
C... find the daughters of this particle and add them
          call STDDAUTRLST(IREV(I),NDAU,LSIZE,LDTR)
          if(NDAU.GT.0)then
            LD = MIN(NDAU,LSIZE)
            do K=1,LD
              IP = LDTR(K)
C... has this particle been listed already?
              if(LIST(IP).EQ.0)then
                IN = IN+1
                LIST(IP) = IN
                IREV(IN) = IP
                IST(IN) = ISTHEP(IP)
                ID(IN) = IDHEP(IP)
                do J=1,5
                  P1(J,IN) = PHEP(J,IP)
                enddo
                do J=1,4
                  V1(J,IN) = VHEP(J,IP)
                enddo
              else
                write(lnhout,1001) IP
              endif
            enddo
          endif
        enddo
        JN1 = JN2+1
        JN2 = IN
      enddo
C...did we miss anything?
      do I=1,NHEP
        if(LIST(I).EQ.0)then
D         write(lnhout,1002) I
          IN = IN+1
          LIST(I) = IN
          IREV(IN) = I
          IST(IN) = ISTHEP(I)
          ID(IN) = IDHEP(I)
          do J=1,5
            P1(J,IN) = PHEP(J,I)
          enddo
          do J=1,4
            V1(J,IN) = VHEP(J,I)
          enddo
C...get daughters of this particle
          if(JDAHEP(1,I).GT.0)then
      JN1 = IN
      JN2 = IN
      do WHILE (JN2.GE.JN1)
        do L=JN1,JN2
C... find the daughters of this particle and add them
          call STDDAUTRLST(IREV(L),NDAU,LSIZE,LDTR)
          if(NDAU.GT.0)then
            LD = MIN(NDAU,LSIZE)
            do K=1,LD
              IP = LDTR(K)
C... has this particle been listed already?
              if(LIST(IP).EQ.0)then
                IN = IN+1
                LIST(IP) = IN
                IREV(IN) = IP
                IST(IN) = ISTHEP(IP)
                ID(IN) = IDHEP(IP)
                do J=1,5
                  P1(J,IN) = PHEP(J,IP)
                enddo
                do J=1,4
                  V1(J,IN) = VHEP(J,IP)
                enddo
              else
                write(lnhout,1001) IP
              endif
            enddo
          endif
        enddo
        JN1 = JN2+1
        JN2 = IN
      enddo
          endif
        endif
      enddo
C...get mother/daughter info
      do I=1,IN
        do K=1,2
          JTMP = JDAHEP(K,IREV(I))
          JDA(K,I) = LIST(JTMP)
          JTMP = JMOHEP(K,IREV(I))
          JMO(K,I) = LIST(JTMP)
        enddo
      enddo
C...put it all back into the common block
      if(IN.NE.NHEP) write(lnhout,1003) IN,NHEP
      do I=1,IN
        ISTHEP(I) = IST(I)
        IDHEP(I) = ID(I)
        do J=1,2
          JMOHEP(J,I) = JMO(J,I)
          JDAHEP(J,I) = JDA(J,I)
        enddo
        do J=1,5
          PHEP(J,I) = P1(J,I)
        enddo
        do J=1,4
          VHEP(J,I) = V1(J,I)
        enddo
      enddo
      return
1001  format(' STDSORT: particle ',I5,' is out of order')
1002  format(' STDSORT: particle ',I5,
     1      ' was not found in daughter search')
1003  format(' STDSORT: WARNING found only ',I4,' of ',I4,' particles')
      end
