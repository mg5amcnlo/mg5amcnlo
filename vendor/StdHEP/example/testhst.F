      SUBROUTINE TSTFLEM(IOFF)
C... fill histograms
#include "stdhep.inc"

      INTEGER IOFF,I

      CALL HFILL(IOFF+11,FLOAT(NEVHEP),0.,1.)
      CALL HFILL(IOFF+12,FLOAT(NHEP),0.,1.)
      DO 100 I=1,NHEP
        CALL HFILL(IOFF+13,FLOAT(IDHEP(I)),0.,1.)
        CALL HFILL(IOFF+14,FLOAT(ISTHEP(I)),0.,1.)
        CALL HFILL(IOFF+15,FLOAT(JMOHEP(1,I)),0.,1.)
        CALL HFILL(IOFF+16,FLOAT(JMOHEP(2,I)),0.,1.)
        CALL HFILL(IOFF+17,FLOAT(JDAHEP(1,I)),0.,1.)
        CALL HFILL(IOFF+18,FLOAT(JDAHEP(2,I)),0.,1.)
        CALL HFILL(IOFF+21,SNGL(PHEP(1,I)),0.,1.)
        CALL HFILL(IOFF+22,SNGL(PHEP(2,I)),0.,1.)
        CALL HFILL(IOFF+23,SNGL(PHEP(3,I)),0.,1.)
        CALL HFILL(IOFF+24,SNGL(PHEP(4,I)),0.,1.)
        CALL HFILL(IOFF+25,SNGL(PHEP(5,I)),0.,1.)
        CALL HFILL(IOFF+26,SNGL(VHEP(1,I)),0.,1.)
        CALL HFILL(IOFF+27,SNGL(VHEP(2,I)),0.,1.)
        CALL HFILL(IOFF+28,SNGL(VHEP(3,I)),0.,1.)
        CALL HFILL(IOFF+29,SNGL(VHEP(4,I)),0.,1.)
 100  CONTINUE
      RETURN
      END
      SUBROUTINE TSTFLXSEC(IOFF)
C... fill histograms
#include "stdhep.inc"
#include "stdcm1.inc"

      INTEGER IOFF
      REAL BN

      BN = FLOAT(NEVHEP)+0.5
      CALL HFILL(IOFF+20,BN,0.,STDXSEC)
      RETURN
      END
      SUBROUTINE TSTBKEM(IOFF)
C--- book histograms
C             IOFF =  0  --> Herwig
C             IOFF = 20  --> Pythia
C             IOFF = 40  --> Isajet
C             IOFF = 60  --> PYTHIA
C             IOFF = -1  --> all
C
      IMPLICIT NONE
C---   The HBOOK common
      REAL HMEMOR
      COMMON/PAWC/HMEMOR(50000)
      INTEGER IOFF
C---   Setup the HBOOK memory limit.
      CALL HLIMIT(50000)
C---   Book histograms.
      IF(IOFF.EQ.20) GOTO 100
      IF(IOFF.EQ.40) GOTO 200
      CALL HBOOK1(11,'Herwig: NEVHEP',100,0.0,100.0,0.)
      CALL HBOOK1(12,'Herwig: NHEP',100,0.,1000.,0.)
      CALL HBOOK1(13,'Herwig: IDHEP',100,0.,100.,0.)
      CALL HBOOK1(14,'Herwig: ISTHEP',100,0.,200.,0.)
      CALL HBOOK1(15,'Herwig: JMOHEP(1,)',100,0.,1000.,0.)
      CALL HBOOK1(16,'Herwig: JMOHEP(2,)',100,0.,1000.,0.)
      CALL HBOOK1(17,'Herwig: JDAHEP(1,)',100,0.,1000.,0.)
      CALL HBOOK1(18,'Herwig: JDAHEP(2,)',100,0.,1000.,0.)
      CALL HBOOK1(20,'Herwig: cross-section',100,0.,100.,0.)
      CALL HBOOK1(21,'Herwig: PHEP(1,)',100,-100.,100.,0.)
      CALL HBOOK1(22,'Herwig: PHEP(2,)',100,-100.,100.,0.)
      CALL HBOOK1(23,'Herwig: PHEP(3,)',100,-1000.,1000.,0.)
      CALL HBOOK1(24,'Herwig: PHEP(4,)',100,-1000.,1000.,0.)
      CALL HBOOK1(25,'Herwig: PHEP(5,)',50,-250.,250.,0.)
      CALL HBOOK1(26,'Herwig: VHEP(1,)',10,-1.,1.,0.)
      CALL HBOOK1(27,'Herwig: VHEP(2,)',10,-1.,1.,0.)
      CALL HBOOK1(28,'Herwig: VHEP(3,)',10,-1.,1.,0.)
      CALL HBOOK1(29,'Herwig: VHEP(4,)',10,0.,1.,0.)
      IF(IOFF.GE.0) GOTO 400
 100  CALL HBOOK1(31,'Pythia: NEVHEP',100,0.,100.,0.)
      CALL HBOOK1(32,'Pythia: NHEP',100,0.,200.,0.)
      CALL HBOOK1(33,'Pythia: IDHEP',100,0.,100.,0.)
      CALL HBOOK1(34,'Pythia: ISTHEP',20,0.,20.,0.)
      CALL HBOOK1(35,'Pythia: JMOHEP(1,)',100,0.,100.,0.)
      CALL HBOOK1(36,'Pythia: JMOHEP(2,)',100,0.,100.,0.)
      CALL HBOOK1(37,'Pythia: JDAHEP(1,)',100,0.,100.,0.)
      CALL HBOOK1(38,'Pythia: JDAHEP(2,)',100,0.,100.,0.)
      CALL HBOOK1(41,'Pythia: PHEP(1,)',100,-100.,100.,0.)
      CALL HBOOK1(42,'Pythia: PHEP(2,)',100,-100.,100.,0.)
      CALL HBOOK1(43,'Pythia: PHEP(3,)',100,-100.,100.,0.)
      CALL HBOOK1(44,'Pythia: PHEP(4,)',100,-100.,100.,0.)
      CALL HBOOK1(45,'Pythia: PHEP(5,)',100,-100.,100.,0.)
      CALL HBOOK1(46,'Pythia: VHEP(1,)',100,-50.,50.,0.)
      CALL HBOOK1(47,'Pythia: VHEP(2,)',100,-50.,50.,0.)
      CALL HBOOK1(48,'Pythia: VHEP(3,)',100,-50.,50.,0.)
      CALL HBOOK1(49,'Pythia: VHEP(4,)',100,0.,100.,0.)
      IF(IOFF.GE.0) GOTO 400
 200  CALL HBOOK1(51,'Isajet: NEVHEP',100,0.,100.,0.)
      CALL HBOOK1(52,'Isajet: NHEP',100,0.,1000.,0.)
      CALL HBOOK1(53,'Isajet: IDHEP',100,0.,100.,0.)
      CALL HBOOK1(54,'Isajet: ISTHEP',50,0.,50.,0.)
      CALL HBOOK1(55,'Isajet: JMOHEP(1,)',100,0.,1000.,0.)
      CALL HBOOK1(56,'Isajet: JMOHEP(2,)',100,0.,1000.,0.)
      CALL HBOOK1(57,'Isajet: JDAHEP(1,)',100,0.,1000.,0.)
      CALL HBOOK1(58,'Isajet: JDAHEP(2,)',100,0.,1000.,0.)
      CALL HBOOK1(60,'Isajet: cross-section',100,0.,100.,0.)
      CALL HBOOK1(61,'Isajet: PHEP(1,)',100,-500.,500.,0.)
      CALL HBOOK1(62,'Isajet: PHEP(2,)',100,-500.,500.,0.)
      CALL HBOOK1(63,'Isajet: PHEP(3,)',100,-500.,500.,0.)
      CALL HBOOK1(64,'Isajet: PHEP(4,)',100,-500.,500.,0.)
      CALL HBOOK1(65,'Isajet: PHEP(5,)',100,-500.,500.,0.)
      CALL HBOOK1(66,'Isajet: VHEP(1,)',10,-1.,1.,0.)
      CALL HBOOK1(67,'Isajet: VHEP(2,)',10,-1.,1.,0.)
      CALL HBOOK1(68,'Isajet: VHEP(3,)',10,-1.,1.,0.)
      CALL HBOOK1(69,'Isajet: VHEP(4,)',10,0.,1.,0.)
 400  CONTINUE
      RETURN
      END
