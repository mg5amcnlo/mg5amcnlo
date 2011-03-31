*
* $Id: lenocc.f,v 1.1 2009/07/30 22:46:16 madgraph Exp $
*
* $Log: lenocc.f,v $
* Revision 1.1  2009/07/30 22:46:16  madgraph
* JA: Implemented CKKW-style matching with Pythia pT-ordered showers
*
* Revision 1.1.1.1  1996/02/15 17:49:49  mclareni
* Kernlib
*
*
      FUNCTION LENOCC (CHV)
C
C CERN PROGLIB# M507    LENOCC          .VERSION KERNFOR  4.21  890323
C ORIG. March 85, A.Petrilli, re-write 21/02/89, JZ
C
C-    Find last non-blank character in CHV

      CHARACTER    CHV*(*)

      N = LEN(CHV)

      DO 17  JJ= N,1,-1
      IF (CHV(JJ:JJ).NE.' ') GO TO 99
   17 CONTINUE
      JJ = 0

   99 LENOCC = JJ
      RETURN
      END
