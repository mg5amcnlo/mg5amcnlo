*
* $Id: mcjet.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: mcjet.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:32 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.00/00 04/06/90 18.55.55 by Jorge L. Rodriguez
*-- Author :
C ---------------------------------------------------------------------
C CLMCX:MCJET.INC
C ---------------------------------------------------------------------
C ICLASS(36) qqbar class (1-10)
C PSPIN(10) Pseudoscalar fraction per cless
C CMIX1(3,6) SU(3) mixing for uu, dd, etc
C CMLT1(5) 1st coefficient in mult. formula
C CMLT2(5) 2nd coefficient in mult. formula
C WIDTH(5) 3rd coefficient in mult. formula
      INTEGER ICLASS
      REAL PSPIN, CMIX1, CMLT1, CMLT2, WIDTH
      COMMON/JET3PA/
     * ICLASS(36),
     * PSPIN(10),
     * CMIX1(3,6),
     * CMLT1(5),
     * CMLT2(5),
     * WIDTH(5)
