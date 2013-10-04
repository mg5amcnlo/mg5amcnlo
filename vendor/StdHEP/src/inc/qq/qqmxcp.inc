*
* $Id: qqmxcp.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqmxcp.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:33 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 15/10/93 09.31.21 by Paul Avery
*-- Author :
* 30/10/96 Lynn Garren: Change CTTAU from real to double precision.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQMXCP.INC
*
* Contains info on mixing and CP violation within an event
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* ILDECA(*) Decay channel of particle from mixing/CP
* CTTAU(*) Generated C*T in center of mass using mixing/CP info
      INTEGER ILDECA
      DOUBLE PRECISION CTTAU
      COMMON /QQMXC1/
     * CTTAU(250), ILDECA(250)
