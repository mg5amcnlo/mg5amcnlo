*
* $Id: qqsngl.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqsngl.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:33 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 05/10/94 11.04.14 by Peter C Kim
*CMZ : 08/10/93 10.46.33 by Paul Avery
*>> Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQSNGL.INC
*
* Setup for single particle generation
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* PSINGL(2) Momentum range
* STHET(2) Theta angle range (0 - pi)
* SPHI(2) PHI angle range (0 - 2pi)
*
* CSINGL Single particle name
      REAL PSINGL,STHET,SPHI
      COMMON /QQSNG1/
     * PSINGL(2),STHET(2),SPHI(2)
      CHARACTER*10 CSINGL
      COMMON /QQSNG2/ CSINGL
      SAVE /QQSNG1/, /QQSNG2/
