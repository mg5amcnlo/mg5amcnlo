*
* $Id: qqbmst.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqbmst.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1 1994/10/07 23:57:34 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 03/10/94 17.35.16 by Paul Avery
*CMZ : 08/10/93 11.16.15 by Paul Avery
*>> Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQBMST.INC
*
* Initial beam conditions
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* ENERQQ Nominal CM energy
* BEAMQQ ENERQQ/2
* BMPSQQ Nominal positron beam energy
* BMNGQQ Nominal electron beam energy
* EWIDQQ CM energy 1 sigma spread (BWPSQQ, BWNGQQ actually used
* in event generation)
* BWPSQQ Positron energy 1 sigma spread
* BWNGQQ Electron energy 1 sigma spread
* BPOSQQ(3) Beam centroid (meters)
* BSIZQQ(3) Beam sigma (meters)
*
      REAL ENERQQ, BEAMQQ, BMPSQQ, BMNGQQ, EWIDQQ, BWPSQQ, BWNGQQ
      REAL BPOSQQ, BSIZQQ
      COMMON/QQBMS1/
     * ENERQQ, BEAMQQ, BMPSQQ, BMNGQQ, EWIDQQ, BWPSQQ, BWNGQQ,
     * BPOSQQ(3), BSIZQQ(3)
