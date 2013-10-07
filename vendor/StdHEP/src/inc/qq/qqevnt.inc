*
* $Id: qqevnt.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqevnt.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.6 1996/05/01 07:05:43 zfiles
* BDISPX, BDISPY (beam dispersion) and P4PRQ1, P4PRQ2 (primary quark 4 momenta)
* are saved.
*
* Revision 1.5 1996/02/21 20:59:06 lkg
* IVRSCG and IBNCMC now use spare words in qqevnt/qqinfo/roar fields -->
* backwards compatibility with old executables.
*
* Revision 1.4 1996/02/16 20:03:39 zfiles
* Added IQQFLV
*
* Revision 1.3 1996/01/15 22:20:27 lkg
* Add CLEOG version number IVRSCG (qqinfo) and MC generated bunch number (qqevnt)
* Update roar fields to accommodate the new variables.
*
* Revision 1.2 1995/04/25 13:58:07 zfiles
* New variable XANGQQ and IDKMEC
*
* Revision 1.1 1994/10/07 23:57:35 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 03/10/94 17.34.41 by Paul Avery
*CMZ : 08/10/93 10.27.52 by Paul Avery
*>> Author :
* 15/10/96 Lynn Garren: Add double precision conditionals.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQEVNT.INC
*
* QQ parameters at the start of an event
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* IEVTQQ QQ Event number
* IRUNQQ QQ Run number
* IRANQQ(2) Random seeds at start of QQ event
* IRANMC(2) Random seeds at start of event in subsequent analysis
* IRANCC(2) Historical. Not used
* IQQFLV e+e- --> qq= quark type (JETSET continuum)
* IRS2QQ(4) Reserved for future use
* IBMRAD 0 ==> No photon radiated 1 ==> Photon radiated
* IBNCMC generated bunch number of event
*
* ECM CM invariant mass after radiation
* P4CMQQ(4) 4 vector of CM in lab
* P4PHQQ(4) 4 vector of radiated photon
* ENERNW CM energy after radiation
* BEAMNW ENERNW/2
* BEAMP Positron energy
* BEAMN Electron energy
* XANGQQ Beam crossing angle (in mrad)
* BDISPX Beam dispersion in x direction (in mrad)
* BDISPY Beam dispersion in y direction (in mrad)
* P4PRQ1(4) 4 vector of the primary quark in LUND
* P4PRQ2(4) 4 vector of the primary anti-quark in LUND
      INTEGER IEVTQQ, IRUNQQ
      INTEGER IRANQQ, IRANMC, IRANCC, IQQFLV, IRS2QQ
      INTEGER IBMRAD, IBNCMC
      COMMON/QQEVT1/
     * IEVTQQ, IRUNQQ,
     * IRANQQ(2), IRANMC(2), IRANCC(2), IQQFLV, IBNCMC, IRS2QQ(3),
     * IBMRAD
      DOUBLE PRECISION P4CMQQ, P4PHQQ
      REAL ECM, ENERNW, BEAMNW, BEAMP, BEAMN
      REAL XANGQQ, BDISPX, BDISPY, P4PRQ1, P4PRQ2
      COMMON/QQEVT2/
     * P4CMQQ(4), P4PHQQ(4), ECM,
     * ENERNW, BEAMNW, BEAMP, BEAMN, XANGQQ, BDISPX, BDISPY,
     * P4PRQ1(4), P4PRQ2(4)
