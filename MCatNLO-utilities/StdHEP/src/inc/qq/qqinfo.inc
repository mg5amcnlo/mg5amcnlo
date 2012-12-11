*
* $Id: qqinfo.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqinfo.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.5 1996/06/05 07:55:00 clib
* Added BDSIGX, BDSIGY (sigmas for beam dispersion)
*
* Revision 1.4 1996/05/01 07:03:58 zfiles
* Decay.dec version number is saved as IVRSDD.
*
* Revision 1.3 1996/02/21 20:59:08 lkg
* IVRSCG and IBNCMC now use spare words in qqevnt/qqinfo/roar fields -->
* backwards compatibility with old executables.
*
* Revision 1.2 1996/01/15 22:20:28 lkg
* Add CLEOG version number IVRSCG (qqinfo) and MC generated bunch number (qqevnt)
* Update roar fields to accommodate the new variables.
*
* Revision 1.1 1994/10/07 23:57:36 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 05/10/94 02.20.16 by Peter Kim
*CMZ : 05/10/94 02.20.00 by Peter Kim
*>> Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQINFO.INC
*
* QQ information
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* IVRSQQ QQ Version number
* IORGQQ Origin of QQ events
* Save the Model number used.
*
* IVRSCG CLEOG version number
* IVRSDD DECAY.DEC version number
* IRS1QQ Reserved for future use
      INTEGER IVRSQQ, IORGQQ, IVRSCG, IVRSDD, IRS1QQ
      COMMON /QQINF1/
     * IVRSQQ, IORGQQ, IVRSCG, IVRSDD, IRS1QQ
* DATEQQ Date the QQ job was run
* TIMEQQ Time the QQ job was run
      CHARACTER DATEQQ*20, TIMEQQ*20
      COMMON/QQINF2/
     * DATEQQ, TIMEQQ
* BDSIGX Sigma for Beam dispersion in X direction
* BDSIGY Sigma for Beam dispersion in Y direction
      REAL BDSIGX, BDSIGY
      COMMON/QQINF3/
     * BDSIGX, BDSIGY
