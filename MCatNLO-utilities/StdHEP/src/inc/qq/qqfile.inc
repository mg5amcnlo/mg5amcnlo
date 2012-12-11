*
* $Id: qqfile.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqfile.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1 1994/10/07 23:57:35 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 05/10/94 03.07.47 by Peter Kim
*CMZ : 08/10/93 10.40.20 by Paul Avery
*>> Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQFILE.INC
*
* Names and contents of files that were used in this QQ run
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* FOUTQQ Name of QQ printout file (*.DAT)
* FCTLQQ Name of QQ control file (*.ctr)
* FDECQQ Name of initial decay file (*.dec)
* FGEOQQ Geometry file (not used in QQ)
*
* NCTLQQ # lines in FCTLQQ
* NDECQQ # lines in NEWDEC
* CCTLQQ(*) Contents of FCTLQQ
* CDECQQ(*) Contents of NEWDEC
      CHARACTER FOUTQQ*80, FCTLQQ*80, FDECQQ*80
      CHARACTER FGEOQQ*80
      COMMON/QQFIL1/
     * FOUTQQ, FCTLQQ, FDECQQ, FGEOQQ
      INTEGER NCTLQQ, NDECQQ
      COMMON/QQFIL2/
     * NCTLQQ, NDECQQ
      CHARACTER CCTLQQ*80, CDECQQ*80
      COMMON/QQFIL3/
     * CCTLQQ(MCTLQQ), CDECQQ(MDECQQ)
*
