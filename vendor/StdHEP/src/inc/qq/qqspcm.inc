*
* $Id: qqspcm.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqspcm.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:33 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 21/09/94 22.58.11 by Paul Avery
*-- Author : Lynn Garren 06/12/93
* 16/10/96 Lynn Garren: Add double precision conditionals.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQSPCM.INC
*
* LOCAL VARIABLES THAT ARE USED IN SUBROUTINE SPECTR
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
      DOUBLE PRECISION P2QRK, EMB, EMBQ, EMNU, EMCL, EMCQ, PBAR, EBEAM
      COMMON /SPECOM/P2QRK(4,4),EMB,EMBQ,EMNU,EMCL,EMCQ,PBAR,EBEAM
C---------------------------------------------------------------------
