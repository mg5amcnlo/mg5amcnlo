*
* $Id: qqluns.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqluns.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:32 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 21/09/94 22.55.49 by Paul Avery
*CMZ : 1.03/70 26/07/90 20.38.43 by Paul Avery
*-- Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQLUNS.INC
*
* QQ unit numbers
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* LTTIQQ Fortran unit number for terminal input
* LTTOQQ Fortran unit number for terminal output
* LOUTQQ Fortran unit number for file output
* LTINQQ Fortran unit number for data input
* LTOUQQ Fortran unit number for data output
* LTMPQQ Temporary Fortran unit number
C
      INTEGER LTTIQQ, LTTOQQ, LOUTQQ, LTINQQ, LTOUQQ, LTMPQQ
C
      COMMON /QQLUNS/ LTTIQQ, LTTOQQ, LOUTQQ, LTINQQ, LTOUQQ, LTMPQQ
