*
* $Id: qqbuffer.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqbuffer.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:32 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.00/00 23/08/90 13.44.03 by Paul Avery
*-- Author :
* IRCIQQ 0 if input file was just opened
* 1 if last input record is begin run
* 2 if last input record is event
* 3 if last input record is end run
* 4 if last input record is header
* 5 if last input record is EOF
* 6 if last input record is end of dataset (default)
* IRCOQQ 0 if output file was just opened
* 1 if last output record is begin run
* 2 if last output record is event
* 3 if last output record is end run
* 4 if last output record is header
* 5 if last output record is EOF
* 6 if last output record is end of dataset (default)
* BUFRQQ Special buffer user to store input
* BUFWQQ Special buffer user to store output
      INTEGER IRCIQQ, IRCOQQ
      REAL BUFRQQ, BUFWQQ
C
      COMMON /QQC111/ IRCIQQ, IRCOQQ, BUFRQQ(1000), BUFWQQ(1000)
      INTEGER IBFRQQ(1000), IBFWQQ(1000)
      EQUIVALENCE (BUFRQQ, IBFRQQ)
      EQUIVALENCE (BUFWQQ, IBFWQQ)
