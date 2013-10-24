*
* $Id: qqpars.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqpars.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.4 1996/07/17 07:31:20 clib
* Increase the size of arrays for Helicity cards (500 --> 1000)
*
* Revision 1.3 1994/10/27 16:59:35 zfiles
* MCNUM of 1000 does not fit in a ROAR record.
*
* Revision 1.2 1994/10/25 13:59:38 zfiles
* At Peter and Paul's request, we set MCNUM to 1000. (from 500)'
*
* Revision 1.1 1994/10/07 23:57:38 zfiles
* New include files for QQ.
*
*
*CMZ : 1.02/61 03/10/94 17.34.06 by Paul Avery
*CMZ : 1.03/70 11/10/93 17.13.22 by Paul Avery
*-- Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQPARS.INC
*
* Parameters used in most dimensions in QQ
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* MCTRK Max number of QQ tracks
* NTRKS Ditto
* MCVRTX Max number of QQ vertices
* NVTXS Ditto
* MCHANS Total number of decay channels (BRLIST, MLLIST)
* MCDTRS Total number of daughters (IDLIST)
* MPOLQQ Total number of polarized decays
* MCNUM Max number of stable particles (ISTBLE, XMAS, IPLIST0
* MCSTBL Max number of stable particle codes
* MCSTAB Max length of stable particle list
* MCTLQQ Max # of lines to store from QQ control file
* MDECQQ Max # of lines to store from QQ decay file
* MHLPRB Max length of HELPRB
* MHLLST Max length of HELLST
* MHLANG Max length of HELANG
* MCPLST Max length of CPLIST
* MFDECA Max number of decay file names to store
      INTEGER MCTRK, NTRKS, MCVRTX, NVTXS
      INTEGER MCHANS, MCDTRS, MPOLQQ
      INTEGER MCNUM, MCSTBL, MCSTAB, MCTLQQ, MDECQQ
      INTEGER MHLPRB, MHLLST, MHLANG, MCPLST, MFDECA
      INTEGER MCNUM1
      PARAMETER (MCTRK = 512)
      PARAMETER (NTRKS = MCTRK)
      PARAMETER (MCVRTX = 256)
      PARAMETER (NVTXS = MCVRTX)
      PARAMETER (MCHANS = 4000)
      PARAMETER (MCDTRS = 8000)
      PARAMETER (MPOLQQ = 300)
      PARAMETER (MCNUM = 600)
      PARAMETER (MCSTBL = 40)
      PARAMETER (MCSTAB = 512)
      PARAMETER (MCTLQQ = 100)
      PARAMETER (MDECQQ = 300)
      PARAMETER (MHLPRB = 1000)
      PARAMETER (MHLLST = 2000)
      PARAMETER (MHLANG = 1000)
      PARAMETER (MCPLST = 200)
      PARAMETER (MFDECA = 5)
      PARAMETER (MCNUM1 = MCNUM + 21)
