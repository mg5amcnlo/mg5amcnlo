*
* $Id: qqbrat.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqbrat.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.1.1.1 1994/10/08 02:21:32 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 22/09/94 00.06.59 by Paul Avery
*-- Author :
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQBRAT.INC
*
* Information on decay channels
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* Decay information per particle
* IPLIST(1,*) Pntr to first entry in MLLIST/BRLIST per particle (0=stable)
* (2,*) # of decay channels per particle (0 if stable)
* (3,*) Pntr to first entry in CPLIST per particle (0=no CP decays)
* (4,*) # of CP eigenstate channels in CPLIST
      INTEGER IPLIST
      COMMON/QQBRA1/
     * IPLIST(4,-20:MCNUM)
* Decay information per channel
* NBRLST # of decay channels defined
* BRLIST(*) Branching fraction per channel (in cumulative order)
* AGLIST(7,*) User defined parameters for channel. For 2 body decays they
* specify the angular distribution:
* dN/dcos(theta) = A1 + ... + A7*cos(theta)**6
* MLLIST(1-9,*) Decay information per channel
* 1 # daughters
* 2 pointer to first daughter position in IDLIST
* 3 MATRX = matrix element ID number
* 4 Number of helicity states defined
* 5 Number of helicity angular distributions
* 6 Pointer to first helicity state in HELPRB (0 if none)
* 7 Pointer to helicity angular dist. list HELANG (0 if none)
* 8 Pointer to corresponding entry in CPLIST (CP eig. BR list)
* 9 Sum of daughter type ID's. Used in inclusive B decays'
* LCPTAG(*) TRUE if daughter decays to be carried out as tag + CP eig.
*
      INTEGER NBRLST, MLLIST
      LOGICAL LCPTAG
      REAL BRLIST, AGLIST
      COMMON/QQBRA2/
     * NBRLST,
     * BRLIST(MCHANS), AGLIST(7,MCHANS), MLLIST(9,MCHANS),
     * LCPTAG(MCHANS)
* Daughter list pointed to by all channels
* NDGHTR Total number of daughters defined
* IDLIST(*) List of daughters
      INTEGER NDGHTR, IDLIST
      COMMON/QQBRA3/
     * NDGHTR, IDLIST(MCDTRS)
* Helicity ang. distributions (set by ANGULAR_HELICITY commd.)
* NHLANG Length of HELANG list
* HELANG(*) Helicity of this decay
* COFANG(7,*) A1, A2, etc. used in dN/dcos = A1 + ... + A7*cos(theta)**6
      INTEGER NHLANG
      REAL HELANG, COFANG
      COMMON/QQBRA4/
     * NHLANG, HELANG(MHLANG), COFANG(7,MHLANG)
* Helicity state probability list (set by HELICITY command)
* NHLPRB Length of HELPRB
* IHLPRB(*) Pointer to first daughter helicity in HELLST
* HELPRB(*) Probability (stored in cumulative order)
      INTEGER NHLPRB, IHLPRB
      REAL HELPRB
      COMMON/QQBRA5/
     * NHLPRB, IHLPRB(MHLPRB), HELPRB(MHLPRB)
* Helicity states of certain daughters
* NHLLST Length of HELLST
* HELLST(*) List of helicities (pointed to by IHLPRB(I))
      INTEGER NHLLST
      REAL HELLST
      COMMON/QQBRA6/
     * NHLLST, HELLST(MHLLST)
* List of CP eigenstatechannels
* NCPLST Number of CP eigenstate channels
* CPLIST(*) CP branching fraction per channel (in cumulative order)
* ICPLST(*) Pointer to corresponding entry in MLLIST
* CPSNPH(*) CP sin(phi) per channel
      INTEGER NCPLST, ICPLST
      REAL CPLIST, CPSNPH
      COMMON/QQBRA7/
     * NCPLST, CPLIST(MCPLST), ICPLST(MCPLST), CPSNPH(MCPLST)
* NFDECA # of decay files read in
* CFDECA(*) List of decay files read in
      INTEGER NFDECA
      COMMON/QQBRA8/
     * NFDECA
      CHARACTER*80 CFDECA
      COMMON/QQBRA9/
     * CFDECA(MFDECA)
