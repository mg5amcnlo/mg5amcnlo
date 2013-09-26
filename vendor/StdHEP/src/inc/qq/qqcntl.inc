*
* $Id: qqcntl.inc,v 1.1 2005/09/29 19:27:22 garren Exp $
*
* $Log: qqcntl.inc,v $
* Revision 1.1  2005/09/29 19:27:22  garren
* 5.04.02
*
* Revision 1.2 1995/02/03 16:15:45 zfiles
* Mods from PCK. Call GTPIR and a few other routines after TRKSIM
* as options
*
* Revision 1.1.1.1 1994/10/08 02:21:33 zfiles
* first version of qqlib in CVS
*
*
*CMZ : 1.04/00 05/10/94 11.05.39 by Peter C Kim
*CMZ : 1.03/74 07/01/94 13.29.47 by Brian Heltsley
* Add bhlumi variables
*CMZ : 1.03/70 11/10/93 09.57.47 by Paul Avery
*CMZ : 08/10/93 11.43.41 by Paul Avery
*>> Author :
* 31/10/96 Lynn GArren: Make CEN, FWID, C1B, and C2B double precision.
*
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
* QQCNTL.INC
*
* Control information for QQ run
* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
*
* NDUMP Number of events to dump
* NDO Number of events to generate
* IRAN Initial random seed (0 ==> generate from daytime)
* IRAND Initial random seed actually used
* IDQQ ID number of this process
* MODEL Model # we are using to generate events
      INTEGER NDUMP, NDO, IRAN, IRAND, IDQQ
      INTEGER MODEL
      COMMON /QQCNT1/
     * NDUMP, NDO, IRAN, IRAND, IDQQ, MODEL
* SPREAD Energy spread (sigma) at 10 GeV (scales like E**2). Used
* to generate BWPSQQ,BWNGQQ only if BMPSQQ = 0.
* ENRGMN Min CM energy allowed after radiation
* ENRGMX Max CM energy allowed after radiation
* KEMIN Min. radiated photon energy in percent of E(beam)
* KEMAX Max. radiated photon energy in percent of E(beam)
* BHEMIN Min e+ or e- energy/ebeam (for BHLUMI only)
* CRSANG Beam crossing angle (in mrad)
      REAL SPREAD, ENRGMN, ENRGMX, KEMIN, KEMAX, BHEMIN, CRSANG
      COMMON/QQCNT3/
     * SPREAD, ENRGMN, ENRGMX, KEMIN, KEMAX, BHEMIN, CRSANG
* LTAPE TRUE ==> Write events to file
* BMRAD TRUE ==> Allow initial state radiation
* LWIDE TRUE ==> Use FWHM defined in decay file
* TRACE TRUE ==> Call Sub. ERRSET to trace errors
* LSHAPE TRUE ==> Fill shape 4 vectors
* LSELCT TRUE ==> Call function ISELCT to select events
* LPHASE TRUE ==> Call Sub. PHASE for decays
* (otherwise don't and get no 4 vectors)'
* LPRDEC TRUE ==> Print decays defined in decay file
* LRESPR TRUE ==> Use resonance for initial VPHO energy
* LONGLF TRUE ==> Decay long-lived (ctau>1cm) particles in QQ,
* otherwise decayed in CLEOG
* LTRKSM TRUE ==> Do TRKSIM/GLBFIL and save ANL* banks
* LTSMP0 TRUE ==> after TRKSIM, call GTPIR
* LTSMVF TRUE ==> after TRKSIM, call VFINDR
* LTSMEI TRUE ==> after TRKSIM, call GETEID
* ADDPQQ TRUE ==> Add final state radiation via PHOTOS
* LPROMP TRUE ==> User can type in UDECAY file at run time.
      LOGICAL LTAPE, BMRAD, LWIDE, TRACE, LSHAPE, LSELCT
      LOGICAL LPHASE, LPRDEC, LRESPR, LONGLF, LTRKSM, ADDPQQ, LPROMP
      LOGICAL LTSMP0, LTSMVF, LTSMEI
      COMMON /QQCNT4/
     * LTAPE, BMRAD, LWIDE, TRACE, LSHAPE, LSELCT,
     * LPHASE, LPRDEC, LRESPR, LONGLF, LTRKSM, ADDPQQ, LPROMP,
     * LTSMP0, LTSMVF, LTSMEI
* IFL Used in quark or gluon jet processes
* IMODE Used in quark or gluon jet processes
* ANGMIN Min angle (degrees) for QED radiation
* ANGMAX Max angle (degrees) for QED radiation
* CEN(2) Mult. parameter for hadron W vertices
* FWID(2) Mult. parameter for hadron W vertices
* C1B(3) Mult. parameter
* C2B(3) Mult. parameter
* CQMAS(7) Quark masses
* RESPRO(2) Mass and width of initial production resonance
* Default is Y(4S)
      INTEGER IFL, IMODE
      DOUBLE PRECISIONCEN, FWID, C1B, C2B
      REAL ANGMIN, ANGMAX, CQMAS
      REAL RESPRO
      COMMON /QQCNT5/
     * CEN(2), FWID(2), C1B(3), C2B(3),
     * IFL, IMODE,
     * ANGMIN, ANGMAX, CQMAS(7), RESPRO(2)
      SAVE /QQCNT1/, /QQCNT3/, /QQCNT4/, /QQCNT5/
