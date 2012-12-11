C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)

C  
C	Les Houches accord 2001
C
C	NUP - number of particle entries in this event
C	IDPRUP - ID of the process for this event 
C		(ID's are generator-specific)
C	XWGTUP - event weight
C	SCALUP - scale of the event in GeV, as used to calculate PDFs
C	AQEDUP - QED coupling used for this event
C	AQCDUP - QCD coupling used for this event
C	IDUP - particle ID according to PDG convention
C	ISTUP - status code:
C		-1 incoming particle
C		+1 outgoing final state particle
C		-2 intermediate space like propagator
C		+2 intermediate resonance, mass should be preserved
C		+3 intermediate resonance for documentation only
C		-9 incoming beam particles (generally not needed)
C	MOTHUP - index of first and last mother
C	ICOLUP - tag for color flow lines
C	PUP - lab frame 4 momentum and mass in GeV
C	VTIMUP - invariant lifetime
C	SPINUP - cos of angle between spin-vector of particle and 3 momentum of decaying particle
C
