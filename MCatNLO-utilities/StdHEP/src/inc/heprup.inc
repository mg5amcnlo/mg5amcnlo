C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)

C  
C	Les Houches accord 2001
C
C	MAXPUP - max. number of different processes to be interfaced at one time
C	IDBMUP - ID of beam particles 1 and 2 according to the PDG convention
C	EBMUP - energy in GeV of beam particles 1 and 2
C	PDFGUP - author group for beam particles 1 and 2 according to PDFlib specifications
C	PDFSUP - PDF set ID for beam particles 1 and 2 according to PDFlib specifications
C
C	For e+e- or when the SHG defaults are to be used, 
C 	set PDFGUP=-1 and PDFSUP=-1
C
C	IDWTUP - master switch dictating how the event weights are interpreted
C	NPRUP - number of different user subprocesses
C	XSECUP - cross section for process in pb
C	XERRUP - statistical error associated with XSECUP
C	XMAXUP - maximum XWGTUP (in common block HEPEUP) for this process
C	LPRUP - user process ID's for this run
C
C		event		control of
C	IDWTUP	selection	mixing or	XWGTUP		output
C		criteria	unweighting	input
C	+1	XMAXUP		SHG		+weighted	+1
C	-1	XMAXUP		SHG		+/-weighted	+/-1
C	+2	XSECUP		SHG		+weighted	+1
C	-2	XSECUP		SHG		+/-weighted	+/-1
C	+3			user interface	+1		+1
C	-3			user interface	+/-1		+/-1
C	+4			user interface	+weighted	+weighted
C	14			user interface	+/-weighted	+/-weighted
C
