/*******************************************************************************
*									       *
* phaseP.h -- Nirvana Phase Space Event Display	(private include file)         *
*									       *
* Copyright (c) 1991 Universities Research Association, Inc.		       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warrenty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
* Fermilab Nirvana GUI Library						       *
* August 10 1991							       *
*									       *
* Written by Mark Edel							       *
* February 1994, by Paul Lebrun						       *
*               Augmented to support StdHep Readout and Vertex display	       *
*									       *
*									       *
*******************************************************************************/
/* SCCS ID: phaseP.h 1.1 4/6/92 */

enum DisplayMode {BY_MOMENTUM, BY_PT, BY_RAPIDITY,
		  BY_PSEUDORAPIDITY, BY_USER_ROUTINE};
		  
enum StdHepWindowType {STDHEP_PHASE, STDHEP_SPACE, STDHEP_PARA};

enum HighlightTracksMode{TRACK_NODE, TRACK_DAUGHTERS, TRACK_DESCENDANTS,
			 TRACK_MOTHER, TRACK_ANCESTORS};
		  
#define NATIVE_STDHEP 0 /* Std Hep file, native binary format */
#define ZEBRA_STDHEP 1 /* Std Hep file, Zebra FZ, exchange format */
#define XDR_STDHEP 2 /* Std Hep file, xdr binary format */

#define CHARSET (XmStringCharSet)XmSTRING_DEFAULT_CHARSET

#define MKSTRING(string) \
	XmStringCreateLtoR(string, XmSTRING_DEFAULT_CHARSET)
	
#define SET_ONE_RSRC(widget, name, newValue) \
{ \
    static Arg args[1] = {{name, (XtArgVal)0}}; \
    args[0].value = (XtArgVal)newValue; \
    XtSetValues(widget, args, 1); \
}	

#define GET_ONE_RSRC(widget, name, valueAddr) \
{ \
    static Arg args[1] = {{name, (XtArgVal)0}}; \
    args[0].value = (XtArgVal)valueAddr; \
    XtGetValues(widget, args, 1); \
}

#define SQR(n) ((n)*(n))

typedef struct _PhaseWindow {
    struct _PhaseWindow *next;
    int type;                   /* type of windo : Phase space here */
    int modetreedisp;
    Widget colorWindowShell;    /* Informatio for ColorCode Display */
    Widget dtreecolorcode;
    Widget streecolorcode;
    Widget treecolorcode;
    int *treeheadcolorcode;
    PhaseEvent colorcode;
    int nEvents;
    int nMaxEvts;
    int eventNum;
    PhaseEvent event;
    Widget eventTreeButton;
    Widget colorCodeButton;
    Widget dtree;
    Widget stree;
    Widget tree;
    Widget nodeWindowShell;
    int *treehead;
    int (*openProc)();
    void (*closeProc)();
    PhaseEvent *(*getEventProc)();
    void (*exitProc)();
    FILE file;
    int filemode;
    char *filename;
    int zebLrec;
    int xdrStream;
    int selnodeNumTrack;
    int *selnodeTracks;
    Widget shell;
    Widget spin;
    Widget eventNumText;
    Widget eventSelectorLabel;
    Widget statsLabel;
    Widget scaleArea;
    Widget scaleLabel;
    Widget btnRotationPanel;
    Widget absRotationPanel;
    Widget absRotationLabel;
    Widget absRotFields[3];
    double buttonRotateDegrees;
    char stable, unstable;
    char charged, neutral;
    char electrons, muons, gammas, neutrinos, quarks, hadrons, wz;
    GC scaleGC;
    GC highlightGC;
    int selectedTrack;
    int *selectedNode;
    Widget trackWindowShell;
    Widget trackWindowLabel;
    Widget newPhaseButton;
    Widget newParaButton;
    int highlightTrackMode; /* End of StdHep Window */
    Widget vectorLengthPanel;
    int displayMode;
    double vectorLength;
} PhaseWindow;

typedef struct  _SpaceWindow {
    struct _SpaceWindow *next;
    int type;                   /* type of windo : Real space here */
    int modetreedisp;
    Widget colorWindowShell;    /* Informatio for ColorCode Display */
    Widget dtreecolorcode;
    Widget streecolorcode;
    Widget treecolorcode;
    int *treeheadcolorcode;
    PhaseEvent colorcode;
    int nEvents;
    int nMaxEvts;
    int eventNum;
    PhaseEvent event;
    Widget eventTreeButton;
    Widget colorCodeButton;
    Widget dtree;
    Widget stree;
    Widget tree;
    Widget nodeWindowShell;
    int *treehead;
    int (*openProc)();
    void (*closeProc)();
    PhaseEvent *(*getEventProc)();
    void (*exitProc)();
    FILE file;
    int filemode;
    char *filename;
    int zebLrec;
    int xdrStream;
    int selnodeNumTrack;
    int *selnodeTracks;
    Widget shell;
    Widget spin;
    Widget eventNumText;
    Widget eventSelectorLabel;
    Widget statsLabel;
    Widget scaleArea;
    Widget scaleLabel;
    Widget btnRotationPanel;
    Widget absRotationPanel;
    Widget absRotationLabel;
    Widget absRotFields[3];
    double buttonRotateDegrees;
    char stable, unstable;
    char charged, neutral;
    char electrons, muons, gammas, neutrinos, quarks, hadrons, wz;
    GC scaleGC;
    GC highlightGC;
    int selectedTrack;
    int *selectedNode;
    Widget trackWindowShell;
    Widget trackWindowLabel;
    Widget newPhaseButton;
    Widget newParaButton;
    int highlightTrackMode; /* End of StdHep Window */
    Widget scaleSliderScale;
    Widget xTranslSliderScale;
    Widget yTranslSliderScale;
    Widget zTranslSliderScale;
    Widget momToSpaceSliderScale;
    Widget longToTrSliderScale;
    Widget scaleSliderValue;
    Widget xTranslSliderValue;
    Widget yTranslSliderValue;
    Widget zTranslSliderValue;
    Widget momToSpaceSliderValue;
    Widget longToTrSliderValue;
    SpaceVertex *vertices; /* Specific to Vertices */
    float maxScale, currentScale;
    float minTransl[3], maxTransl[3], currentTransl[3];
    float maxMomToSpace, currentMomToSpace;
    float minLongToTr, maxLongToTr, currentLongToTr;
    float maxMomentum;
    int showVertices;
    int numRealVertices;
    SpaceVertex *realVertices;
    Widget detectorButton;
    int nDetectorSegments;
    SpinSegment *detectorSegments;
} SpaceWindow;

typedef struct  _ParaWindow {
    struct _ParaWindow *next;
    int type;                   /* type of windo : Para space here */
    int modetreedisp;
    Widget colorWindowShell;    /* Informatio for ColorCode Display */
    Widget dtreecolorcode;
    Widget streecolorcode;
    Widget treecolorcode;
    int *treeheadcolorcode;
    PhaseEvent colorcode;
    int nEvents;
    int nMaxEvts;
    int eventNum;
    PhaseEvent event;
    Widget eventTreeButton;
    Widget colorCodeButton;
    Widget dtree;
    Widget stree;
    Widget tree;
    Widget nodeWindowShell;
    int *treehead;
    int (*openProc)();
    void (*closeProc)();
    PhaseEvent *(*getEventProc)();
    void (*exitProc)();
    FILE file;
    int filemode;
    char *filename;
    int zebLrec;
    int xdrStream;
    int selnodeNumTrack;
    int *selnodeTracks;
    Widget shell;
    Widget spin;
    Widget eventNumText;
    Widget eventSelectorLabel;
    Widget statsLabel;
    Widget scaleArea;
    Widget scaleLabel;
    Widget btnRotationPanel;
    Widget absRotationPanel;
    Widget absRotationLabel;
    Widget absRotFields[3];
    double buttonRotateDegrees;
    char stable, unstable;
    char charged, neutral;
    char electrons, muons, gammas, neutrinos, quarks, hadrons, wz;
    GC scaleGC;
    GC highlightGC;
    int selectedTrack;
    int *selectedNode;
    Widget trackWindowShell;
    Widget trackWindowLabel;
    Widget newPhaseButton;
    Widget newParaButton;
    int highlightTrackMode; /* End of StdHep Window */
    Widget zTranslSliderScale;
    Widget rapToPtSliderScale;
    Widget zTranslSliderValue;
    Widget rapToPtSliderValue;
    float minTranslz, maxTranslz, currentTranslz;
    float minRapToPt, maxRapToPt, currentRapToPt;
    float maxMomentum;
} ParaWindow;
/* The common part of these two type of windows  */

typedef struct _StdHepWindow {
    struct _StdHepWindow *next;  /* The next window                     */
    int type;                   /* type of windo : Phase of Vertex space */
    int modetreedisp;
    Widget colorWindowShell;    /* Information for ColorCode Display */
    Widget dtreecolorcode;
    Widget streecolorcode;
    Widget treecolorcode;
    int *treeheadcolorcode;
    PhaseEvent colorcode;
    int nEvents;
    int nMaxEvts;
    int eventNum;
    PhaseEvent event;
    Widget eventTreeButton;
    Widget colorCodeButton;
    Widget dtree;
    Widget stree;
    Widget tree;
    Widget nodeWindowShell;
    int *treehead;
    int (*openProc)();
    void (*closeProc)();
    PhaseEvent *(*getEventProc)();
    void (*exitProc)();
    FILE file;
    int filemode;
    char *filename;
    int zebLrec;
    int xdrStream;
    int selnodeNumTrack;
    int *selnodeTracks;
    Widget shell;	/* Just the top shell */
    Widget spin;
    Widget eventNumText;
    Widget eventSelectorLabel;
    Widget statsLabel;
    Widget scaleArea;
    Widget scaleLabel;
    Widget btnRotationPanel;
    Widget absRotationPanel;
    Widget absRotationLabel;
    Widget absRotFields[3];
    double buttonRotateDegrees;
    char stable, unstable;
    char charged, neutral;
    char electrons, muons, gammas, neutrinos, quarks, hadrons, wz;
    GC scaleGC;
    GC highlightGC;
    int selectedTrack;
    int *selectedNode;
    Widget trackWindowShell;
    Widget trackWindowLabel;
    Widget newPhaseButton;
    Widget newParaButton;
    int highlightTrackMode;
} StdHepWindow;

extern StdHepWindow *WindowList;
extern Display *TheDisplay;
