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
* August 10, 1991							       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
/* SCCS ID: phaseP.h 1.1 4/6/92 */

		  
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

typedef struct  SpaceWindow {
    struct _SpaceWindow *next;
    Widget shell;
    Widget spin;
    Widget eventNumText;
    Widget eventSelectorLabel;
    Widget statsLabel;
    Widget scaleArea;
    Widget scaleLabel;
    Widget trackWindowShell;
    Widget trackWindowLabel;
    Widget vectorLengthPanel;
    Widget momtospaceSlider;
    Widget xTranslSlider;
    Widget yTransSlider;
    Widget zTranslider;
    Widget momToSpaceSlider;
    Widget longToTrSlider;
    Widget btnRotationPanel;
    Widget absRotationPanel;
    Widget absRotationLabel;
    Widget absRotFields[3];
    Widget eventTreeButton;
    Widget colorCodeButton;
    Widget dtree;
    Widget stree;
    Widget tree;
    Widget nodeWindowShell;
    Widget colorWindowShell;
    Widget dtreecolorcode;
    Widget streecolorcode;
    Widget treecolorcode;
    int *treehead;  /* too complicate to load the nodehep struct here  */
    int *treeheadcolorcode;
    int modetreedisp;
    GC scaleGC;
    GC highlightGC;
    int nEvents;
    int nMaxEvts;
    int eventNum;
    int selectedTrack;
    int *selectedNode;
    int selnodeNumTrack;
    int *selnodeTracks;
    double vectorLength;
    double buttonRotateDegrees;
    PhaseEvent event;
    PhaseEvent colorcode;
    int (*openProc)();
    void (*closeProc)();
    PhaseEvent *(*getEventProc)();
    void (*exitProc)();
    FILE file;
    int filemode;
    char *filename;
    int zebLrec;
    int xdrStream;
    char stable, unstable;
    char electrons, muons, gammas, neutrinos, quarks, hadrons, wz;
} PhaseWindow;

extern PhaseWindow *WindowList;
extern Display *TheDisplay;
