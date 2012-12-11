/*******************************************************************************
*									       *
* panel.c -- Nirvana Phase Space Event Display				       *
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
* August 15, 1991							       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)panel.c	1.2	4/10/92";
#include <stdio.h>
#include <stdarg.h>
#include <sys/param.h>
#include <limits.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <X11/Xatom.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/Frame.h>
#include <Xm/ArrowB.h>
#include <Xm/Text.h>
#include <Xm/PushBG.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>
#include <Xm/DrawingA.h>
#include "spin/Spin.h"
#include "util/DialogF.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "panel.h"
#include "menu.h"
#include "pick.h"
#include "rotation.h"

#define INITIAL_VECTOR_LENGTH .40	/* % of vector shown towards center */
#define INITIAL_DISPLAY_MODE BY_MOMENTUM/* Display mode at window creation */
#define INITIAL_ROT_INCR 10.		/* Key and button rotation (degrees) */

static Widget createContents(Widget parent, PhaseWindow *window, int hasEventBar,
			      char *eventBarText);
static void closeCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void momentumCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void ptCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void rapidityCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void pseudorapCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void stableCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void unstableCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void chargedCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void neutralCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void electronsCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void muonsCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void neutrinosCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void gammasCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void quarksCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void hadronsCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void wzCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void userRoutineCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void viewRotXPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void viewRotYPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void viewRotZPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void viewRotXNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void viewRotYNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void viewRotZNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotXPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotYPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotZPosCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotXNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotYNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void coordRotZNegCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void scaleUpCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void scaleDownCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void resetRotationCB(Widget w, PhaseWindow *window, caddr_t call_data); 
static void scaleExposeCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void resizeCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void redisplayCB(Widget w, PhaseWindow *window, caddr_t call_data);
static void spinSelectCB(Widget w, PhaseWindow *window, SpinCallbackStruct *cb);
static void nextEvtCB(Widget w, PhaseWindow *window,caddr_t call_data);
static void previousEvtCB(Widget w, PhaseWindow *window,caddr_t call_data);

/*
** Create a new event display window
*/
PhaseWindow *CreatePanel(Display *display, int setsEventNum,
    int canOpenFiles, int canExit, char *windowTitle, char *eventSelectorText)
{
    Widget appShell, main, menuBar, form;
    PhaseWindow *window;
    static Atom wmpAtom, dwAtom = NULL;
    Arg al[20];
    int ac;
    XGCValues values;

    /* Allocate some memory for the new window data structure */
    window = (PhaseWindow *)XtMalloc(sizeof(PhaseWindow));
    
    /* Create an toplevel shell to hold the window */
    ac = 0;
    XtSetArg(al[ac], XmNtitle, windowTitle); ac++;
    XtSetArg(al[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    XtSetArg(al[ac], XmNiconName, windowTitle); ac++;
    appShell = XtAppCreateShell ("phase", "Phase",
		applicationShellWidgetClass, display, al, ac);
    	    
    /*
    ** create a main window holding a menu bar and a form with the rest of
    ** the window contents.
    */
    main = XmCreateMainWindow(appShell, "main", NULL, 0);
    XtManageChild(main);
    menuBar = CreateMenuBar(main, (StdHepWindow *) window,
                            canOpenFiles, canExit, STDHEP_PHASE);
    XtManageChild(menuBar);
    form = createContents(main, window, setsEventNum,
			  eventSelectorText);
    XtManageChild(form);

    /* add the window to the global window list */
    AddToWindowList((StdHepWindow *) window);
    
    /* realize all of the widgets in the new window */
    XtRealizeWidget(appShell);

    /* set up closeCB to be called when the user selects close from the
       window menu.  The close menu item usually activates f.kill which
       sends a WM_DELETE_WINDOW protocol request for the window. */
    if (dwAtom == NULL) {
    	wmpAtom = XmInternAtom(display, "WM_PROTOCOLS", TRUE);
    	dwAtom = XmInternAtom(display, "WM_DELETE_WINDOW", TRUE);
    }
    XmAddProtocolCallback(appShell, wmpAtom, dwAtom, closeCB, window);

    /* initialize window structure, including a read-only graphics contexts
       for drawing in scale area and highlighting in spin widget */
    window->shell = appShell;
    window->selectedTrack = NO_TRACK;
    window->trackWindowShell = NULL;
    window->vectorLengthPanel = NULL;
    window->btnRotationPanel = NULL;
    window->absRotationPanel = NULL;
    window->vectorLength = INITIAL_VECTOR_LENGTH;
    window->displayMode = INITIAL_DISPLAY_MODE;
    window->buttonRotateDegrees = INITIAL_ROT_INCR;
    window->event.nParticles = 0;
    window->event.particles = NULL;
    window->colorcode.nParticles = 0;
    window->colorcode.particles = NULL;
    ac = 0;
    XtSetArg(al[ac], XmNbackground, &values.background); ac++;
    XtSetArg(al[ac], XmNforeground, &values.foreground); ac++;
    XtGetValues(window->scaleArea, al, ac);
    window->scaleGC =
    	XtGetGC(window->scaleArea, GCForeground|GCBackground, &values);
    window->highlightGC = SpinCopyGC(window->spin);
    XSetLineAttributes(display, window->highlightGC, 5, LineSolid,
    		       CapRound, JoinMiter);
    return window;
}

/*
** Add a window to the the window list.
*/
void AddToWindowList(StdHepWindow *window) 
{
    StdHepWindow *temp;

    temp = WindowList;
    WindowList = window;
    window->next = temp;
}

/*
** Remove a window from the list of windows
*/
void RemoveFromWindowList(StdHepWindow *window)
{
    StdHepWindow *temp;

    if (WindowList == window)
	WindowList = window->next;
    else {
	for (temp = WindowList; temp != NULL; temp = temp->next) {
	    if (temp->next == window) {
		temp->next = window->next;
		break;
	    }
	}
    }
}

int NWindows()
{
    StdHepWindow *win;
    int n;
    
    for (win=WindowList, n=0; win!=NULL; win=win->next, n++);
    return n;
}

static void closeCB (Widget w, PhaseWindow *window, caddr_t call_data) 
{
    CloseWindow((StdHepWindow *) window);
}

static Widget createContents(Widget parent, PhaseWindow *window,
                              int hasEventBar, char *eventBarText)
{
    Arg    	args[50];
    int    	ac;
    XmString	s1, s2;
    char ctmp[10];
    Widget	form, displayMode, stability, particleType, rotation, stats;
    Widget	controls, trackInfoFrame, trackInfo;
    Widget	scaleFrame, scaleRC, scaleLabel, scaleArea;
    Widget	dispModeLabel, particleTypeLabel;
    Widget	eventCtrlFrame, eventCtrlForm, eventNumLabel, upArrow;
    Widget	eventNumText, downArrow, eventCtrlText;
    Widget	widget, spin;
    Pixel	foreground, background;
    Pixmap	pm;

    /* Create the form onto which everything goes */
    ac = 0;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    form = XmCreateForm(parent, "form", args, ac);
    XtManageChild(form);
  
    if (hasEventBar) {
        ac = 0;
	XtSetArg(args[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
	eventCtrlFrame = XmCreateFrame(form, "eventCtrlFrame", args, ac);
	XtManageChild(eventCtrlFrame);
        
        ac = 0;
	eventCtrlForm = XmCreateForm(eventCtrlFrame, "eventCtrlForm", args, ac);
	XtManageChild(eventCtrlForm);
	
        ac = 0;
	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Event #"))); ac++;
	eventNumLabel = (Widget) XmCreateLabelGadget(eventCtrlForm, "eventNumLabel",
					    args, ac);
	XmStringFree(s1);
	XtManageChild(eventNumLabel);
	
	ac = 0;
	XtSetArg(args[ac], XmNarrowDirection, XmARROW_UP); ac++;
	XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNleftWidget, eventNumLabel); ac++;
	upArrow = XmCreateArrowButton(eventCtrlForm, "upArrow", args, ac);
        XtAddCallback(upArrow, XmNactivateCallback, 
        		(XtCallbackProc) nextEvtCB, window); 
	XtManageChild(upArrow);
	
	ac = 0;
	
	sprintf(ctmp,"  %d  ", ac);
	XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING(ctmp))); ac++;
	XtSetArg(args[ac], XmNcolumns, 7); ac++;
	XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
	XtSetArg(args[ac], XmNleftWidget, upArrow); ac++;
	eventNumText = (Widget) XmCreateLabelGadget(
	                           eventCtrlForm, "eventNumText",
					    args, ac);
	XtManageChild(eventNumText);
	
	ac = 0;
	XtSetArg(args[ac], XmNarrowDirection, XmARROW_DOWN); ac++;
	XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNleftWidget, eventNumText); ac++;
	downArrow = XmCreateArrowButton(eventCtrlForm, "downArrow", args, ac);
        XtAddCallback(downArrow, XmNactivateCallback,
                        (XtCallbackProc) previousEvtCB, window); 
	XtManageChild(downArrow);
	
	ac = 0;
	XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING(eventBarText))); ac++;
	XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
	eventCtrlText = (Widget) XmCreateLabelGadget(
	                           eventCtrlForm, "eventCtrlText",
					    args, ac);
	XmStringFree(s1);
	XtManageChild(eventCtrlText);
    }
    
    /* Create a Form for the controls along the left edge of window */
    ac = 0;
    if (hasEventBar) {
    	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    	XtSetArg(args[ac], XmNtopWidget, eventCtrlFrame); ac++;
    } else {
    	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    }
    controls = XmCreateForm(form, "controls", args, ac);
    XtManageChild(controls);

    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Display Mode"))); ac++;
    dispModeLabel = (Widget) XmCreateLabelGadget(controls,
                             "dispModeLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(dispModeLabel);

    ac = 0;
    XtSetArg(args[ac], XmNradioBehavior, True); ac++;
    XtSetArg(args[ac], XmNradioAlwaysOne, True); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, dispModeLabel); ac++;
    displayMode = XmCreateRowColumn(controls, "displayMode", args, ac);
    XtManageChild(displayMode);
 
    ac = 0;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, displayMode); ac++;
    stability = XmCreateRowColumn(controls, "stability", args, ac);
    XtManageChild(stability);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Particle Types"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, stability); ac++;
    particleTypeLabel = (Widget) XmCreateLabelGadget(controls, 
                                        "particleTypeLabel",
    					    args, ac);
    XmStringFree(s1);
    XtManageChild(particleTypeLabel);

    ac = 0;
    XtSetArg(args[ac], XmNborderWidth, (Dimension)0); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, particleTypeLabel); ac++;
    particleType = XmCreateRowColumn(controls, "particleType", args, ac);
    XtManageChild(particleType);
 
    ac = 0;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNadjustLast, False); ac++;    
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNnumColumns, (short)3); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, particleType); ac++;
    rotation = XmCreateRowColumn(controls, "rotation", args, ac);
    XtManageChild(rotation);

    ac = 0;
    XtSetArg(args[ac], XmNhighlightThickness, 0); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNorientation, XmHORIZONTAL); ac++;
    if (hasEventBar) {
    	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    	XtSetArg(args[ac], XmNtopWidget, eventCtrlFrame); ac++;
    } else {
    	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
    }
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNleftWidget, controls); ac++;
    XtSetArg(args[ac], XmNtopOffset, (int)2); ac++;
    XtSetArg(args[ac], XmNleftOffset, (int)0); ac++;
    stats = XmCreateRowColumn(form, "stats", args, ac);
    XtManageChild(stats);

#define TOGGLE_BTN(parent, label, name, cb, set) \
    ac = 0; \
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING(label))); ac++; \
    XtSetArg(args[ac], XmNset, set); ac++; \
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++; \
    widget = XmCreateToggleButtonGadget(parent, name, args, ac); \
    XmStringFree(s1); \
    XtAddCallback(widget, XmNvalueChangedCallback, \
    (XtCallbackProc) cb, window); \
    XtManageChild(widget); \

    TOGGLE_BTN(displayMode, "Momentum", "momentum", momentumCB, True);
    TOGGLE_BTN(displayMode, "Pt", "pt", ptCB, False);
    TOGGLE_BTN(displayMode, "Rapidity", "rapidity", rapidityCB, False);
    TOGGLE_BTN(displayMode, "PseodoRapidity", "pseudorap", pseudorapCB, False);
    TOGGLE_BTN(displayMode, "User Routine", "userRoutine", userRoutineCB, False);
/*     if (!hasUserRoutine) Always set to insensitive in this version.. */
     SET_ONE_RSRC(widget, XmNsensitive, False);
    TOGGLE_BTN(stability, "Stable", "stable", stableCB, True);
    TOGGLE_BTN(stability, "Unstable", "unstable", unstableCB, True);
    TOGGLE_BTN(particleType, "Charged", "charged", chargedCB, True);
    TOGGLE_BTN(particleType, "Neutral", "neutral", neutralCB, True);
    TOGGLE_BTN(particleType, "Electrons", "electrons", electronsCB, True);
    TOGGLE_BTN(particleType, "Muons", "muons", muonsCB, True);
    TOGGLE_BTN(particleType, "Neutrinos", "neutrinos", neutrinosCB, True);
    TOGGLE_BTN(particleType, "Gammas", "gammas", gammasCB, True);
    TOGGLE_BTN(particleType, "Quarks/Gluons", "quarks", quarksCB, True);
    TOGGLE_BTN(particleType,"Hadrons", "hadrons", hadronsCB, True);
    TOGGLE_BTN(particleType,"W/Z", "wz", wzCB, True);

    /* Get foreground an background colors for pixmap buttons */
    ac = 0;
    XtSetArg(args[ac], XmNforeground, &foreground); ac++;
    XtSetArg(args[ac], XmNbackground, &background); ac++;
    XtGetValues(controls, args, ac);

#define ROTATE_BTN(pmName, widgetName, cb) \
    pm = XmGetPixmap(XtScreen(parent), pmName, foreground, background); \
    ac = 0; \
    XtSetArg(args[ac], XmNlabelType, XmPIXMAP); ac++; \
    XtSetArg(args[ac], XmNlabelPixmap, pm); ac++; \
    widget = XmCreatePushButtonGadget(rotation, widgetName, args, ac); \
    XtAddCallback(widget, XmNactivateCallback, \
                  (XtCallbackProc) cb, window); \
    XtManageChild(widget);
 
    ROTATE_BTN("viewRotXPos", "viewRotXPos", viewRotXPosCB) 
    ROTATE_BTN("viewRotXNeg", "viewRotXNeg", viewRotXNegCB)
    ROTATE_BTN("coordRotXPos", "coordRotXPos", coordRotXPosCB)
    ROTATE_BTN("coordRotXNeg", "coordRotXNeg", coordRotXNegCB)
    ROTATE_BTN("scaleUp", "scaleUp", scaleUpCB)
    ROTATE_BTN("viewRotYPos", "viewRotYPos", viewRotYPosCB)
    ROTATE_BTN("viewRotYNeg", "viewRotYNeg", viewRotYNegCB)
    ROTATE_BTN("coordRotYPos", "coordRotYPos", coordRotYPosCB)
    ROTATE_BTN("coordRotYNeg", "coordRotYNeg", coordRotYNegCB)
    ROTATE_BTN("scaleDown", "scaleDown", scaleDownCB)
    ROTATE_BTN("viewRotZPos", "viewRotZPos", viewRotZPosCB)
    ROTATE_BTN("viewRotZNeg", "viewRotZNeg", viewRotZNegCB)
    ROTATE_BTN("coordRotZPos", "coordRotZPos", coordRotZPosCB)
    ROTATE_BTN("coordRotZNeg", "coordRotZNeg", coordRotZNegCB)
    ROTATE_BTN("resetRotation", "resetRotation", resetRotationCB)

    ac = 0;
    XtSetArg(args[ac], XmNshadowType, XmSHADOW_IN); ac++;
    XtSetArg(args[ac], XmNshadowThickness, 1); ac++;
    scaleFrame = XmCreateFrame(stats, "scaleFrame", args, ac);
    XtManageChild(scaleFrame);
    trackInfoFrame = XmCreateFrame(stats, "trackInfoFrame", args, ac);
    XtManageChild(trackInfoFrame);
    
    ac = 0;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNentryAlignment, XmALIGNMENT_CENTER); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    scaleRC = XmCreateRowColumn(scaleFrame, "scaleRC", args, ac);
    XtManageChild(scaleRC);

    ac = 0;
    scaleArea = XmCreateDrawingArea(scaleRC, "scaleArea", args, ac);
    XtAddCallback(scaleArea, XmNexposeCallback,
                  (XtCallbackProc) scaleExposeCB, window);
    XtManageChild(scaleArea);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("x"))); ac++;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    scaleLabel = (Widget) XmCreateLabelGadget(scaleRC, "scaleLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(scaleLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNalignment, XmALIGNMENT_BEGINNING); ac++;
    XtSetArg(args[ac], XmNrecomputeSize, False); ac++;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING(
     "00000 tracks, 00000 are stable\nx\nTrack of max. xxxxx is a(n) xxxx"))); ac++;
     trackInfo = (Widget) XmCreateLabelGadget(trackInfoFrame,
                                              "trackInfo", args, ac);
    XmStringFree(s1);
    XtManageChild(trackInfo);
    
    ac = 0;
    XtSetArg(args[ac], XmNshowAxes, True); ac++;
    XtSetArg(args[ac], XmNperspectiveOn, True); ac++;
    XtSetArg(args[ac], XmNdoubleBuffer, False); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_FORM); ac++;
    XtSetArg(args[ac], XmNleftWidget, controls); ac++;
    XtSetArg(args[ac], XmNtopWidget, stats); ac++;
    XtSetArg(args[ac], XmNtopOffset, (int)0); ac++;
    XtSetArg(args[ac], XmNbottomOffset, (int)0); ac++;
    XtSetArg(args[ac], XmNleftOffset, (int)0); ac++;
    XtSetArg(args[ac], XmNrightOffset, (int)0); ac++;
    spin = XtCreateManagedWidget("spin", spinWidgetClass, form, args, ac);
    XtAddCallback(spin, XmNresizeCallback,
                   (XtCallbackProc) resizeCB, window);
    XtAddCallback(spin, XmNredisplayCallback,
                     (XtCallbackProc)  redisplayCB, window);
    XtAddCallback(spin, XmNbtn2Callback, 
                       (XtCallbackProc) spinSelectCB, window);

    window->spin = spin;
    window->eventNumText = eventNumText;
    window->eventSelectorLabel = eventCtrlText;
    window->statsLabel = trackInfo;
    window->scaleArea = scaleArea;
    window->scaleLabel = scaleLabel;
    
    window->stable = True;
    window->unstable = True;
    window->charged = True;
    window->neutral = True;
    window->electrons = True;
    window->muons = True;
    window->gammas = True;
    window->neutrinos = True;
    window->quarks = True;
    window->hadrons = True;
    window->wz = True;
    
    return(form);
}

static void momentumCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    if (XmToggleButtonGetState(w))
    	SetDisplayMode(window, BY_MOMENTUM);
}

static void ptCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    if (XmToggleButtonGetState(w))
    	SetDisplayMode(window, BY_PT);
}

static void rapidityCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    if (XmToggleButtonGetState(w))
    	SetDisplayMode(window, BY_RAPIDITY);
}

static void pseudorapCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    if (XmToggleButtonGetState(w))
    	SetDisplayMode(window, BY_PSEUDORAPIDITY);
}

static void userRoutineCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    if (XmToggleButtonGetState(w))
    	SetDisplayMode(window, BY_USER_ROUTINE);
}

static void stableCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->stable = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void unstableCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->unstable = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void chargedCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->charged = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void neutralCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->neutral = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void electronsCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->electrons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void muonsCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->muons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void gammasCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->gammas = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void neutrinosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->neutrinos = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void quarksCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->quarks = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void hadronsCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->hadrons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void wzCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    window->wz = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void viewRotXPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateX(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotYPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateY(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotZPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateZ(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotXNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateX(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void viewRotYNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateY(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void viewRotZNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateZ(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotXPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateX(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotYPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateY(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotZPosCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateZ(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotXNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateX(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotYNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateY(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotZNegCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateZ(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void scaleUpCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinSetScale(window->spin, SpinGetScale(window->spin) * (5./4.));
    DrawScale((StdHepWindow *) window);
}
 
static void scaleDownCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinSetScale(window->spin, SpinGetScale(window->spin) * (4./5.));
    DrawScale((StdHepWindow *) window);
}
 
static void resetRotationCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinRestore(window->spin);
}
 
static void scaleExposeCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    DrawScale((StdHepWindow *) window);
}
 
static void resizeCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    DrawScale((StdHepWindow *) window);
}
 
static void redisplayCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    ShowSelectedTrack((StdHepWindow *) window);
    UpdateRotationPanel((StdHepWindow *) window);
}
 
static void spinSelectCB(Widget w, PhaseWindow *window, SpinCallbackStruct *cb)
{
    int particleIndex;
    
    particleIndex = FindTrack((StdHepWindow *) window,
                    cb->event->xbutton.x,cb->event->xbutton.y);
    SelectTrack((StdHepWindow *) window, particleIndex);
}
 
static void nextEvtCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    int ievt = window->event.eventNum;
    ievt++;
    if (ievt > window->nEvents) {
        DialogF(DF_INF, w, 1,
         "You reached the end of this file",
                        "OK");
        return;
     }
    GetSetStdHep((StdHepWindow *) window, ievt);
}
 
static void previousEvtCB(Widget w, PhaseWindow *window, caddr_t call_data)
{
    int ievt = window->event.eventNum;
    ievt--;
    if (ievt <= 0)  {
        DialogF(DF_INF, w, 1,
         "You went back to the beginning this file",
                        "OK");
        return;
     }
    GetSetStdHep((StdHepWindow *) window, ievt);
} 

