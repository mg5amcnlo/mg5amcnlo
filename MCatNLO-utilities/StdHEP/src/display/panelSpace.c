/*******************************************************************************
*									       *
* panelSpace.c -- Nirvana Space-Vertices Event Display		               *
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
* February 1994							               *
*									       *
* Clone from panel.c ( by M. Edel), by P. Lebrun.	 	  	       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)panel.c	1.2	4/10/92";
#include <stdio.h>
#include <math.h>
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
#include <Xm/Scale.h>
#include <Xm/Label.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>
#include <Xm/DrawingA.h>
#include "spin/Spin.h"
#include "util/DialogF.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "panelSpace.h"
#include "menu.h"
#include "pick.h"
#include "rotation.h"

#define INITIAL_MOMTOSP_RAT 0.1         /* The initial scale factor for the 
						conversion momentum to space */
#define INITIAL_LONGTOTR_RAT 10.       /* The initial longitudinal to transverse
						ratio */
#define INITIAL_ROT_INCR 10.		/* Key and button rotation (degrees) */

static Widget createContents(Widget parent, SpaceWindow *window,
                             int hasEventBar, char *eventBarText);
static void closeCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void stableCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void unstableCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void chargedCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void neutralCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void electronsCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void muonsCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void neutrinosCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void gammasCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void quarksCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void hadronsCB(Widget w, SpaceWindow *window, caddr_t call_data);
static Widget createSpaceSlider(Widget *scaleret, Widget *valueret, 
			 Widget parent, Widget attached,
			 char *namef, char *namel,
                         int initval, SpaceWindow *window, 
                          void (*gensliderCB) ());
static void wzCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void scaleCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void xTransCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void yTransCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void zTransCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void momToSpaceCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void longToTrCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void viewRotXPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void viewRotYPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void viewRotZPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void viewRotXNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void viewRotYNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void viewRotZNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotXPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotYPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotZPosCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotXNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotYNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void coordRotZNegCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void scaleUpCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void scaleDownCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void resetRotationCB(Widget w, SpaceWindow *window, caddr_t call_data); 
static void scaleExposeCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void resizeCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void redisplayCB(Widget w, SpaceWindow *window, caddr_t call_data);
static void spinSelectCB(Widget w, SpaceWindow *window, SpinCallbackStruct *cb);
static void nextEvtCB(Widget w, SpaceWindow *window,caddr_t call_data);
static void previousEvtCB(Widget w, SpaceWindow *window,caddr_t call_data);

/*
** Create a new event display window
*/
SpaceWindow *CreatePanelSp(Display *display, int setsEventNum,
    int canOpenFiles, int canExit, char *windowTitle, char *eventSelectorText)
{
    Widget appShell, main, menuBar, form;
    SpaceWindow *window;
    static Atom wmpAtom, dwAtom = NULL;
    Arg al[20];
    int type, ac;
    XGCValues values;

    /* Allocate some memory for the new window data structure */
    window = (SpaceWindow *)XtMalloc(sizeof(SpaceWindow));
    
    /* Create an toplevel shell to hold the window */
    ac = 0;
    XtSetArg(al[ac], XmNtitle, windowTitle); ac++;
    XtSetArg(al[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    XtSetArg(al[ac], XmNiconName, windowTitle); ac++;
    /* Keep the phase name here, to same valid XDefaults.. */
    appShell = XtAppCreateShell ("phase", "Space",
		applicationShellWidgetClass, display, al, ac);
    	    
    /*
    ** create a main window holding a menu bar and a form with the rest of
    ** the window contents.
    */
    main = XmCreateMainWindow(appShell, "main", NULL, 0);
    XtManageChild(main);
    type = STDHEP_SPACE;
    menuBar = CreateMenuBar(main, (StdHepWindow *) window,
                                   canOpenFiles, canExit, type);
    XtManageChild(menuBar);
    form = createContents(main, window, setsEventNum, eventSelectorText);
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
    window->btnRotationPanel = NULL;
    window->absRotationPanel = NULL;
    window->buttonRotateDegrees = INITIAL_ROT_INCR;
    window->event.nParticles = 0;
    window->event.particles = NULL;
    window->colorcode.nParticles = 0;
    window->colorcode.particles = NULL;
    window->nDetectorSegments = 0;
    window->detectorSegments = NULL;
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

static void closeCB (Widget w, SpaceWindow *window, caddr_t call_data) 
{
    CloseWindow((StdHepWindow *) window);
}

static Widget createContents(Widget parent, SpaceWindow *window,
			     int hasEventBar, char *eventBarText)
{
    Arg    	args[50];
    int    	ac;
    XmString	s1, s2;
    char ctmp[10];
    Widget	form, displayMode, stability, particleType, rotation, stats;
    Widget	formt1, allSliders;
    Widget	controls, trackInfoFrame, trackInfo;
    Widget	scaleFrame, scaleRC, scaleLabel, scaleArea;
    Widget	stabilityLabel, particleTypeLabel;
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
	eventNumLabel = (Widget) XmCreateLabelGadget(
	                         eventCtrlForm, "eventNumLabel",
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
                      (XtCallbackProc)  nextEvtCB, window); 
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
/*	eventNumText = XmCreateText(eventCtrlForm, "eventNumText", args, ac); */
	XtManageChild(eventNumText);
	
	ac = 0;
	XtSetArg(args[ac], XmNarrowDirection, XmARROW_DOWN); ac++;
	XtSetArg(args[ac], XmNleftAttachment, XmATTACH_WIDGET); ac++;
	XtSetArg(args[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNbottomAttachment, XmATTACH_FORM); ac++;
	XtSetArg(args[ac], XmNleftWidget, eventNumText); ac++;
	downArrow = XmCreateArrowButton(eventCtrlForm, "downArrow", args, ac);
        XtAddCallback(downArrow, XmNactivateCallback,
                      (XtCallbackProc)  previousEvtCB, window); 
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
    XtSetArg(args[ac], XmNborderWidth, (Dimension)0); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNnumColumns, (short)2); ac++;
    stability = XmCreateRowColumn(controls, "stability", args, ac);
    XtManageChild(stability);
 
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("Particle Types"))); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, stability); ac++;
    particleTypeLabel = (Widget) XmCreateLabelGadget(
                                  controls, "particleTypeLabel",
    					    args, ac);
    XmStringFree(s1);
    XtManageChild(particleTypeLabel);

    ac = 0;
    XtSetArg(args[ac], XmNborderWidth, (Dimension)0); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNnumColumns, (short)2); ac++;
    XtSetArg(args[ac], XmNtopWidget, particleTypeLabel); ac++;
    particleType = XmCreateRowColumn(controls, "particleType", args, ac);
    XtManageChild(particleType);

 
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
                  (XtCallbackProc)  cb, window); \
    XtManageChild(widget); \

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
    
    /* Do now the slider.. X, Y ,Z translation first. */
    ac = 0;
    XtSetArg(args[ac], XmNborderWidth, (Dimension)0); ac++;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
/*    XtSetArg(args[ac], XmNrightAttachment, XmATTACH_WIDGET); ac++; */
    XtSetArg(args[ac], XmNnumColumns, (short)1); ac++;
    XtSetArg(args[ac], XmNtopWidget, particleType); ac++;
/*    XtSetArg(args[ac], XmNrightWidget, particleType); ac++; */
    allSliders = XmCreateRowColumn(controls, "allSliders", args, ac);
    XtManageChild(allSliders);
     
                                 
    ac = 0;
    XtSetArg(args[ac], XmNpacking, XmPACK_COLUMN); ac++;
    XtSetArg(args[ac], XmNadjustLast, False); ac++;    
    XtSetArg(args[ac], XmNspacing, 0); ac++;
    XtSetArg(args[ac], XmNnumColumns, (short)5); ac++;
    XtSetArg(args[ac], XmNtopAttachment, XmATTACH_WIDGET); ac++;
    XtSetArg(args[ac], XmNtopWidget, allSliders); ac++;
    rotation = XmCreateRowColumn(controls, "rotation", args, ac);
    XtManageChild(rotation);

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
                   (XtCallbackProc)  cb, window); \
    XtManageChild(widget);
    
    ROTATE_BTN("viewRotXPos", "viewRotXPos", viewRotXPosCB) 
    ROTATE_BTN("viewRotYPos", "viewRotYPos", viewRotYPosCB)
    ROTATE_BTN("viewRotZPos", "viewRotZPos", viewRotZPosCB)
    ROTATE_BTN("viewRotXNeg", "viewRotXNeg", viewRotXNegCB)
    ROTATE_BTN("viewRotYNeg", "viewRotYNeg", viewRotYNegCB)
    ROTATE_BTN("viewRotZNeg", "viewRotZNeg", viewRotZNegCB)
    ROTATE_BTN("coordRotXPos", "coordRotXPos", coordRotXPosCB)
    ROTATE_BTN("coordRotYPos", "coordRotYPos", coordRotYPosCB)
    ROTATE_BTN("coordRotZPos", "coordRotZPos", coordRotZPosCB)
    ROTATE_BTN("coordRotXNeg", "coordRotXNeg", coordRotXNegCB)
    ROTATE_BTN("coordRotYNeg", "coordRotYNeg", coordRotYNegCB)
    ROTATE_BTN("coordRotZNeg", "coordRotZNeg", coordRotZNegCB)
    ROTATE_BTN("scaleUp", "scaleUp", scaleUpCB)
    ROTATE_BTN("scaleDown", "scaleDown", scaleDownCB)
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
                    (XtCallbackProc)  scaleExposeCB, window);
    XtManageChild(scaleArea);
    
    ac = 0;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING("x"))); ac++;
    XtSetArg(args[ac], XmNmarginHeight, 0); ac++;
    scaleLabel = (Widget) XmCreateLabelGadget(
                            scaleRC, "scaleLabel", args, ac);
    XmStringFree(s1);
    XtManageChild(scaleLabel);
    
    ac = 0;
    XtSetArg(args[ac], XmNalignment, XmALIGNMENT_BEGINNING); ac++;
    XtSetArg(args[ac], XmNrecomputeSize, False); ac++;
    XtSetArg(args[ac], XmNlabelString, (s1=MKSTRING(
     "00000 tracks, 00000 are stable\nx\nTrack of max. xxxxx is a(n) xxxx"))); ac++;
    trackInfo = (Widget) XmCreateLabelGadget(
                           trackInfoFrame, "trackInfo", args, ac);
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
                   (XtCallbackProc) redisplayCB, window);
    XtAddCallback(spin, XmNbtn2Callback,
                   (XtCallbackProc)  spinSelectCB, window);

/* Place now the Sliders, right attached to the spin widget */ 

    formt1 = createSpaceSlider(&window->scaleSliderScale, 
				&window->scaleSliderValue, 
			        allSliders, spin,
			        "scaleScale", "Overall Scale", 
                                SLIDER_MAX, window, scaleCB);  
    formt1 = createSpaceSlider(&window->xTranslSliderScale, 
				&window->xTranslSliderValue, 
			        allSliders, spin,
			        "scaleX", "X transl.", 
                                (SLIDER_MAX/2), window, xTransCB);  
    formt1 = createSpaceSlider(&window->yTranslSliderScale, 
				&window->yTranslSliderValue, 
			        allSliders, spin,
			        "scaleY", "Y transl.", 
                                (SLIDER_MAX/2), window, yTransCB);  
    formt1 = createSpaceSlider(&window->zTranslSliderScale, 
				&window->zTranslSliderValue, 
			        allSliders, spin,
			        "scaleZ", "Z transl.", 
                                (SLIDER_MAX/2), window, zTransCB);  
    formt1 = createSpaceSlider(&window->momToSpaceSliderScale, 
				&window->momToSpaceSliderValue, 
			        allSliders, spin,
			        "momtospace", "P to Dist", 
                               (SLIDER_MAX/10), window, momToSpaceCB);  
    formt1 = createSpaceSlider(&window->longToTrSliderScale, 
				&window->longToTrSliderValue, 
			        allSliders, spin,
			        "LongToTr",
			         "Aspect ratio Long/Tr.          ", 
                                (SLIDER_MAX/10), window, longToTrCB);
/* This last string is in fact constraining the appearance of the widget */
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

static Widget createSpaceSlider(Widget *scaleret, Widget *valueret, 
			 Widget parent, Widget attached,
			 char *namef, char *namel,
                         int initval, SpaceWindow *window, 
                          void (*gensliderCB) ())
{
    Widget bulletin, scale, name, value;
    int i;
    Arg    	args[20];
    int    	ac;
    XmString s1;
     	
    /* create the contents of the slider window */
    scale = XtVaCreateManagedWidget("scale", xmScaleWidgetClass, parent,
    			XmNorientation, XmHORIZONTAL,
    			XmNminimum, 0,
    			XmNmaximum, SLIDER_MAX,
    			XmNvalue, initval,
    			XmNtopOffset, 3,
    			XmNtopAttachment, XmATTACH_WIDGET,
    			XmNtopWidget, parent,
    			XmNleftAttachment, XmATTACH_WIDGET,
    			XmNleftWidget, parent,
    			XmNrightAttachment, XmATTACH_WIDGET,
    			XmNrightWidget, attached,
    			XmNuserData, 0, 0);
	    XtAddCallback(scale, XmNdragCallback,
	    		  (XtCallbackProc)gensliderCB, (caddr_t) window);
	    XtAddCallback(scale, XmNvalueChangedCallback,
	    		  (XtCallbackProc)gensliderCB, (caddr_t) window);
     *scaleret = scale;    		  
    value = XtVaCreateManagedWidget("value", xmLabelWidgetClass, parent, 
    			XmNlabelString,s1=XmStringCreateSimple(namel),
    			XmNalignment, XmALIGNMENT_END,
    			XmNrecomputeSize, False, 0);
      *valueret = value;
      XmStringFree(s1);
    	
    return bulletin;
}

static void chargedCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->charged = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void neutralCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->neutral = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}


static void stableCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->stable = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void unstableCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->unstable = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void electronsCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->electrons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void muonsCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->muons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void gammasCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->gammas = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void neutrinosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->neutrinos = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void quarksCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->quarks = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void hadronsCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->hadrons = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void wzCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    window->wz = XmToggleButtonGetState(w);
    DrawEvent((StdHepWindow *) window, False);
}

static void scaleCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int k, sliderValue;
     float curval, factor;
     XmString s1;
     char line[40];
     
    if (window->filename == NULL) return;
    
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = window->maxScale * 0.3678794 * 
                    exp (-1.0 * SLIDER_MAX / sliderValue );
    if (curval == window->currentScale)  return;
    window->currentScale = curval;
    sprintf(line,"Overall scale = %12.5e",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->scaleSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    /* 
    ** Reset the defaul and scale of other sliders
    **
    */
    window->minTransl[2] = -curval/2.; window->maxTransl[2] = curval/2.;
    window->currentTransl[2] = 0.;
     /*
      ** Set now the Transverse to Longitudinal aspect ratio. 
      ** Currently, at the outset, this is a fixed quantity.. 
      */
     for (k=0; k<2; k++){   
       window->minTransl[k] = -1. *  curval/( 2. * window->currentLongToTr); 
       window->maxTransl[k] = curval/( 2. * window->currentLongToTr);
       window->currentTransl[k] = 0.;
    }
    
    window->currentMomToSpace = 
          2.0 * window->maxTransl[2] /window->maxMomentum;
    window->maxMomToSpace = 10. *  window->currentMomToSpace;
    SetScaleSpaceSliders(window);
    DrawEvent((StdHepWindow *) window, True);
}

static void xTransCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int sliderValue;
     float curval;
     XmString s1;
     char line[40];
     
    if (window->filename == NULL) return;
    
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = window->minTransl[0] +
     sliderValue * (window->maxTransl[0] - window->minTransl[0])/SLIDER_MAX;
    if (curval == window->currentTransl[0])  return;
    window->currentTransl[0] = curval;
    sprintf(line,"X Trans. = %12.5e",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->xTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    DrawEvent((StdHepWindow *) window, False);
}

static void yTransCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int sliderValue;
     char line[40];
     XmString s1;
     float curval;
     
    if (window->filename == NULL) return;
     
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = window->minTransl[1] +
    sliderValue * (window->maxTransl[1] - window->minTransl[1])/SLIDER_MAX;
    if (curval == window->currentTransl[1])  return;
    window->currentTransl[1] = curval;
    sprintf(line,"Y Trans. = %12.5e",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->yTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    DrawEvent((StdHepWindow *) window, False);
}

static void zTransCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int sliderValue;
     char line[40];
     XmString s1;
     float curval;
     
    if (window->filename == NULL) return;
     
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = window->minTransl[2] +
      sliderValue * (window->maxTransl[2] - window->minTransl[2])/SLIDER_MAX;
    if (curval == window->currentTransl[2])  return;
    window->currentTransl[2] = curval;
    sprintf(line,"Z Trans. = %12.5e",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->zTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    DrawEvent((StdHepWindow *) window, False);
}
     
static void momToSpaceCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int sliderValue;
     char line[40];
     XmString s1;
     float curval;
     
    if (window->filename == NULL) return;
     
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = 
         sliderValue * (window->maxMomToSpace)/SLIDER_MAX;
    if (curval == window->currentMomToSpace)  return;
    window->currentMomToSpace = curval;
    sprintf(line,"P to Dist = %12.5e",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->momToSpaceSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    DrawEvent((StdHepWindow *) window, False);
    DrawScale((StdHepWindow *) window);
}
     
static void longToTrCB(Widget w, SpaceWindow *window,caddr_t call_data)
{
     int sliderValue;
     char line[40];
     XmString s1;
     float curval;
     
    if (window->filename == NULL) return;
     
    XtVaGetValues(w, XmNvalue, &sliderValue, 0);
    curval = window->minLongToTr +
       sliderValue * (window->maxLongToTr - window->minLongToTr)/SLIDER_MAX;
    if (curval == window->currentLongToTr)  return;
    window->currentLongToTr = curval;
    sprintf(line,"Pl to Pt = %f",curval);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->longToTrSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    DrawEvent((StdHepWindow *) window, False);
}
     

static void viewRotXPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateX(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotYPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateY(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotZPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateZ(window->spin, window->buttonRotateDegrees);
}
 
static void viewRotXNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateX(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void viewRotYNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateY(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void viewRotZNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinViewRotateZ(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotXPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateX(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotYPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateY(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotZPosCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateZ(window->spin, window->buttonRotateDegrees);
}
 
static void coordRotXNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateX(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotYNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateY(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void coordRotZNegCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinCoordRotateZ(window->spin, 0.-window->buttonRotateDegrees);
}
 
static void scaleUpCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinSetScale(window->spin, SpinGetScale(window->spin) * (5./4.));
    DrawScale((StdHepWindow *) window);
}
 
static void scaleDownCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinSetScale(window->spin, SpinGetScale(window->spin) * (4./5.));
    DrawScale((StdHepWindow *) window);
}
 
static void resetRotationCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    SpinStopSpinning(window->spin);
    SpinRestore(window->spin);
}
 
static void scaleExposeCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    DrawScale((StdHepWindow *) window);
}
 
static void resizeCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    DrawScale((StdHepWindow *) window);
}
 
static void redisplayCB(Widget w, SpaceWindow *window, caddr_t call_data)
{
    ShowSelectedTrack((StdHepWindow *) window);
    UpdateRotationPanel((StdHepWindow *) window);
}
 
static void spinSelectCB(Widget w, SpaceWindow *window, SpinCallbackStruct *cb)
{
    int particleIndex;
    particleIndex = FindTrack((StdHepWindow *)window,
                        cb->event->xbutton.x,cb->event->xbutton.y);
    SelectTrack((StdHepWindow *) window, particleIndex);
}
 
static void nextEvtCB(Widget w, SpaceWindow *window, caddr_t call_data)
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
 
static void previousEvtCB(Widget w, SpaceWindow *window, caddr_t call_data)
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

