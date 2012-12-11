/*******************************************************************************
*									       *
* menu.c -- Nirvana event display menus					       *
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
static char SCCSID[] = "@(#)menu.c	1.2	5/1/92";
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>
#include <X11/Xatom.h>
#include <Xm/Xm.h>
#include <Xm/MainW.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/LabelG.h>
#include <Xm/CascadeB.h>
#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Separator.h>
#include <Xm/FileSB.h>
#include <Xm/TextF.h>
#include <Xm/DialogS.h>
#include "util/printUtils.h"
#include "util/getfiles.h"
#include "util/DialogF.h"
#include "util/misc.h"
#include "spin/Spin.h"
#include "util/help.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "menu.h"
#include "panel.h"
#include "settings.h"
#include "rotation.h"
#include "stdHepFiles.h"
#include "pick.h"
#include "dispTree.h"
#include "help.h"
#include "drawEvent.h"
/*
* Disgusting Fortran stuff
*/

extern void closestdhep_();

static void newPhaseCB(Widget w, PhaseWindow *window, caddr_t callData);
static void newParaCB(Widget w, StdHepWindow *window, caddr_t callData);
static void detectorSketchCB(Widget w, SpaceWindow *window, caddr_t callData);
static void openXCB(Widget w, PhaseWindow *window, caddr_t callData);
static void closeCB(Widget w, PhaseWindow *window, caddr_t callData);
static void printCB(Widget w, PhaseWindow *window, caddr_t callData);
static void exitCB(Widget w, PhaseWindow *window, caddr_t callData);
static void statisticsCB(Widget w, PhaseWindow *window, caddr_t callData);
static void rotationAnglesCB(Widget w, PhaseWindow *window, caddr_t callData); 
static void colorCodesCB(Widget w, StdHepWindow *window, caddr_t callData);
static void showAxesCB(Widget w, PhaseWindow *window, caddr_t callData);
static void showVerticesCB(Widget w, SpaceWindow *window, caddr_t callData);
static void perspectiveCB(Widget w, PhaseWindow *window, caddr_t callData);
static void bufferCB(Widget w, PhaseWindow *window, caddr_t callData);
static void vecLengthCB(Widget w, PhaseWindow *window, caddr_t callData); 
static void rotationIncrCB(Widget w, StdHepWindow *window, caddr_t callData);
static int getFZFilename(Widget w, char *filename, int *blockSize); 
static void eventTreeCB(Widget w, StdHepWindow *window, caddr_t callData); 
static void highlightNodeCB(Widget w, StdHepWindow *window, caddr_t callData); 
static void highlightDaughtersCB(Widget w,
                                 StdHepWindow *window, caddr_t callData); 
static void highlightDescendantsCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData); 
static void highlightMotherCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData); 
static void highlightAncestorsCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData);
                                                                    
/*
** Create the menu bar
*/
Widget CreateMenuBar(Widget parent, StdHepWindow *winsthep, int canOpenFiles,
		     int canExit, int type)
{
    Widget	menuBar;	/*  RowColumn	 		*/
    Widget	cascade;	/*  CascadeButton		*/
    Widget	menuPane;	/*  RowColumn	 		*/
    Widget      subPane;	/*  PullDown		 	 */
    Widget      radioHighlight;	/*  Radio for track mode options */
    Widget 	form;		/* Generic form			*/
    Widget	button;		/*  PushButton			*/

    Arg		al[10];		/*  arg list			*/
    int		ac;		/*  arg count			*/
    XmString	st1, st2;
    Display *display;
    Screen *screen;
    XVisualInfo info;
    PhaseWindow *window = (PhaseWindow *) winsthep;
    SpaceWindow *winspace = (SpaceWindow *) winsthep;
   
    display = XtDisplay(parent);
    screen = XtScreen(parent); 

#define ITEM(parent, name, label, mnemonic, acc, accText, callback) \
    ac = 0; \
    XtSetArg(al[ac], XmNlabelString, st1=MKSTRING(label)); ac++; \
    XtSetArg(al[ac], XmNmnemonic, mnemonic); ac++; \
    XtSetArg(al[ac], XmNacceleratorText, st2=MKSTRING(accText)); ac++; \
    XtSetArg(al[ac], XmNaccelerator, acc); ac++; \
    button = XmCreatePushButton(parent, name, al, ac); \
    XtAddCallback(button, XmNactivateCallback, \
                  (XtCallbackProc) callback, window); \
    XtManageChild(button); \
    XmStringFree(st1); \
    XmStringFree(st2);

#define TOGGLE_ITEM(parent, name, label, mnemonic, acc, accText, callback, set) \
    ac = 0; \
    XtSetArg(al[ac], XmNlabelString, st1=MKSTRING(label)); ac++; \
    XtSetArg(al[ac], XmNmnemonic, mnemonic); ac++; \
    XtSetArg(al[ac], XmNacceleratorText, st2=MKSTRING(accText)); ac++; \
    XtSetArg(al[ac], XmNaccelerator, acc); ac++; \
    XtSetArg(al[ac], XmNset, set); ac++; \
    button = XmCreateToggleButton(parent, name, al, ac); \
    XtAddCallback(button, XmNvalueChangedCallback, \
                          (XtCallbackProc) callback, window); \
    XtManageChild (button); \
    XmStringFree(st1); \
    XmStringFree(st2);

#define SEPARATOR(parent, name) \
    ac = 0; \
    button = XmCreateSeparator(menuPane, "separator1", al, ac); \
    XtManageChild (button);

    /*
    ** Create MenuBar widget.
    */
    ac = 0;
    menuBar = XmCreateMenuBar(parent, "menuBar", al, ac);

    /*
    ** Create "File" pull down menu.
    */
    ac = 0;
    menuPane = XmCreatePulldownMenu(menuBar, "file", al, ac);
    if (type == STDHEP_PHASE) {
      ac = 0; 
      XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show New Display...")); ac++;
      XtSetArg(al[ac], XmNmnemonic, 'N'); ac++;
      XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl N")); ac++; 
      XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> n" ); ac++; 
      XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
      button = XmCreatePushButton(menuPane, "newPhase", al, ac); 
      XtAddCallback(button, XmNactivateCallback, 
                     (XtCallbackProc) newPhaseCB, winsthep); 
      XtManageChild(button);
      winspace->newPhaseButton = button; 
      XmStringFree(st1); 
      XmStringFree(st2);
    }
    ITEM(menuPane, "close", "Close", 'C', "Ctrl <Key> w", "Ctrl W", closeCB)
    if (canOpenFiles) {
	SEPARATOR(menuPane, "separator1")
	ITEM(menuPane, "openx", "Open, Xdr StdHep...", 'X',
	                         "Ctrl <Key> x", "Ctrl X", openXCB)
    }
    SEPARATOR(menuPane, "separator2")
    ITEM(menuPane, "print", "Print...", 'P', "Ctrl <Key> p", "Ctrl P", printCB)

    if (canExit) {
	SEPARATOR(menuPane, "separator3")
	ITEM(menuPane, "exit", "Exit", 'E', "<Key>F3:", "F3", exitCB)
    }
    
    ac = 0;
    XtSetArg (al[ac], XmNsubMenuId, menuPane);  ac++;
    if (canOpenFiles) {
	XtSetArg(al[ac], XmNlabelString, st1=MKSTRING("File")); ac++;
	XtSetArg(al[ac], XmNmnemonic, 'F'); ac++;
	cascade = XmCreateCascadeButton (menuBar, "fileB", al, ac);
    } else {
	XtSetArg(al[ac], XmNlabelString, st1=MKSTRING("Window")); ac++;
	XtSetArg(al[ac], XmNmnemonic, 'W'); ac++;
	cascade = XmCreateCascadeButton (menuBar, "windowB", al, ac);
    }
    XtManageChild (cascade);
    XmStringFree(st1);
    
    /*
    ** Create the Event pull down menu
    */
    menuPane = XmCreatePulldownMenu(menuBar, "event", al, ac);
#ifdef notdef /* not implemented yet */
    ITEM(menuPane, "statistics", "Statistics...", 'S',
    		"Ctrl <Key> s", "Ctrl S", statisticsCB)
#endif
    ITEM(menuPane, "rotationAngles", "Rotation Angles...", 'R',
    	 "Ctrl <Key> r", "Ctrl R", rotationAnglesCB)
    ac = 0; 
    XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show Event Tree...")); ac++;
    XtSetArg(al[ac], XmNmnemonic, 'T'); ac++;
    XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl T")); ac++; 
    XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> t" ); ac++; 
    XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
    button = XmCreatePushButton(menuPane, "eventTree", al, ac); 
    XtAddCallback(button, XmNactivateCallback, 
                  (XtCallbackProc) eventTreeCB, window); 
    XtManageChild(button);
    window->eventTreeButton = button; 
    XmStringFree(st1); 
    XmStringFree(st2);
    /*
    ** The Eta-Pt Phase Space 
    */
    ac = 0; 
    XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show Eta-Pt Display...")); ac++;
    XtSetArg(al[ac], XmNmnemonic, 'E'); ac++;
    XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl E")); ac++; 
    XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> e" ); ac++; 
    XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
    button = XmCreatePushButton(menuPane, "newPara", al, ac); 
    XtAddCallback(button, XmNactivateCallback,
                  (XtCallbackProc)  newParaCB, winsthep); 
    XtManageChild(button);
    winsthep->newParaButton = button; 
    XmStringFree(st1); 
    XmStringFree(st2);
    
    if ((type == STDHEP_SPACE)) {
      ac = 0; 
      XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show Phase Display...")); ac++;
      XtSetArg(al[ac], XmNmnemonic, 'D'); ac++;
      XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl D")); ac++; 
      XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> d" ); ac++; 
      XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
      button = XmCreatePushButton(menuPane, "newPhase", al, ac); 
      XtAddCallback(button, XmNactivateCallback,
                    (XtCallbackProc)  newPhaseCB, winsthep); 
      XtManageChild(button);
      winspace->newPhaseButton = button; 
      XmStringFree(st1); 
      XmStringFree(st2);
      
      ac = 0; 
      XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show Detector Sketch...")); ac++;
      XtSetArg(al[ac], XmNmnemonic, 'S'); ac++;
      XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl S")); ac++; 
      XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> s" ); ac++; 
      XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
      button = XmCreatePushButton(menuPane, "detectorSketch", al, ac); 
      XtAddCallback(button, XmNactivateCallback, 
                      (XtCallbackProc) detectorSketchCB, winsthep); 
      XtManageChild(button);
      winspace->detectorButton = button; 
      XmStringFree(st1); 
      XmStringFree(st2);
     }
    
/* Check that we have a color screen .. */   
    if (XMatchVisualInfo(display, XScreenNumberOfScreen(screen),
	    DefaultDepthOfScreen(screen), PseudoColor, &info)) {
     ac = 0; 
     XtSetArg(al[ac], XmNlabelString,
                st1=MKSTRING("Show Color Code...")); ac++;
     XtSetArg(al[ac], XmNmnemonic, 'C'); ac++;
     XtSetArg(al[ac], XmNacceleratorText,
                 st2=MKSTRING("Ctrl C")); ac++; 
     XtSetArg(al[ac], XmNaccelerator,"Ctrl <Key> c" ); ac++; 
     XtSetArg(al[ac], XmNsensitive, FALSE); ac++; 
     button = XmCreatePushButton(menuPane, "colorCode", al, ac); 
     XtAddCallback(button, XmNactivateCallback,
                     (XtCallbackProc) colorCodesCB, window); 
     XtManageChild(button);
     window->colorCodeButton = button; 
     XmStringFree(st1); 
     XmStringFree(st2);
    } else window->colorCodeButton = NULL;
    
    ac = 0;
    XtSetArg (al[ac], XmNsubMenuId, menuPane);  ac++;
    XtSetArg(al[ac], XmNlabelString, st1=MKSTRING("Event")); ac++;
    XtSetArg(al[ac], XmNmnemonic, 'E'); ac++;
    cascade = XmCreateCascadeButton (menuBar, "eventB", al, ac);
    XtManageChild (cascade);
    XmStringFree(st1);
    
    /*
    ** Create the Preferences pull down menu
    */
    menuPane = XmCreatePulldownMenu(menuBar, "preferences", al, ac);
    TOGGLE_ITEM(menuPane, "showAxes", "Show Axes", 'A',
    		"Ctrl <Key> a", "Ctrl A", showAxesCB, True)
    if (type == STDHEP_SPACE) {
      TOGGLE_ITEM(menuPane, "showVertices", "Show Vertices", 'V',
    		"Ctrl <Key> v", "Ctrl V", showVerticesCB, True)
    }
/* The perspective option is buggy, skip it..
    TOGGLE_ITEM(menuPane, "perspective", "Perspective", 'P',
    		"Ctrl <Key> x", "Ctrl X", perspectiveCB, True) */
    TOGGLE_ITEM(menuPane, "bufferGraphics", "Buffer Graphics", 'B',
    		NULL, "", bufferCB, False)
    if (type == STDHEP_PHASE) {
    ITEM(menuPane, "vectorLength", "% Vectors Shown...", '%',
    		NULL, "", vecLengthCB)
    }
    ITEM(menuPane, "rotationIncrements", "Rotation Increments...", 'R',
    		NULL, "", rotationIncrCB)
    
    winsthep->highlightTrackMode = TRACK_NODE;    
    subPane = AddSubMenu(menuPane, "trackHighlight", "Track Highlight", 'H');
    XtVaSetValues(subPane, XmNradioBehavior, True, 0);
    button = XtVaCreateManagedWidget("trackNode",
                xmToggleButtonWidgetClass, subPane,
                XmNlabelString, st1=XmStringCreateSimple("Node -> Track"),
                XmNset, True, 0);
    XtAddCallback(button, XmNvalueChangedCallback,
                   (XtCallbackProc)  highlightNodeCB, window); 
    XmStringFree(st1);
    button = XtVaCreateManagedWidget("trackDaughters",
                xmToggleButtonWidgetClass, subPane,
                XmNlabelString, st1=XmStringCreateSimple("Node -> Daughters"),
                XmNset, False, 0);
    XtAddCallback(button, XmNvalueChangedCallback,
                  (XtCallbackProc) highlightDaughtersCB, window); 
    XmStringFree(st1);
    button = XtVaCreateManagedWidget("trackDescendants",
                xmToggleButtonWidgetClass, subPane,
                XmNlabelString, st1=XmStringCreateSimple("Node -> Descendants"),
                XmNset, False, 0);
    XtAddCallback(button, XmNvalueChangedCallback,
                 (XtCallbackProc)  highlightDescendantsCB, window); 
    XmStringFree(st1);     
    button = XtVaCreateManagedWidget("trackMother",
                xmToggleButtonWidgetClass, subPane,
                XmNlabelString, st1=XmStringCreateSimple("Node -> Mother"),
                XmNset, False, 0);
    XtAddCallback(button, XmNvalueChangedCallback,
                 (XtCallbackProc)  highlightMotherCB, window); 
    XmStringFree(st1);
    button = XtVaCreateManagedWidget("trackAncestors",
                xmToggleButtonWidgetClass, subPane,
                XmNlabelString, st1=XmStringCreateSimple("Node -> Ancestors"),
                XmNset, False, 0);
    XtAddCallback(button, XmNvalueChangedCallback,
                  (XtCallbackProc) highlightAncestorsCB, window); 
    XmStringFree(st1);
     
    ac = 0;
    XtSetArg (al[ac], XmNsubMenuId, menuPane);  ac++;
    XtSetArg(al[ac], XmNlabelString, st1=MKSTRING("Preferences")); ac++;
    XtSetArg(al[ac], XmNmnemonic, 'P'); ac++;
    cascade = XmCreateCascadeButton (menuBar, "preferencesB", al, ac);
    XtManageChild (cascade);
    XmStringFree(st1);
    
    /* Create the Help menu */
    if (type == STDHEP_PHASE) 
       CreateHelpPulldownMenu(menuBar, MainMenuHelp);
    if (type == STDHEP_SPACE) 
       CreateHelpPulldownMenu(menuBar, MainMenuHelpSpace);
    if (type == STDHEP_PARA)
       CreateHelpPulldownMenu(menuBar, MainMenuHelpPara);
       
    return (menuBar);
}

void GetSetStdHep(StdHepWindow *window, int evtnum)
{
    XmString mString;
    char *ctmp;
    PhaseWindow *winp = (PhaseWindow *) window;
    SpaceWindow *wins = (SpaceWindow *) window;
    
/*
* If a track has been selected, make sure it taking care off first 
*/
    if (window->selectedTrack != NO_TRACK)
        XtUnmapWidget(window->trackWindowShell);
/*
* If a Tree window exists, scrap it
*/
     if (window->dtree != NULL) CloseHepTreeWindow(NULL, window);
/*
* Now get the new stuff..
*/        
     OpenGetStdHep(window, evtnum);
     
/* Set the Tree button Sensitive, newPhase ( and Color tree ) */
    
    XtSetSensitive(window->eventTreeButton, TRUE);                           
    if(window->colorCodeButton != NULL)
        XtSetSensitive(window->colorCodeButton, TRUE);                           
    XtSetSensitive(window->newPhaseButton, TRUE);
    XtSetSensitive(window->newParaButton, TRUE);
    if (window->type == STDHEP_SPACE) 
       XtSetSensitive(wins->detectorButton, TRUE);
    
/* Set the Slider value corresponding to this event */
                               
     if (window->type == STDHEP_SPACE) { 
       SetScaleSpaceSliders(wins);
       SET_ONE_RSRC( wins->scaleSliderScale, XmNvalue, SLIDER_MAX);
    } else if (window->type == STDHEP_PARA) {
       SetScaleParaSliders((ParaWindow *) wins);
    }    
/*
* Set the eventnumber information line.. and event number. 
*/
    ctmp = (char *) XtMalloc(80);
    if (window->nEvents == window->nMaxEvts )
    sprintf(ctmp, "Out of at least %d, events, StdHep evt %d ", 
                       window->nMaxEvts, window->event.userData);
    else   sprintf(ctmp, "Out of %d, events, StdHep evt %d ", 
                       window->nEvents, window->event.userData);

    mString = XmStringCreateSimple(ctmp);
    XtFree((char *) ctmp);
    XtVaSetValues(window->eventSelectorLabel, XmNlabelString, mString, 0);
    XmStringFree(mString);
    
    ctmp = (char *) XtMalloc(20);
    sprintf(ctmp, " %d ", evtnum);
    mString = XmStringCreateSimple(ctmp);
    XtFree((char *) ctmp);
    XtVaSetValues(window->eventNumText, XmNlabelString, mString, 0);
    XmStringFree(mString);
    
    window->selectedTrack = NO_TRACK;
    window->selnodeNumTrack = 0;
    window->selnodeTracks = NULL;
    
    DrawEvent(window, True);
}

static void newPhaseCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    char title[40];
/* 
* Start a new window, with the same event, but clone the window 
* structure to avoid clashes with the other(s) existing windows. 
* Since we have only one bacpointer to TreeHep...
*/
    sprintf(title, "StdHep Event %d ", window->event.userData);
    DisplayOneEvent(XtDisplay(w), &window->event, title, STDHEP_PHASE,
    			window->exitProc);    
}

static void newParaCB(Widget w, StdHepWindow *window, caddr_t callData) 
{
    char title[40];
/* 
* Start a new window, with the same event, but clone the window 
* structure to avoid clashes with the other(s) existing windows. 
* Since we have only one bacpointer to TreeHep...
*/
    sprintf(title, "StdHep Event %d ", window->event.userData);
    DisplayOneEvent(XtDisplay(w), &window->event, title, STDHEP_PARA,
    			window->exitProc);    
}

static void detectorSketchCB(Widget w, SpaceWindow *window, caddr_t callData) 
{
/*
**  Read a Detector-Sketch file, and redraw the event. 
*/
	GetDetectorSketch(window);
	DrawEvent((StdHepWindow *) window, False);
}

static void openXCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
   
   int ifile, resp, ievt;
   int istr = 0;
   
  /* Set the Event Tree button insensitive */
  
    XtSetSensitive(window->eventTreeButton, FALSE);
    if (window->colorCodeButton != NULL) 
       XtSetSensitive(window->colorCodeButton, FALSE);
    
    if (window->filename != NULL) {
        resp = DialogF(DF_WARN, w, 2,
                "Opening a new file will delete current display ",
                        "Close", "Cancel");
        if (resp == 2)
            return;
        XtFree((char *) window->filename);
    }
    window->filename = (char *) XtMalloc(255);    
    ifile = GetExistingFilename(w, "Xdr StdHep File", window->filename);
    if (ifile == GFN_CANCEL) {
       XtFree((char *) window->filename);
       window->filename = NULL;
       return;
    }
    
/* Set the maximum number of events "check through STDHEP i/o to xx, aribtrary
   at this point.. To set by a preference pull down menu */
   
    window->nMaxEvts = 100;
    window->xdrStream = istr;
    window->nEvents = 
       OpenCheckXdrStdHep(window->filename, window->nMaxEvts, istr);
    if (window->nEvents <= 0) {
        resp = DialogF(DF_INF, w, 1,
         "This file does not contains a single StdHep event, closing it ",
                        "OK");
          closestdhep_();
          window->nEvents = 0;
          return;
    }
    ievt = 1;
    window->filemode = XDR_STDHEP;
/* 
* Set the title window to the filename
*/
    XtVaSetValues(window->shell, XmNtitle, window->filename,
                 XmNiconName, window->filename, 0);
                                   
    GetSetStdHep((StdHepWindow *) window, ievt);  
}

static int getFZFilename(Widget w, char *filename, int *blockSize)
{
    int resp, bsOK, newBlockSize;
    Widget fileSB, form, label, text;
    XmString s1, s2;
    int n;
    Arg args[10];
    
    while(True) {
	n=0;
	XtSetArg(args[n], XmNlistLabelString,
    		s1=XmStringCreateSimple("FZ file (XLI) to open:")); n++;
	XtSetArg(args[n], XmNdialogStyle, XmDIALOG_FULL_APPLICATION_MODAL); n++;
	XtSetArg(args[n], XmNdialogTitle, s2=XmStringCreateSimple(" ")); n++;
	fileSB = XmCreateFileSelectionDialog(w, "hbFileSB", args, n);
	XmStringFree(s1); XmStringFree(s2);
	XtUnmanageChild(XmFileSelectionBoxGetChild(fileSB, XmDIALOG_TEXT)); 
	XtUnmanageChild(XmFileSelectionBoxGetChild(fileSB,
		XmDIALOG_SELECTION_LABEL));
	form = XtVaCreateManagedWidget("bsForm", xmFormWidgetClass, fileSB, 0);
	label = XtVaCreateManagedWidget("bsLabel", xmLabelGadgetClass, form,
    		XmNlabelString,
    		    s1=XmStringCreateSimple("FZ file lrec (words):"),
    		XmNtopAttachment, XmATTACH_FORM,
    		XmNbottomAttachment, XmATTACH_FORM, 0);
	text = XtVaCreateManagedWidget("bsText", xmTextFieldWidgetClass, form,
    		XmNleftAttachment, XmATTACH_WIDGET,
    		XmNrightAttachment, XmATTACH_FORM,
    		XmNleftWidget, label, 0);
    	RemapDeleteKey(text);
    	*blockSize = 900;
    	SetIntText(text, *blockSize);
    
	resp = HandleCustomExistFileSB(fileSB, filename);
	XmStringFree(s1);		/* JMK - 6/28/93	*/
    	if (resp == GFN_CANCEL)
    	    return GFN_CANCEL ;
    	bsOK = GetIntText(text, &newBlockSize);
    	if (bsOK == TEXT_READ_OK) {
	    *blockSize = newBlockSize;
    	    return GFN_OK;
	} else if (bsOK == TEXT_IS_BLANK)
	    DialogF(DF_ERR, w, 1, "Please enter record length\n\
for FZ file",  "Acknowledged");
	else 
	    DialogF(DF_ERR, w, 1, "Can't read value for FZ\n\
file record length",  "Acknowledged");
    }
}

void CloseWindow(StdHepWindow *window)
{
     int resp, iok;
     SpaceWindow *wins;
     PhaseWindow *winp;
     
    /* If no other windows opened, this means that the users will de-facto
    	exit the program.. Make sure that's what he wants.. */
    	
      if (((WindowList == window) && (window->next == NULL)) ) {
        resp = DialogF(DF_WARN, window->shell, 2,
                "Closing this window will terminate the show",
                        "exit", "Cancel");
        if (resp == 2) return;
        if (window->exitProc != NULL)
    	      (*window->exitProc)();
    }
    /* Remove the Tree Window (if there ) */
    
    CloseHepTreeWindow(NULL, window);
    
    /* Remove the colorcode tree */ 
    if (window->treeheadcolorcode != NULL) 
        FreeTree((nodehep *) window->treeheadcolorcode);
        
    if (window->colorcode.nParticles != 0 )
        XtFree((char *) window->colorcode.particles); 
     
    if (window->selnodeTracks != NULL) XtFree((char *) window->selnodeTracks);
       
    /* remove and deallocate all of the widgets associated with window */
    XtDestroyWidget(window->shell);
    
    /* free the event track data */
    if (window->event.nParticles != 0)
    	XtFree((char *) window->event.particles);
    	
    /* Free the verices if a Space Window */
    if (window->type == STDHEP_SPACE) { 
        wins = (SpaceWindow *) window;
        XtFree((char *) wins->vertices);
        XtFree((char *) wins->realVertices);
        wins->vertices = NULL;
        wins->realVertices = NULL;
        if (wins->detectorSegments != NULL) {
          XtFree((char *) wins->detectorSegments);
          wins->detectorSegments = NULL;
          }
     }	
    /* remove the window from the global window list */
    RemoveFromWindowList(window);
    
    /* deallocate the window data structure */
    XtFree((char *) window);
}

void SetScaleSpaceSliders(SpaceWindow *window)
{
      XmString s1;
      int ival;
      char line[40];

    sprintf(line,"Overall Scale = %12.5e",window->currentScale);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->scaleSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    sprintf(line,"X Trans. = %12.5e",window->currentTransl[0]);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->xTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    sprintf(line,"Y Trans. = %12.5e",window->currentTransl[1]);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->yTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    sprintf(line,"Z Trans. = %12.5e",window->currentTransl[2]);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->zTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    sprintf(line,"P to Dist = %12.5e",window->currentMomToSpace);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->momToSpaceSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    sprintf(line,"Pl to Pt = %f", window->currentLongToTr);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->longToTrSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    SET_ONE_RSRC( window->xTranslSliderScale, XmNvalue, SLIDER_MAX/2);
    SET_ONE_RSRC( window->yTranslSliderScale, XmNvalue, SLIDER_MAX/2);
    SET_ONE_RSRC( window->zTranslSliderScale, XmNvalue, SLIDER_MAX/2);
    SET_ONE_RSRC( window->momToSpaceSliderScale, XmNvalue, SLIDER_MAX/10);
    SET_ONE_RSRC( window->longToTrSliderScale, XmNvalue, SLIDER_MAX/10);
}

 void SetScaleParaSliders(ParaWindow *window)
{
      XmString s1;
      int i, ival;
      char line[40];
      PhaseParticle *p;
      double eta, pt, etamax, ptmax;

/*
** Compute the maximum Pt and eta in the event, this set the scale 
**     
*/
    etamax = 0.; ptmax = 0.;
    p = window->event.particles;
    for (i=0; i < window->event.nParticles; i++, p++) {
      eta = ParticleRapidity(p->px, p->py, p->pz, p->mass);
      if (fabs(eta) > etamax) etamax = fabs(eta);
      pt = ParticlePT(p->px, p->py);
      if (pt > ptmax) ptmax = pt;
    }
    window->minTranslz = -etamax;
    window->maxTranslz = etamax;
    window->currentTranslz = 0.;

    sprintf(line,"Eta Translation = %12.5e",window->currentTranslz);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->zTranslSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    window->currentRapToPt = ptmax/etamax;
    window->minRapToPt = 0.2 * window->currentRapToPt;
    window->maxRapToPt = 5. * window->currentRapToPt;
     window->currentRapToPt = window->minRapToPt + 
     0.5 * (window->maxRapToPt - window->minRapToPt);
    sprintf(line,"Eta to Dist = %12.5e",window->currentRapToPt);
    s1=XmStringCreateSimple(line);
    XtVaSetValues(window->rapToPtSliderValue, XmNlabelString, s1, 0);
    XmStringFree(s1);
    
    SET_ONE_RSRC( window->zTranslSliderScale, XmNvalue, SLIDER_MAX/2);
    SET_ONE_RSRC( window->rapToPtSliderScale, XmNvalue, SLIDER_MAX/2);
}

static void closeCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    CloseWindow((StdHepWindow *) window);
}

static void printCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    char tmpFileName[L_tmpnam];    /* L_tmpnam defined in stdio.h */
    char *windowTitle;

    tmpnam(tmpFileName);
    SpinPrintContents(window->spin, tmpFileName);
    GET_ONE_RSRC(window->shell, XmNtitle, &windowTitle)
    PrintFile(window->spin, tmpFileName, windowTitle);
    remove(tmpFileName);

}

static void exitCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    if (window->exitProc != NULL)
    	(*window->exitProc)();
}

static void statisticsCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    printf("statisticsCB called\n");
}

static void rotationAnglesCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    ShowAbsRotationPanel((StdHepWindow *) window);
}

static void colorCodesCB(Widget w, StdHepWindow *window, caddr_t callData) 
{
    window->modetreedisp = TREEDISPCODE;
    if (window->colorcode.nParticles == 0)
       LoadColorCodeEvent((StdHepWindow *) window);
    DispTree(w, window);
}

static void eventTreeCB(Widget w, StdHepWindow *window, caddr_t callData) 
{
    window->modetreedisp = TREEDISPREAL;
    DispTree(w, window);
}

static void showAxesCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    SET_ONE_RSRC(window->spin, XmNshowAxes, XmToggleButtonGetState(w));
}

static void showVerticesCB(Widget w, SpaceWindow *window, caddr_t callData) 
{
	if (XmToggleButtonGetState(w)) window->showVertices = True;
	else window->showVertices = False;
	if ((window->numRealVertices > 0) && (window->vertices != NULL))
	           DrawEvent((StdHepWindow*) window, False);
}

static void perspectiveCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    SET_ONE_RSRC(window->spin, XmNperspectiveOn, XmToggleButtonGetState(w));
}

static void bufferCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    SET_ONE_RSRC(window->spin, XmNdoubleBuffer, XmToggleButtonGetState(w));
}

static void vecLengthCB(Widget w, PhaseWindow *window, caddr_t callData) 
{
    ShowVectorLengthPanel(window);
}

static void rotationIncrCB(Widget w, StdHepWindow *window, caddr_t callData) 
{
    ShowButtonRotationPanel(window);
}
static void highlightNodeCB(Widget w, StdHepWindow *window, caddr_t callData)
{
    if (XmToggleButtonGetState(w)) window->highlightTrackMode = TRACK_NODE;
}

static void highlightDaughtersCB(Widget w,
                                 StdHepWindow *window, caddr_t callData) 
{
    if (XmToggleButtonGetState(w)) 
     window->highlightTrackMode = TRACK_DAUGHTERS;
}

static void highlightDescendantsCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData) 
{
    if (XmToggleButtonGetState(w)) 
    window->highlightTrackMode = TRACK_DESCENDANTS;
}

static void highlightMotherCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData) 
{
    if (XmToggleButtonGetState(w)) 
    window->highlightTrackMode = TRACK_MOTHER;
}

static void highlightAncestorsCB(Widget w, 
                                 StdHepWindow *window, caddr_t callData) 

{
    if (XmToggleButtonGetState(w)) 
    window->highlightTrackMode = TRACK_ANCESTORS;
}


