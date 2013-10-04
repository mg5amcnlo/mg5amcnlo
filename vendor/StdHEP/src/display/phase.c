/*******************************************************************************
*									       *
* phase.c -- Nirvana Phase Space Event Display				       *
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
static char SCCSID[] = "@(#)phase.c	1.2	5/22/92";
#include <sys/param.h>
#include <stdio.h>
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>		/* only for SpinSegment reqd by drawEvent.h */
#include "spin/Spin.h"	/* "" "" */
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "panel.h"
#include "panelPara.h"
#include "pixmaps.h"
#include "pick.h"
#include "drawEvent.h"
#include "dispTree.h"

static int ColorsAndBitmapsInitialized = False;

static void initColorsAndBitmaps(Display *display);
static void initWindow(StdHepWindow *window);

/*
** DisplayOneEvent
**
** Puts up a phase space event display window to display a single event.
** DisplayOneEvent must be called from a properly initialized Motif environment.
** After calling DisplayOneEvent, the program must enter a standard Motif event
** loop (XtMainLoop, XtAppMainLoop, or a user written main loop).
**
** Parameters
**
**	display		The X display to use
**	event		The event to display (phase.h has details).  The
**			event data is copied including the particle list,
**			so the caller does not need to maintain it after
**			the call.
**	windowTitle	A title for the event display window
**      mode 		A int to declare the display type or mode, 
**			either PHASE or PARA ( Eta-Pt) plot.
**	exitProc	A procedure to execute when the user selects "Exit"
**			from the "File", or "Window" menu of the event
**			display window.  Pass NULL to omit "Exit" from
**			the menu.
**			
*/
EventWindow *DisplayOneEvent(Display *display, PhaseEvent *event,
    char *windowTitle, int mode, void (*exitProc)())
{
    char title[MAXPATHLEN + 10];
    char eventControlText[MAXPATHLEN + 30];
    PhaseWindow *window;
    ParaWindow *winpa;
    StdHepWindow *winsthep;
    int particleBytes;

    if (ColorsAndBitmapsInitialized == False)
    	initColorsAndBitmaps(display);

    if (mode == STDHEP_PHASE) {
      window = CreatePanel(display, False, False, 
    			 exitProc != NULL, windowTitle, NULL);
      window->type = STDHEP_PHASE;
      winsthep = (StdHepWindow *) window;
    } else if ( mode == STDHEP_PARA) { 
      winpa = CreatePanelPa(display, False, False, 
    			 exitProc != NULL, windowTitle, NULL);
      winsthep = (StdHepWindow *) winpa;
      winpa->type = STDHEP_PARA;
    }
    winsthep->exitProc = exitProc;
    winsthep->getEventProc = NULL;
    winsthep->openProc = NULL;
    initWindow(winsthep);
    
    /* Copy the event so caller doesn't have to maintain it */
    winsthep->event = *event;
    particleBytes = event->nParticles * sizeof(PhaseParticle);
    winsthep->event.particles = (PhaseParticle *)XtMalloc(particleBytes);
    memcpy(winsthep->event.particles, event->particles, particleBytes);
    
    if (mode == STDHEP_PARA) SetScaleParaSliders(winpa);
    DrawEvent(winsthep, True);
    if ( mode == STDHEP_PARA) {
      XtSetSensitive(winsthep->eventTreeButton, TRUE);                           
      if(winsthep->colorCodeButton != NULL)
        XtSetSensitive(winsthep->colorCodeButton, TRUE);
        }                           
    return (EventWindow *)window;
}

/*
** DisplayEventFile
**
** openProc should return # of events in file, or -1 if can't open
** should this allow an initial filename to be specified?
*/
EventWindow *DisplayEventFile(Display *display, void (*exitProc)(),
    int (*openProc)(char *filename),
    void (*closeProc)(FILE *fs), PhaseEvent *(*getEventProc)(int eventNum))
{
    static char title[] = "Event Phase Space Display (no file open)";
    static char barText[] = "No file open";
    PhaseWindow *window;
   
    if (ColorsAndBitmapsInitialized == False)
    	initColorsAndBitmaps(display);

    window = CreatePanel(display, True, True,
    			 exitProc != NULL, title, barText);
    window->type = STDHEP_PHASE;
    window->exitProc = exitProc;
    window->getEventProc = getEventProc;
    window->openProc = openProc;
    window->closeProc = closeProc;
    initWindow( (StdHepWindow *) window);
    return (EventWindow *)window;
}


static void initColorsAndBitmaps(Display *display)
{   
    /*
    ** Put the button bitmaps into the motif bitmap cache for later use
    ** in creating buttons with bitmap labels
    */
    RegisterPhaseBitmaps(DefaultScreenOfDisplay(display));
    
    /*
    ** Allocate color cells for particle colors
    */
    AllocateColors(DefaultScreenOfDisplay(display));
    ColorsAndBitmapsInitialized = True;
}

static void initWindow(StdHepWindow *window)
{
    window->filename=NULL;
    window->event.particles = NULL;
    window->event.eventNum = 0;
    window->event.nParticles = 0;
    window->dtree = NULL;
    window->stree = NULL;
    window->tree = NULL;
    window->dtreecolorcode = NULL;
    window->streecolorcode = NULL;
    window->treecolorcode = NULL;
    window->nodeWindowShell = NULL;
    window->colorWindowShell = NULL;
    window->treehead = NULL;
    window->selectedTrack = NO_TRACK;
    window->selectedNode = NULL;
    window->selnodeNumTrack = 0;
    window->selnodeTracks = NULL;
    window->treeheadcolorcode = NULL;
    window->modetreedisp = TREEDISPREAL;
}    
