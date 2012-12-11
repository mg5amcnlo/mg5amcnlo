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
#include "panelSpace.h"
#include "pixmaps.h"
#include "pick.h"
#include "drawEvent.h"
#include "dispTree.h"

static int ColorsAndBitmapsInitialized = False;

static void initColorsAndBitmaps(Display *display);
static void initWindow(StdHepWindow *window);

/*
** DisplayEventFile
**
** openProc should return # of events in file, or -1 if can't open
** should this allow an initial filename to be specified?
*/
EventSpaceWindow *DisplayEventFileSp(Display *display, void (*exitProc)())
{
    static char title[] = "Event Real Space Display (Track-Vertices)";
    static char barText[] = "No file open";
    SpaceWindow *window;
   
    if (ColorsAndBitmapsInitialized == False)
    	initColorsAndBitmaps(display);

   window = CreatePanelSp(display, True, True,
    			 exitProc != NULL, title, barText);
    window->showVertices = True;
    window->type = STDHEP_SPACE;
    window->exitProc = exitProc;
    initWindow((StdHepWindow *)window);
    return (EventSpaceWindow *)window;
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
    /*
    ** Compute some sin and cos for Vertex display 
    */
    SetPinVertex();
    ColorsAndBitmapsInitialized = True;
}

static void initWindow(StdHepWindow *window)
{
    SpaceWindow *swindow;
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
    if(window->type == STDHEP_SPACE) {
      swindow = (SpaceWindow*) window;
      swindow->maxMomToSpace = 100.;
      swindow->currentMomToSpace = 1.;
    }
}    
