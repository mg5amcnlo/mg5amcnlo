/*******************************************************************************
*									       *
* pick.c -- Pick tracks on event display by pointing middle mouse button       *
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
static char SCCSID[] = "@(#)pick.c	1.1	4/6/92";
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/Label.h>
#include <Xm/DialogS.h>
#include <float.h>	/* for FLT_MAX */
#include <math.h>
#include "spin/Spin.h"
#include "util/stringUtils.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "pick.h"
#include "drawEvent.h"
#include "dispTree.h"

#define MIN_PICK_DIST 15	/* Farthest from a track that user can pick
				   and still select the track, in pixels */
#define MAXCHAR_HEPNAM 30  /* for use with hepnam_ */


static void createTrackInfoWindow(StdHepWindow *window, XmString contents,
				  char *title);
static void closeCB(Widget w, StdHepWindow *window, caddr_t call_data);

/*
** Highlight the selected track by drawing it thicker
*/
void ShowSelectedTrack(StdHepWindow *window)
{
    SpinSegment seg;
    short x1, y1, x2, y2;
    Arg wargs[2];
    double length;
    PhaseParticle *particle;
    nodehep *seln1, *seln2, *htop;
    int n, i, *list;
    Widget highnode;
    PhaseWindow *winp = (PhaseWindow *) window;
    SpaceWindow *wins = (SpaceWindow *) window;
    SpaceVertex *vert;
     
    
    /* Widget spin = window->spin; Causes Ultrix cc to "schain botch 3" */
    Display *display = XtDisplay(window->spin);
    Window spinWindow = XtWindow(window->spin);
    
    if ((window->selectedTrack == NO_TRACK) &&
        (window->selnodeNumTrack <= 0)) return;
    
    if (window->selectedTrack != NO_TRACK) {
      particle = &window->event.particles[window->selectedTrack];
      if (ParticleVisible(window, particle->id, particle->stable)) {
	if (window->type == STDHEP_PHASE ) 
	  ParticleToSegment(winp, particle, &seg, &length);
	else if (window->type == STDHEP_PARA) { 
	  ParticleToParaSegment((ParaWindow *) wins, particle, &seg, &length);
	   } else {
        vert = wins->vertices; vert += window->selectedTrack;
	   TrackToSegment(wins, particle, vert, &seg, &length);
	 }  
	SpinTransformPoint(window->spin, seg.x1, seg.y1, seg.z1, &x1, &y1);
	SpinTransformPoint(window->spin, seg.x2, seg.y2, seg.z2, &x2, &y2);
	XSetForeground(display, window->highlightGC, seg.pixel);
	XDrawLine(display, spinWindow, window->highlightGC, x1, y1, x2, y2);
     }
     if ((window->dtree != NULL) && (window->treehead !=NULL)) {
	/* If previous select, set the node to normal shadow value */
	
	/* located the node, and heighlight it.. if different from 
		the one already selected, to avoid flashing  */
            htop = (nodehep *) window->treehead;
	    seln1 = SearchNodes(htop, window->selectedTrack);
	    seln2 = NULL;
	    if (window->selectedNode != NULL) {
	     seln2 = (nodehep *)  window->selectedNode;
	     if (seln1 !=seln2) { 
	       highnode = (Widget) *(seln2->stwidget);
               XtSetArg(wargs[0], XmNshadowThickness, 2);
	       XtSetValues(highnode, wargs, 1);
	      }
	     }
	  
	   if ((seln1 != NULL) && (seln1 != seln2)) {
	     window->selectedNode = (int *) seln1;
	     highnode = (Widget)  *(seln1->stwidget);
             XtSetArg(wargs[0], XmNshadowThickness, 5);
	     XtSetValues(highnode, wargs, 1);
	   } 
     }
      
    }
    if ( window->selnodeNumTrack > 0) {
     list = window->selnodeTracks; 
      for (i=0; i < window->selnodeNumTrack; i++, list++) {
        particle = &window->event.particles[*list];
        if (ParticleVisible(window, particle->id, particle->stable)) {
	 if (window->type == STDHEP_PHASE ) 
	   ParticleToSegment(winp, particle, &seg, &length);
	 else if (window->type == STDHEP_PARA) {
	   ParticleToParaSegment((ParaWindow *) wins, particle, &seg, &length);
	   } else {
            vert = wins->vertices; vert += (*list);
	   TrackToSegment(wins, particle, vert, &seg, &length);
	  }  
	 SpinTransformPoint(window->spin, seg.x1, seg.y1, seg.z1, &x1, &y1);
	 SpinTransformPoint(window->spin, seg.x2, seg.y2, seg.z2, &x2, &y2);
	 XSetForeground(display, window->highlightGC, seg.pixel);
	 XDrawLine(display, spinWindow, window->highlightGC, x1, y1, x2, y2);
      }
     }
    }
          
}

/*
** Highlight the selected track by drawing it thicker
*/
void SelectTrack(StdHepWindow *window, int trackIndex)
{
    if (trackIndex != window->selectedTrack) {
	window->selectedTrack = trackIndex;
	SpinRedisplay(window->spin);
	ShowTrackStats(window, trackIndex);
    }
}

int FindTrack(StdHepWindow *window, int x, int y)
{
    PhaseParticle *p;
    SpinSegment seg;
    double length;
    int i, minIndex, iok;
    short xs1, ys1, xs2, ys2;	/* window coordinates of a track segment */
    double rp, thetap;		/* pick point translated by xs1, ys1 & polar */
    double rs, thetas;		/* segment translated by xs1, ys1 & polar */
    double segLength;		/* length of segment in window coordinates */
    double xpPrime, ypPrime;	/* pick pt translated and rotated */
    double dist;		/* distance of current segment from pick pt */
    double minDist = FLT_MAX;	/* distance of closest segment so far */
    double l12a, lo1sq, lo2sq, l12, l12sq, dperpsq, dmidsq, xm12, ym12;		
    PhaseWindow *winp = (PhaseWindow *) window;
    SpaceWindow *wins = (SpaceWindow *) window;
    SpaceVertex *vert;
    
    /*
    ** Loop through all of the particles, finding the one which displays
    ** closest to the pick point
    */
    if (window->type == STDHEP_SPACE ) vert = wins->vertices;
    for (i=0, p=window->event.particles; i<window->event.nParticles;
         i++, p++) {
      if (ParticleVisible(window, p->id, p->stable)) {
    	/* get the line segment representing the particle in the spin widget */
	 if (window->type == STDHEP_PHASE ) 
	   iok = ParticleToSegment(winp, p, &seg, &length);
	   else if (window->type == STDHEP_PARA) {
	   iok = 
	     ParticleToParaSegment((ParaWindow *) wins, p, &seg, &length);
	   } else {
	   iok = TrackToSegment(wins, p, vert, &seg, &length);
	  }  
    	if (iok) {
    	    /* translate it to window coordinates */
    	    SpinTransformPoint(window->spin, seg.x1,seg.y1,seg.z1 ,&xs1, &ys1);
    	    SpinTransformPoint(window->spin, seg.x2,seg.y2,seg.z2, &xs2, &ys2);
    	    /*
    	    ** The criteria for mindistance depends on the type of plots..
    	    ** For Phase Display, work in Polar coordinates, ( Mark Edel),
    	    */
    	    if (window->type == STDHEP_PHASE) {
    	    
    	    /* translate the pick point and the line segment into a new
    	       coordinate system centered on xs1, ys1 and rotated so the
    	       line segment lies along the x axis		     	  */
    	    CartToPolar((double)(x - xs1), (double)(y - ys1), &thetap, &rp);
    	    CartToPolar((double)(xs2 - xs1), (double)(ys2 - ys1), &thetas, &rs);
    	    segLength = rs;
    	    PolarToCart(thetap - thetas, rp, &xpPrime, &ypPrime);
    	    
    	    /* distance can now be figured as distance between point and
    	       the x axis (if the point is within the bounds of the segment),
    	       or the distance from the closer endpoint (if it is not within
    	       the bounds of the segment) */
    	    if (xpPrime < 0)
    	    	dist = sqrt(SQR(xpPrime) + SQR(ypPrime));
    	    else if (xpPrime > segLength)
    	    	dist = sqrt(SQR(xpPrime - segLength) + SQR(ypPrime));
    	    else
    	    	dist = fabs(ypPrime);
    	    } else {
    	    /* 
    	    ** Compute the distance at midpoint. Divide by five to 
    	    ** increase Pick probability..
    	    */
    	    xm12 = (xs1 + xs2)/2.; ym12 = (ys1 + ys2)/2.;
    	    dist = 0.2 *sqrt((x - xm12) * (x - xm12) + (y -ym12) * (y -ym12));
    	    }	
    	    /* remember the particle of minimum distance from the pick point */
    	    if (dist < minDist) {
    	    	minDist = dist;
    	    	minIndex = i;
    	    }
    	}
      }
      if (window->type == STDHEP_SPACE ) vert++;
    }
    /* 
    ** One now has to convert the Segment index to the track Index, 
    ** which depends on the current visibility of the choosen particle.
    */
    if (minDist > MIN_PICK_DIST) return NO_TRACK;
    else  return minIndex;
}

void ShowTrackStats(StdHepWindow *window, int trackNum)
{
    PhaseParticle *p;
    SpaceVertex *vert;
    char nameText[MAXCHAR_HEPNAM], statText[512], *stabilityText, userValueText[40];
    char titleText[40];
    XmString cStatText;
    double momentum, pt, rapidity, pseudorap, userValue;
    int outColor;
    double outVec[3];
    XmFontList fontList;
    PhaseWindow *winp = (PhaseWindow *) window;
    SpaceWindow *wins = (SpaceWindow *) window;
    
    if (trackNum == NO_TRACK) {
    	if (window->trackWindowShell)
    	    XtUnmapWidget(window->trackWindowShell);
    	return;
    }
    
    p = &window->event.particles[trackNum];
    hepnam_(&p->id, nameText, MAXCHAR_HEPNAM);
    *(char *)strchr(nameText, ' ') = '\0';
    sprintf(titleText, "%s (%d)", nameText, trackNum);  

    /* If greek font is available, use greek letters in particle name */
    GET_ONE_RSRC(window->statsLabel, XmNfontList, &fontList);
    if (CharsetAvailable("greek", fontList))
    	hepnmg_(&p->id, nameText);
    
    if (p->stable)
    	stabilityText = "stable";
    else
    	stabilityText = "unstable";
    	
    momentum = ParticleMomentum(p->px, p->py, p->pz);
    pt = ParticlePT(p->px, p->py);
    rapidity = ParticleRapidity(p->px, p->py, p->pz, p->mass);
    pseudorap = ParticlePseudorapidity(p->px, p->py, p->pz); 
/*     if (window->userMapProc) {
	(*window->userMapProc)(window->event, outVec, &outColor);
	userValue = sqrt(pow(outVec[0],2)+pow(outVec[1],2)+pow(outVec[2],2));
	sprintf(userValueText, "\nUser Value = %lf", userValue);
    } else {
    	sprintf(userValueText, "");
    } */
    
    /* In this version, no User text.. */
    sprintf(userValueText, "");
    sprintf(statText, "Particle: %s\nPx = %f\nPy = %f\nPz = %f\n\
Momentum = %f\nPt = %f\nRapidity = %f\nPseudoRapidity = %f%s",
	nameText, p->px, p->py, p->pz, momentum, pt, rapidity,
	pseudorap, userValueText);
    if (window->type == STDHEP_SPACE) { 
    /* Add one line to specify the vertex */
      vert = wins->vertices; vert += trackNum;
       sprintf(statText, "Particle: %s\nPx = %f\nPy = %f\nPz = %f\n\
Momentum = %f\nPt = %f\nRapidity = %f\nPseudoRapidity = %f\n\
\n\Origin: \nX = %f\nY = %f\nZ = %f\nLife = %f",
	nameText, p->px, p->py, p->pz, momentum, pt, rapidity,
	pseudorap,vert->x, vert->y, vert->z, vert->time);
    } 
    cStatText = MultiFontString(statText);

    if (window->trackWindowShell == NULL) {
    	createTrackInfoWindow(window, cStatText, titleText);
    } else {
    	SET_ONE_RSRC(window->trackWindowShell, XmNtitle, titleText);
    	SET_ONE_RSRC(window->trackWindowLabel, XmNlabelString, cStatText);
    	XtMapWidget(window->trackWindowShell);
    }
    XmStringFree(cStatText);
}

static void createTrackInfoWindow(StdHepWindow *window, XmString contents,
				  char *title)
{
    Arg args[50];
    int ac;
    Widget label, shell;
    short mainX, mainY, mainWidth, mainHeight;
    Atom wmpAtom, dwAtom;

    ac = 0;
    XtSetArg(args[ac], XmNx, &mainX); ac++;
    XtSetArg(args[ac], XmNy, &mainY); ac++;
    XtSetArg(args[ac], XmNwidth, &mainWidth); ac++;
    XtSetArg(args[ac], XmNheight, &mainHeight); ac++;
    XtGetValues(window->shell, args, ac);
    
    ac = 0;
    XtSetArg(args[ac], XmNx, mainX + mainWidth - 150); ac++;
    XtSetArg(args[ac], XmNy, mainY + mainHeight - 150); ac++;
#ifdef MOTIF10
    XtSetArg(args[ac], XmNallowShellResize, True); ac++;
    XtSetArg(args[ac], XmNwidth, 300); ac++;
    XtSetArg(args[ac], XmNheight, 300); ac++;
#endif
    XtSetArg(args[ac], XmNdeleteResponse, XmDO_NOTHING); ac++;
    XtSetArg(args[ac], XmNtitle, title); ac++;
    shell = XmCreateDialogShell(window->shell, "", args, ac);

    ac = 0;
    XtSetArg(args[ac], XmNalignment, XmALIGNMENT_BEGINNING); ac++;
    XtSetArg(args[ac], XmNlabelString, contents); ac++;
    label = XmCreateLabel(shell, "trackWindowShell", args, ac);

    /* Make close command in window menu gracefully prompt for close */
    wmpAtom = XmInternAtom(XtDisplay(window->shell), "WM_PROTOCOLS", TRUE);
    dwAtom  = XmInternAtom(XtDisplay(window->shell), "WM_DELETE_WINDOW", TRUE);
    XmAddProtocolCallback(shell, wmpAtom, dwAtom, closeCB, window);

    XtManageChild(label);
    XtManageChild(shell);

    window->trackWindowLabel = label;
    window->trackWindowShell = shell;
}   

static void closeCB(Widget w, StdHepWindow *window, caddr_t call_data)
{
    XtUnmapWidget(window->trackWindowShell);
}
