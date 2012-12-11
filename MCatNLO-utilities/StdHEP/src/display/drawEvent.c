/*******************************************************************************
*									       *
* drawEvent.c -- Map the event - ie., the list of four-momentum vectors	to an  *
*		 array of points or segments for displaying on the spin widget *
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
* June 17, 1991, August 22, 1991  					       *
*									       *
* Written by Paul Lebrun & Mark Edel				               *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)drawEvent.c	1.1	4/6/92";
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <Xm/Xm.h>
#include "spin/Spin.h"
#include "util/stringUtils.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "drawEvent.h"

#define MAXPARTCOLOR 30		/* Size of particle color table */
#define EPSILON 0.1e-10

#define FULL_ON 0xffff
#define HALF_ON (0xffff/2)
#define OFF 0

#define NUMPINSPHI 8
#define NUMPINSTETA 7
#define MAXCHAR_HEPNAM 30  /* for use with hepnam_ */

extern float stdchg_(long *id);

static Pixel allocRGB(Screen *screen,
	unsigned short red, unsigned short green, unsigned short blue);
static Pixel particlePixel(int particleID);
static void detectorToSegment(SpaceWindow *window, SpinSegment *segdt, 
					SpinSegment *seg);

/*
** AllocateColors
**
** Generate the HEP color convention : This routines fills the 
** Global array ParticleColors, which is simply an list of color indices generated
** by calls to XAllocColor. 
**
** This convention is the following : 
**
** 	The photon is always black.  If a color can not be allocated, Black 
** 	is chosen.
**
** 	For GrayScale Display, the charged leptons are white, the other 
** 	neutrinos are black and the quarks and hadrons are gray. 
**
** 	For Color display,  the charged leptons have the three fundamental 
** 	RGB colors: electron is blue, muon is green and tau is red. 
** 	The neutrinos are gray. The quarks are characterised a mixture
** 	where one componenet of RGB is set to 0, the other two are 
** 	half of the full intensity.  For instance, the up quark has 
** 	no blue compoenet, and is half green, half red.
** 	The Weak bosons and the gluons are made of a disgusting 
** 	mixture of all RGB colors ( pink for the W andZ bosons, 
** 	brown for the gluons).
**	The color of a hadron is set the by color of the heaviest quark 
**	in this hadron.
*/
void AllocateColors(Screen *screen)
{
    Display *display = DisplayOfScreen(screen);
    Pixel black = BlackPixelOfScreen(screen);
    Pixel white = WhitePixelOfScreen(screen);
    Pixel color;
    int i;
    XVisualInfo info;

    /* As a fallback, initialize all particle colors to black */
    for (i=0; i<MAXPARTCOLOR; i++)
	ParticleColors[i]=black;

    /* Assign colors based on type of screen available */
    if (XMatchVisualInfo(display, XScreenNumberOfScreen(screen),
	    DefaultDepthOfScreen(screen), PseudoColor, &info)) {
	/*
	** Screen can handle PseudoColor (color lookup table)
	*/
	/* The Up quark */
	ParticleColors[1] = allocRGB(screen, FULL_ON, FULL_ON, OFF);

	/* The Down quark */
	ParticleColors[2] = allocRGB(screen, FULL_ON, HALF_ON, OFF);

	/* The Strange quark */
	ParticleColors[3] = allocRGB(screen, OFF, FULL_ON, FULL_ON);

	/* The Charm quark */
	ParticleColors[4] = allocRGB(screen, OFF, FULL_ON, HALF_ON);

	/* The B  quark */
	ParticleColors[5] = allocRGB(screen, FULL_ON, OFF, FULL_ON);

	/* The Top quark */
	ParticleColors[6] = allocRGB(screen, HALF_ON, OFF, FULL_ON);

	/* The electron  */
	ParticleColors[11] = allocRGB(screen, OFF, OFF, FULL_ON);

	/* The muon */
	ParticleColors[13] = allocRGB(screen, OFF, FULL_ON, OFF);

	/* The Tau  */
	ParticleColors[15] = allocRGB(screen, FULL_ON, OFF, OFF);

	/* Neutrinos  */
	color = allocRGB(screen, FULL_ON/3, FULL_ON/3, FULL_ON/3);
	ParticleColors[16] = ParticleColors[14] = ParticleColors[12] = color;

	/* The photon quark */
	ParticleColors[22] = black;

	/* The W boson */
	ParticleColors[24] = allocRGB(screen, FULL_ON, HALF_ON, HALF_ON);

	/* The Z Boson */
	ParticleColors[23] = allocRGB(screen, FULL_ON, FULL_ON, HALF_ON);

	/* Gluons */
	color = allocRGB(screen, FULL_ON, HALF_ON, 2*FULL_ON/3);
	ParticleColors[21] = ParticleColors[9] = color;

    } else if (XMatchVisualInfo(display, XScreenNumberOfScreen(screen),
	    DefaultDepthOfScreen(screen), GrayScale, &info)) {
	/*
	** Screen can't handle color but can handle GrayScale
	*/
	/* Quarks */
	color = allocRGB(screen, OFF, FULL_ON, FULL_ON);
	for (i=1; i<7; i++)
	    ParticleColors[i] = color;
	ParticleColors[9] = color;
	ParticleColors[21]= color;

	/* Leptons */
	ParticleColors[11] = white;
	ParticleColors[13] = white;
	ParticleColors[15] = white;
    }
}
/*
** Produce a bunch of Radial coordinates to build a small icon describing the 
** this vertex.
*/
void SetPinVertex()
{
  int i,j;
  double pi, dphi, dteta, phi, teta;
  
  pi = acos(-1.0);
  dphi = 2.0 * pi / NUMPINSPHI; dteta = pi / NUMPINSTETA;
  for (i=0, phi = 0. ; i < NUMPINSTETA; i++, phi += dphi) {
    for (j=0, teta = 0. ; j < NUMPINSPHI; j++, teta += dteta) {
      XPinVertexPoints[j][i] = sin(teta) * cos(phi);
      YPinVertexPoints[j][i] = sin(teta) * sin(phi);
      ZPinVertexPoints[j][i] = cos(teta);
    }
  }
}

/*
** DrawEvent
**
** Map an event to 3D segments for the spin widget.  From a 4-momentum vector,
** (i.e., the 3-momentum vector and the unique identifier from which a physical
** mass can be obtained), An array of 3-dimensional vector and track type is
** deduced).  The structure PhaseEvent is described in phase.h.  The setting
** of the displayMode toggle on the event display panel determines how the
** event is mapped onto the screen:
** 
** BY_MOMENTUM:		The length is proportional the  |P|, the norm of the 3-d
**		      	momentum vector. This is the simplest and default option.
**
** BY_PT:	 	The length of the vector is proportional to Pt, the 
**			transverse momentum. This trick greatly enhances 
**			the "high Pt" Physics in the event.
**
** BY_RAPIDITY: 	The length of the vector is propotional to the rapidity 
**			of the particle, as defined in the PDG bible.
**
** BY_PSEUDORAPIDITY:	The length of the vector is proportional to the pseudo
**			rapidity, i.e., the mass of the particle is 
**			neglected.
**
** BY_USER_ROUTINE :	The user function specified in the DisplayEvent call
**			is called to determine what the new vector ought to be... 
*/
void DrawEvent(StdHepWindow *window, int setScale)
{
    int i, maxIndex, nSegments, nns, npvis;
    int doPoints;
    double scale, innerScale, length, maxLength, outVec[3];
    SpinSegment *segment, *segments, *segdt, temp;
    SpinPoint *point, *points;
    PhaseParticle *p, *particles = window->event.particles;
    int nStable = 0, nParticles = window->event.nParticles;
    SpaceVertex *pinv, *v;
    char nameText[MAXCHAR_HEPNAM], statText[255];
    XmString cStatText;
    /* There is a very wierd Ultrix compiler problem concerning this statement:*/
    /*    PhaseEvent *event; */
    static char *unitText[6] = {"p", "Pt", "Eta", "Eta", "value", "Eta-Pt"};
    XmFontList fontList;
    PhaseWindow *winp;
    SpaceWindow *wins;
    ParaWindow *winpa;
    
    /* If a Space window, find out the number of vertices in this event, 
       so we can allocate more space to draw short segments to represent 
       a vertex.. */
       
    if (window->type == STDHEP_SPACE) {
       wins = (SpaceWindow *) window;
    /* allocate space for the maximum possible number of segments or points */
       nns = NUMPINSTETA * NUMPINSPHI * wins->numRealVertices + nParticles;
       nns += wins->nDetectorSegments;
       segments = (SpinSegment *)XtMalloc(sizeof(SpinSegment) * nns);
       doPoints = False;
     } else if (window->type == STDHEP_PARA) {
        winpa = (ParaWindow *) window;
	segments = (SpinSegment *)XtMalloc(sizeof(SpinSegment) * nParticles);
	doPoints = False;
     } else {
        winp =(PhaseWindow *) window;
        doPoints = (winp->vectorLength == 0.);
        if (doPoints)
	    points = (SpinPoint *)XtMalloc(sizeof(SpinPoint) * nParticles);
        else
	   segments = (SpinSegment *)XtMalloc(sizeof(SpinSegment) * nParticles);
     }
    
    /* Loop over all of the particles, translating them into 3-D line segments
       or points for the Spin widget.  Also find the particle of maximum value
       (for setting the scale and printing statistics), and the number of stable
       particles (for printing statistics later) */
    nSegments = 0;
    npvis = 0;
    maxLength = -1.;
    if (doPoints) {
	point = points;
	for (i=0, p=particles; i<nParticles; i++, p++) {
    	    if (p->stable)
    		nStable++;
    	    if (ParticleToSegment(winp, p, &temp, &length)) {
    	        point->x = temp.x1;
    	        point->y = temp.y1;
    	        point->z = temp.z1;
    	        point->pixel = temp.pixel;
    		point++;
    		nSegments++;
    		npvis++;
    		if (length > maxLength) {
    		    maxLength = length;
    		    maxIndex = i;
    		}
    	    }
	}
	/* Display the points */
	SpinSetPoints(window->spin, points, nSegments);
	XtFree((char *) points);
    } else {
      if (window->type == STDHEP_PHASE) {
	segment=segments;
	for (i=0, p=particles; i<nParticles; i++, p++) {
    	    if (p->stable)
    		nStable++;
    	    if (ParticleToSegment(winp, p, segment, &length)) {
    		segment++;
    		nSegments++;
    		npvis++;
    		if (length > maxLength) {
    		    maxLength = length;
    		    maxIndex = i;
    		}
    	    }
	}
       } else if (window->type == STDHEP_PARA) {
	 segment=segments;
	 for (i=0, p=particles; i<nParticles; i++, p++) {
    	    if (p->stable)
    		nStable++;
    	    if (ParticleToParaSegment(winpa, p, segment, &length)) {
    		segment++;
    		nSegments++;
    		npvis++;
    		if (length > maxLength) {
    		    maxLength = length;
    		    maxIndex = i;
    		}
    	    }
    	  }
       } else {            
	   segment=segments;
           maxLength = 0.;
	   for (i=0, p=particles, v=wins->vertices; 
	                     i<nParticles; i++, p++, v++) {
    	    if (p->stable)
    		nStable++;
    	    if (TrackToSegment(wins, p, v, segment, &length)) {
    		segment++;
    		nSegments++;
    		npvis++;
    		if (length > maxLength) {
    		    maxLength = length;
    		    maxIndex = i;
    		}
    	    }
	   }
	   if (wins->nDetectorSegments > 0) {
	    for (i=0, segdt = wins->detectorSegments;
	          i < wins->nDetectorSegments;
	           i++, segdt++, segment++, nSegments++)  
	          detectorToSegment(wins, segdt, segment);
	   } 
	   if (wins->showVertices) {
	   for (i=0, pinv=wins->realVertices;
	         i < wins->numRealVertices; i++, pinv++){
	          if (fabs((double) pinv->z) < wins->currentScale) {
	            VertexToSegment(wins, pinv, &segment);
	             nSegments += NUMPINSPHI *NUMPINSTETA;
	          } 
             }
           }
         }
	}    
	/* Display the line segments with the spin widget */
	SpinSetSegments(window->spin, segments, nSegments);
	XtFree((char *) segments);
	/* 
	* Currently, the spin widget Perspective resource is buggy, 
	* turn if off.. 
	*/
        SET_ONE_RSRC(window->spin, XmNperspectiveOn, FALSE);
    
    /* If requested, set the scale to show the data most effectively */
    if (setScale) {
    	SpinSetScale(window->spin, (maxLength > 0.) ? 1./maxLength : 1.);
    	DrawScale(window);
    }
    
    /* Fill in the statistics window */
    if (npvis == 0) {
    	sprintf(statText, "%d tracks, %d are stable.\nNo Tracks are visible",
    		nParticles, nStable);
    } else {
        if ((window->type == STDHEP_PHASE) || (window->type == STDHEP_PARA)) { 
	    GET_ONE_RSRC(window->statsLabel, XmNfontList, &fontList);
	    if (CharsetAvailable("greek", fontList)) {
    	        hepnmg_(&particles[maxIndex].id, nameText);
	    } else {
    	        hepnam_(&particles[maxIndex].id, nameText, MAXCHAR_HEPNAM);
	         *strchr(nameText, ' ') = '\0';
	    }
	    if (window->type == STDHEP_PHASE) {
	    sprintf(statText,
"%d tracks, %d are stable,\n%d are now displayed\nTrack of max. %s is a(n) %s",
    	        nParticles, nStable, nSegments,
    	        unitText[winp->displayMode], nameText);
    	    } else { 
	    sprintf(statText,
"%d tracks, %d are stable,\n%d are now displayed\nTrack of max. %s is a(n) %s",
    	        nParticles, nStable, nSegments,
    	        unitText[5], nameText);
    	    }
    	  } else {
    	    if (wins->showVertices)
    	      nns = nSegments
    	         - (NUMPINSPHI * NUMPINSTETA * wins->numRealVertices);
    	    else nns = nSegments; 
	    sprintf(statText,
"%d tracks, %d are stable,\n%d are now displayed\n %d distinct vertices found",
    	        nParticles, nStable, nns, wins->numRealVertices);
    	  }
    }
    cStatText = MultiFontString(statText);
    SET_ONE_RSRC(window->statsLabel, XmNlabelString, cStatText);
    XmStringFree(cStatText);
}

int ParticleToSegment(PhaseWindow *window, PhaseParticle *p,
		      SpinSegment *seg, double *length)
{
    int outColor;
    double scale, innerScale, outVec[3];

    if (!ParticleVisible((StdHepWindow *) window, p->id, p->stable))
    	return False;

    switch (window->displayMode) {
      case BY_MOMENTUM:
	*length = ParticleMomentum(p->px, p->py, p->pz);
	break;
      case BY_PT:
	*length = ParticlePT(p->px, p->py);
	break;
      case BY_RAPIDITY:
	*length = ParticleRapidity(p->px, p->py, p->pz, p->mass);
	break;
      case BY_PSEUDORAPIDITY:
	*length = ParticlePseudorapidity(p->px, p->py, p->pz);
	break;
/*       case BY_USER_ROUTINE:
	(*window->userMapProc)(window->event, outVec, &outColor);
	*length = sqrt(pow(outVec[0],2) + pow(outVec[1],2) + pow(outVec[2],2));
	break; */
    }
    if (window->displayMode == BY_USER_ROUTINE) {
    	seg->x1 = outVec[0];
    	seg->y1 = outVec[1];
    	seg->z1 = outVec[2];
    	seg->x2 = outVec[0] * (1. - window->vectorLength);
    	seg->y2 = outVec[0] * (1. - window->vectorLength);
    	seg->z2 = outVec[0] * (1. - window->vectorLength);
    	seg->pixel = ParticleColors[outColor];
    } else {
    	scale = *length / ParticleMomentum(p->px, p->py, p->pz);
    	innerScale = scale * (1. - window->vectorLength);
    	seg->x1 = p->px * scale;
    	seg->y1 = p->py * scale;
    	seg->z1 = p->pz * scale;
    	seg->x2 = p->px * innerScale;
    	seg->y2 = p->py * innerScale;
    	seg->z2 = p->pz * innerScale;
    	seg->pixel = particlePixel(p->id);
    }
    return True;
}

int ParticleToParaSegment(ParaWindow *window, PhaseParticle *p,
		      SpinSegment *seg, double *length)
{
    if (!ParticleVisible((StdHepWindow *) window, p->id, p->stable))
    	return False;

     seg->x1 = 0.;
     seg->y1 = 0.;
     seg->x2 = p->px;
     seg->y2 = p->py;
     if (p->pz < 0.) 
         seg->z1 = window->currentTranslz - 
               ( ParticleRapidity(p->px, p->py, p->pz, p->mass) * 
                window->currentRapToPt);
     else 
         seg->z1 = window->currentTranslz + 
               ( ParticleRapidity(p->px, p->py, p->pz, p->mass) * 
                window->currentRapToPt);
     seg->z2 = seg->z1;
     seg->pixel = particlePixel(p->id);
     *length = ParticlePT(p->px, p->py);
    return True;
}

int TrackToSegment(SpaceWindow *window,  PhaseParticle *p, SpaceVertex *v,
		      SpinSegment *seg, double *length)
{
	double scaleZ, scaleT;
	
        if (!ParticleVisible((StdHepWindow *)window, p->id, p->stable))
    	return False;
	
	scaleZ = window->currentMomToSpace;
	scaleT = window->currentMomToSpace * window->currentLongToTr;
	seg->x1 = window->currentLongToTr * v->x;
	seg->y1 = window->currentLongToTr * v->y;
	seg->z1 = v->z;
	/*
	** Check now that we are in the visible range
	**
	*/
	if (fabs(seg->z1) > window->currentScale) return False;
    	/*
    	** Now translate 
    	*/
    	seg->x1 = seg->x1 + window->currentLongToTr * window->currentTransl[0];
    	seg->y1 = seg->y1 + window->currentLongToTr * window->currentTransl[1];
    	seg->z1 = seg->z1 + window->currentTransl[2];
    	/* 
    	** The end of the tracks is set by the momentum
    	*/
	seg->x2 = seg->x1 + scaleT * p->px;
	seg->y2 = seg->y1 + scaleT * p->py;
	seg->z2 = seg->z1 + scaleZ * p->pz;
	*length = sqrt ((seg->x2 * seg->x2) + (seg->y2 * seg->y2) + 
			 (seg->z2 * seg->z2));
    	seg->pixel = particlePixel(p->id);
	return True;
}

void VertexToSegment(SpaceWindow *window,  SpaceVertex *v,
		      SpinSegment **seg)
{
	double scaleZ, scaleT;
	int i,j;
	SpinSegment *segt;
	Pixel black;
        Screen *screen;
        
        screen = XtScreen(window->shell);
        black = BlackPixelOfScreen(screen);	 
	scaleZ = PINSIZE * window->maxTransl[2];
/* 	scaleT = PINSIZE * window->maxTransl[0]; */
	segt = *seg;
        for (i=0; i < NUMPINSTETA; i++) {
          for (j=0; j < NUMPINSPHI; j++) {
	    segt->x1 = window->currentLongToTr * v->x;
	    segt->y1 = window->currentLongToTr * v->y;
	    segt->z1 = v->z;
    	/*
    	** Now translate 
    	*/
    	    segt->x1 = segt->x1 + 
    	       window->currentLongToTr * window->currentTransl[0];
    	    segt->y1 = segt->y1 + 
    	       window->currentLongToTr * window->currentTransl[1];
    	    segt->z1 = segt->z1 + 
    	       window->currentTransl[2];
    	    /*
    	    ** The end of the pins.. 
    	    */
	    segt->x2 = segt->x1 + scaleZ * XPinVertexPoints[j][i];
	    segt->y2 = segt->y1 + scaleZ * YPinVertexPoints[j][i];
	    segt->z2 = segt->z1 + scaleZ * ZPinVertexPoints[j][i];
	    segt->pixel = black;
	    segt++;
	    }
	 }
	 *seg = segt;
}

static void detectorToSegment(SpaceWindow *window, SpinSegment *segdt, 
					SpinSegment *seg)
{
	int i,j;
	Pixel black;
        Screen *screen;
        
        screen = XtScreen(window->shell);
        black = BlackPixelOfScreen(screen);	 
	seg->pixel = black;
	seg->x1 = window->currentLongToTr * segdt->x1;
	seg->y1 = window->currentLongToTr * segdt->y1;
	seg->z1 = segdt->z1;
	seg->x2 = window->currentLongToTr * segdt->x2;
	seg->y2 = window->currentLongToTr * segdt->y2;
	seg->z2 = segdt->z2;
    	/*
    	** Now translate 
    	*/
    	seg->x1 = seg->x1 + 
    	       window->currentLongToTr * window->currentTransl[0];
    	seg->y1 = seg->y1 + 
    	       window->currentLongToTr * window->currentTransl[1];
    	seg->z1 = seg->z1 + 
    	       window->currentTransl[2];
    	seg->x2 = seg->x2 + 
    	       window->currentLongToTr * window->currentTransl[0];
    	seg->y2 = seg->y2 + 
    	       window->currentLongToTr * window->currentTransl[1];
    	seg->z2 = seg->z2 + 
    	       window->currentTransl[2];
     	return;
}


	
void SetDisplayMode(PhaseWindow *window, int newMode)
{
    int changeScale, oldMode = window->displayMode;
    
    changeScale = (((newMode == BY_MOMENTUM || newMode == BY_PT) &&
    	  (oldMode != BY_MOMENTUM && oldMode != BY_PT)    ) ||
         ((newMode == BY_RAPIDITY || newMode == BY_PSEUDORAPIDITY) &&
          (oldMode != BY_RAPIDITY && oldMode != BY_PSEUDORAPIDITY)    ) ||
         (newMode == BY_USER_ROUTINE && oldMode != BY_USER_ROUTINE));
    window->displayMode = newMode;
    DrawEvent((StdHepWindow *) window, changeScale);
}

void DrawScale(StdHepWindow *window)
{
    Widget scaleWidget = window->scaleArea;
    Window scaleWindow = XtWindow(scaleWidget);
    Display *display = XtDisplay(scaleWidget);
    double scale, scaleAreaWidth, scaleBarLength = 1., barMultiplier = 2.;
    double tempMom;
    short widgetWidth, spinWidth, widgetHeight;
    int leftX, rightX, topY, midY, bottomY;
    GC gc = window->scaleGC;
    char scaleText[100];
    XmString cScaleText;
    SpaceWindow *wins;
    double scaleVertex;
    PhaseWindow *winp;
    ParaWindow *winpa;

    /* Get the scale factor for translating spin coordinates to pixels */
    GET_ONE_RSRC(window->spin, XmNwidth, &spinWidth);
    scale = (1./(SpinGetScale(window->spin)/2.)) / (double)spinWidth;
    
    /* Calculate the width of the scale draw area in spin coordinates */
    GET_ONE_RSRC(scaleWidget, XmNwidth, &widgetWidth);
    scaleAreaWidth = widgetWidth * scale;
    
    /* choose a sensible length for the scale bar, such as 1, 5, 10, 50 etc. */
    if (scaleAreaWidth < 1.) {
    	while (scaleBarLength > scaleAreaWidth) {
    	    scaleBarLength /= barMultiplier;
    	    barMultiplier = (barMultiplier == 2.) ? 5. : 2.;
    	}
    } else {
    	while (scaleBarLength < scaleAreaWidth) {
    	    barMultiplier = (barMultiplier == 2.) ? 5. : 2.;
    	    scaleBarLength *= barMultiplier;
    	}
    	scaleBarLength /= barMultiplier;
    }
    
    /* draw the scale bar in the middle of the scale area widget */
    GET_ONE_RSRC(scaleWidget, XmNheight, &widgetHeight);
    leftX = ((scaleAreaWidth - scaleBarLength) / 2.) / scale;
    rightX = leftX + scaleBarLength / scale;
    topY = widgetHeight/4;
    midY = widgetHeight/2;
    bottomY = midY + widgetHeight/4;
    XClearWindow(display, scaleWindow);
    XDrawLine(display, scaleWindow, gc, leftX, topY, leftX, bottomY);
    XDrawLine(display, scaleWindow, gc, rightX, topY, rightX, bottomY);
    XDrawLine(display, scaleWindow, gc, leftX, midY, rightX, midY);
    
    /* put in the text to label the scale */
    if (window->type == STDHEP_PHASE) {
      winp = (PhaseWindow *) window;
      if (winp->displayMode == BY_MOMENTUM || winp->displayMode == BY_PT)
    	sprintf(scaleText, "%G Gev/c", scaleBarLength);
      else
    	sprintf(scaleText, "%G Unit Eta", scaleBarLength);
    } else if (window->type == STDHEP_PARA) {
      winpa = (ParaWindow *) window;
      scaleVertex = scaleBarLength / winpa->currentRapToPt;
      sprintf (scaleText, "%G eta = %G Gev/c",scaleBarLength, scaleVertex);
    
    } else {
      wins = (SpaceWindow *) window;
      tempMom = wins->currentMomToSpace;
      if( tempMom == 0.0) tempMom = EPSILON;
      scaleVertex = scaleBarLength / tempMom;
      sprintf (scaleText, "%G cm = %G Gev/c",scaleBarLength, scaleVertex);
    } 
    cScaleText = MKSTRING(scaleText);
    SET_ONE_RSRC(window->scaleLabel, XmNlabelString, cScaleText);
    XmStringFree(cScaleText);
}


static Pixel allocRGB(Screen *screen,
	unsigned short red, unsigned short green, unsigned short blue)
{
    Display *display = DisplayOfScreen(screen);
    XColor color;
    
    color.red = red;
    color.green = green;
    color.blue = blue;
    if (XAllocColor(display, DefaultColormapOfScreen(screen), &color))
    	return color.pixel;
    else 
     	return BlackPixelOfScreen(screen);
}

int ParticleVisible(StdHepWindow *window, long id, int stable)
{
    long absID = abs(id);
    float charge = stdchg_(&id);
    
    if (stable && !window->stable)
    	return False;
    if (!stable && !window->unstable)
   	return False;
   	
    if ((fabs(charge) < 0.1) && !window->neutral)
    	return False;
    if ((fabs(charge) > 0.1) && !window->charged)
   	return False;
     
    if (absID == 11 && !window->electrons)
	return False;
    if (absID == 13 && !window->muons)
	return False;
    if ((absID == 12 || absID == 14 || absID == 16) && !window->neutrinos)
	return False;
    if ((absID <=  6 || absID == 9 || absID == 21) && !window->quarks)
	return False;
    if ((absID == 23 || absID == 24) && !window->wz)
	return False; 
    if (absID == 22 && !window->gammas)
	return False;
    if (absID >= 100 && !window->hadrons)
        return False;
    return True;
}

static Pixel particlePixel(int particleID)
{
    return ParticleColors[ParticleColorIndex(particleID)];
}

int ParticleColorIndex(int particleID)
{
    long absID = abs(particleID);
    long i, aid, kq[3], kqj;
    double akqx, akq3, akq2, akq1, akqj; 
    
    /*
    ** For non hadrons, return the particle id as the color
    */
    if (absID < 100) {
    	if (absID > (MAXPARTCOLOR - 1))
    	    return MAXPARTCOLOR - 1;
    	return absID;
    }
    
    /*
    ** For hadrons, return the largest digit? (ask paul...)
    */
    akqj = ((double)absID)/10.;

    aid = (long)akqj;
    akq1 = akqj/10.;
    kq[0] = aid - ((long)akq1) * 10;

    aid = (long)akq1;
    akq2 = akq1/10.;
    kq[1] = aid - ((long)akq2) * 10;

    aid = (long)akq2;
    akq3 = akq2/10.;
    kq[2] = aid - ((long)akq3) * 10;

    kqj = 0;
    for (i=0; i<3; i++)
        if (kq[i] > kqj) kqj = kq[i];  

    if (kqj > (MAXPARTCOLOR-1))
    	return MAXPARTCOLOR-1;
    else
    	return kqj;
}

double ParticleMomentum(double px, double py, double pz)
{
    double pTmp = sqrt(SQR(px) + SQR(py) + SQR(pz));
    if (pTmp == 0.0) return EPSILON;
    else return pTmp;
}

double ParticlePT(double px, double py)
{
    return sqrt(SQR(px) + SQR(py));
}

double ParticleRapidity(double px, double py, double pz, double mass)
{
    double p, pt, e;
    
    p = ParticleMomentum(px, py, pz);
    e = sqrt(p*p + mass*mass);
    return fabs(0.5*log(fabs((e+pz+0.0001)/(e-pz+0.0000001))));
}

double ParticlePseudorapidity(double px, double py, double pz)
{
    double p, pt, teta;
    
    p = ParticleMomentum(px, py, pz);
    pt = ParticlePT(px,py);
    teta = acos(pz/(p+ 0.00001));
    return fabs(log(fabs(tan(0.5*teta))));
}
