/* testPhase.c */
static char SCCSID[] = "@(#)testPhase.c	1.1	4/6/92";
#include <stdio.h>
#include <X11/Intrinsic.h>
#include "spin/Spin.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "drawEvent.h"

#define MAXPARTCOLOR 30

Pixel ParticleColors[MAXPARTCOLOR]; /* Size of the color table. */
double XPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
double YPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
double ZPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
StdHepWindow *WindowList = NULL;

extern PhaseEvent TestEvent;

extern void exit();

void main (int argc, char **argv)
{
    XtAppContext context;
    Display *display;
    
    /*
    ** Initialize toolkit and open display.
    */
    XtToolkitInitialize ();
    context = XtCreateApplicationContext();
    display = XtOpenDisplay (context, NULL, "phase", "Phase", NULL,
                                0, &argc, argv);
    if (!display) {
	XtWarning("phase: Can't open display");
	exit(0);
    }
    
    /*
    ** Create an event display panel
    */
    DisplayEventFile(display, exit, NULL, NULL, NULL);
    /*
    ** Process events.
    */
    XtAppMainLoop (context);
}

