/*******************************************************************************
*									       *
* Spin.c -- 3D "virtual sphere" display widget				       *
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
* July 22, 1991								       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)Spin.c	1.5	8/31/92";
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <sys/time.h>
#include <X11/Xlib.h>
#include <X11/Intrinsic.h>
#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>
#include <Xm/Xm.h>
#include <Xm/XmP.h>
#include "geometry.h"
#include "SpinP.h"
#include "../util/psUtils.h"

#define MIN_REFRESH_TIME 33	/* How often, ideally, to refresh animated
				   rotation in msec (33msec = (1/30) second)  */
#define NO_ROTATE_RADIUS .8	/* radius inside which to damp Z rotation     */
#define MIN_TOSS_SPEED 10 	/* min mouse "toss" to start auto-rotation
				   in degrees/second			      */
#define KEY_SPEED_CHANGE 10	/* deg/sec rotatn adjustment for shift+arrow  */
#define PERSPECTIVE_FACTOR .3	/* How much perspecive to apply		      */

enum direction {LEFT, RIGHT, UP, DOWN};

#define RADIANS(x)  (M_PI * 2.0 * (x) / 360.0)
#define DEGREES(x)  ((x) / (M_PI * 2.0) * 360.0)

static void motionAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void btnUpAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void btn2AP(SpinWidget w, XEvent *event, char *args, int n_args);
static void btn3AP(SpinWidget w, XEvent *event, char *args, int n_args);
static void spinUpAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void spinDownAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void spinLeftAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void spinRightAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void stopAP(SpinWidget w, XEvent *event, char *args, int n_args);
static void keySpin(SpinWidget w, XEvent *event, int direction);
static void rotateProc(SpinWidget w, XtIntervalId *id);
static void initialize(SpinWidget request, SpinWidget new);
static void redisplay(SpinWidget w, XEvent *event, Region region);
static void redisplayContents(SpinWidget w);
static void redrawContents(SpinWidget w, int outDevice);
static void destroy(SpinWidget w);
static void resize(SpinWidget w);
static Boolean setValues(SpinWidget current, SpinWidget request,SpinWidget new);
static double normalizeAngle(double angle);
static void drawAxes(SpinWidget w, Drawable drawBuf, int outDevice);
static void transform3DtoWindow(SpinWidget w, double x, double y, double z, 
				double m[3][3], short *xOut, short *yOut);
static int compareSegments(SpinSegment *seg1, SpinSegment *seg2);
static int comparePoints(SpinPoint *pt1, SpinPoint *pt2);
static void updateBufferAllocation(SpinWidget w);

static char defaultTranslations[] = 
    "<Btn1Motion>: Motion()\n\
     <Btn1Down>: Motion()\n\
     <Btn1Up>: BtnUp()\n\
     <Btn2Down>: Btn2Press()\n\
     <Btn3Down>: Btn3Press()\n\
     <Key>Return: StopSpin()\n\
     <Key>space: StopSpin()\n\
     <Key>k: SpinUp()\n\
     <Key>j: SpinDown()\n\
     <Key>h: SpinLeft()\n\
     <Key>l: SpinRight()\n\
     <Key>Up: SpinUp()\n\
     <Key>Down: SpinDown()\n\
     <Key>Left: SpinLeft()\n\
     <Key>Right: SpinRight()\n";

static XtActionsRec actionsList[] = {
    {"Motion", (XtActionProc)motionAP},
    {"BtnUp", (XtActionProc)btnUpAP},
    {"Btn2Press", (XtActionProc)btn2AP},
    {"Btn3Press", (XtActionProc)btn3AP},
    {"StopSpin", (XtActionProc)stopAP},
    {"SpinLeft", (XtActionProc)spinLeftAP},
    {"SpinRight", (XtActionProc)spinRightAP},
    {"SpinUp", (XtActionProc)spinUpAP},
    {"SpinDown", (XtActionProc)spinDownAP}
};

static XtResource resources[] = {
    {XmNkeyRotateDegrees, XmCKeyRotateDegrees, XmRInt, sizeof(int),
      XtOffset(SpinWidget, spin.keyRotateDegrees), XmRString, "10"},
    {XmNperspectiveOn, XmCPerspectiveOn, XmRBoolean, sizeof(Boolean),
      XtOffset(SpinWidget, spin.perspectOn), XmRString, "False"},
    {XmNdoubleBuffer, XmCDoubleBuffer, XmRBoolean, sizeof(Boolean),
      XtOffset(SpinWidget, spin.doubleBuffer), XmRString, "False"},
    {XmNshowAxes, XmCShowAxes, XmRBoolean, sizeof(Boolean),
      XtOffset(SpinWidget, spin.showAxes), XmRString, "True"},
    {XmNfontList, XmCFontList, XmRFontList, sizeof(XmFontList),
      XtOffset(SpinWidget, spin.font), XmRImmediate, NULL},
    {XmNxAxisLabel, XmCXAxisLabel, XmRXmString, sizeof (XmString), 
      XtOffset(SpinWidget, spin.axisLabels[0]), XmRString, "X"},
    {XmNyAxisLabel, XmCYAxisLabel, XmRXmString, sizeof (XmString), 
      XtOffset(SpinWidget, spin.axisLabels[1]), XmRString, "Y"},
    {XmNzAxisLabel, XmCZAxisLabel, XmRXmString, sizeof (XmString), 
      XtOffset(SpinWidget, spin.axisLabels[2]), XmRString, "Z"},
    {XmNresizeCallback, XmCCallback, XmRCallback, sizeof(caddr_t),
      XtOffset (SpinWidget, spin.resize), XtRCallback, NULL},
    {XmNbtn2Callback, XmCCallback, XmRCallback, sizeof(caddr_t),
      XtOffset (SpinWidget, spin.btn2), XtRCallback, NULL},
    {XmNbtn3Callback, XmCCallback, XmRCallback, sizeof(caddr_t),
      XtOffset (SpinWidget, spin.btn3), XtRCallback, NULL},
    {XmNredisplayCallback, XmCCallback, XmRCallback, sizeof(caddr_t),
      XtOffset (SpinWidget, spin.redisplay), XtRCallback, NULL},
};

SpinClassRec  spinClassRec = {
     /* CoreClassPart */
  {
    (WidgetClass) &xmPrimitiveClassRec,  /* superclass            */
    "Spin",                         /* class_name            */
    sizeof(SpinRec),                /* widget_size           */
    NULL,                           /* class_initialize      */
    NULL,                           /* class_part_initialize */
    FALSE,                          /* class_inited          */
    (XtInitProc) initialize,        /* initialize            */
    NULL,                           /* initialize_hook       */
    XtInheritRealize,               /* realize               */
    actionsList,                    /* actions               */
    XtNumber(actionsList),          /* num_actions           */
    resources,                      /* resources             */
    XtNumber(resources),            /* num_resources         */
    NULLQUARK,                      /* xrm_class             */
    TRUE,                           /* compress_motion       */
    TRUE,                           /* compress_exposure     */
    TRUE,                           /* compress_enterleave   */
    TRUE,                           /* visible_interest      */
    (XtWidgetProc) destroy,         /* destroy               */
    (XtWidgetProc) resize,          /* resize                */
    (XtExposeProc) redisplay,       /* expose                */
    (XtSetValuesFunc) setValues,    /* set_values            */
    NULL,                           /* set_values_hook       */
    XtInheritSetValuesAlmost,       /* set_values_almost     */
    NULL,                           /* get_values_hook       */
    NULL,                           /* accept_focus          */
    XtVersion,                      /* version               */
    NULL,                           /* callback private      */
    defaultTranslations,            /* tm_table              */
    NULL,                           /* query_geometry        */
    NULL,                           /* display_accelerator   */
    NULL,                           /* extension             */
  },
  /* Motif primitive class fields */
  {
     (XtWidgetProc) _XtInherit,         /* Primitive border_highlight   */
     (XtWidgetProc) _XtInherit,         /* Primitive border_unhighlight */
      XtInheritTranslations,		/* translations                 */
     (XtActionProc)motionAP,		/* arm_and_activate             */
     NULL,				/* get resources      		*/
     0,					/* num get_resources  		*/
     NULL,         			/* extension                    */
  },
  /* Spin class part */
  {
    0,                              	/* ignored	                */
  }
};

WidgetClass spinWidgetClass = (WidgetClass) &spinClassRec;

/*
** Widget initialize method
*/
static void initialize(SpinWidget request, SpinWidget new)
{
    XGCValues values;
    int i;
    XFontStruct *fs;
   
    /* Make sure the window size is not zero. The Core 
       initialize() method doesn't do this. */
    if (request->core.width == 0)
    	new->core.width = 100;
    if (request->core.height == 0)
   	new->core.height = 100;

    /* Make a local copy of the font list, or get the default if not specified */
    if (new->spin.font == NULL)
#ifdef MOTIF10
	new->spin.font = XmFontListCreate(
	    XLoadQueryFont(XtDisplay(new), "fixed"),
	    XmSTRING_DEFAULT_CHARSET);
#else
    	new->spin.font = XmFontListCopy(
    		_XmGetDefaultFontList((Widget) new, XmLABEL_FONTLIST));
#endif
    else
        new->spin.font = XmFontListCopy(new->spin.font);

    /* Create the graphics contexts */
#ifdef MOTIF10
    fs = new->spin.font->font;
#else
    _XmFontListGetDefaultFont(new->spin.font, &fs);
#endif
    values.font = fs->fid;
    values.foreground = new->primitive.foreground;
    values.background = new->core.background_pixel;
    new->spin.gc = XCreateGC(XtDisplay(new), XDefaultRootWindow(XtDisplay(new)),
    			     GCForeground|GCBackground|GCFont, &values);  

    /* Make local copies of the XmStrings */
    for (i=0; i<3; i++) {
	if (new->spin.axisLabels[i] != NULL)
    	    new->spin.axisLabels[i] = XmStringCopy(new->spin.axisLabels[i]);
    }

    /* Initialize the transformation matrix and scale */
    CopyM(Identity3x3, new->spin.matrix);
    new->spin.scale = 1.;

    /* Set size dependent items */
    new->spin.drawBuffer = NULL;
    resize(new);

    /* Initialize the drawing */
    new->spin.nSegments = 0;
    new->spin.nPoints = 0;
    
    /* No drag or spin operation started yet */
    new->spin.dragging = False;
    new->spin.spinning = False;

    /* miscellaneous items */
    new->spin.lastX = 0.;
    new->spin.lastY = 0.;
    new->spin.lastAzimuth = 0.;
    new->spin.lastTimeStamp = 0;
    new->spin.axialCorr = 0.;
}

/*
** Widget destroy method
*/
static void destroy(SpinWidget w)
{
    int i;
    
    if (w->spin.spinning)
    	SpinStopSpinning((Widget)w);
    for (i=1; i<3; i++)
    	XmStringFree(w->spin.axisLabels[i]);
    XFreeGC(XtDisplay(w), w->spin.gc);
    if (w->spin.font != NULL)
    	XmFontListFree(w->spin.font);
    XtRemoveAllCallbacks ((Widget) w, XmNresizeCallback);
    XtRemoveAllCallbacks ((Widget) w, XmNbtn2Callback);
    XtRemoveAllCallbacks ((Widget) w, XmNbtn3Callback);
    XtRemoveAllCallbacks ((Widget) w, XmNredisplayCallback);
}

/*
** Widget resize method
*/
static void resize(SpinWidget w)
{
    XRectangle clipRect;
    int borderWidth =
    	w->primitive.shadow_thickness+w->primitive.highlight_thickness;
    
    /* calculate the center of the widget */
    w->spin.centerX = w->core.width/2; 
    w->spin.centerY = w->core.height/2;
    w->spin.drawWidth = w->core.width - 2 * borderWidth;
    /* resize the drawing buffer, an offscreen pixmap for smoother animation */
    updateBufferAllocation(w); 

    /* set drawing gc to clip drawing before motif shadow and highlight */
    clipRect.x = borderWidth;
    clipRect.y = borderWidth;
    clipRect.width = w->core.width - 2 * borderWidth;
    clipRect.height = w->core.height - 2 * borderWidth;
    XSetClipRectangles(XtDisplay(w), w->spin.gc, 0, 0, &clipRect, 1, Unsorted);
    
    /* call the resize callback */
    if (XtIsRealized(w))
    	XtCallCallbacks((Widget) w, XmNresizeCallback, NULL);
}

/*
** Widget redisplay method
*/
static void redisplay(SpinWidget w, XEvent *event, Region region)
{
    /* Draw the Motif required shadows and highlights */
    if (w->primitive.shadow_thickness > 0) {
	_XmDrawShadow (XtDisplay(w), XtWindow(w), 
		       w->primitive.bottom_shadow_GC,
		       w->primitive.top_shadow_GC,
                       w->primitive.shadow_thickness,
                       w->primitive.highlight_thickness,
                       w->primitive.highlight_thickness,
                       w->core.width - 2 * w->primitive.highlight_thickness,
                       w->core.height-2 * w->primitive.highlight_thickness);
    }
    if (w->primitive.highlighted)
	_XmHighlightBorder((Widget)w);
    else if (_XmDifferentBackground((Widget)w, XtParent((Widget)w)))
	_XmUnhighlightBorder((Widget)w);
    
    /* Now draw the contents of the spin widget */
    redisplayContents(w);
}

/*
** Widget setValues method
*/
static Boolean setValues(SpinWidget current, SpinWidget request, SpinWidget new)
{
    XGCValues values;
    Boolean redraw = False;
    int i;

    /* If the colors have changed, change the drawing GC */
    if (new->core.background_pixel != current->core.background_pixel) {
    	XSetBackground(XtDisplay(new), new->spin.gc,
    		       new->core.background_pixel);
    	redraw = TRUE;
    }
    if (new->primitive.foreground != current->primitive.foreground) {
    	XSetForeground(XtDisplay(new), new->spin.gc,
    		       new->primitive.foreground);
	redraw = TRUE;  
    }
    /* if labels are changed, free the old ones and copy the new ones */
    for (i=0; i<3; i++) {
	if (new->spin.axisLabels[i] != current->spin.axisLabels[i]) {
    	    if (current->spin.axisLabels[i] != NULL)
    		XmStringFree(current->spin.axisLabels[i]);
    	    new->spin.axisLabels[i] = XmStringCopy(new->spin.axisLabels[i]);
	}
    }
    /* if double buffering changes, allocate or deallocate offscreen pixmap */
    if (new->spin.doubleBuffer != current->spin.doubleBuffer)
    	updateBufferAllocation(new);
    /* redraw to show or remove axes if showAxes has changed */
    if (new->spin.showAxes != current->spin.showAxes)
    	redraw = TRUE;
    /* redraw if perspective is turned on or off */
    if (new->spin.perspectOn != current->spin.perspectOn)
    	redraw = TRUE;
    /* if highlight thickness or shadow thickness changed, resize and redraw */
    if  ((new->primitive.highlight_thickness != 
          current->primitive.highlight_thickness) ||
         (new -> primitive.shadow_thickness !=
          current->primitive.shadow_thickness)) {
    	redraw = TRUE;
        resize (new);
    }
    return (redraw); 
} 

/*
** Button press and button motion action proc.
*/
static void motionAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    double newM[3][3];
    double polarAngle, azimuth, x, y, radius, normFactor, dx, dy, dAzimuth;

    SpinStopSpinning((Widget)w);

    if (event->type == ButtonPress)
#ifdef MOTIF10
    	_XmProcessTraversal(w, XmTRAVERSE_CURRENT);
#else
    	XmProcessTraversal((Widget)w, XmTRAVERSE_CURRENT);
#endif   
    
    if (event->type == ButtonPress || event->type == MotionNotify) {
	/* calculate coordinates of button press normalized to unit circle */
	normFactor = (double)w->spin.drawWidth / 2.;
	x = ((double)(event->xbutton.x - w->spin.centerX))/normFactor;
	y = ((double)(event->xbutton.y - w->spin.centerY))/normFactor;
	/* calculate corresponding point in spherical polar coordinate sys. */
	if (x == 0. && y == 0.)
	    azimuth = w->spin.lastAzimuth;
	else
	    azimuth = atan2(y, x);
	radius = sqrt(x*x + y*y);
	if (radius > 1.) radius = 1.;
	polarAngle = asin(radius);
	/* get the change since the last rotation */
	dx = x - w->spin.lastX;
	dy = y - w->spin.lastY;
	dAzimuth = azimuth - w->spin.lastAzimuth;
	dAzimuth = normalizeAngle(dAzimuth);
	/* Inside of a set radius, damp out rotation along the axis perpendicular
	   to the plane of the window */
	if (radius < NO_ROTATE_RADIUS)
	    /* damp rotation by setting up counter rotation along the line
	       from the center of rotation to the mouse */
	    w->spin.axialCorr = normalizeAngle(w->spin.axialCorr - dAzimuth);
	if (w->spin.dragging) {
	    CopyM(w->spin.startMatrix, newM);
	    ViewRotZ(w->spin.axialCorr, newM);
	    ViewRotY(polarAngle, newM);
	    ViewRotZ(azimuth, newM);
	    CopyM(newM, w->spin.matrix);
	} else {
	    /* just starting the drag operation */
	    CopyM(w->spin.matrix, newM);
	    ViewRotZ(0.-azimuth, newM);
	    ViewRotY(0.-polarAngle, newM);
	    CopyM(newM, w->spin.startMatrix);
	    w->spin.dragging = True;
	    w->spin.axialCorr = 0.;
	}
	/* redraw from new rotated view */
	if (XtIsRealized(w))
    	    redisplayContents(w);
	/* save information for calculating differences next time */
	w->spin.lastAzimuth = azimuth;
	w->spin.lastX = x;
	w->spin.lastY = y;
	/* save information for deciding whether to auto-rotate */
	w->spin.lastDX = dx;
	w->spin.lastDY = dy;
	w->spin.lastDT = event->xbutton.time - w->spin.lastTimeStamp;
	w->spin.lastTimeStamp = event->xbutton.time;
    }  
} 

/*
** Button up action proc.
*/
static void btnUpAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    double x, y, normFactor, dx, dy, radius, angRot, angDist;
    int interval, degPerSec, rotationType;

    if (w->spin.dragging) {
	w->spin.dragging = False;
	interval = (event->xbutton.time - w->spin.lastTimeStamp) +
		   w->spin.lastDT;
	dx = w->spin.lastDX;
	dy = w->spin.lastDY;
	/* find the STARTING x, y by subtracting dx and dy from current pos */
	normFactor = (double)w->core.width / 2.;
	x = ((double)(event->xbutton.x - w->spin.centerX))/normFactor - dx;
	y = ((double)(event->xbutton.y - w->spin.centerY))/normFactor - dy;
	radius = sqrt(x*x + y*y);
	if (radius < NO_ROTATE_RADIUS) {
	    angRot = atan2(dy, dx);
	    angDist = asin(sqrt(dx*dx + dy*dy));
	    rotationType = SPIN_AXIS_XY;
	} else {
	    angRot = 0.;
	    angDist = ((fabs(x)>.5) ? atan(dy/x) : 0)  - 
		      ((fabs(y)>.5) ? atan(dx/y) : 0);
	    rotationType = SPIN_AXIS_Z;
	}
	degPerSec = (double)DEGREES(angDist) / ((double)interval/1000.);
	if (abs(degPerSec) >= MIN_TOSS_SPEED) {
	    SpinStartSpinning((Widget)w, rotationType, (int)DEGREES(angRot),
			      degPerSec);
	}
    }
}

static void btn2AP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    SpinCallbackStruct cbStruct;
    
#ifdef MOTIF10
    	_XmProcessTraversal(w, XmTRAVERSE_CURRENT);
#else
    	XmProcessTraversal((Widget)w, XmTRAVERSE_CURRENT);
#endif   

    /* Just call the callback */
    cbStruct.reason = XmCR_INPUT;
    cbStruct.event = event;
    XtCallCallbacks ((Widget) w, XmNbtn2Callback, &cbStruct);
}

static void btn3AP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    SpinCallbackStruct cbStruct;
    
#ifdef MOTIF10
    	_XmProcessTraversal(w, XmTRAVERSE_CURRENT);
#else
    	XmProcessTraversal((Widget)w, XmTRAVERSE_CURRENT);
#endif   

    /* Just call the callback */
    cbStruct.reason = XmCR_INPUT;
    cbStruct.event = event;
    XtCallCallbacks ((Widget) w, XmNbtn3Callback, &cbStruct);
}

/*
** Arrow key action procs.
*/
static void spinUpAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    keySpin(w, event, UP);
}
static void spinDownAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    keySpin(w, event, DOWN);
}
static void spinLeftAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    keySpin(w, event, LEFT);
}
static void spinRightAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    keySpin(w, event, RIGHT);
}

/*
** Stop spinning action proc. 
*/
static void stopAP(SpinWidget w, XEvent *event, char *args, int n_args)
{
    SpinStopSpinning((Widget)w);
}

/*
** SpinStartSpinning
**
** Make a spin widget begin rotating on its own (use SpinStopSpinning to make it
** stop).  The widget can rotate its display about the view Z axis (the axis
** out of the plane of the window), or about an axis at any given angle in
** the XY plane.
**
** Parameters
**
** 	w		A spin widget
**	rotationType	Specify either SPIN_AXIS_XY or SPIN_AXIS_Z.  SPIN_AXIS_Z
**			selects spinning around the view Z axis (zAngle is
**		        ignored).  SPIN_AXIS_XY selects rotation about an axis
**			in the XY plane.  Set the angle of this axis in zAngle.
**	zAngle		The angle of the rotation axis in the XY plane for
**			SPIN_AXIS_XY rotation.  The angle is measured in degrees
**			from the X axis.
**	degPerSec	The requested rotation speed in degrees per second.
*/
void SpinStartSpinning(Widget w, int rotationType, int zAngle, int degPerSec)
{
    SpinWidget sw = (SpinWidget)w;
    struct timeval tp;
    struct timezone tzp;

    /* Set up the starting matrix and starting time for the rotation proc */
    gettimeofday(&tp, &tzp);
    CopyM(sw->spin.matrix, sw->spin.startMatrix);
    sw->spin.rStartSeconds = tp.tv_sec;
    sw->spin.rStartMilliseconds = tp.tv_usec / 1000;
    sw->spin.rAngle = RADIANS(zAngle);
    sw->spin.rType = rotationType;
    sw->spin.rIncrement = RADIANS(degPerSec) / 1000.; 
    sw->spin.spinning = True;

    /* Set the timeout to activate the rotation procedure, once started, the
       rotation procedure will re-request the timeout until told to stop     */
    sw->spin.timeOutID = XtAppAddTimeOut(XtWidgetToApplicationContext(w),
    		                         MIN_REFRESH_TIME, 
    		                         (XtTimerCallbackProc) rotateProc, w);
}

/*
** SpinStopSpinning
**
** Make a spin widget stop auto rotating
**
** Parameters
**
** 	w		A spin widget
*/
void SpinStopSpinning(Widget w)
{
    SpinWidget sw = (SpinWidget)w;
    
    if (sw->spin.spinning) {
    	sw->spin.spinning = False;
    	XtRemoveTimeOut(sw->spin.timeOutID);
    }
}

/*
** SpinSetScale
**
** Set the scale factor for displaying data in a spin widget.  The coordinate
** system scales with the window size, so that a scale factor of 1.0 results
** in a range viewable of -1 to 1 in X, Y, and Z regardless of window size.
**
** Parameters
**
** 	w		A spin widget
**	scale		A scale factor to apply to all displayed data
*/
void SpinSetScale(Widget w, double scale)
{
    ((SpinWidget)w)->spin.scale = scale;
    if (XtIsRealized(w))
    	redisplayContents((SpinWidget)w);
}

/*
** SpinGetScale
**
** Get the scale factor for displaying data in a spin widget.  The coordinate
** system scales with the window size, so that a scale factor of 1.0 results
** in a range viewable of -1 to 1 in X, Y, and Z regardless of window size.
**
** Parameters
**
** 	w		A spin widget
*/
double SpinGetScale(Widget w)
{
    return ((SpinWidget)w)->spin.scale;
}

/*
** SpinSetSegments
**
** Specify the contents of the spin widget in the form of line segments in
** 3D space.  Takes an array of SpinSegment data structures which contain
** the X,Y, and Z coordinates of the starting and ending points of each
** segment.  This call is mutually exclusive with SpinSetPoints as an
** alternate method of specifying the contents of the widget.  For points
** rather than lines, use SpinSetPoints.
**
** Parameters
**
** 	w		A spin widget
**	segments	An array of line segments to display
**	nSegments	The number of segments specified in segments
*/
void SpinSetSegments(Widget w, SpinSegment *segments, int nSegments)
{
    SpinWidget sw = (SpinWidget)w;
    
    if (sw->spin.nSegments != 0) {
    	XtFree((char *) sw->spin.segments);
    	sw->spin.nSegments = 0;
    }
    if (sw->spin.nPoints != 0) {
    	XtFree((char *) sw->spin.points);
    	sw->spin.nPoints = 0;
    }
    if (nSegments != 0) {
    	sw->spin.segments =
    		(SpinSegment *)XtMalloc(sizeof(SpinSegment) * nSegments);
    	memcpy(sw->spin.segments, segments, sizeof(SpinSegment) * nSegments);
	sw->spin.nSegments = nSegments;
    }
    
    /* Sort segs by color so the drawing routines can batch drawing calls */
    qsort(sw->spin.segments, nSegments, sizeof(SpinSegment), compareSegments);

    if (XtIsRealized(w))
    	redisplayContents(sw);
}

/*
** SpinSetPoints
**
** Specify the contents of the spin widget in the form of poins in 3D space
** measured from the origin.  Takes an array of SpinPoint data structures
** which contain the X,Y, and Z coordinates of the  end points of each
** point.  This call is mutually exclusive with SpinSetSegments.  Use
** SpinSetSegments instead, if you want to display line segments that do not
** start at the origin.
**
** Parameters
**
** 	w		A spin widget
**	points		An array of points to display
**	nPoints		The number of elements specified in points
*/
void SpinSetPoints(Widget w, SpinPoint *points, int nPoints)
{
    SpinWidget sw = (SpinWidget)w;
    
    if (sw->spin.nSegments != 0) {
    	XtFree((char *) sw->spin.segments);
    	sw->spin.nSegments = 0;
    }
    if (sw->spin.nPoints != 0) {
    	XtFree((char *) sw->spin.points);
    	sw->spin.nPoints = 0;
    }
    if (nPoints != 0) {
    	sw->spin.points =
    		(SpinPoint *)XtMalloc(sizeof(SpinPoint) * nPoints);
    	memcpy(sw->spin.points, points, sizeof(SpinPoint) * nPoints);
	sw->spin.nPoints = nPoints;
    }
    
    /* Sort points by color so the drawing routines can batch drawing calls */
    qsort(sw->spin.points, nPoints, sizeof(SpinPoint), comparePoints);
    
    if (XtIsRealized(w))
    	redisplayContents(sw);
}

/*
** SpinSetTransform, SpinGetTransform
**
** Set or get the current 3x3 rotation matrix that the spin widget is using
** to display data.  Note that this matrix does not include scaling.
**
** Parameters
**
** 	w		A spin widget
**	matrix		A 3x3 matrix of type double.  Values are copied into
**			(SpinGetTransform) or out of (SpinSetTransform) the
**			elements of this matrix.
*/
void SpinGetTransform(Widget w, double matrix[3][3])
{
    CopyM(((SpinWidget)w)->spin.matrix, matrix);
}
void SpinSetTransform(Widget w, double matrix[3][3])
{
    CopyM(matrix, ((SpinWidget)w)->spin.matrix);
    if (XtIsRealized(w))
    	redisplayContents((SpinWidget)w);
}

/*
** SpinRestore
**
** Restore the rotation of a spin widget to its initial orientation, Y axis
** pointing up, X axis pointing right, and Z axis pointing out of the page.
**
** Parameters
**
** 	w		A spin widget
*/
void SpinRestore(Widget w)
{
    CopyM(Identity3x3, ((SpinWidget)w)->spin.matrix);
    if (XtIsRealized(w))
    	redisplayContents((SpinWidget)w);
}

/*
** SpinTransformPoint
**
** Transform a point using the current rotation scaling and perspective of the
** spin widget.  Returns the point in (integer) window coordinates.
**
** Parameters
**
** 	w		A spin widget
**	x, y, z		A point in 3D space.
**	xOut, yOut	Returned values for the point mapped to window
**			coordinates.
*/
void SpinTransformPoint(Widget w, double x, double y, double z,
			short *xOut, short *yOut)
{
    SpinWidget sw = (SpinWidget)w;

    double scale = sw->spin.scale * ((double)sw->spin.drawWidth/2.);
    
    x *= scale; y *= scale; z *= scale;
    transform3DtoWindow(sw, x, y, z, sw->spin.matrix, xOut, yOut);
}

/*
** SpinViewRotateX, SpinViewRotateY, SpinViewRotateZ,
** SpinCoordRotateX, SpinCoordRotateY, SpinCoordRotateZ
**
** Rotate the spin widget display.  View routines rotate with respect to
** the current view of the widget, Coord routines rotate with respect to the
** inherent coordinate system of the data.
**
** Parameters
**
**	w		A spin widget
**	degrees		Angle to rotate by in degrees (can be negative)
*/
void SpinViewRotateX(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    ViewRotX(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}
void SpinViewRotateY(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    ViewRotY(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}
void SpinViewRotateZ(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    ViewRotZ(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}
void SpinCoordRotateX(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    CoordRotX(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}
void SpinCoordRotateY(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    CoordRotY(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}
void SpinCoordRotateZ(Widget w, double degrees)
{
    double m[3][3];
    
    SpinGetTransform(w, m);
    CoordRotZ(RADIANS(degrees), m);
    SpinSetTransform(w, m);
}

/*
** SpinRedisplay
**
** Force the Spin widget to repaint it's display.  For programs which draw
** on the spin widget to erase what they've drawn.
**
** Parameters
**
**	w		A spin widget
*/
void SpinRedisplay(Widget w)
{
    if (XtIsRealized(w))
    	redisplayContents((SpinWidget)w);
}

/*
** SpinPrintContents
**
** Prints the contents Spin widget to a PostScript file.
**
** Parameters
**
**	w		A spin widget
**	psFileName	Name for the PostScript file that will be created
**	
*/
void SpinPrintContents(Widget w, char *psFileName)
{
    FILE *ps;

    ps = OpenPS(psFileName, w->core.width, w->core.height);
    if (ps != NULL) {
	redrawContents((SpinWidget)w, PS_PRINTER);
	EndPS();
    }    
}

/*
** SpinCopyGC
**
** Copy the graphics context that the spin widget uses for drawing.  For
** programs that draw on the spin widget.
**
** Parameters
**
**	w		A spin widget
*/
GC SpinCopyGC(Widget w)
{
    GC gc;
    
    gc = XCreateGC(XtDisplay(w), XtWindow(w), 0, NULL);
    XCopyGC(XtDisplay(w), ((SpinWidget)w)->spin.gc, GCForeground|GCBackground|
    	GCFont|GCClipMask|GCClipXOrigin|GCClipYOrigin, gc);
    return gc;
}

/*
** Redisplays the contents part of the widget, without the motif shadows and
** highlights.
*/
static void redisplayContents(SpinWidget w)
{        
     redrawContents(w, X_SCREEN);

    /* Call the redisplay callback so an application which
       draws on the spin widget can refresh it's graphics */
    if (XtIsRealized(w))
    	XtCallCallbacks((Widget) w, XmNredisplayCallback, NULL);
}

/*
** Redisplays the contents part of the widget, without the motif shadows and
** highlights. Outputs to screen or to a PostScript printer file.
*/
static void redrawContents(SpinWidget w, int outDevice)
{
    SpinSegment *seg;
    SpinPoint *pt;
    XSegment *xSegments, *xSeg, *colorSeg;
    XPoint *xPoints, *xPt, *colorPt;
    Display *display = XtDisplay(w);
    Pixel lastPixel;
    Drawable drawBuf;
    GC gc = w->spin.gc;
    int i, width = w->core.width, height = w->core.height;
    int nSegColors = 0, nPtColors = 0;
    double m[3][3];
    struct {
    	XSegment *start;
    	Pixel color;
    	int nSegs;
    } segColors[MAX_COLORS], *segColor;
    struct {
    	XPoint *start;
    	Pixel color;
    	int nPts;
    } ptColors[MAX_COLORS], *ptColor;

    /* Save some energy if the widget isn't visible */
    if (!w->core.visible)
        return;

    /* Set destination for drawing commands, offscreen pixmap or window */
    if (w->spin.doubleBuffer)
    	drawBuf = w->spin.drawBuffer;
    else
    	drawBuf = XtWindow(w);

    /* Copy matrix to save from perspective and scale transformations */
    CopyM(w->spin.matrix, m);

    /* Add scaling (axes are drawn w/same rotation at different scale) */
    ScaleM(w->spin.scale * ((double)w->spin.drawWidth/2.), m);

    /*
    ** Build up lists of XSegments and XPoints to draw.  Also remember
    ** where color changes occur so drawing requests can be batched.
    */
    xSegments = (XSegment *)XtMalloc(sizeof(XSegment)*(w->spin.nSegments));
    xPoints = (XPoint *)XtMalloc(sizeof(XPoint)*(w->spin.nPoints));
    
    xSeg = xSegments;
    lastPixel = -1;
    for (i=0, seg=w->spin.segments; i<w->spin.nSegments; i++, seg++) {
    	if (seg->pixel != lastPixel) {
    	    segColors[nSegColors].start = xSeg;
    	    segColors[nSegColors].color = seg->pixel;
    	    segColors[nSegColors].nSegs = 0;
    	    lastPixel = seg->pixel;
    	    nSegColors++;
    	}
    	transform3DtoWindow(w, seg->x1, seg->y1, seg->z1, m,
    	    		    &xSeg->x1, &xSeg->y1);
    	transform3DtoWindow(w, seg->x2, seg->y2, seg->z2, m,
    	    		    &xSeg->x2, &xSeg->y2);
        segColors[nSegColors-1].nSegs += 1;
        xSeg++;
    }
    xPt = xPoints;
    lastPixel = -1;
    for (i=0, pt=w->spin.points; i<w->spin.nPoints; i++, pt++) {
    	if (pt->pixel != lastPixel) {
    	    ptColors[nPtColors].start = xPt;
    	    ptColors[nPtColors].color = pt->pixel;
    	    ptColors[nPtColors].nPts = 0;
    	    lastPixel = pt->pixel;
    	    nPtColors++;
    	}
    	transform3DtoWindow(w, pt->x,pt->y,pt->z, m, &xPt->x, &xPt->y);
        ptColors[nPtColors-1].nPts += 1;
        xPt++;
    }

    /* Clear the drawing buffer or window */
    if (outDevice == X_SCREEN) {
    	XSetForeground(display, gc, w->core.background_pixel);
        XFillRectangle(display, drawBuf, gc, 0, 0, width, height);
    }

    /* Draw the Axes if they are turned on */
    if (w->spin.showAxes)
    	drawAxes(w, drawBuf, outDevice);

    /* Draw the saved segments and points, one color at a time */
    for (i=0, segColor = segColors; i<nSegColors; i++, segColor++) {
	XSetForeground(display, gc, segColor->color);
        if (outDevice == X_SCREEN) {
	    XDrawSegments(display, drawBuf, gc, segColor->start,
	    		  segColor->nSegs);
        } else if (outDevice == PS_PRINTER) {
            PSDrawSegments(display, drawBuf, gc, segColor->start,
           		   segColor->nSegs);
        }
    }
    XtFree((char *) xSegments);
    for (i=0, ptColor = ptColors; i<nPtColors; i++, ptColor++) {
   	XSetForeground(display, gc, ptColor->color);
        if (outDevice == X_SCREEN)
    	    XDrawPoints(display, drawBuf, gc, ptColor->start, ptColor->nPts, 0);
        else if (outDevice == PS_PRINTER)
            PSDrawPoints(display, drawBuf, gc, ptColor->start, ptColor->nPts,0); 
    }
    XtFree((char *) xPoints);
    
    /* For double buffering, now copy offscreen pixmap to screen */
    if (outDevice == X_SCREEN) {
	if (w->spin.doubleBuffer) {
    	    XCopyArea(XtDisplay(w), drawBuf, XtWindow(w), gc,
    	  	      0, 0, width, height, 0, 0);
	}
    }
}
    	
/*
** This procedure is called when the rotation time out expires.  It rotates
** the rotation matrix and redraws the contents of the widget in the new
** rotated position.  After completing, it re-establishes the time out
** to activate it again after MIN_REFRESH_TIME
*/ 
static void rotateProc(SpinWidget w, XtIntervalId *id)
{
    struct timeval tp;
    struct timezone tzp;
    long dt;
    double newM[3][3];
    
    /* If the widget is no longer supposed to be auto-
       rotating, return and don't reset the time out   */
    if (!w->spin.spinning)
    	return;
    	
    /* Rotate the the widget if it's realized and visible */
    if (XtIsRealized(w) && w->core.visible) {
    	/* Make sure the server is done with the last rotation */
    	XSync(XtDisplay(w), False);
	/* Use the current time to calculate the amount
	   to rotate, so rotation appears smooth 	*/
    	gettimeofday(&tp, &tzp);
	dt = 1000 * (tp.tv_sec - w->spin.rStartSeconds) +
    	     (tp.tv_usec / 1000 - w->spin.rStartMilliseconds);
	CopyM(w->spin.startMatrix, newM);
	if (w->spin.rType == SPIN_AXIS_XY) {
	    ViewRotZ(0.-w->spin.rAngle, newM);
	    ViewRotY(dt*w->spin.rIncrement, newM);
	    ViewRotZ(w->spin.rAngle, newM);
	} else {
	    ViewRotZ(dt*w->spin.rIncrement, newM);
	}
	CopyM(newM, w->spin.matrix);
	/* Redisplay in rotated position */
	redisplayContents(w);
    }
    
    /* Set up the next time out to execute this procedure again */
    w->spin.timeOutID = 
         XtAppAddTimeOut(XtWidgetToApplicationContext((Widget) w),
    					MIN_REFRESH_TIME,
    					(XtTimerCallbackProc)  rotateProc, w);
}

/*
** Process arrow key presses
*/
static void keySpin(SpinWidget w, XEvent *event, int direction)
{
    int rAngle, speed;
    
    if (event->xbutton.state & ShiftMask) {
    	/* shift key pressed means adjust spin speed */
	if (w->spin.spinning) {
	    /* already spinning, adjust roatation speed */
    	    rAngle = DEGREES(normalizeAngle(w->spin.rAngle));
    	    if   (((direction == LEFT) && (rAngle < -90 || rAngle > 90)) ||
    	    	  ((direction == RIGHT) && (rAngle > -90 && rAngle < 90)) ||
    	    	  ((direction == UP) && (rAngle < 0 || rAngle > 180)) ||
    	    	  ((direction == DOWN) && (rAngle > 0 && rAngle < 180))) {
		speed = DEGREES(1000.*w->spin.rIncrement) + KEY_SPEED_CHANGE;
 	    } else {
		speed = DEGREES(1000.*w->spin.rIncrement) - KEY_SPEED_CHANGE;
	    }
	    /* Stop and start again to recalculate with new rotation speed */
	    SpinStopSpinning((Widget)w);
	    if (speed >= KEY_SPEED_CHANGE)
	    	SpinStartSpinning((Widget)w, w->spin.rType, rAngle, speed);
	} else {
    	    if (direction == LEFT) rAngle = 180;
 	    else if (direction == RIGHT) rAngle = 0;
 	    else if (direction == UP) rAngle = -90;
 	    else /* DOWN */ rAngle = 90;
	    SpinStartSpinning((Widget)w, SPIN_AXIS_XY, rAngle, KEY_SPEED_CHANGE);
    	}
    } else {
    	/* shift key not pressed */
    	SpinStopSpinning((Widget)w);
    	if (direction == LEFT)
	    ViewRotY(RADIANS(0 - w->spin.keyRotateDegrees), w->spin.matrix);
 	else if (direction == RIGHT)
    	    ViewRotY(RADIANS(w->spin.keyRotateDegrees), w->spin.matrix);
 	else if (direction == UP)
	    ViewRotX(RADIANS(0- w->spin.keyRotateDegrees), w->spin.matrix);
 	else /* DOWN */
    	    ViewRotX(RADIANS(w->spin.keyRotateDegrees), w->spin.matrix);
	redisplayContents(w);
    }
}

/*
** Put an angle back in the range of -PI to PI.  Don't use this for degrees!
*/
static double normalizeAngle(double angle)
{
    if (angle > PI)
	return angle - 2*PI;
    if (angle < 0.-PI)
	return angle + 2*PI;
    return angle;
}

/*
** Draw the coordinate axes on the display
*/
static void drawAxes(SpinWidget w, Drawable drawBuf, int outDevice)
{
    double m[3][3];
    short tx1, ty1, tx2, ty2;
    Display *display = XtDisplay(w);
    GC gc = w->spin.gc;
    enum {X_ORIENT, Y_ORIENT, Z_ORIENT} orient;
    int i, nLines = 0, width = w->core.width, height = w->core.height;
    SpinSegment *seg;
    static SpinSegment	axisSegments[] =
  	{{0, 0,0,0, .8,0,0}, {0, .8,0,0, .77,0,.008}, {0, .8,0,0, .77,0,-.008},
  	{0, .8,0,0, .77,.008,0}, {0, .8,0,0, .77,-.008,0}};
    static SpinPoint labelLocs[3] = {{0, .8,0,0}, {0, 0,.8,0}, {0, 0,0,.8}};
    XSegment *line, lines[XtNumber(axisSegments)*3];

    XSetForeground(display, gc, w->primitive.foreground);

    /* Draw the axis lines and arrows */
    CopyM(w->spin.matrix, m);
    ScaleM((double)width/2., m);
    line = lines;
    for (orient=X_ORIENT; orient<=Z_ORIENT; orient++) {
    	if (orient == Y_ORIENT)
    	    CoordRotZ(-PI/2, m);
    	if (orient == Z_ORIENT)
    	    CoordRotY(PI/2, m);
	for (seg=axisSegments, i=0; i<XtNumber(axisSegments); i++, seg++) {
    	    transform3DtoWindow(w, seg->x1, seg->y1, seg->z1, m,
    	    			&line->x1, &line->y1);
    	    transform3DtoWindow(w, seg->x2, seg->y2, seg->z2, m,
    	    			&line->x2, &line->y2);
    	    line++;
    	    nLines++;
	}
    }
    if (outDevice == X_SCREEN)
        XDrawSegments(display, drawBuf, gc, lines, nLines);
    else if (outDevice == PS_PRINTER)
        PSDrawSegments(display, drawBuf, gc, lines, nLines);

    /* Draw the axis labels */
    CopyM(w->spin.matrix, m);
    ScaleM((double)width/2., m);
    for (i=0; i<3; i++) {
   	transform3DtoWindow(w, labelLocs[i].x, labelLocs[i].y, labelLocs[i].z, 
    			    m, &tx1, &ty1);
        if (outDevice == X_SCREEN) {
	    XmStringDrawImage(display, drawBuf, w->spin.font,
	    	      w->spin.axisLabels[i], gc, tx1, ty1, 100,
	    	      XmALIGNMENT_BEGINNING, XmSTRING_DIRECTION_L_TO_R,  NULL);
        } else if (outDevice == PS_PRINTER) {
            PSDrawXmString(display, drawBuf, w->spin.font,
	    	      w->spin.axisLabels[i], gc, tx1, ty1, 100,
	    	      XmALIGNMENT_BEGINNING);
        }
    }
}

/*
** Transform 3D coordinates to 2D window coordinates using matrix m with
** no scaling applied.  This does apply perspective if it is turned on in
** the widget.
*/
static void transform3DtoWindow(SpinWidget w, double x, double y, double z, 
				double m[3][3], short *xOut, short *yOut)
{
    double tx, ty, tz;
    
    TransformPoint(x, y, z, &tx, &ty, &tz, m);

    /* X gets wierd when points get near it's precision limit.  If users zoom
       in far enough that points go past this threshold, we must clip them.
       Since this routine deals with points rather than segments, it can't 
       clip entirely correctly.  Lines which are not radial, with endpoints
       that lies outside the clipped area, will be distorted.  (Of course
       the X server would have distorted them even more, so we're still
       better off) */
#define MAXRES (SHRT_MAX/8)
#define MINRES (-SHRT_MAX/8)
    if (w->spin.perspectOn)
    	Perspective(&tx, &ty, &tz, PERSPECTIVE_FACTOR);
    if (tx > MAXRES) {
    	ty *= MAXRES / tx; tx = MAXRES;
    }
    if (ty > MAXRES) {
    	tx *= MAXRES / ty; ty = MAXRES;
    }
    if (tx < MINRES) {
    	ty *= MINRES / tx; tx = MINRES;
    }
    if (ty < MINRES) {
    	tx *= MINRES / ty; ty = MINRES;
    }
    *xOut = w->spin.centerX + (int)tx;
    *yOut = w->spin.centerY - (int)ty;
}

static int compareSegments(SpinSegment *seg1, SpinSegment *seg2)
{
    if (seg1->pixel < seg2->pixel)
    	return -1;
    else if (seg1->pixel == seg2->pixel)
    	return 0;
    else
    	return 1;
}

static int comparePoints(SpinPoint *pt1, SpinPoint *pt2)
{
    if (pt1->pixel < pt2->pixel)
    	return -1;
    else if (pt1->pixel == pt2->pixel)
    	return 0;
    else
    	return 1;
}

static void updateBufferAllocation(SpinWidget w)
{ 
    if (w->spin.drawBuffer)
    	XFreePixmap(XtDisplay(w), w->spin.drawBuffer);
    if (w->spin.doubleBuffer) {
    	w->spin.drawBuffer = XCreatePixmap(XtDisplay(w),
		DefaultRootWindow(XtDisplay(w)), w->core.width, w->core.height,
    	 	DefaultDepthOfScreen(XtScreen(w)));   
    } else {
    	w->spin.drawBuffer = NULL;
    }
}
