/*******************************************************************************
*									       *
* SpinP.h - Spin Widget (3D "virtual sphere" display) Public Header File	       *
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
* July 23, 1991								       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
/* SCCS ID: SpinP.h 1.2 5/1/92 */
#ifndef SPINP_H
#define SPINP_H

#include "Spin.h"
#include <Xm/XmP.h>
#if XmVersion >= 1002
#include <Xm/PrimitiveP.h>
#endif

typedef struct _SpinClassPart{
    int ignore;
} SpinClassPart;

typedef struct _SpinClassRec{
    CoreClassPart  core_class;
    XmPrimitiveClassPart primitive_class;
    SpinClassPart  spin_class;
} SpinClassRec;

extern SpinClassRec spinClassRec;

typedef struct _SpinPart {
    int centerX;		/* coordinates of the center of the window   */
    int centerY;
    int drawWidth;		/* width of drawable area of widget */	
    GC gc;	   	       	/* the main graphics context */
    double scale;
    int keyRotateDegrees;	/* how far to rotate with each arrow press */
    double matrix[3][3];	/* The rotation matrix */
    double startMatrix[3][3];	/* Saved matrix from start of user mouse drag
    				   normalized to polarAngle = 0, azimuth = 0 */
    Pixmap drawBuffer;		/* double buffering for non-flashing draws */
    double axialCorr;
    double lastAzimuth;
    double lastX;
    double lastY;
    long rStartSeconds;
    int rStartMilliseconds;
    double rAngle;
    int rType;
    double rIncrement;
    double lastDX;
    double lastDY;
    double lastDT;		/* in milliseconds */
    XtIntervalId timeOutID;
    Time lastTimeStamp;
    XmFontList font;
    SpinSegment *segments;	/* Displayed object expressed as segments */
    int nSegments;		/* Number of segments */
    SpinPoint *points;		/* Object expressed as lines from 0,0,0 */
    int nPoints;		/* Number of points */
    double perspect;		/* The Perspective parameter, related to distance
			       	   of the observer to the center of the event */
    XmString axisLabels[3];	/* Compound string labels for axes */
    XtCallbackList resize;	/* callbacks */
    XtCallbackList btn2;
    XtCallbackList btn3;
    XtCallbackList redisplay;
    Boolean dragging;		/* Set at the start of a mouse drag */
    Boolean spinning;		/* Set when auto rotating */
    Boolean perspectOn;		/* Set if perspective viewing is on */
    Boolean doubleBuffer;	/* When set, draw first to offscreen pixmap */
    Boolean showAxes;		/* When set, display the coordinate axes */
} SpinPart;

typedef struct _SpinRec {
   CorePart        core;
   XmPrimitivePart primitive;
   SpinPart        spin;
} SpinRec;

#endif /* SPINP_H */
