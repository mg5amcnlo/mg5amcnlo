/*******************************************************************************
*									       *
* Spin.h - Spin Widget (3D "virtual sphere" display) Public Header File	       *
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
* May 10, 1991								       *
*									       *
* Written by Mark Edel							       *
*									       *
*******************************************************************************/
/* SCCS ID: Spin.h 1.2 5/1/92 */

#ifndef  SPIN_H
#define  SPIN_H

enum SpinAxisType {SPIN_AXIS_XY, SPIN_AXIS_Z};

#define MAX_COLORS 256		/* Maximum # of colors widget can display */

/* Resource strings */
#define XmNkeyRotateDegrees "keyRotateDegrees"
#define XmCKeyRotateDegrees "KeyRotateDegrees"
#define XmNdoubleBuffer "doubleBuffer"
#define XmCDoubleBuffer "DoubleBuffer"
#define XmNperspectiveOn "perspectiveOn"
#define XmCPerspectiveOn "PerspectiveOn"
#define XmNshowAxes "showAxes"
#define XmCShowAxes "ShowAxes"
#define XmNxAxisLabel "xAxisLabel"
#define XmCXAxisLabel "XAxisLabel"
#define XmNyAxisLabel "yAxisLabel"
#define XmCYAxisLabel "YAxisLabel"
#define XmNzAxisLabel "zAxisLabel"
#define XmCZAxisLabel "ZAxisLabel"
#define XmNbtn2Callback "btn2Callback"
#define XmCBtn2Callback "Btn2Callback"
#define XmNbtn3Callback "btn3Callback"
#define XmCBtn3Callback "Btn3Callback"
#define XmNredisplayCallback "redisplayCallback"
#define XmCRedisplayCallback "RedisplayCallback"

extern WidgetClass spinWidgetClass;

typedef struct _SpinClassRec *SpinWidgetClass;
typedef struct _SpinRec *SpinWidget;

typedef struct _SpinSegment {
    Pixel pixel;
    double x1, y1, z1, x2, y2, z2;
} SpinSegment;

typedef struct _SpinPoint {
    Pixel pixel;
    double x, y, z;
} SpinPoint;

typedef struct {
    int     reason;
    XEvent *event;
} SpinCallbackStruct;

void SpinSetSegments(Widget w, SpinSegment *segments, int nSegments);
void SpinSetPoints(Widget w, SpinPoint *points, int nPoints);
void SpinStartSpinning(Widget w, int rotationType, int zAngle, int degPerSec);
void SpinStopSpinning(Widget w);
void SpinGetTransform(Widget w, double matrix[3][3]);
void SpinSetTransform(Widget w, double matrix[3][3]);
void SpinRestore(Widget w);
void SpinSetScale(Widget w, double scale);
double SpinGetScale(Widget w);
void SpinTransformPoint(Widget w, double x, double y, double z,
			short *xOut, short *yOut);
void SpinRedisplay(Widget w);
GC SpinCopyGC(Widget w);
void SpinViewRotateX(Widget w, double degrees);
void SpinViewRotateY(Widget w, double degrees);
void SpinViewRotateZ(Widget w, double degrees);
void SpinCoordRotateX(Widget w, double degrees);
void SpinCoordRotateY(Widget w, double degrees);
void SpinCoordRotateZ(Widget w, double degrees);
void SpinPrintContents(Widget w, char *PSfname);

#endif
