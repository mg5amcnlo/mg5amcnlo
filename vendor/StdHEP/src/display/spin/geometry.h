/*******************************************************************************
*									       *
* geometry.h -- Utilities for coordinate transformation			       *
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
* March 18, 1991							       *
*									       *
* Written by Paul Lebrun							       *
*									       *
*******************************************************************************/
/* SCCS ID: geometry.h 1.1 4/6/92 */

/* Constants */
#define PI 3.1415927
#define PIDEGREE 180.

/* Global Variables */
extern double Identity3x3[3][3];

/* Function Prototypes */
void ViewRotX(double theta, double m[3][3]);
void ViewRotY(double theta, double m[3][3]);
void ViewRotZ(double theta, double m[3][3]);
void CoordRotX(double theta, double m[3][3]);
void CoordRotY(double theta, double m[3][3]);
void CoordRotZ(double theta, double m[3][3]);
void Perspective(double *x, double *y, double *z, double pp);
void ScaleM(double factor, double m[3][3]);
void TransformPoint(double x, double y, double z, double *x1, double *y1,
		    double *z1, double m[3][3]);
void MultM(double a[3][3], double b[3][3], double result[3][3]);
void CopyM(double srcM[3][3], double dstM[3][3]);
void CartToPolar(double x, double y, double *theta, double *radius);
void PolarToCart(double theta, double radius, double *x, double *y);
