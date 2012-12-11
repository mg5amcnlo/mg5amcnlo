/*******************************************************************************
*									       *
* geometry.c -- Utilities for coordinate transformation		   	       *
*							      		       *
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
* June 12, 1991	         						       *
*									       *
* by Paul Lebrun & Mark Edel						       *
*									       *
* June 17 : add the Perspective tranformation.				       *
*									       *
*******************************************************************************/
static char SCCSID[] = "@(#)geometry.c	1.1	4/6/92";
#include <stdio.h>
#include <math.h>
#include "geometry.h"

double Identity3x3[3][3] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

/* Function Prototypes */
void genXRotMatrix(double theta, double m[3][3]);
void genYRotMatrix(double theta, double m[3][3]);
void genZRotMatrix(double theta, double m[3][3]);
void genScaleMatrix(double scale, double m[3][3]);

/*
** Routines for generating matricies for rotation and scaling.  The matrix
** parameter to each takes a 3x3 array which the caller has allocated.
*/
void genXRotMatrix(double theta, double m[3][3])
{
    m[0][0] = 1.0;	m[0][1] = 0.;		m[0][2] = 0.;
    m[1][0] = 0.;	m[1][1] = cos(theta);	m[1][2] = sin(theta);
    m[2][0] = 0.;	m[2][1] = 0. - m[1][2];	m[2][2] = m[1][1];
}
void genYRotMatrix(double theta, double m[3][3])
{
    m[0][0] = cos(theta);	m[0][1] = 0.;	m[0][2] = 0. - sin(theta);
    m[1][0] = 0.;		m[1][1] = 1.;	m[1][2] = 0.;
    m[2][0] = 0. - m[0][2];	m[2][1] = 0.;	m[2][2] = m[0][0];
}
void genZRotMatrix(double theta, double m[3][3])
{
    m[0][0] = cos(theta);	m[0][1] = sin(theta);	m[0][2] = 0.;
    m[1][0] = 0. - m[0][1];	m[1][1] = m[0][0];	m[1][2] = 0.;
    m[2][0] = 0.;		m[2][1] = 0.;		m[2][2] = 1.;
}
void genScaleMatrix(double scale, double m[3][3])
{
    m[0][0] = scale;	m[0][1] = 0.;	    m[0][2] = 0.;
    m[1][0] = 0.;	m[1][1] = scale;    m[1][2] = 0.;
    m[2][0] = 0.;	m[2][1] = 0.;	    m[2][2] = scale;
}

/*
** ViewRotX, ViewRotY, ViewRotZ, CoordRotX, CoordRotY, CoordRotZ, ScaleM
**
** These routines take an angle or a scale factor, and apply a transformation
** to matrix m.  The ViewRot... routines apply a rotation relative the the
** current orientation implied by the matrix.  The CoordRot... routines apply
** a rotation relative to the actual axes of the data.
**
** Parameters
**
**	theta, scale	An amount in radians to rotate by, or a scale factor
**			to apply.
**	m		A matrix to apply the transformation to.
*/
void ViewRotX(double theta, double m[3][3])
{
    double a[3][3];

    genXRotMatrix(theta, a);
    MultM(a, m, m);   
}
void ViewRotY(double theta, double m[3][3])
{
    double a[3][3];

    genYRotMatrix(theta, a);
    MultM(a, m, m);   
}
void ViewRotZ(double theta, double m[3][3])
{
    double a[3][3];

    genZRotMatrix(theta, a);
    MultM(a, m, m);   
}
void CoordRotX(double theta, double m[3][3])
{
    double a[3][3];

    genXRotMatrix(theta, a);
    MultM(m, a, m);   
}
void CoordRotY(double theta, double m[3][3])
{
    double a[3][3];

    genYRotMatrix(theta, a);
    MultM(m, a, m);   
}
void CoordRotZ(double theta, double m[3][3])
{
    double a[3][3];

    genZRotMatrix(theta, a);
    MultM(m, a, m);   
}
void ScaleM(double factor, double m[3][3])
{
    double a[3][3];

    genScaleMatrix(factor, a);
    MultM(m, a, m);   
}

/*
** Perspective
**
** Apply a perspective transformation to a point.  This should be called only
** after rotation and scaling have been applied.
**
** Parameters
**
**	x, y, z		Point to be adjusted
**	pp		Amount of perspective adjustment to apply.  This value
**			can range from 0.0 to 1.0 with 1 being the most severe
*/
void Perspective(double *x, double *y, double *z, double pp)
{
    double ss, lpt;

    lpt = sqrt(*x**x + *y**y + *z**z); 
    /* if point is less than one pixel from the origin,
       ignore it to keep the z/lpt term from blowing up */
    if (lpt < 1.)
        return;   
    ss = 1. - (*z/lpt) * pp;
    *x *= ss;
    *y *= ss; 
}

/*
** TransformPoint
**
** Apply matrix M to a point (multiply the vector x, y, z by matrix m).
**
** Parameters
**
**	x, y, z		A point to transform.
**	x1, y1, z1	The resulting point.
**	m		The matrix to multiply by.
*/
void TransformPoint(double x, double y, double z, double *x1, double *y1,
		    double *z1, double m[3][3])
{
        *x1 = x * m[0][0] + y * m[0][1] + z * m[0][2];
        *y1 = x * m[1][0] + y * m[1][1] + z * m[1][2];
        *z1 = x * m[2][0] + y * m[2][1] + z * m[2][2];
}

/*
** MultM
**
** Multiply two 3x3 matricies
**
** Parameters
**
**	a, b		The two 3x3 matricies to multiply.
**	result		A 3x3 matrix to hold the result. Allocated by the caller.
*/
void MultM(double a[3][3], double b[3][3], double result[3][3])
{
     int i, j;
     double at[3][3];

    /* c could be equal to a or b, place the result in a temporary array.. */
    for (i=0; i<=2; i++) {
       for (j=0; j<=2; j++) 
          at[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2]*b[2][j];
    }
    CopyM(at, result);
}

/*
** CopyM
**
** Copy a 3x3 matrix
**
** Parameters
**
**	srcM	Matrix to copy from.
**	dstM	Matrix to copy into.  Allocated by the caller.
*/
void CopyM(double srcM[3][3], double dstM[3][3])
{
    int i, j;
    
    for (i=0; i<=2; i++)
	for (j=0; j<=2; j++) 
            dstM[i][j] = srcM[i][j];
}

/*
** CartToPolar, PolarToCart
**
** Transform cartesian coordinates to polar coordinates and visa versa
**
** Parameters
**
**	x, y		Cartesian coordinates
**	theta, radius	Polar coordinates
*/
void CartToPolar(double x, double y, double *theta, double *radius)
{
    if (x == 0.) {
	*radius = fabs(y);
	*theta = PI/2;
	if (y < 0.) *theta = 3*PI/2;
    } else {
	*theta = atan(y/x);
	if (x < 0.) *theta = *theta+ PI;
	*radius = sqrt(x*x + y*y);
    }
}
void PolarToCart(double theta, double radius, double *x, double *y)
{
    *x = radius * cos(theta);
    *y = radius * sin(theta);
}
