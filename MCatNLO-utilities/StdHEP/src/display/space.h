/*******************************************************************************
*									       *
* phase.h -- Nirvana Phase Space Event Display				       *
*									       *
* Copyright (c) 1992 Universities Research Association, Inc.		       *
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
* Modification History:							       *
*									       *
*	5/19/92 - JMK - Add XDR filter routines for structures defined here    *
*									       *
*******************************************************************************/
/* SCCS ID: phase.h 1.2 5/22/92 */


typedef struct _SpaceVertex {
    float x,y,z; /* the x, ,y, z Origin of a particle */
    float time; /* Production time */
} SpaceVertex;

#define SLIDER_MAX 1000         /* Arbitrary max value for animation */

typedef void *EventSpaceWindow;

EventSpaceWindow *DisplayEventFileSp(Display *display, void (*exitProc)());
