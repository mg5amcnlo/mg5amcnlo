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

typedef struct _PhaseParticle {
    long id;
    int stable;
    double px, py, pz, mass;
    int mother; 	/* the Mother index */
    int firstdaughter;
    int lastdaughter;
    long userData;
} PhaseParticle;

typedef struct _PhaseEvent {
    int eventNum;
    int nParticles;
    long userData;
    PhaseParticle *particles;
} PhaseEvent;

typedef void *EventWindow;

EventWindow *DisplayOneEvent(Display *display, PhaseEvent *event,
    char *windowTitle, int mode,  void (*exitProc)());
EventWindow *DisplayEventFile(Display *display, void (*exitProc)(),
    int (*openProc)(char *filename),
    void (*closeProc)(FILE *fs), PhaseEvent *(*getEventProc)(int eventNum));

#ifdef XDR_FILTERS_HERE

static bool_t xdr_PhaseEvent(XDR *xdrs, PhaseEvent *event);
static bool_t xdr_PhaseParticle(XDR *xdrs, PhaseParticle *particles);

static bool_t xdr_PhaseEvent(XDR *xdrs, PhaseEvent *event)
{
    int i;
    
    if (xdrs->x_op == XDR_FREE)		/* nothing to free */
    	return (TRUE);
    if     (xdr_int(xdrs, &event->eventNum) &&
            xdr_int(xdrs, &event->nParticles) &&
            xdr_long(xdrs, &event->userData) &&
            xdr_long(xdrs, &event->particles)	) {
        if (xdrs->x_op == XDR_DECODE) {
            event->particles = (PhaseParticle *) malloc((event->nParticles)
        		* sizeof(PhaseParticle));
	}
    }
    else
        return (FALSE);
    for (i = 0; i < event->nParticles; ++i) {
    	if (!xdr_PhaseParticle(xdrs, &event->particles[i]))
            return (FALSE);
    }
    return (TRUE);
}

static bool_t xdr_PhaseParticle(XDR *xdrs, PhaseParticle *particles)
{
    return (xdr_long(xdrs, &particles->id) && 
            xdr_int(xdrs, &particles->stable) &&
            xdr_double(xdrs, &particles->px) &&
            xdr_double(xdrs, &particles->py) &&
            xdr_double(xdrs, &particles->pz) &&
            xdr_long(xdrs, &particles->userData)  );
}
#endif	/* XDR_FILTERS_HERE */
