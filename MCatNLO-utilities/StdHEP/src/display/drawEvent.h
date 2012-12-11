/*******************************************************************************
*									       *
* drawEvent.h -- Map the event - ie., the list of four-momentum vectors	to an  *
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
/* SCCS ID: drawEvent.h 1.1 4/6/92 */
void DrawEvent(StdHepWindow *window, int setScale);
int ParticleToSegment(PhaseWindow *window, PhaseParticle *p,
		      SpinSegment *seg, double *length);
int TrackToSegment(SpaceWindow *window,  PhaseParticle *p, SpaceVertex *v,
		      SpinSegment *seg, double *length);
void VertexToSegment(SpaceWindow *window, SpaceVertex *v, 
		      SpinSegment **seg);
void SetDisplayMode(PhaseWindow *window, int newMode);
void DrawScale(StdHepWindow *window);
void AllocateColors(Screen *screen);
void SetPinVertex();
int ParticleVisible(StdHepWindow *window, long id, int stable);
double ParticleMomentum(double px, double py, double pz);
double ParticlePT(double px, double py);
double ParticleRapidity(double px, double py, double pz, double mass);
double ParticlePseudorapidity(double px, double py, double pz);
int ParticleColorIndex(int id);

#define MAXPARTCOLOR 30             /* Size of particle color table */
#define NUMPINSPHI 8
#define NUMPINSTETA 7
#define DEFAULTLONGTOTR 10.
#define PINSIZE 0.005
#define EXPPINTR 1.5

extern Pixel ParticleColors[MAXPARTCOLOR];
extern double XPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
extern double YPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
extern double ZPinVertexPoints[NUMPINSPHI][NUMPINSTETA];
