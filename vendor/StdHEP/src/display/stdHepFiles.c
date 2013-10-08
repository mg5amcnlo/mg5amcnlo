/*******************************************************************************
*									       *
* stdHepFiles.c -- Nirvana Phase Space Event Display, Accessing the StdHep     *
*                  files and records					       *   
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
* January 28 1994							       *
*									       *
* Written by Paul Lebrun						       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>		/* only for SpinSegment reqd by drawEvent.h */
#include "spin/Spin.h"
#include "util/getfiles.h"
#include "util/DialogF.h"
#include "util/misc.h"
#include "phase.h"
#include "space.h"
#include "phaseP.h"
#include "drawEvent.h"
#include "stdhep.h"
#include "stdhep_mcfio.h"

/* Function declarations to get the STDHEP, Fortran */

extern void opennatstdhep_(char *file, int *ierr, int l);
extern void closestdhep_();

/*
** Check the a STDHEP file is readable, and see if it has some
**  events.  the menu bar
*/
int OpenCheckXdrStdHep(char *filename, int nmax, int istr)
{
   int nn, i, lok, ierr, flag;
   
   ierr = StdHepXdrReadInit(filename, nmax, istr);
   nn = 0;
   if (ierr != 0 ) return nn;
   /* get the first nmax events */
   for (i=0; i <= nmax; i++) {
     lok = StdHepXdrRead(&flag, istr);
     if (lok != 0) {
       StdHepXdrEnd(istr);
       return nn;
       }
     nn ++;
   }
   StdHepXdrEnd(istr);
   return nn;
}   

void OpenGetStdHep(StdHepWindow *window, int nevt)
{
   int i, j, k, ierr,  l;
   int istr, flag;
   int lok = 0;
   char * ctmp;
   SpaceWindow *wins;
   SpaceVertex *vertex, *vTempReal, *vtt, *vtt2 ;
   PhaseParticle *particle;
   double xvv, yvv, zvv, dxsq, dysq, dzsq, dsq, dtemp, dmax;
   double pmax;

/* Scrap the previous event, if there */
   
   if (window->event.particles != NULL) {
     XtFree((char *)window->event.particles);
     if ((PhaseWindow *) window->selnodeTracks != NULL) 
             XtFree((char *)(PhaseWindow *) window->selnodeTracks);
     window->selnodeNumTrack =0;
     window->event.eventNum = 0;
     window->event.nParticles = 0;
     window->event.particles = NULL;
     if (window->type == STDHEP_SPACE) {
       wins = (SpaceWindow *) window;
       XtFree((char *)wins->vertices); XtFree((char *)wins->realVertices);
       wins->numRealVertices = 0;
     }
   }
   
/* Open the file again, and read until we find the one we want.. */

   l = strlen(window->filename);
   switch(window->filemode) {
    case XDR_STDHEP:
      istr = window->xdrStream;
      ierr = StdHepXdrReadOpen(window->filename, nevt, istr);
      break;
    default :
     printf (" Internal Error, Inconsistent file mode \n");
     return;
    }
   if (ierr != 0 ) return;
   for (i=0; i < nevt; i++) {
    if (window->filemode == XDR_STDHEP) {
       flag = 0;	/* reset flag before getting next event */
       while ((flag != 1 && flag != 2 && flag !=4 && flag != 5 ) && lok == 0) {
          lok = StdHepXdrRead(&flag, istr);
          }
       }
    if (lok != 0) {
       if (window->filemode == XDR_STDHEP) 
          StdHepXdrEnd(istr);
       else
          closestdhep_();
       return;
       }
   }
   if (window->filemode == XDR_STDHEP) 
     StdHepXdrEnd(istr);
   else
     closestdhep_();
/*
* Copy the buffer..
*/
   if (hepevt_.nhep <= 0 ) return;
   window->event.eventNum = nevt;
   window->event.userData = hepevt_.nevhep;
   window->event.nParticles = hepevt_.nhep;
   window->event.particles = (PhaseParticle *) 
   XtMalloc (hepevt_.nhep * sizeof(PhaseParticle));
   particle = window->event.particles;
   for (i=0; i< hepevt_.nhep; i++, particle++) {
       particle->id = hepevt_.idhep[i];
       particle->stable = False;
       if (hepevt_.isthep[i] == 1) particle->stable = True;
       particle->px = hepevt_.phep[i][0];
       particle->py = hepevt_.phep[i][1];
       particle->pz = hepevt_.phep[i][2];
       particle->mass = hepevt_.phep[i][4];
       particle->mother = hepevt_.jmohep[i][0];
       particle->firstdaughter = hepevt_.jdahep[i][0];
       particle->lastdaughter = hepevt_.jdahep[i][1];
       particle->userData = i;
     }
     if (window->type == STDHEP_SPACE) { 
     /* 
     * Load Vertex information
     */
       wins = (SpaceWindow *) window;
       wins->vertices = (SpaceVertex *) 
          XtMalloc (hepevt_.nhep * sizeof(SpaceVertex));
      vertex = wins->vertices;
      for (i=0; i< hepevt_.nhep; i++, vertex++) {
        vertex->x = hepevt_.vhep[i][0];
        vertex->y = hepevt_.vhep[i][1];
        vertex->z = hepevt_.vhep[i][2];
        vertex->time = hepevt_.vhep[i][3];
      }
      /*
      ** Set now reasonable values for slider ranges. Start by computing 
      ** the real number of vertices. If two vertices are separate by less 
      ** 1 micron, call it one vertex.. 
      */
      vertex = wins->vertices;
      dmax = 0.;
      wins->numRealVertices = 1;
      vTempReal = (SpaceVertex *) 
          XtMalloc (hepevt_.nhep * sizeof(SpaceVertex));
      vtt = vTempReal;
      vtt->x = hepevt_.vhep[0][0];
      vtt->y = hepevt_.vhep[0][1];
      vtt->z = hepevt_.vhep[0][2];  
      vtt->time = hepevt_.vhep[0][3];
      for (i=0; i < hepevt_.nhep; i++, vertex++) {
        xvv = vertex->x; yvv = vertex->y; zvv = vertex->z;
        k = 1;
        vtt2 = vTempReal; 
        for (j=0; j<wins->numRealVertices; j++, vtt2++) {
          dxsq = (xvv - vtt2->x)*(xvv - vtt2->x);
          dysq = (yvv - vtt2->y)*(yvv - vtt2->y);
          dzsq = (zvv - vtt2->z)*(zvv - vtt2->z);
          dsq = dxsq + dysq + dzsq;
          if (dsq < 1.e-12) {
            k= 0;
            break;
          }
         }
         if (k == 1) {
            dtemp = sqrt(dsq);
            if (dtemp > dmax ) dmax = dtemp;
            wins->numRealVertices++; vtt++;
            vtt->x = hepevt_.vhep[i][0];
            vtt->y = hepevt_.vhep[i][1];
            vtt->z = hepevt_.vhep[i][2];  
            vtt->time = hepevt_.vhep[i][3];
            }
      }
      /* Realloc and copy to shorten that buffer */
      wins->realVertices = (SpaceVertex *) 
          XtMalloc (wins->numRealVertices * sizeof(SpaceVertex));
      vtt = vTempReal; vtt2 = wins->realVertices; 
        for (j=0 ; j<wins->numRealVertices; j++, vtt++, vtt2++) {
      	  vtt2->x = vtt->x;  vtt2->y = vtt->y;  vtt2->z = vtt->z;
      	  vtt2->time = vtt->time;
      	}
      XtFree((char *)vTempReal);
      /*
      ** Compute stuff to set the scale now... At minimum, the real world 
      ** coordinate will arbitrarily set to 5 cm 
      **  ( in case only 1 vertex shows up )
      */
      if (dmax < 0.5) dmax = 1.;       
      dmax = 5.0 * dmax;
      wins->minTransl[2] = -dmax/2.; wins->maxTransl[2] = dmax/2.;
      wins->currentTransl[2] = 0.;
      wins->currentScale = dmax;
      wins->maxScale = dmax;
      /*
      ** Set now the Transverse to Longitudinal aspect ratio. 
      ** Currently, at the outset, this is a fixed quantity.. 
      */
      wins->minLongToTr = 1.;  wins->maxLongToTr = 10. * DEFAULTLONGTOTR;
      wins->currentLongToTr = DEFAULTLONGTOTR; 
      for (k=0; k<2; k++){   
       wins->minTransl[k] = -1. *  dmax/( 2. * wins->currentLongToTr); 
       wins->maxTransl[k] = dmax/( 2. * wins->currentLongToTr);
       wins->currentTransl[k] = 0.;
      }
      /*
      ** Loop over thetracks, compute the largest momentum.
      ** in order to guess the appropriate gev to cm conversion factor.
      */
       pmax = 0.;
       particle = window->event.particles;
       for (i=0; i< hepevt_.nhep; i++, particle++) {
         dsq = ParticleMomentum(particle->px, particle->py, particle->pz);
         if (dsq > pmax) pmax = dsq;
        }
      pmax = pmax * 2.5;
      wins->maxMomentum = pmax;
      wins->currentMomToSpace = 2.0 * wins->maxTransl[2] /pmax;
      wins->maxMomToSpace = 10. *  wins->currentMomToSpace;
              
     }
     return;
}   

void GetDetectorSketch(SpaceWindow *window)
{
    int i, j, ifile, npts, ok, nSegments;
    char filename[255];
    char line[132];
    char *ll;
    FILE *fileD;
    float x1, y1, z1, x2, y2, z2;
    SpinSegment *spdt;
    Pixel black; /* For possible expansion ( I hate this.. ) */
    

    /*
    ** Scrap the previous detector sketch
    **
    */    
    if (window->detectorSegments != NULL)
         XtFree((char *)window->detectorSegments);
    window->nDetectorSegments = 0;
    ifile = GetExistingFilename(window->shell, 
                     "Detector Sketch File", filename);
    if (ifile == GFN_CANCEL) {
       return;
    }
    /*
    ** Scan the file once, to read the number of segments one has to 
    **	allocate.
    */
    fileD = fopen(filename, "r");
    nSegments = 0;
    ok = True;
    while (ok) {
      ll = fgets(line, 131, fileD);
      if (ll == NULL) ok = False;
      if (ferror(fileD) != 0) {
        perror(" Error Reading Detector Sketch file, check file or format ");
        ok = False;
        }
      npts = sscanf(line, " %f %f %f %f %f %f ", 
            &x1, &y1, &z1, &x2,  &y2, &z2);
      if (ok && (npts == 6)) nSegments++;
    }
    fclose(fileD);
    if (nSegments == 0) return;
    window->nDetectorSegments = nSegments;
    window->detectorSegments = 
           (SpinSegment *)XtMalloc(sizeof(SpinSegment) * nSegments);
    ok = True;
    spdt = window->detectorSegments;
    fileD = fopen(filename, "r");
    while (ok) {
      ll = fgets(line, 131, fileD);
      if (ll == NULL) ok = False;
      npts = sscanf(line, " %f %f %f %f %f %f ", 
            &x1, &y1, &z1, &x2,  &y2, &z2);
      if (ok && (npts == 6)) {
        spdt->x1 = (double) x1;
        spdt->y1 = (double) y1;
        spdt->z1 = (double) z1;
        spdt->x2 = (double) x2;
        spdt->y2 = (double) y2;
        spdt->z2 = (double) z2;
        spdt++;
       }
    }
    fclose(fileD);
} 
    

void LoadColorCodeEvent(PhaseWindow *window)
{
	PhaseParticle  *pt;
	
	window->colorcode.eventNum = -9999;
	window->colorcode.nParticles = 36;
	window->colorcode.userData = 0;
	window->colorcode.particles = 
	(PhaseParticle *) XtMalloc(sizeof(PhaseParticle) *
	                 window->colorcode.nParticles);  
        pt = window->colorcode.particles;
        
        /* Top Level : a gluino, a neutral Higgs, a charged Higgs, a gluon */ 
        		 
	pt->id = 47; pt->mother = 0; pt->firstdaughter = 0; pt++;
	pt->id = 25; pt->mother = 0; pt->firstdaughter = 5; pt++;
	pt->id = 33; pt->mother = 0; pt->firstdaughter = 14; pt++;
	pt->id = 9; pt->mother = 0; pt->firstdaughter = 23; pt++;
	
        /* 4 Part. next : Electrweak Bosons from neutral Higgs  */ 
        		 
	pt->id = -24; pt->mother = 2; pt->firstdaughter = 8; pt++;
	pt->id = 24; pt->mother = 2; pt->firstdaughter = 10; pt++;
	pt->id = 23; pt->mother = 2; pt->firstdaughter = 12; pt++;
	pt->id = 16; pt->mother = 5; pt->firstdaughter = 0; pt++;
	pt->id = -15; pt->mother = 5; pt->firstdaughter = 0; pt++;
	pt->id = 14; pt->mother = 6; pt->firstdaughter = 0; pt++;
	pt->id = 13; pt->mother = 6; pt->firstdaughter = 0; pt++;
	pt->id = 11; pt->mother = 7; pt->firstdaughter = 0; pt++;
	pt->id = -11; pt->mother = 7; pt->firstdaughter =0; pt++;
	
        /* 13 Part., next,
          Electrweak Bosons from Charged Higgs: into bare quarks.  */ 
        		 
	pt->id = -24; pt->mother = 3; pt->firstdaughter = 17; pt++;
	pt->id = 24; pt->mother = 3; pt->firstdaughter = 19; pt++;
	pt->id = 24; pt->mother = 3; pt->firstdaughter = 21; pt++;
	pt->id = 5; pt->mother = 14; pt->firstdaughter = 0; pt++;
	pt->id = 6; pt->mother = 14; pt->firstdaughter = 0; pt++;
	pt->id = 4; pt->mother = 15; pt->firstdaughter = 0; pt++;
	pt->id = 3; pt->mother = 15; pt->firstdaughter = 0; pt++;
	pt->id = 2; pt->mother = 16; pt->firstdaughter = 0; pt++;
	pt->id = 1; pt->mother = 16; pt->firstdaughter =0; pt++;
	
        /* 22 Part, next : Gluon fragments into B Bbar, B decay 
        		First B...  */ 
        		 
	pt->id = -511; pt->mother = 4; pt->firstdaughter = 25; pt++;
	pt->id = 511; pt->mother = 4; pt->firstdaughter = 31; pt++;
	pt->id = 443; pt->mother = 23; pt->firstdaughter = 27; pt++;
	pt->id = 310; pt->mother = 23; pt->firstdaughter = 28; pt++;
	pt->id = -13; pt->mother = 25; pt->firstdaughter = 0; pt++;
	pt->id = 13; pt->mother = 25; pt->firstdaughter = 0; pt++;
	pt->id = 211; pt->mother = 26; pt->firstdaughter = 0; pt++;
	pt->id = -211; pt->mother = 26; pt->firstdaughter = 0; pt++;

        /* 30 Part. defined, the second B decays */
        
	pt->id = 421; pt->mother = 24; pt->firstdaughter = 33; pt++;
	pt->id = 111; pt->mother = 24; pt->firstdaughter = 35; pt++;
	pt->id = -321; pt->mother = 31; pt->firstdaughter = 0; pt++;
	pt->id = -211; pt->mother = 31; pt->firstdaughter = 0; pt++;
	pt->id = 22; pt->mother = 32; pt->firstdaughter = 0; pt++;
	pt->id = 22; pt->mother = 32; pt->firstdaughter = 0; pt++;
}
