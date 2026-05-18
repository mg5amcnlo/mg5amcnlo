/*******************************************************************************
*									       *
* copy_stdevent.c -- copy from stdevent.h to stdhep.h                           *
*									       *
* Copyright (c) 1995 Universities Research Association, Inc.		       *
* All rights reserved.							       *
* 									       *
* This material resulted from work developed under a Government Contract and   *
* is subject to the following license:  The Government retains a paid-up,      *
* nonexclusive, irrevocable worldwide license to reproduce, prepare derivative *
* works, perform publicly and display publicly by or for the Government,       *
* including the right to distribute to other Government contractors.  Neither  *
* the United States nor the United States Department of Energy, nor any of     *
* their employees, makes any warranty, express or implied, or assumes any      *
* legal liability or responsibility for the accuracy, completeness, or         *
* usefulness of any information, apparatus, product, or process disclosed, or  *
* represents that its use would not infringe privately owned rights.           *
*                                        				       *
*									       *
* Written by Lynn Garren    					       	       *
*									       *
*									       *
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
/* 
*   StdHep definitions and include files
*/
#include "stdhep.h"
#include "stdevent.h"

int copy_stdevent_( void )
{
    int nh, i, k, icol;
    if (hepevt_.nhep + stdevent_.nhep > NMXHEP) {
        fprintf(stderr,
          "     copy_stdevent: event would overflow HEPEVT array size\n");
        fprintf(stderr,"     copy_stdevent: event %d has been lost\n",
               stdevent_.nevhep);
        return 5;
    }
    hepevt_.nevhep = stdevent_.nevhep;
    nh = hepevt_.nhep;
    for (i = 0; i < stdevent_.nhep; i++) {
        hepevt_.isthep[nh+i] = stdevent_.isthep[i];
        hepevt_.idhep[nh+i] = stdevent_.idhep[i];
        hepevt_.jmohep[nh+i][0] = stdevent_.jmohep1[i];
        hepevt_.jmohep[nh+i][1] = stdevent_.jmohep2[i];
        hepevt_.jdahep[nh+i][0] = stdevent_.jdahep1[i];
        hepevt_.jdahep[nh+i][1] = stdevent_.jdahep2[i];
        for (k = 0; k < 4; k++)
            hepevt_.phep[nh+i][k] = stdevent_.phep[i][k];
        hepevt_.phep[nh+i][5] = stdevent_.mass[i];
        for (k = 0; k < 4; k++)
            hepevt_.vhep[nh+i][k] = stdevent_.vcr[i][k];
        hepev2_.jmulti[nh+i] = stdevent_.jcol[i];
    }
    hepev2_.nmulti = stdevent_.ncol;
    if (hepev2_.nmulti <= NMXMLT ) {
        icol = 1;
        for (i = 0; i < stdevent_.ncol; i++) {
	    hepev3_.nevmulti[i] = stdevent_.nevcol[i];
	    hepev3_.itrkmulti[i] = icol;
	    hepev3_.mltstr[i] = stdevent_.iostr[i];
	    icol += stdevent_.nhepcol[i];
	}
    } else {
	fprintf(stderr," copy_stdevent: %d multiple interactions in this event\n",
	     hepev2_.nmulti );  
	fprintf(stderr," copy_stdevent: only %d multiple interactions are allowed\n",
	     NMXMLT );  
    }
    return 0;
}
